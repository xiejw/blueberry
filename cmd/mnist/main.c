#include <stdio.h>
#include <string.h>

// bb
#include "bb.h"

// -----------------------------------------------------------------------------
// Helper Prototype.
// -----------------------------------------------------------------------------

// Experimental way to create layers.
//
// Thoughts: This creates a bunch of layers. It is good, but only for sequential
// layers. For residual network, how to express?
#define BB_TAG_NULL  0
#define BB_TAG_DENSE 1
#define BB_TAG_SCEL  2

struct bb_layer_config_t {
        int   tag;
        void* config;
};

static error_t bbCreateLayers(struct vm_t*              vm,
                              struct bb_layer_config_t* layer_configs,
                              vec_t(struct bb_layer_t*) * layers);

#define NE(err) _NE_IMPL(err, __FILE__, __LINE__)

#define _NE_IMPL(err, file, line)                                          \
        if ((err)) {                                                       \
                errDump("unexected error at file %s line %d", file, line); \
                goto cleanup;                                              \
        }

// -----------------------------------------------------------------------------
// MNIST.
// -----------------------------------------------------------------------------
#define TOTOL_IMAGES 60000
#define BATCH_SIZE   32
#define IMAGE_SIZE   (28 * 28)
#define LABEL_SIZE   (10)

// -----------------------------------------------------------------------------
// Main.
// -----------------------------------------------------------------------------
int
main()
{
        printf("hello bb.\n");
        struct vm_t*        vm           = bbVmInit();
        struct bb_context_t ctx          = {.is_training = 1};
        vec_t(struct bb_layer_t*) layers = vecNew();
        struct bb_program_t* p           = bbProgNew();
        sds_t                s           = sdsEmpty();

        struct bb_seq_module_t* m = malloc(sizeof(struct bb_seq_module_t));
        memset(m, 0, sizeof(struct bb_seq_module_t));

        m->r = srng64New(123);

        struct shape_t* sp_x = R2S(vm, BATCH_SIZE, IMAGE_SIZE);
        struct shape_t* sp_y = R2S(vm, BATCH_SIZE, LABEL_SIZE);

        m->x = vmTensorNew(vm, F32, sp_x);
        m->y = vmTensorNew(vm, F32, sp_y);

        NE(bbCreateLayers(
            vm,
            (struct bb_layer_config_t[]){
                {
                    .tag = BB_TAG_DENSE,
                    .config =
                        &(struct bb_dense_config_t){
                            .input_dim   = (IMAGE_SIZE),
                            .output_dim  = 64,
                            .kernel_init = BB_INIT_STD_NORMAL,
                            .bias_init   = BB_INIT_ZERO,
                            .actn        = BB_ACTN_RELU},
                },
                {
                    .tag = BB_TAG_DENSE,
                    .config =
                        &(struct bb_dense_config_t){
                            .input_dim   = 64,
                            .output_dim  = 64,
                            .kernel_init = BB_INIT_STD_NORMAL,
                            .bias_init   = BB_INIT_ZERO,
                            .actn        = BB_ACTN_RELU},
                },
                {
                    .tag = BB_TAG_DENSE,
                    .config =
                        &(struct bb_dense_config_t){
                            .input_dim   = 64,
                            .output_dim  = LABEL_SIZE,
                            .kernel_init = BB_INIT_STD_NORMAL,
                            .bias_init   = BB_INIT_NULL,
                            .actn        = BB_ACTN_NONE},
                },
                {.tag = BB_TAG_SCEL,
                 .config =
                     &(struct bb_scel_config_t){.reduction = BB_REDUCTION_SUM}},
                {.tag = BB_TAG_NULL}},
            &layers));

        m->loss = layers[vecSize(layers) - 1];    // Take the loss out.
        vecSetSize(layers, vecSize(layers) - 1);  // Shrink size.
        m->layers = layers;                       // Move ownership

        NE(bbOptNew(vm, BB_OPT_SGD, 0.005, &m->opt));
        NE(bbAUCMetric(vm, &m->metric));
        NE(bbCompileSeqModule(&ctx, p, m));

        bbProgDump(p, &s);
        printf("%s", s);

        float auc;
        NE(m->metric->ops.summary(m->metric, &auc, BB_FLAG_RESET));
        printf("auc: %f", auc);

cleanup:
        sdsFree(s);
        bbSeqModuleFree(m);
        bbProgFree(p);
        vmFree(vm);
        return OK;
}

// -----------------------------------------------------------------------------
// Helper impl.
// -----------------------------------------------------------------------------

error_t
bbCreateLayers(struct vm_t* vm, struct bb_layer_config_t* layer_configs,
               vec_t(struct bb_layer_t*) * layers)
{
        error_t            err;
        struct bb_layer_t* layer;

        struct bb_layer_config_t* curr = layer_configs;
        while (curr->tag != BB_TAG_NULL) {
                switch (curr->tag) {
                case BB_TAG_DENSE:
                        err = bbDenseLayer(
                            vm, (struct bb_dense_config_t*)curr->config,
                            &layer);
                        if (err) {
                                return errEmitNote("failed to create layer.");
                        }
                        vecPushBack(*layers, layer);
                        break;
                case BB_TAG_SCEL:
                        err = bbSCELLayer(
                            vm, (struct bb_scel_config_t*)curr->config, &layer);
                        if (err) {
                                return errEmitNote("failed to create layer.");
                        }
                        vecPushBack(*layers, layer);
                        break;
                default:
                        return errNew("config tag is not supported: %d",
                                      curr->tag);
                }
                curr++;
        }
        return OK;
}
