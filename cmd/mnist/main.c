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
// Main.
// -----------------------------------------------------------------------------
int
main()
{
        printf("hello bb.\n");
        struct vm_t*        vm           = bbVmInit();
        struct bb_context_t ctx          = {.is_training = 1};
        struct srng64_t*    r            = NULL;
        vec_t(struct bb_layer_t*) layers = vecNew();
        struct bb_layer_t*   loss_layer  = NULL;
        struct bb_program_t* p           = NULL;
        sds_t                s           = sdsEmpty();

        r = srng64New(123);
        p = bbProgNew();

        struct shape_t* sp = R2S(vm, 32, 10);

        int x = vmTensorNew(vm, F32, sp);
        int y = vmTensorNew(vm, F32, sp);

        NE(bbCreateLayers(
            vm,
            (struct bb_layer_config_t[]){
                {
                    .tag = BB_TAG_DENSE,
                    .config =
                        &(struct bb_dense_config_t){
                            .input_dim   = 10,
                            .output_dim  = 20,
                            .kernel_init = BB_INIT_STD_NORMAL,
                            .bias_init   = BB_INIT_ZERO,
                            .actn        = BB_ACTN_RELU},
                },
                {.tag = BB_TAG_SCEL,
                 .config =
                     &(struct bb_scel_config_t){.reduction = BB_REDUCTION_SUM}},
                {.tag = BB_TAG_NULL}},
            &layers));

        // Take the loss out.
        loss_layer = layers[vecSize(layers) - 1];
        vecSetSize(layers, vecSize(layers) - 1);

        NE(bbCompileSeqModule(&ctx, p, x, y, layers, loss_layer, NULL, r));

        bbProgDump(p, &s);
        printf("%s", s);

cleanup:

        sdsFree(s);
        for (size_t i = 0; i < vecSize(layers); i++) {
                bbLayerFree(layers[i]);
        }
        bbLayerFree(loss_layer);
        vecFree(layers);
        bbProgFree(p);
        srng64Free(r);
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
