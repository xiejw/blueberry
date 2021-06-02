#include <stdio.h>

// bb
#include "bb.h"

// -----------------------------------------------------------------------------
// Helper Prototype.
// -----------------------------------------------------------------------------

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
        struct vm_t*            vm  = bbVmInit();
        struct bb_context_t     ctx = {.is_training = 1};
        struct bb_program_t*    p   = bbProgNew();
        sds_t                   s   = sdsEmpty();
        struct bb_seq_module_t* m   = bbSeqModuleNew();

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
            &m->layers));

        m->loss = vecPopBack(m->layers);

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
