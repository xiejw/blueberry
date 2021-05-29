#include <stdio.h>

// bb
#include "bb.h"

int
main()
{
        printf("hello bb.\n");
        struct vm_t*         vm    = bbVmInit();
        struct bb_context_t  ctx   = {.is_training = 1};
        struct srng64_t*     r     = NULL;
        struct bb_layer_t*   dense = NULL;
        struct bb_program_t* p     = NULL;
        vec_t(int) inputs          = vecNew();
        vec_t(int) outputs         = vecNew();

        error_t err = bbDenseLayer(
            vm,
            &(struct bb_dense_config_t){.input_dim   = 10,
                                        .output_dim  = 20,
                                        .kernel_init = BB_INIT_STD_NORMAL,
                                        .bias_init   = BB_INIT_ZERO,
                                        .actn        = BB_ACTN_RELU},
            &dense);

        if (err) {
                errDump("failed to create dense layer\n");
                goto cleanup;
        }

        r = srng64New(123);

        err = dense->ops->init(dense, &ctx, r);
        if (err) {
                errDump("failed to init dense layer\n");
                goto cleanup;
        }

        struct shape_t* sp = R2S(vm, 32, 10);
        int             x  = vmTensorNew(vm, F32, sp);

        p = bbProgNew();
        vecPushBack(inputs, x);

        err = dense->ops->jit(dense, &ctx, p, BB_FORWARD, inputs, &outputs);
        if (err) {
                errDump("failed to jit dense layer\n");
                goto cleanup;
        }

cleanup:

        vecFree(inputs);
        vecFree(outputs);
        if (p) bbProgFree(p);
        if (r) srng64Free(r);
        if (dense) {
                dense->ops->release(dense, &ctx);
                free(dense);
        };
        vmFree(vm);
        return OK;
}
