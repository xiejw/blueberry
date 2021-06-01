#include <stdio.h>
#include <string.h>

// bb
#include "bb.h"

int
main()
{
        printf("hello bb.\n");
        struct vm_t*        vm           = bbVmInit();
        struct bb_context_t ctx          = {.is_training = 1};
        struct srng64_t*    r            = NULL;
        struct bb_layer_t*  dense        = NULL;
        struct bb_layer_t*  scel         = NULL;
        vec_t(struct bb_layer_t*) layers = vecNew();
        struct bb_program_t* p           = NULL;
        sds_t                s           = sdsEmpty();

        r = srng64New(123);
        p = bbProgNew();

        struct shape_t* sp = R2S(vm, 32, 10);

        int x = vmTensorNew(vm, F32, sp);
        int y = vmTensorNew(vm, F32, sp);

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

        err = bbSCELLayer(
            vm, &(struct bb_scel_config_t){.reduction = BB_REDUCTION_SUM},
            &scel);
        if (err) {
                errDump("failed to create scel layer\n");
                goto cleanup;
        }

        vecPushBack(layers, dense);
        err = compileSeqModule(&ctx, p, x, y, layers, scel, NULL, r);
        if (err) {
                errDump("failed to compile module\n");
                goto cleanup;
        }

        bbProgDump(p, &s);
        printf("%s", s);

cleanup:

        sdsFree(s);
        for (size_t i = 0; i < vecSize(layers); i++) {
                bbLayerFree(layers[i]);
        }
        vecFree(layers);
        bbLayerFree(scel);
        bbProgFree(p);
        srng64Free(r);
        vmFree(vm);
        return OK;
}
