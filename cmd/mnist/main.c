#include <stdio.h>
#include <string.h>

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
        struct bb_layer_t*   scel  = NULL;
        struct bb_program_t* p     = NULL;
        vec_t(int) outputs1        = vecNew();
        sds_t s                    = sdsEmpty();

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

        r = srng64New(123);
        p = bbProgNew();

        struct shape_t* sp = R2S(vm, 32, 10);
        int             x  = vmTensorNew(vm, F32, sp);
        int             y  = vmTensorNew(vm, F32, sp);
        vecPushBack(p->inputs, x);
        vecPushBack(p->labels, y);

        {
                err = dense->ops.init(dense, &ctx, r);
                if (err) {
                        errDump("failed to init dense layer\n");
                        goto cleanup;
                }
                err = scel->ops.init(scel, &ctx, r);
                if (err) {
                        errDump("failed to init scel layer\n");
                        goto cleanup;
                }
                err = dense->ops.weights(dense, &p->weights);
                if (err) {
                        errDump("failed to obtain dense layer weights\n");
                        goto cleanup;
                }
                err = scel->ops.weights(scel, &p->weights);
                if (err) {
                        errDump("failed to obtain scel  layer weights\n");
                        goto cleanup;
                }
                err = dense->ops.grads(dense, &p->grads);
                if (err) {
                        errDump("failed to obtain dense layer grads\n");
                        goto cleanup;
                }
                err = scel->ops.grads(scel, &p->grads);
                if (err) {
                        errDump("failed to obtain scel layer grads\n");
                        goto cleanup;
                }
        }

        err = dense->ops.jit(dense, &ctx, p, BB_FORWARD, p->inputs, &outputs1);
        if (err) {
                errDump("failed to jit dense layer\n");
                goto cleanup;
        }

        // concat the labels + dense layouer outputs.
        vec_t(int) loss_inputs = vecNew();
        size_t total_size      = vecSize(outputs1) + vecSize(p->labels);
        size_t size_1          = vecSize(p->labels);
        vecReserve(loss_inputs, total_size);

        memcpy(loss_inputs, p->labels, sizeof(int) * size_1);
        memcpy(loss_inputs + size_1, outputs1,
               sizeof(int) * (total_size - size_1));
        vecSetSize(loss_inputs, total_size);

        err =
            scel->ops.jit(scel, &ctx, p, BB_FORWARD, loss_inputs, &p->outputs);
        if (err) {
                errDump("failed to jit dense layer\n");
                goto cleanup;
        }

        bbProgDump(p, &s);
        printf("%s", s);

cleanup:

        sdsFree(s);
        vecFree(outputs1);
        if (p) bbProgFree(p);
        if (r) srng64Free(r);
        if (dense) bbLayerFree(dense);
        if (scel) bbLayerFree(scel);
        vmFree(vm);
        return OK;
}
