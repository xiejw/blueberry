#include "bb.h"

#define SWAP(x, y) \
        t   = (x); \
        (x) = (y); \
        (y) = t;
#define CLEAR(x) vecSetSize((x), 0)

error_t
bbCompileSeqModule(const struct bb_context_t *ctx, struct bb_program_t *p,
                   int x, int y, vec_t(struct bb_layer_t *) layers,
                   struct bb_layer_t *loss, struct bb_opt_t *opt,
                   struct srng64_t *r)
{
        size_t  num_layers = vecSize(layers);
        error_t err        = OK;
        vec_t(int) inputs  = vecNew();
        vec_t(int) outputs = vecNew();
        vec_t(int) t;

        vecPushBack(p->inputs, x);
        vecPushBack(p->labels, y);

        // init all layers. num_layers + 1.
        for (int i = 0; i <= num_layers; i++) {
                struct bb_layer_t *l = i < num_layers ? layers[i] : loss;

                err = l->ops.init(l, ctx, r);
                if (err) {
                        errEmitNote("failed to init %d-th layer", i);
                        goto cleanup;
                }
                err = l->ops.weights(l, &p->weights);
                if (err) {
                        errEmitNote("failed to get weights of %d-th layer", i);
                        goto cleanup;
                }
                err = l->ops.grads(l, &p->grads);
                if (err) {
                        errEmitNote("failed to get grads of %d-th layer", i);
                        goto cleanup;
                }
        }

        // init optimizer.
        err = bbOptInit(opt, p->weights, p->grads);
        if (err) {
                errEmitNote("failed to init optimizer.");
                goto cleanup;
        }

        vecPushBack(inputs, x);

        int direction = BB_FORWARD;
        for (int i = 0; i < num_layers; i++) {
                struct bb_layer_t *l = layers[i];
                err = l->ops.jit(l, ctx, p, direction, inputs, &outputs);
                if (err) {
                        errEmitNote("failed to jit %d-th layer", i);
                        goto cleanup;
                }

                if (i != num_layers - 1) {
                        SWAP(inputs, outputs);
                        CLEAR(outputs);
                }
        }

        assert(vecSize(outputs) == 1);

        CLEAR(inputs);
        vecPushBack(inputs, y);
        vecPushBack(inputs, outputs[0]);

        CLEAR(outputs);

        struct bb_layer_t *l = loss;
        err = l->ops.jit(l, ctx, p, direction, inputs, &outputs);
        if (err) {
                errEmitNote("failed to jit loss");
                goto cleanup;
        }

        SWAP(outputs, p->outputs);

        direction = BB_BACKWARD;
        CLEAR(outputs);
        CLEAR(inputs);
        vecPushBack(inputs, 1);  // start grads as ones (td: 1).
        err = l->ops.jit(l, ctx, p, direction, inputs, &outputs);

        for (int i = num_layers - 1; i >= 0; i--) {
                SWAP(inputs, outputs);
                CLEAR(outputs);

                struct bb_layer_t *l = layers[i];
                err = l->ops.jit(l, ctx, p, direction, inputs, &outputs);
                if (err) {
                        errEmitNote("failed to jit %d-th layer", i);
                        goto cleanup;
                }
        }

        err = bbOptApply(opt, p);
        if (err) {
                errEmitNote("failed to apply optimizer.");
                goto cleanup;
        }
cleanup:
        vecFree(inputs);
        vecFree(outputs);
        return err;
}
