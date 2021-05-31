#include "bb.h"

error_t
moduleSeqNew(const struct bb_context_t *ctx, struct bb_program_t *p, int x,
             int y, vec_t(struct bb_layer_t *) layers, struct bb_layer_t *loss,
             struct bb_opt_t *opt, struct srng64_t *r)
{
        size_t  num_layers = vecSize(layers);
        error_t err        = OK;

        // init all layers. num_layers + 1 (optimizer).
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

cleanup:
        return err;
}
