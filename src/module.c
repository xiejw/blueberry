#include "bb.h"

#include <stdio.h>
#include <string.h>
#include "opt/fn.h"

#define SWAP(x, y) \
        t   = (x); \
        (x) = (y); \
        (y) = t;
#define CLEAR(x) vecSetSize((x), 0)

struct bb_seq_module_t *
bbSeqModuleNew()
{
        struct bb_seq_module_t *m = malloc(sizeof(struct bb_seq_module_t));
        if (m == NULL) return NULL;
        memset(m, 0, sizeof(struct bb_seq_module_t));
        return m;
}

void
bbSeqModuleFree(struct bb_seq_module_t *m)
{
        if (m == NULL) return;

        bbLayerFree(m->loss);
        bbOptFree(m->opt);
        bbLayerFree(m->metric);
        srng64Free(m->r);

        for (size_t i = 0; i < vecSize(m->layers); i++) {
                bbLayerFree(m->layers[i]);
        }
        vecFree(m->layers);
        free(m);
}

error_t
bbCompileSeqModule(const struct bb_context_t *ctx, struct bb_program_t *p,
                   struct bb_seq_module_t *m)
{
        int x                             = m->x;
        int y                             = m->y;
        vec_t(struct bb_layer_t *) layers = m->layers;
        struct bb_layer_t *loss           = m->loss;
        struct bb_opt_t   *opt            = m->opt;
        struct bb_layer_t *metric         = m->metric;
        struct srng64_t   *r              = m->r;

        assert(x != 0 && y != 0);
        assert(layers != NULL);
        assert(loss != NULL);
        assert(opt != NULL);
        assert(metric != NULL);
        assert(r != NULL);

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
                err = l->ops.states(l, &p->states);
                if (err) {
                        errEmitNote("failed to get states of %d-th states", i);
                        goto cleanup;
                }
        }

        // init optimizer.
        err = bbOptInit(opt, p->weights, p->grads);
        if (err) {
                errEmitNote("failed to init optimizer.");
                goto cleanup;
        }

        vecExtend(p->states, opt->states);

        // init metric.
        err = metric->ops.init(metric, ctx, r);
        if (err) {
                errEmitNote("failed to init metric.");
                goto cleanup;
        }
        err = metric->ops.states(metric, &p->states);
        if (err) {
                errEmitNote("failed to get states of metrics.");
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

        err = metric->ops.jit(metric, ctx, p, direction, inputs, NULL);
        if (err) {
                errEmitNote("failed to jit metricoss");
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

        struct bb_fn_t *fn = bbFnNew();

        vecExtend(fn->inputs, p->inputs);
        vecExtend(fn->inputs, p->labels);
        vecExtend(fn->inputs, p->weights);
        vecExtend(fn->inputs, p->states);

        vecExtend(fn->outputs, p->weights);
        vecExtend(fn->outputs, p->grads);
        vecExtend(fn->outputs, p->states);
        vecExtend(fn->outputs, p->outputs);

#define SWAP_INST_LIST()                                   \
        {                                                  \
                struct bb_inst_list_t tmp = fn->inst_list; \
                fn->inst_list             = p->inst_list;  \
                p->inst_list              = tmp;           \
        }

        SWAP_INST_LIST();

        int debug = 1;
        int changed;
        if (runDCEPass(fn, NULL, debug, &changed)) {
                errFatalAndExit1("something wrong.");
        }
        if (runMathPass(fn, NULL, debug, &changed)) {
                errFatalAndExit1("something wrong.");
        }

        SWAP_INST_LIST();
        bbFnFree(fn);

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
