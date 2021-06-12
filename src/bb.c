#include "bb.h"

#include <assert.h>
#include <string.h>

#include "rng/srng64.h"

// -----------------------------------------------------------------------------
// Impl for VM.
// -----------------------------------------------------------------------------

struct vm_t *
bbVmInit()
{
        struct vm_t *vm = vmNew();
        if (vm == NULL) return NULL;

        // allocate some common tensors.
        // TD: 0, scalar, zero.
        // TD: 1, scalar, one.
        {
                struct shape_t *sp = R1S(vm, 1);
                int             td = vmTensorNew(vm, F32, sp);
                assert(td == 0);

                error_t err = vmExec(vm, OP_FILL, NULL, td, -1, -1);
                if (err) {
                        errFatalAndExit("init vm failed: %d", err);
                }

                td = vmTensorNew(vm, F32, sp);
                assert(td == 1);

                struct opopt_t opt = {.mode = OPT_MODE_F_BIT, .f = 1.0};
                err                = vmExec(vm, OP_FILL, &opt, td, -1, -1);
                if (err) {
                        errFatalAndExit("init vm failed: %d", err);
                }
        }
        return vm;
}

// -----------------------------------------------------------------------------
// Impl for Layer.
// -----------------------------------------------------------------------------
void
bbLayerFree(struct bb_layer_t *p)
{
        p->ops.release(p);
        free(p);
}

error_t
bbCreateLayers(struct vm_t *vm, const struct bb_layer_config_t *layer_configs,
               vec_t(struct bb_layer_t *) * layers)
{
        error_t            err;
        struct bb_layer_t *layer;

        const struct bb_layer_config_t *curr = layer_configs;
        while (curr->tag != BB_TAG_NULL) {
                switch (curr->tag) {
                case BB_TAG_DENSE:
                        err = bbDenseLayer(
                            vm, (const struct bb_dense_config_t *)curr->config,
                            &layer);
                        if (err) {
                                return errEmitNote("failed to create layer.");
                        }
                        vecPushBack(*layers, layer);
                        break;
                case BB_TAG_SCEL:
                        err = bbSCELLayer(
                            vm, (const struct bb_scel_config_t *)curr->config,
                            &layer);
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

// -----------------------------------------------------------------------------
// Impl for Helpers.
// -----------------------------------------------------------------------------
error_t
_bbLayerWeights(struct bb_layer_t *this, vec_t(int) * tds)
{
        return vecExtend(*tds, this->weights);
}

error_t
_bbLayerGrads(struct bb_layer_t *this, vec_t(int) * tds)
{
        return vecExtend(*tds, this->grads);
}

error_t
_bbLayerStates(struct bb_layer_t *this, vec_t(int) * tds)
{
        return vecExtend(*tds, this->states);
}

error_t
_bbLayerRelease(struct bb_layer_t *this)
{
        struct vm_t *vm = this->vm;

#define RELEAE_TDS(tds)                                          \
        {                                                        \
                vec_t(int) handles = (tds);                      \
                int     size       = vecSize(handles);           \
                error_t err;                                     \
                for (int i = 0; i < size; i++) {                 \
                        err = vmTensorFree(vm, handles[i]);      \
                        if (err) {                               \
                                return errEmitNote(              \
                                    "failed to release tensor"); \
                        }                                        \
                }                                                \
                vecFree(handles);                                \
                (tds) = vecNew();                                \
        }

        RELEAE_TDS((this->weights));
        RELEAE_TDS((this->grads));
        RELEAE_TDS((this->states));
        RELEAE_TDS((this->ivs));
#undef RELEAE_TDS
        return OK;
}
