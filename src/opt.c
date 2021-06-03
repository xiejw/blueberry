#include "bb.h"

#include <string.h>

// -----------------------------------------------------------------------------
// Prototype.
// -----------------------------------------------------------------------------

static error_t _bbOptSGDInit(struct bb_opt_t *opt);
static error_t _bbOptSGDApply(struct bb_opt_t *opt, struct bb_program_t *p);
static error_t _bbOptRMSPropInit(struct bb_opt_t *opt);
static error_t _bbOptRMSPropApply(struct bb_opt_t *opt, struct bb_program_t *p);

// -----------------------------------------------------------------------------
// Public APIs.
// -----------------------------------------------------------------------------
error_t
bbOptNew(struct vm_t *vm, int type, float32_t lr, void *cfg,
         struct bb_opt_t **out)
{
        struct bb_opt_t *opt = malloc(sizeof(struct bb_opt_t));
        memset(opt, 0, sizeof(struct bb_opt_t));

        opt->lr     = lr;
        opt->vm     = vm;
        opt->type   = type;
        opt->config = cfg;

        *out = opt;
        return OK;
}

error_t
bbOptInit(struct bb_opt_t *opt, vec_t(int) weights, vec_t(int) grads)
{
        if (vecSize(weights) != vecSize(grads))
                return errNew("opt expects len(weights) == len(grads).");

        opt->weights = weights;
        opt->grads   = grads;

        switch (opt->type) {
        case BB_OPT_SGD:
                return _bbOptSGDInit(opt);
        case BB_OPT_RMSPROP:
                return _bbOptRMSPropInit(opt);
        default:
                return errNew("opt type not supported in init (yet): %d",
                              opt->type);
        }
}

error_t
bbOptApply(struct bb_opt_t *opt, struct bb_program_t *p)
{
        switch (opt->type) {
        case BB_OPT_SGD:
                return _bbOptSGDApply(opt, p);
        case BB_OPT_RMSPROP:
                return _bbOptRMSPropApply(opt, p);
        default:
                return errNew("opt type not supported in apply (yet): %d",
                              opt->type);
        }
}

void
bbOptFree(struct bb_opt_t *opt)
{
        if (opt == NULL) return;
        struct vm_t *vm = opt->vm;

        {  // release states.
                vec_t(int) handles = opt->states;
                int size           = vecSize(handles);
                for (int i = 0; i < size; i++) {
                        vmTensorFree(vm, handles[i]);
                }
                vecFree(handles);
        }

        {  // release ivs.
                vec_t(int) handles = opt->ivs;
                int size           = vecSize(handles);
                for (int i = 0; i < size; i++) {
                        vmTensorFree(vm, handles[i]);
                }
                vecFree(handles);
        }

        free(opt->config);
        free(opt->private_data);
        free(opt);
}

// -----------------------------------------------------------------------------
// SGD Impl.
// -----------------------------------------------------------------------------

struct bb_opt_sgd_t {
        size_t weights_count;
};

static error_t
_bbOptSGDInit(struct bb_opt_t *opt)
{
        struct bb_opt_sgd_t *data = malloc(sizeof(struct bb_opt_sgd_t));

        // we will reuse the grads. so record the count only here.
        int weights_count   = vecSize(opt->weights);
        data->weights_count = weights_count;

        opt->private_data = data;
        return OK;
}

static error_t
_bbOptSGDApply(struct bb_opt_t *opt, struct bb_program_t *p)
{
        size_t weights_count =
            ((struct bb_opt_sgd_t *)opt->private_data)->weights_count;
        vec_t(int) weights   = opt->weights;
        vec_t(int) grads     = opt->grads;
        struct opopt_t opopt = {.mode = OPT_MODE_F_BIT, .f = opt->lr};

        // g = lr * g
        // w = w - g

        for (size_t i = 0; i < weights_count; i++) {
                bbProgAppend(p, &(struct oparg_t){OP_MUL, grads[i], grads[i],
                                                  -1, 1, opopt});
                bbProgAppend(p, &(struct oparg_t){OP_MINUS, weights[i],
                                                  weights[i], grads[i], 0});
        }
        return OK;
}

// -----------------------------------------------------------------------------
// RMSProp Impl
// -----------------------------------------------------------------------------
struct bb_opt_rmsprop_t {
        size_t weights_count;
};

error_t
_bbOptRMSPropInit(struct bb_opt_t *opt)
{
        struct bb_opt_rmsprop_t *data = malloc(sizeof(struct bb_opt_rmsprop_t));

        int weights_count   = vecSize(opt->weights);
        data->weights_count = weights_count;

        // need one state.
        vecReserve(opt->states, weights_count);
        vecSetSize(opt->states, weights_count);
        // need one iv
        vecReserve(opt->ivs, weights_count);
        vecSetSize(opt->ivs, weights_count);

        error_t         err;
        struct vm_t *   vm = opt->vm;
        struct shape_t *sp;
        for (size_t i = 0; i < weights_count; i++) {
                err = vmTensorInfo(vm, opt->weights[i], /*dtype=*/NULL, &sp);
                if (err) return errEmitNote("failed to obtain weight shape.");
                int state = vmTensorNew(vm, F32, sp);
                err       = vmExec(vm, OP_FILL, NULL, state, -1, -1);
                if (err) return errEmitNote("failed to zero opt state.");
                opt->states[i] = state;
                opt->ivs[i]    = vmTensorNew(vm, F32, sp);
        }

        opt->private_data = data;
        return OK;
}

error_t
_bbOptRMSPropApply(struct bb_opt_t *opt, struct bb_program_t *p)
{
        size_t weights_count =
            ((struct bb_opt_sgd_t *)opt->private_data)->weights_count;

        // s = rho * s
        // t1 = g * g
        // t1 = (1 - rho) * t1
        // s = s + t1
        // t1 = 1/sqrt(s + epsilon)
        // t1 = t1 * lr
        // t1 = t1 * g
        // w = w - t1

        struct bb_opt_rmsprop_config_t *cfg =
            (struct bb_opt_rmsprop_config_t *)opt->config;

        assert(cfg != NULL);
        float rho = cfg->rho;
        assert(rho < 1 && rho > 0);
        float epsilon = cfg->epsilon;
        assert(epsilon > 0);

        struct opopt_t opopt_rho         = {.mode = OPT_MODE_F_BIT, .f = rho};
        struct opopt_t opopt_1_minus_rho = {.mode = OPT_MODE_F_BIT,
                                            .f    = (1 - rho)};
        struct opopt_t opopt_epsilon = {.mode = OPT_MODE_F_BIT, .f = epsilon};
        struct opopt_t opopt_lr      = {.mode = OPT_MODE_F_BIT, .f = opt->lr};

        vec_t(int) weights = opt->weights;
        vec_t(int) grads   = opt->grads;
        vec_t(int) states  = opt->states;
        vec_t(int) ivs     = opt->ivs;

        for (size_t i = 0; i < weights_count; i++) {
                bbProgAppend(p, &(struct oparg_t){OP_MUL, states[i], states[i],
                                                  -1, 1, opopt_rho});
                bbProgAppend(p, &(struct oparg_t){OP_MUL, ivs[i], grads[i],
                                                  grads[i], 0});
                bbProgAppend(p, &(struct oparg_t){OP_MUL, ivs[i], ivs[i], -1, 1,
                                                  opopt_1_minus_rho});
                bbProgAppend(p, &(struct oparg_t){OP_ADD, states[i], states[i],
                                                  ivs[i], 0});
                bbProgAppend(p, &(struct oparg_t){OP_ISQRT, ivs[i], states[i],
                                                  ivs[i], 1, opopt_epsilon});

                bbProgAppend(p, &(struct oparg_t){OP_MUL, ivs[i], ivs[i], -1, 1,
                                                  opopt_lr});
                bbProgAppend(
                    p, &(struct oparg_t){OP_MUL, ivs[i], ivs[i], grads[i], 0});

                bbProgAppend(p, &(struct oparg_t){OP_MINUS, weights[i],
                                                  weights[i], ivs[i], 0});
        }
        return OK;
}
