#include "bb.h"

#include <string.h>

// -----------------------------------------------------------------------------
// Prototype.
// -----------------------------------------------------------------------------

static error_t _bbOptSGDInit(struct bb_opt_t *opt);
static error_t _bbOptSGDApply(struct bb_opt_t *opt, struct bb_program_t *p);
static error_t _bbOptRMSPropInit(struct bb_opt_t *opt);
static error_t _bbOptRMSPropApply(struct bb_opt_t *opt, struct bb_program_t *p);
static error_t _bbOptAdamPropInit(struct bb_opt_t *opt);
static error_t _bbOptAdamPropApply(struct bb_opt_t     *opt,
                                   struct bb_program_t *p);

// -----------------------------------------------------------------------------
// Public APIs.
// -----------------------------------------------------------------------------
error_t
bbOptNew(struct vm_t *vm, int type, float32_t lr, void *cfg,
         struct bb_opt_t **out)
{
        struct bb_opt_t *opt = malloc(sizeof(struct bb_opt_t));
        memset(opt, 0, sizeof(struct bb_opt_t));

        opt->lr   = lr;
        opt->vm   = vm;
        opt->type = type;

        size_t cfg_size;
        if (cfg != NULL) {
                void *ptr;
                switch (opt->type) {
                case BB_OPT_SGD:
                        return errNew(
                            "Opt SGD does not need config (expect NULL).");
                case BB_OPT_RMSPROP:
                        cfg_size = sizeof(struct bb_opt_rmsprop_config_t);
                        ptr      = malloc(cfg_size);
                        memcpy(ptr, cfg, cfg_size);
                        opt->config = ptr;
                        break;
                case BB_OPT_ADAM:
                        cfg_size = sizeof(struct bb_opt_adam_config_t);
                        ptr      = malloc(cfg_size);
                        memcpy(ptr, cfg, cfg_size);
                        opt->config = ptr;
                        break;
                default:
                        return errNew("Opt type not supported in (yet): %d",
                                      opt->type);
                }
        }

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
        case BB_OPT_ADAM:
                return _bbOptAdamPropInit(opt);
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
        case BB_OPT_ADAM:
                return _bbOptAdamPropApply(opt, p);
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
                // clang-format off
                // reuse grad (bad idea?)
                bbProgAppend(p, &(struct oparg_t){OP_MUL,   grads[i],   grads[i],   -1,       1, opopt});
                bbProgAppend(p, &(struct oparg_t){OP_MINUS, weights[i], weights[i], grads[i], 0});
                // clang-format on
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
        struct vm_t    *vm = opt->vm;
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
            ((struct bb_opt_rmsprop_t *)opt->private_data)->weights_count;

        // math:
        //
        //     s_t  = rho * s_{t-1} + (1 - rho) * g_t^2
        //     w_t  = w_{t-1}  - lr / sqrt(s_t + epsilon) * g_t
        //
        // mlvm code:
        //
        //     s = rho * s
        //     t1 = g * g
        //     t1 = (1 - rho) * t1
        //     s = s + t1
        //     t1 = 1/sqrt(s + epsilon)
        //     t1 = t1 * lr
        //     t1 = t1 * g
        //     w = w - t1

        struct bb_opt_rmsprop_config_t *cfg =
            (struct bb_opt_rmsprop_config_t *)opt->config;

        assert(cfg != NULL);
        float rho = cfg->rho;
        assert(rho < 1 && rho > 0);
        float epsilon = cfg->epsilon;
        assert(epsilon > 0);

        const struct opopt_t opopt_rho = {.mode = OPT_MODE_F_BIT, .f = rho};
        const struct opopt_t opopt_1_minus_rho = {.mode = OPT_MODE_F_BIT,
                                                  .f    = (1 - rho)};
        const struct opopt_t opopt_epsilon     = {.mode = OPT_MODE_F_BIT,
                                              .f    = epsilon};
        const struct opopt_t opopt_lr = {.mode = OPT_MODE_F_BIT, .f = opt->lr};

        vec_t(int) weights = opt->weights;
        vec_t(int) grads   = opt->grads;
        vec_t(int) states  = opt->states;
        vec_t(int) ivs     = opt->ivs;

        for (size_t i = 0; i < weights_count; i++) {
                // clang-format off
                bbProgAppend(p, &(struct oparg_t){OP_MUL,   states[i],  states[i],  -1,       1, opopt_rho});
                bbProgAppend(p, &(struct oparg_t){OP_MUL,   ivs[i],     grads[i],   grads[i], 0});
                bbProgAppend(p, &(struct oparg_t){OP_MUL,   ivs[i],     ivs[i],     -1,       1, opopt_1_minus_rho});
                bbProgAppend(p, &(struct oparg_t){OP_ADD,   states[i],  states[i],  ivs[i],   0});
                bbProgAppend(p, &(struct oparg_t){OP_ISQRT, ivs[i],     states[i],  ivs[i],   1, opopt_epsilon});
                bbProgAppend(p, &(struct oparg_t){OP_MUL,   ivs[i],     ivs[i],     -1,       1, opopt_lr});
                bbProgAppend(p, &(struct oparg_t){OP_MUL,   ivs[i],     ivs[i],     grads[i], 0});
                bbProgAppend(p, &(struct oparg_t){OP_MINUS, weights[i], weights[i], ivs[i],   0});
                // clang-format on
        }
        return OK;
}

// -----------------------------------------------------------------------------
// Adam Impl
// -----------------------------------------------------------------------------
struct bb_opt_adam_t {
        size_t weights_count;
        // index for state
        size_t m1_i;  // zero init
        size_t m2_i;  // zero init
        size_t b1_i;  // minus one init
        size_t b2_i;  // minus one init
        // index for iv
        size_t t1_i;
        size_t t2_i;
        size_t s1_i;
        size_t s2_i;
};

error_t
_bbOptAdamPropInit(struct bb_opt_t *opt)
{
        struct bb_opt_adam_t *data = malloc(sizeof(struct bb_opt_adam_t));

        int weights_count   = vecSize(opt->weights);
        data->weights_count = weights_count;

        // need two states for each weight and two scalar for betas (in total).
        vecReserve(opt->states, 2 * weights_count + 2);
        vecSetSize(opt->states, 2 * weights_count + 2);
        data->m1_i = 0;
        data->m2_i = weights_count;
        data->b1_i = 2 * weights_count;
        data->b2_i = 2 * weights_count + 1;

        // need two ivs for each weight and two scalars.
        vecReserve(opt->ivs, 2 * weights_count + 2);
        vecSetSize(opt->ivs, 2 * weights_count + 2);
        data->t1_i = 0;
        data->t2_i = weights_count;
        data->s1_i = 2 * weights_count;
        data->s2_i = 2 * weights_count + 1;

        error_t         err;
        struct vm_t    *vm = opt->vm;
        struct shape_t *sp;
        size_t          offset = weights_count;
        for (size_t i = 0; i < weights_count; i++) {
                err = vmTensorInfo(vm, opt->weights[i], /*dtype=*/NULL, &sp);
                if (err) return errEmitNote("failed to obtain weight shape.");

                // state

                int state = vmTensorNew(vm, F32, sp);
                err       = vmExec(vm, OP_FILL, NULL, state, -1, -1);
                if (err) return errEmitNote("failed to zero opt state.");
                opt->states[i] = state;

                state = vmTensorNew(vm, F32, sp);
                err   = vmExec(vm, OP_FILL, NULL, state, -1, -1);
                if (err) return errEmitNote("failed to zero opt state.");
                opt->states[offset + i] = state;

                // iv

                opt->ivs[i]          = vmTensorNew(vm, F32, sp);
                opt->ivs[offset + i] = vmTensorNew(vm, F32, sp);
        }

        sp = R1S(vm, 1);

        opt->states[data->b1_i] = vmTensorNew(vm, F32, sp);
        opt->states[data->b2_i] = vmTensorNew(vm, F32, sp);
        opt->ivs[data->s1_i]    = vmTensorNew(vm, F32, sp);
        opt->ivs[data->s2_i]    = vmTensorNew(vm, F32, sp);

        spDecRef(sp);

        struct opopt_t opopt_one = {.mode = OPT_MODE_F_BIT, .f = -1};
        err = vmExec(vm, OP_FILL, &opopt_one, opt->states[data->b1_i], -1, -1);
        if (err) return errEmitNote("failed to init opt state.");
        err = vmExec(vm, OP_FILL, &opopt_one, opt->states[data->b2_i], -1, -1);
        if (err) return errEmitNote("failed to init opt state.");

        opt->private_data = data;
        return OK;
}

error_t
_bbOptAdamPropApply(struct bb_opt_t *opt, struct bb_program_t *p)
{
        struct bb_opt_adam_t *data = (struct bb_opt_adam_t *)opt->private_data;
        size_t                weights_count = data->weights_count;

        // math:
        //
        //     m_t  = beta_1 * m_{t-1} + (1 - beta_1) * g_t
        //     v_t  = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
        //
        //     mh_t = m_t / (1 - beta_1^t)
        //     vh_t = v_t / (1 - beta_2^t)
        //
        //     w_t  = w_{t-1}  - lr / (sqrt(vh_t) + epsilon) * mh^t
        //
        // mlvm code:
        //
        //     ivs: t1 t2 s1 s2
        //
        //     - b1 = - b1 * beta_1
        //     - b2 = - b2 * beta_2
        //
        //     m1 = beta_1 * m1
        //     m2 = beta_2 * m2
        //
        //     t1 = (1 - beta_1) g
        //     t2 = g * g
        //     t2 = (1 - beta_2) * t2
        //
        //     m1 = m1 + t1
        //     m2 = m2 + t2
        //
        //     s1 = 1 - b1 = -b1 + 1
        //     t1 = m1 / s1
        //     s2 = 1 - b2 = -b2 + 1
        //     t2 = m2 / s2
        //
        //     t2 = 1/(sqrt(t2) + epsilon)
        //     t2 = t2 * lr
        //     t1 = t1 * t2
        //     w = w - t1

        struct bb_opt_adam_config_t *cfg =
            (struct bb_opt_adam_config_t *)opt->config;

        assert(cfg != NULL);
        float beta_1 = cfg->beta_1;
        assert(beta_1 < 1 && beta_1 > 0);
        float beta_2 = cfg->beta_2;
        assert(beta_2 < 1 && beta_2 > 0);
        float epsilon = cfg->epsilon;
        assert(epsilon > 0);

        const struct opopt_t opopt_beta_1 = {.mode = OPT_MODE_F_BIT,
                                             .f    = beta_1};
        const struct opopt_t opopt_beta_2 = {.mode = OPT_MODE_F_BIT,
                                             .f    = beta_2};

        const struct opopt_t opopt_1_minus_beta_1 = {.mode = OPT_MODE_F_BIT,
                                                     .f    = (1 - beta_1)};
        const struct opopt_t opopt_1_minus_beta_2 = {.mode = OPT_MODE_F_BIT,
                                                     .f    = (1 - beta_2)};

        const struct opopt_t opopt_epsilon = {.mode = 1 | OPT_MODE_F_BIT,
                                              .f    = epsilon};
        const struct opopt_t opopt_lr = {.mode = OPT_MODE_F_BIT, .f = opt->lr};

        int *w = opt->weights;
        int *g = opt->grads;

        int *m1 = opt->states + data->m1_i;
        int *m2 = opt->states + data->m2_i;
        int  b1 = opt->states[data->b1_i];
        int  b2 = opt->states[data->b2_i];

        int *t1 = opt->ivs + data->t1_i;
        int *t2 = opt->ivs + data->t2_i;
        int  s1 = opt->ivs[data->s1_i];
        int  s2 = opt->ivs[data->s2_i];

        bbProgAppend(p, &(struct oparg_t){OP_MUL, b1, b1, -1, 1, opopt_beta_1});
        bbProgAppend(p, &(struct oparg_t){OP_MUL, b2, b2, -1, 1, opopt_beta_2});
        bbProgAppend(p, &(struct oparg_t){OP_ADD, s1, b1, 1, 0});
        bbProgAppend(p, &(struct oparg_t){OP_ADD, s2, b2, 1, 0});

        for (size_t i = 0; i < weights_count; i++) {
                // clang-format off
                bbProgAppend(p, &(struct oparg_t){OP_MUL,   *(m1+i),  *(m1+i),  -1,       1, opopt_beta_1});
                bbProgAppend(p, &(struct oparg_t){OP_MUL,   *(m2+i),  *(m2+i),  -1,       1, opopt_beta_2});
                bbProgAppend(p, &(struct oparg_t){OP_MUL,   *(t1+i),  *(g +i),  -1,       1, opopt_1_minus_beta_1});
                bbProgAppend(p, &(struct oparg_t){OP_MUL,   *(t2+i),  *(g +i),  *(g+i),   0});
                bbProgAppend(p, &(struct oparg_t){OP_MUL,   *(t2+i),  *(t2+i),  -1,       1, opopt_1_minus_beta_2});

                bbProgAppend(p, &(struct oparg_t){OP_ADD,   *(m1+i),  *(m1+i),  *(t1+i),  0});
                bbProgAppend(p, &(struct oparg_t){OP_ADD,   *(m2+i),  *(m2+i),  *(t2+i),  0});
                bbProgAppend(p, &(struct oparg_t){OP_DIVIDE,*(t1+i),  *(m1+i),  s1 ,      0});
                bbProgAppend(p, &(struct oparg_t){OP_DIVIDE,*(t2+i),  *(m2+i),  s2 ,      0});

                bbProgAppend(p, &(struct oparg_t){OP_ISQRT, *(t2+i),  *(t2+i),  -1,       1, opopt_epsilon});
                bbProgAppend(p, &(struct oparg_t){OP_MUL,   *(t2+i),  *(t2+i),  -1,       1, opopt_lr});
                bbProgAppend(p, &(struct oparg_t){OP_MUL,   *(t1+i),  *(t1+i),  *(t2+i),  0});
                bbProgAppend(p, &(struct oparg_t){OP_MINUS, *(w +i),  *(w +i),  *(t1+i),  0});
                // clang-format on
        }
        return OK;
}
