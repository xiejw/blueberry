#include "bb.h"

#include <string.h>

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

        for (size_t i = 0; i < weights_count; i++) {
                bbProgAppend(p, &(struct oparg_t){OP_MUL, grads[i], grads[i],
                                                  -1, 1, opopt});
                bbProgAppend(p, &(struct oparg_t){OP_MINUS, weights[i],
                                                  weights[i], grads[i], 0});
        }
        return OK;
}

error_t
bbOptNew(struct vm_t *vm, int type, float32_t lr, vec_t(int) weights,
         vec_t(int) grads, struct bb_opt_t **out)
{
        struct bb_opt_t *opt = malloc(sizeof(struct bb_opt_t));
        memset(opt, 0, sizeof(struct bb_opt_t));

        if (vecSize(weights) != vecSize(grads))
                return errNew("opt expects len(weights) == len(grads).");
        if (type != BB_OPT_SGD)
                return errNew("unsupported optimizer type: %d", type);

        opt->lr      = lr;
        opt->vm      = vm;
        opt->type    = type;
        opt->weights = weights;
        opt->grads   = grads;

        assert(type == BB_OPT_SGD);
        _bbOptSGDInit(opt);

        *out = opt;
        return OK;
}

error_t
bbOptApply(struct bb_opt_t *opt, struct bb_program_t *p)
{
        assert(opt->type == BB_OPT_SGD);
        return _bbOptSGDApply(opt, p);
}

void
bbOptFree(struct bb_opt_t *opt)
{
        struct vm_t *vm = opt->vm;

        {  // release states.
                vec_t(int) handles = opt->states;
                int size           = vecSize(handles);
                for (int i = 0; i < size; i++) {
                        vmTensorFree(vm, handles[i]);
                }
                vecFree(handles);
        }

        free(opt->private_data);
        free(opt);
}
