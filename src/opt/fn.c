#include "opt/fn.h"

// Op

void
bbInstInputs(struct bb_inst_t* inst, vec_t(int) * inputs)
{
        if (inst->op.t1 >= 0) vecPushBack(*inputs, inst->op.t1);
        if (inst->op.t2 >= 0) vecPushBack(*inputs, inst->op.t2);
}

void
bbInstOutputs(struct bb_inst_t* inst, vec_t(int) * outputs)
{
        vecPushBack(*outputs, inst->op.dst);
        if (inst->op.op == OP_LS_SCEL && inst->op.has_opt &&
            inst->op.opt.mode & OPT_MODE_I_BIT) {
                vecPushBack(*outputs, inst->op.opt.i);
        }
}

// -----------------------------------------------------------------------------
// Fn.
// -----------------------------------------------------------------------------

struct bb_fn_t*
bbFnNew()
{
        size_t          psize = sizeof(struct bb_fn_t);
        struct bb_fn_t* p     = calloc(1, psize);
        bbInstListReset(&p->inst_list);
        return p;
}

void
bbFnFree(struct bb_fn_t* p)
{
        if (p == NULL) return;
        vecFree(p->inputs);
        vecFree(p->outputs);
        bbInstListFree(&p->inst_list);
        free(p);
}

void
bbFnAppend(struct bb_fn_t* p, struct oparg_t* op)
{
        bbInstListAppend(&p->inst_list, op);
}

void
bbFnDump(struct bb_fn_t* fn, sds_t* s)
{
        sdsCatPrintf(s, "fn:\n");

#define PRINT_COLLECTION(collection)                                         \
        {                                                                    \
                sdsCatPrintf(s, "{  // " #collection "\n  ");                \
                size_t size = vecSize(fn->collection);                       \
                if (size) {                                                  \
                        for (int i = 0; i < size; i++) {                     \
                                sdsCatPrintf(s, "%3d, ", fn->collection[i]); \
                        }                                                    \
                        sdsCatPrintf(s, "\n");                               \
                } else {                                                     \
                        sdsCatPrintf(s, "(empty)\n");                        \
                }                                                            \
                sdsCatPrintf(s, "}\n");                                      \
        }

        PRINT_COLLECTION(inputs);
        PRINT_COLLECTION(outputs);

#undef PRINT_COLLECTION

        bbInstListDump(&fn->inst_list, s);
}
