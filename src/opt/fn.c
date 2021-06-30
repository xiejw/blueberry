#include "opt/fn.h"

#include <stdlib.h>  // malloc

// eva
#include "adt/hashing.h"

// -----------------------------------------------------------------------------
// Ctx.
// -----------------------------------------------------------------------------

static struct dict_ty_t ty_fn = {
    .hashFn  = hashFnStr,
    .keyDup  = dupFnStr,
    .valDup  = NULL,
    .keyCmp  = keyCmpFnStr,
    .keyFree = freeFnStr,
    .valFree = NULL,
};

struct bb_fn_ctx_t *
bbFnCtxNew()
{
        struct bb_fn_ctx_t *p = malloc(sizeof(*p));
        p->cfg                = NULL;
        p->fns                = dictNew(&ty_fn, NULL);
        return p;
}

void
bbFnCtxFree(struct bb_fn_ctx_t *p)
{
        if (p == NULL) return;
        dictFree(p->fns);
        free(p);
}

// -----------------------------------------------------------------------------
// Fn.
// -----------------------------------------------------------------------------

struct bb_fn_t *
bbFnNew()
{
        size_t          psize = sizeof(struct bb_fn_t);
        struct bb_fn_t *p     = calloc(1, psize);
        bbInstListReset(&p->inst_list);
        return p;
}

void
bbFnFree(struct bb_fn_t *p)
{
        if (p == NULL) return;
        vecFree(p->inputs);
        vecFree(p->outputs);
        bbInstListFree(&p->inst_list);
        free(p);
}

void
bbFnAppend(struct bb_fn_t *p, struct oparg_t *op)
{
        bbInstListAppend(&p->inst_list, op);
}

void
bbFnDump(struct bb_fn_t *fn, sds_t *s)
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
