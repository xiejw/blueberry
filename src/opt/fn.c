#include "opt/fn.h"

#include "vm_internal.h"  //  MAX_TENSOR_COUNT

// -----------------------------------------------------------------------------
// Map Helpers.
// -----------------------------------------------------------------------------

// a fast map specific for the tensor descriptor. This structure assumes that
// the input must SSA-like and all tensor descriptor is contiguous.
struct td_map_t {
        int cap;
        vec_t(void *) data;
};

struct td_map_t *
bbTdMap()
{
        int              cap = MLVM_MAX_TENSOR_COUNT;
        struct td_map_t *p   = malloc(sizeof(struct td_map_t));
        p->cap               = cap;
        p->data              = NULL;
        vecReserve(p->data, cap);
        memset(p->data, 0, cap * sizeof(void *));  // NULL all slots.
        return p;
}

error_t
bbTdMapFind(struct td_map_t *map, int td, void **data)
{
        if (td < 0) return errNew("td cannot be negative.");
        if (td >= map->cap) return errNew("td is too large.");
        *data = map->data[td];
        return OK;
}

#define BB_TD_MAP_OVERWRITE        0
#define BB_TD_MAP_DO_NOT_OVERWRITE 1

error_t
bbTdMapSet(struct td_map_t *map, int td, void *v, int policy, int *existed)
{
        if (td < 0) return errNew("td cannot be negative.");
        if (td >= map->cap) return errNew("td is too large.");

        *existed = map->data[td] != NULL;

        if (policy == BB_TD_MAP_OVERWRITE) {
                map->data[td] = v;
                return OK;
        }

        assert(policy == BB_TD_MAP_DO_NOT_OVERWRITE);
        if (!*existed) {
                map->data[td] = v;
        }
        return OK;
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

// -----------------------------------------------------------------------------
// Passes.
// -----------------------------------------------------------------------------

error_t
runDCEPass(struct bb_fn_t *fn, void *cfg, int *changed)
{
        *changed = 0;
        return OK;
}
