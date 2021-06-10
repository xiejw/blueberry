#include "opt/fn.h"

// eva
#include "adt/dict.h"

// mlvm
#include "vm_internal.h"  //  MAX_TENSOR_COUNT

#include <stdio.h>  // TODO

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
bbTdMapNew()
{
        int              cap = MLVM_MAX_TENSOR_COUNT;
        struct td_map_t *p   = malloc(sizeof(struct td_map_t));
        p->cap               = cap;
        p->data              = NULL;
        vecReserve(p->data, cap);
        memset(p->data, 0, cap * sizeof(void *));  // NULL all slots.
        return p;
}

void
bbTdMapFree(struct td_map_t *p)
{
        if (p == NULL) return;
        vecFree(p->data);
        free(p);
}

error_t
bbTdMapFind(struct td_map_t *map, int td, void **data)
{
        if (td < 0) return errNew("td cannot be negative.");
        if (td >= map->cap) return errNew("td is too large.");
        *data = map->data[td];
        return OK;
}

error_t
bbTdMapSet(struct td_map_t *map, int td, void *v)
{
        if (td < 0) return errNew("td cannot be negative.");
        if (td >= map->cap) return errNew("td is too large.");

        void **p = &map->data[td];

        if (*p != NULL) return errNew("already have value.");

        *p = v;
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

uint64_t
hashFn(const void *key)
{
        return (intptr_t)key;
}

int
keyCmp(void *privdata, const void *key1, const void *key2)
{
        return key1 == key2;
}

struct dict_ty_t ty = {
    .hashFn  = hashFn,
    .keyDup  = NULL,
    .valDup  = NULL,
    .keyCmp  = keyCmp,
    .keyFree = NULL,
    .valFree = NULL,
};

error_t
runDCEPass(struct bb_fn_t *fn, void *cfg, int *changed)
{
        error_t           err;
        struct td_map_t  *map  = bbTdMapNew();
        struct bb_inst_t *curr = fn->inst_list.head;
        struct bb_inst_t *inst;
        dict_t           *t = dictNew(&ty, NULL);

        // record map from td to inst.
        while (curr != NULL) {
                err = bbTdMapFind(map, curr->op.dst, (void **)&inst);
                if (err) return errEmitNote("failed to look up td.");
                if (inst != NULL) errNew("do not support in-place update.");

                if (bbTdMapSet(map, curr->op.dst, &curr->op)) {
                        errEmitNote("failed to insert td.");
                }
                curr = curr->next;
        }

        // push outputs to criticals.
        vec_t(struct bb_inst_t *) criticals = vecNew();
        size_t output_count                 = vecSize(fn->outputs);
        for (size_t i = 0; i < output_count; i++) {
                int td = fn->outputs[i];
                err    = bbTdMapFind(map, td, (void **)&inst);
                if (err) return errEmitNote("failed to look up td.");
                if (inst == NULL) errNew("expect op generating output.");
                vecPushBack(criticals, inst);
        }

        int existed;
        while (vecSize(criticals) > 0) {
                struct bb_inst_t *inst_src;
                inst = vecPopBack(criticals);
                // mark
                struct dict_entry_t *en = dictAddOrFind(t, inst, &existed);
                assert(!existed);
                dictSetUIntVal(en, 1);

                // put the instruction into criticals if not marked yet.
                err = bbTdMapFind(map, inst->op.t1, (void **)&inst_src);
                if (err) return errEmitNote("failed to look up td.");
                if (inst_src != NULL) {
                        en = dictFind(t, inst_src);
                        if (en == NULL) vecPushBack(criticals, inst_src);
                }

                // put the instruction into criticals if not marked yet.
                err = bbTdMapFind(map, inst->op.t2, (void **)&inst_src);
                if (err) return errEmitNote("failed to look up td.");
                if (inst_src != NULL) {
                        en = dictFind(t, inst_src);
                        if (en == NULL) vecPushBack(criticals, inst_src);
                }
        }

        curr  = fn->inst_list.head;
        int i = 0;
        while (curr != NULL) {
                struct dict_entry_t *en = dictFind(t, &curr->op);
                // TODO (debug?)
                // if (en == NULL) {
                //         printf("inst %d not marked.\n", i);
                // } else {
                //         printf("inst %d marked.\n", i);
                // }
                i++;
                if (en == NULL) {
                        struct bb_inst_t *next = curr->next;

                        bbInstListDelete(&fn->inst_list, curr);

                        curr = next;
                        continue;
                }
                curr = curr->next;
        }

        dictFree(t);
        vecFree(criticals);
        bbTdMapFree(map);
        *changed = 0;
        return OK;
}
