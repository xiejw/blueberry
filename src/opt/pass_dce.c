#include "opt/fn.h"

#include <stdio.h>

// eva
#include "adt/dict.h"

#include "td_map.h"

// -----------------------------------------------------------------------------
// DCE.
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
runDCEPass(struct bb_fn_t *fn, void *cfg, int debug, int *changed)
{
        sds_t s = sdsEmpty();
        if (debug) {
                sdsCatPrintf(&s, "==================\n");
                sdsCatPrintf(&s, "Running DCE Pass.\n");
                sdsCatPrintf(&s, "==================\n");
                bbFnDump(fn, &s);
                printf("%s\n", s);
        }

        error_t           err;
        struct td_map_t  *map  = bbTdMapNew();
        struct bb_inst_t *curr = fn->inst_list.head;
        struct bb_inst_t *inst;
        dict_t           *t = dictNew(&ty, NULL);

        // record map from td to inst.
        while (curr != NULL) {
                err = bbTdMapFind(map, curr->op.dst, (void **)&inst);
                if (err) return errEmitNote("failed to look up td.");
                if (inst != NULL)
                        return errNew("do not support in-place update.");

                if (bbTdMapSet(map, curr->op.dst, &curr->op)) {
                        return errEmitNote("failed to insert td.");
                }

                // special case for LS_SCEL.
                if (curr->op.op == OP_LS_SCEL && curr->op.has_opt &&
                    curr->op.opt.mode & OPT_MODE_I_BIT) {
                        if (bbTdMapSet(map, curr->op.opt.i, &curr->op)) {
                                return errEmitNote("failed to insert td.");
                        }
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
                if (inst != NULL) {
                        vecPushBack(criticals, inst);
                }
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
                assert(inst != NULL);
                if (inst->op.t1 >= 0) {
                        err = bbTdMapFind(map, inst->op.t1, (void **)&inst_src);
                        if (err) return errEmitNote("failed to look up td.");
                        if (inst_src != NULL) {
                                en = dictFind(t, inst_src);
                                if (en == NULL)
                                        vecPushBack(criticals, inst_src);
                        }
                }

                // put the instruction into criticals if not marked yet.
                if (inst->op.t2 >= 0) {
                        err = bbTdMapFind(map, inst->op.t2, (void **)&inst_src);
                        if (err) return errEmitNote("failed to look up td.");
                        if (inst_src != NULL) {
                                en = dictFind(t, inst_src);
                                if (en == NULL)
                                        vecPushBack(criticals, inst_src);
                        }
                }

                if (inst->op.op == OP_LS_SCEL && inst->op.has_opt &&
                    inst->op.opt.mode & OPT_MODE_I_BIT) {
                        err = bbTdMapFind(map, inst->op.opt.i,
                                          (void **)&inst_src);
                        if (err) return errEmitNote("failed to look up td.");
                        if (inst_src != NULL) {
                                en = dictFind(t, inst_src);
                                if (en == NULL)
                                        vecPushBack(criticals, inst_src);
                        }
                }
        }

        int delete_count = 0;
        curr             = fn->inst_list.head;
        while (curr != NULL) {
                struct dict_entry_t *en = dictFind(t, &curr->op);
                if (en == NULL) {
                        struct bb_inst_t *next = curr->next;

                        if (debug) {
                                printf("--> Delete inst for dst = %d\n",
                                       curr->op.dst);
                        }
                        delete_count++;
                        bbInstListDelete(&fn->inst_list, curr);
                        curr = next;
                        continue;
                }
                curr = curr->next;
        }

        if (debug && delete_count > 0) {
                sdsClear(s);
                sdsCatPrintf(&s, "==================\n");
                sdsCatPrintf(&s, "After DCE Pass.\n");
                sdsCatPrintf(&s, "==================\n");
                bbFnDump(fn, &s);
                printf("%s\n", s);
        }

        dictFree(t);
        vecFree(criticals);
        bbTdMapFree(map);
        *changed = delete_count > 0;
        sdsFree(s);
        return OK;
}
