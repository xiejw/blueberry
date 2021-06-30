#include "opt/fn.h"

#include <stdio.h>

// eva
#include "adt/dict.h"

// mlvm
#include "vm_internal.h"  //  MLVM_MAX_TENSOR_COUNT

// -----------------------------------------------------------------------------
// DCE.
// -----------------------------------------------------------------------------

static uint64_t
hashFn(const void *key)
{
        return (intptr_t)key;
}

static int
keyCmp(void *privdata, const void *key1, const void *key2)
{
        return key1 == key2;
}

static struct dict_ty_t ty_ptr = {
    .hashFn  = hashFn,
    .keyDup  = NULL,
    .valDup  = NULL,
    .keyCmp  = keyCmp,
    .keyFree = NULL,
    .valFree = NULL,
};

static struct dict_ty_t ty_i64 = {
    .hashFn  = valueHashFnI64,
    .keyDup  = valueDupI64,
    .valDup  = NULL,
    .keyCmp  = valueCmpI64,
    .keyFree = valueFreeI64,
    .valFree = NULL,
};

// Run DCE Pass on the fn.
//
// The algorithm is simple:
//
// 1. Check SSA compliant.
// 2. criticals = []
// 3. Push all instructions, which generates outputs, into criticals.
// 4. while criticals is not empty:
//       pop inst from criticals.
//       mark inst
//       push instructions, not marked, generating operands into criticals
// 5. remove all instructions, which are not marked.
error_t
runDCEPass(struct bb_fn_t *fn, struct bb_fn_ctx_t *ctx, int *changed)
{
        int   debug = ctx->debug_mode;
        sds_t s     = sdsEmpty();

        if (debug) {
                sdsCatPrintf(&s, "==================\n");
                sdsCatPrintf(&s, "Running DCE Pass.\n");
                sdsCatPrintf(&s, "==================\n");
                bbFnDump(fn, &s);
                printf("%s\n", s);
        }

        dict_t *inst_mark_set  = dictNew(&ty_ptr, NULL);
        dict_t *td_to_inst_map = dictNew(&ty_i64, NULL);
        dictExpand(td_to_inst_map, MLVM_MAX_TENSOR_COUNT);

        struct dict_entry_t *entry;
        struct value_t       key;
        int                  existed;

        vec_t(int) inputs  = vecNew();
        vec_t(int) outputs = vecNew();

        // record map from td to inst.
        struct bb_inst_t *curr = fn->inst_list.head;
        struct bb_inst_t *inst;

        while (curr != NULL) {
                vecSetSize(outputs, 0);  // clear
                bbInstOutputs(curr, &outputs);
                for (int i = 0; i < vecSize(outputs); i++) {
                        int td = outputs[i];
                        valueSetI64(&key, td);

                        entry = dictAddOrFind(td_to_inst_map, &key, &existed);
                        if (existed)
                                return errNew(
                                    "do not support in-place update.");

                        dictSetData(td_to_inst_map, entry, &curr->op);
                }

                curr = curr->next;
        }

        // push outputs to criticals.
        vec_t(struct bb_inst_t *) criticals = vecNew();
        size_t output_count                 = vecSize(fn->outputs);
        for (size_t i = 0; i < output_count; i++) {
                int td = fn->outputs[i];
                valueSetI64(&key, td);
                entry = dictFind(td_to_inst_map, &key);
                if (entry != NULL) {
                        inst = dictGetData(entry);
                        vecPushBack(criticals, inst);
                }
        }

        // recursively push instructions into criticals.
        while (vecSize(criticals) > 0) {
                inst = vecPopBack(criticals);
                assert(inst != NULL);

                // mark current instruction so it will not be eliminated.
                struct dict_entry_t *en =
                    dictAddOrFind(inst_mark_set, inst, &existed);
                assert(!existed);
                dictSetU64(en, 1);

                // put all instructions, which generates inputs, into
                // criticals if not marked yet.
                vecSetSize(inputs, 0);  // clear
                bbInstInputs(inst, &inputs);

                struct bb_inst_t *inst_src;
                for (int i = 0; i < vecSize(inputs); i++) {
                        int td = inputs[i];
                        valueSetI64(&key, td);
                        entry = dictFind(td_to_inst_map, &key);
                        if (entry != NULL) {
                                inst_src = dictGetData(entry);
                                en       = dictFind(inst_mark_set, inst_src);
                                if (en == NULL)
                                        vecPushBack(criticals, inst_src);
                        }
                }
        }

        // delete all instructions, which are not marked..
        int delete_count = 0;
        curr             = fn->inst_list.head;
        while (curr != NULL) {
                struct dict_entry_t *en = dictFind(inst_mark_set, &curr->op);
                if (en == NULL) {
                        struct bb_inst_t *next = curr->next;

                        if (debug) {
                                sdsClear(s);
                                bbInstDump(curr, &s);
                                printf("--> Delete inst: %s\n", s);
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

        dictFree(inst_mark_set);
        dictFree(td_to_inst_map);
        vecFree(inputs);
        vecFree(outputs);
        vecFree(criticals);
        *changed = delete_count > 0;
        sdsFree(s);
        return OK;
}
