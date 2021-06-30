#include "opt/fn.h"

#include <stdio.h>

// eva
#include "adt/dict.h"

#include "td_map.h"

// -----------------------------------------------------------------------------
// Math Local Optimization.
// -----------------------------------------------------------------------------

// Remove unnecessary math operations, e.g., x = x * 1 in fn.
//
// This can often come from gradient tape as the first gradient is always 1.
// Here we take a naive approach to do very simple optimization for this case
// only.
//
// The algorithm
//
// 1. Check SSA complicant.
// 2. Pass 1: Record all td's generators.
// 2. Pass 2: Record all td's uses (count).
// 2. Pass 3: Find all candidates and replace the td if the operand's use is 1.
struct td_info_t {
        int               uses;  // count of uses.
        struct bb_inst_t *src;   // src instruciton generating this td.
        struct td_info_t *next;  // point to next info.
};

error_t
runMathPass(struct bb_fn_t *fn, void *cfg, int debug, int *changed)
{
        sds_t s = sdsEmpty();
        if (debug) {
                sdsCatPrintf(&s, "==================\n");
                sdsCatPrintf(&s, "Running Math Pass.\n");
                sdsCatPrintf(&s, "==================\n");
                bbFnDump(fn, &s);
                printf("%s\n", s);
        }

        error_t           err  = OK;
        struct td_map_t  *map  = bbTdMapNew();  // type is td_info_t*
        struct bb_inst_t *curr = fn->inst_list.head;

        struct td_info_t *head = NULL;
        struct td_info_t *info;

        vec_t(int) inputs                    = vecNew();
        vec_t(int) outputs                   = vecNew();
        vec_t(struct bb_inst_t *) candidates = vecNew();

        while (curr != NULL) {
                vecSetSize(outputs, 0);  // clear
                bbInstOutputs(curr, &outputs);
                int size = vecSize(outputs);

                for (int i = 0; i < size; i++) {
                        int td = outputs[i];
                        err    = bbTdMapFind(map, td, (void **)&info);
                        if (err) {
                                err = errEmitNote("failed to look up td.");
                                goto cleanup;
                        }
                        if (info == NULL) {
                                // create a new one.
                                info       = malloc(sizeof(*info));
                                info->uses = 0;
                                info->src  = curr;
                                info->next = head;
                                head       = info;
                                if (bbTdMapSet(map, td, info)) {
                                        err =
                                            errEmitNote("failed to insert td.");
                                        goto cleanup;
                                }
                        } else if (info->src != curr) {
                                err = errNew("not ssa.");
                                goto cleanup;
                        }
                }

                vecSetSize(inputs, 0);  // clear
                bbInstInputs(curr, &inputs);
                size = vecSize(inputs);
                for (int i = 0; i < size; i++) {
                        int td = inputs[i];
                        err    = bbTdMapFind(map, td, (void **)&info);
                        if (err) {
                                err = errEmitNote("failed to look up td.");
                                goto cleanup;
                        }
                        if (info == NULL) {
                                continue;  // must be fn input or global consts.
                        }
                        info->uses++;
                }

                if (curr->op.op == OP_MUL && curr->op.t2 == 1) {
                        vecPushBack(candidates, curr);
                }

                curr = curr->next;
        }

        int replaced_count = 0;
        while (vecSize(candidates) > 0) {
                curr = vecPopBack(candidates);
                err  = bbTdMapFind(map, curr->op.t1, (void **)&info);
                if (err) {
                        err = errEmitNote("failed to look up td.");
                        goto cleanup;
                }
                if (info == NULL) {
                        continue;  // shall we do replace straightforward?
                        // another cavediate is it doens not rewrite output.
                }
                if (info->uses == 1) {
                        if (debug) {
                                sdsClear(s);
                                sdsCatPrintf(&s, "Candidate: ");
                                bbInstDump(curr, &s);
                                printf("%s\n", s);
                        }
                        // rewrite now.
                        assert(info->src != NULL);
                        info->src->op.dst = curr->op.dst;
                        bbInstListDelete(&fn->inst_list, curr);
                        replaced_count++;
                }
        }

        if (debug) {
                sdsClear(s);
                sdsCatPrintf(&s, "==================\n");
                sdsCatPrintf(&s, "After Math Pass.\n");
                sdsCatPrintf(&s, "==================\n");
                bbFnDump(fn, &s);
                printf("%s\n", s);
        }

        *changed = replaced_count > 0;

cleanup:
        vecFree(inputs);
        vecFree(outputs);
        vecFree(candidates);

        while (head != NULL) {
                info = head->next;
                free(head);
                head = info;
        }

        bbTdMapFree(map);
        sdsFree(s);
        return err;
}
