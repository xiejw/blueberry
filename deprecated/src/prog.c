#include "bb.h"

#include <string.h>

static void
bbOpDump(sds_t *s, struct opopt_t *opt)
{
        // dump like
        //
        //   .opt = {.mode=1}
        //   .opt = {.mode=1|I, .i = 23}
        //   .opt = {.mode=1|F, .f = 23.0}

        sdsCatPrintf(s, ".opt = {.mode = %d", opt->mode & OPT_MODE_UNMASK);
        if (OPT_MODE_GET_I_BIT(*opt)) {
                sdsCatPrintf(s, "|I, .i = %d", opt->i);
        } else if (OPT_MODE_GET_F_BIT(*opt)) {
                sdsCatPrintf(s, "|F, .f = %f", opt->f);
        } else if (OPT_MODE_GET_R_BIT(*opt)) {
                sdsCatPrintf(s, "|R, .r = <to_be_filled>");
        }
        sdsCatPrintf(s, "}");
}

// -----------------------------------------------------------------------------
// Impl for Inst.
// -----------------------------------------------------------------------------

void
bbInstDump(struct bb_inst_t *inst, sds_t *s)
{
        char *opname;
        switch (inst->op.op) {
        case OP_MATMUL:
                opname = "OP_MATMUL";
                break;
        case OP_CMPL:
                opname = "OP_CMPL";
                break;
        case OP_MUL:
                opname = "OP_MUL";
                break;
        case OP_DIVIDE:
                opname = "OP_DIVIDE";
                break;
        case OP_ISQRT:
                opname = "OP_ISQRT";
                break;
        case OP_MINUS:
                opname = "OP_MINUS";
                break;
        case OP_ADD:
                opname = "OP_ADD";
                break;
        case OP_MAX:
                opname = "OP_MAX";
                break;
        case OP_REDUCE:
                opname = "OP_REDUCE";
                break;
        case OP_ARGMAX:
                opname = "OP_ARGMAX";
                break;
        case OP_EQ:
                opname = "OP_EQ";
                break;
        case OP_LS_SCEL:
                opname = "OP_LS_SCEL";
                break;
        default:
                opname = "UNKNOWN";
        }
        struct oparg_t *op = &inst->op;
        sdsCatPrintf(s,
                     "{.op = %2d (%-10s), .dst = %3d, .t1 = "
                     "%3d, .t2 = %3d",
                     op->op, opname, op->dst, op->t1, op->t2);
        if (!op->has_opt) {
                sdsCatPrintf(s, "}");
        } else {
                sdsCatPrintf(s, ", ");
                bbOpDump(s, &op->opt);
                sdsCatPrintf(s, "}");
        }
}

void
bbInstInputs(struct bb_inst_t *inst, vec_t(int) * inputs)
{
        if (inst->op.t1 >= 0) vecPushBack(*inputs, inst->op.t1);
        if (inst->op.t2 >= 0) vecPushBack(*inputs, inst->op.t2);
}

void
bbInstOutputs(struct bb_inst_t *inst, vec_t(int) * outputs)
{
        vecPushBack(*outputs, inst->op.dst);
        if (inst->op.op == OP_LS_SCEL && inst->op.has_opt &&
            inst->op.opt.mode & OPT_MODE_I_BIT) {
                vecPushBack(*outputs, inst->op.opt.i);
        }
}

// -----------------------------------------------------------------------------
// Impl for Inst List.
// -----------------------------------------------------------------------------

void
bbInstListReset(struct bb_inst_list_t *list)
{
        memset(list, 0, sizeof(*list));
}

void
bbInstListFree(struct bb_inst_list_t *list)
{
        struct bb_inst_t *next, *curr;
        curr = list->head;
        while (curr != NULL) {
                next = curr->next;
                free(curr);
                curr = next;
        }
}

void
bbInstListAppend(struct bb_inst_list_t *list, struct oparg_t *op)
{
        struct bb_inst_t *inst = malloc(sizeof(struct bb_inst_t));
        inst->op               = *op;
        inst->next             = NULL;

        struct bb_inst_t *tail = list->tail;

        if (list->head == NULL) {
                inst->prev = NULL;
                list->head = inst;
                list->tail = inst;
        } else {
                inst->prev = tail;
                tail->next = inst;
                list->tail = inst;
        }

        list->count++;
}

void
bbInstListDelete(struct bb_inst_list_t *list, struct bb_inst_t *inst)
{
        assert(list->head != NULL);

        if (list->head == list->tail) {
                // single inst case.
                assert(list->head == inst);
                list->head = NULL;
                list->tail = NULL;
        } else if (list->tail == inst) {
                // tail case.
                assert(inst->next == NULL);
                inst->prev->next = NULL;
                list->tail       = inst->prev;
        } else if (list->head == inst) {
                assert(inst->prev == NULL);
                inst->next->prev = NULL;
                list->head       = inst->next;
        } else {
                // general case.
                inst->prev->next = inst->next;
                inst->next->prev = inst->prev;
        }
        list->count--;
        free(inst);
}

void
bbInstListDump(struct bb_inst_list_t *list, sds_t *s)
{
        sdsCatPrintf(s, "{  // ops\n");
        if (list->head == NULL) {
                sdsCatPrintf(s, "  (empty)\n");
                sdsCatPrintf(s, "}\n");
                return;
        }

        struct bb_inst_t *curr;
        curr = list->head;
        while (curr != NULL) {
                sdsCatPrintf(s, "  ");
                bbInstDump(curr, s);
                sdsCatPrintf(s, "\n");
                curr = curr->next;
        }
        sdsCatPrintf(s, "}\n");
}

// -----------------------------------------------------------------------------
// Impl for Program.
// -----------------------------------------------------------------------------

struct bb_program_t *
bbProgNew()
{
        struct bb_program_t *p = calloc(1, sizeof(struct bb_program_t));
        if (p == NULL) return NULL;
        bbInstListReset(&p->inst_list);
        return p;
}
void
bbProgFree(struct bb_program_t *p)
{
        if (p == NULL) return;
        bbInstListFree(&p->inst_list);
        vecFree(p->inputs);
        vecFree(p->labels);
        vecFree(p->outputs);
        vecFree(p->weights);
        vecFree(p->grads);
        vecFree(p->states);
        free(p);
}

void
bbProgAppend(struct bb_program_t *p, struct oparg_t *op)
{
        bbInstListAppend(&p->inst_list, op);
}

error_t
bbProgCompileToBatchOps(struct bb_program_t *p, int *out_count,
                        struct oparg_t **out)
{
        struct bb_inst_list_t *list  = &p->inst_list;
        size_t                 count = list->count;
        if (count == 0) {
                *out_count = 0;
                *out       = NULL;
                return OK;
        }

        struct oparg_t *ops = malloc(count * sizeof(struct oparg_t));

        struct bb_inst_t *curr = list->head;
        for (size_t i = 0; i < count; i++) {
                assert(curr != NULL);
                *(ops + i) = curr->op;
                curr       = curr->next;
        }

        *out_count = count;
        *out       = ops;
        return OK;
}

void
bbProgDump(struct bb_program_t *p, sds_t *s)
{
        sdsCatPrintf(s, "program:\n");

#define PRINT_COLLECTION(collection)                                        \
        {                                                                   \
                sdsCatPrintf(s, "{  // " #collection "\n  ");               \
                size_t size = vecSize(p->collection);                       \
                if (size) {                                                 \
                        for (int i = 0; i < size; i++) {                    \
                                sdsCatPrintf(s, "%3d, ", p->collection[i]); \
                        }                                                   \
                        sdsCatPrintf(s, "\n");                              \
                } else {                                                    \
                        sdsCatPrintf(s, "(empty)\n");                       \
                }                                                           \
                sdsCatPrintf(s, "}\n");                                     \
        }

        PRINT_COLLECTION(inputs);
        PRINT_COLLECTION(labels);
        PRINT_COLLECTION(outputs);
        PRINT_COLLECTION(weights);
        PRINT_COLLECTION(grads);
        PRINT_COLLECTION(states);

#undef PRINT_COLLECTION

        bbInstListDump(&p->inst_list, s);
}
