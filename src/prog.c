#include "bb.h"

#include <string.h>

// -----------------------------------------------------------------------------
// Impl for Program.
// -----------------------------------------------------------------------------

struct bb_program_t *
bbProgNew()
{
        struct bb_program_t *p = malloc(sizeof(struct bb_program_t));
        if (p == NULL) return NULL;
        memset(p, 0, sizeof(struct bb_program_t));
        return p;
}

void
bbProgFree(struct bb_program_t *p)
{
        if (p == NULL) return;
        struct bb_inst_t *next, *curr;
        curr = p->head;
        while (curr != NULL) {
                next = curr->next;
                free(curr);
                curr = next;
        }
        vecFree(p->inputs);
        vecFree(p->labels);
        vecFree(p->outputs);
        vecFree(p->weights);
        vecFree(p->grads);
        free(p);
}

void
bbProgAppend(struct bb_program_t *p, struct oparg_t *op)
{
        struct bb_inst_t *inst = malloc(sizeof(struct bb_inst_t));
        inst->op               = *op;
        inst->next             = NULL;

        struct bb_inst_t *tail = p->tail;

        if (p->head == NULL) {
                inst->prev = NULL;
                p->head    = inst;
                p->tail    = inst;
        } else {
                inst->prev = tail;
                tail->next = inst;
                p->tail    = inst;
        }
}

void
bbProgDump(struct bb_program_t *p, sds_t *s)
{
        sdsCatPrintf(s, "program:\n");

        {
                sdsCatPrintf(s, "{  // inputs\n  ");
                size_t size = vecSize(p->inputs);
                if (size) {
                        for (int i = 0; i < size; i++) {
                                sdsCatPrintf(s, "%3d, ", p->inputs[i]);
                        }
                        sdsCatPrintf(s, "\n");
                } else {
                        sdsCatPrintf(s, "(empty)\n");
                }
                sdsCatPrintf(s, "}\n");
        }

        {
                sdsCatPrintf(s, "{  // labels\n  ");
                size_t size = vecSize(p->labels);
                if (size) {
                        for (int i = 0; i < size; i++) {
                                sdsCatPrintf(s, "%3d, ", p->labels[i]);
                        }
                        sdsCatPrintf(s, "\n");
                } else {
                        sdsCatPrintf(s, "(empty)\n");
                }
                sdsCatPrintf(s, "}\n");
        }

        {
                sdsCatPrintf(s, "{  // outputs\n  ");
                size_t size = vecSize(p->outputs);
                if (size) {
                        for (int i = 0; i < size; i++) {
                                sdsCatPrintf(s, "%3d, ", p->outputs[i]);
                        }
                        sdsCatPrintf(s, "\n");
                } else {
                        sdsCatPrintf(s, "(empty)\n");
                }
                sdsCatPrintf(s, "}\n");
        }

        {
                sdsCatPrintf(s, "{  // weights\n  ");
                size_t size = vecSize(p->weights);
                if (size) {
                        for (int i = 0; i < size; i++) {
                                sdsCatPrintf(s, "%3d, ", p->weights[i]);
                        }
                        sdsCatPrintf(s, "\n");
                } else {
                        sdsCatPrintf(s, "(empty)\n");
                }
                sdsCatPrintf(s, "}\n");
        }

        {
                sdsCatPrintf(s, "{  // grads\n  ");
                size_t size = vecSize(p->grads);
                if (size) {
                        for (int i = 0; i < size; i++) {
                                sdsCatPrintf(s, "%3d, ", p->grads[i]);
                        }
                        sdsCatPrintf(s, "\n");
                } else {
                        sdsCatPrintf(s, "(empty)\n");
                }
                sdsCatPrintf(s, "}\n");
        }

        {
                sdsCatPrintf(s, "{  // ops\n");
                if (p->head == NULL) {
                        sdsCatPrintf(s, "  (empty)\n");
                        return;
                }

                struct bb_inst_t *curr;
                curr = p->head;
                while (curr != NULL) {
                        char *opname;
                        switch (curr->op.op) {
                        case OP_MATMUL:
                                opname = "OP_MATMUL";
                                break;
                        case OP_MUL:
                                opname = "OP_MUL";
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
                        case OP_LS_SCEL:
                                opname = "OP_LS_SCEL";
                                break;
                        default:
                                opname = "UNKNOWN";
                        }
                        struct oparg_t *op = &curr->op;
                        sdsCatPrintf(s,
                                     "  {.op = %2d (%-10s)}, .dst = %3d, .t1 = "
                                     "%3d, .t2 = %3d",
                                     op->op, opname, op->dst, op->t1, op->t2);
                        if (!op->has_opt) {
                                sdsCatPrintf(s, "}\n");
                        } else {
                                sdsCatPrintf(s, ", .mode = %d}\n",
                                             op->opt.mode);
                        }
                        curr = curr->next;
                }
                sdsCatPrintf(s, "}\n");
        }
}
