#ifndef BB_FN_H
#define BB_FN_H

#include "vm.h"

#include "adt/vec.h"

struct bb_fn_t {
        vec_t(int) inputs;
        vec_t(int) outputs;
        struct bb_inst_t *head;
        struct bb_inst_t *tail;
};

struct bb_fn_t *bbFnNew();
void            bbFnFree();

#endif
