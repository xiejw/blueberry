#ifndef BB_FN_H
#define BB_FN_H

#include "vm.h"

#include "adt/vec.h"

#include "bb.h"

struct bb_fn_t {
        vec_t(int) inputs;
        vec_t(int) outputs;
        struct bb_inst_list_t inst_list;
};

struct bb_fn_t* bbFnNew();
void            bbFnFree();

error_t runDCEPass(struct bb_fn_t* fn, void* cfg, int* changed);

#endif
