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
void            bbFnFree(struct bb_fn_t*);
void            bbFnAppend(struct bb_fn_t*, struct oparg_t* op);
void            bbFnDump(struct bb_fn_t*, sds_t* s);

// pass_dce.c
error_t runDCEPass(struct bb_fn_t* fn, void* cfg, int debug, int* changed);

#endif
