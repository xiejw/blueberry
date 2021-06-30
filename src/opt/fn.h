#ifndef BB_FN_H
#define BB_FN_H

#include "vm.h"

#include "adt/dict.h"
#include "adt/vec.h"

#include "bb.h"

// fn

struct bb_fn_t {
        vec_t(int) inputs;
        vec_t(int) outputs;
        struct bb_inst_list_t inst_list;
};

struct bb_fn_t *bbFnNew();
void            bbFnFree(struct bb_fn_t *);
void            bbFnAppend(struct bb_fn_t *, struct oparg_t *op);
void            bbFnDump(struct bb_fn_t *, _mut_ sds_t *s);

// ctx

struct bb_fn_ctx_t {
        void   *cfg;
        dict_t *fns;  // key is str (owned) value is fn (unowned).
};

struct bb_fn_ctx_t *bbFnCtxNew();
void                bbFnCtxFree(struct bb_fn_ctx_t *);

// passes

// pass shape inf
error_t runShapeInf();
// pass_dce.c
error_t runDCEPass(struct bb_fn_t *fn, void *cfg, int debug,
                   _mut_ int *changed);
// pass_math.c
error_t runMathPass(struct bb_fn_t *fn, void *cfg, int debug,
                    _mut_ int *changed);

#endif
