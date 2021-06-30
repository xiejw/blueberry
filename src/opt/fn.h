#ifndef BB_FN_H
#define BB_FN_H

#include "vm.h"

// eva
#include "adt/dict.h"
#include "adt/vec.h"

#include "bb.h"

// -----------------------------------------------------------------------------
// Fn
// -----------------------------------------------------------------------------

struct bb_fn_t {
        vec_t(int) inputs;
        vec_t(int) outputs;
        struct bb_inst_list_t inst_list;
};

struct bb_fn_t *bbFnNew();
void            bbFnFree(struct bb_fn_t *);
void            bbFnAppend(struct bb_fn_t *, struct oparg_t *op);
void            bbFnDump(struct bb_fn_t *, _mut_ sds_t *s);

// -----------------------------------------------------------------------------
// Ctx
// -----------------------------------------------------------------------------

struct bb_fn_ctx_t;

typedef error_t (*bb_fn_pass_t)(struct bb_fn_t *, struct bb_fn_ctx_t *, int *);

struct bb_fn_ctx_t {
        int     debug_mode;
        dict_t *fns;  // key is str (owned) value is fn (unowned).
        vec_t(bb_fn_pass_t) passes;
};

struct bb_fn_ctx_t *bbFnCtxNew();
void                bbFnCtxFree(struct bb_fn_ctx_t *);

error_t bbFnCtxAddFn(struct bb_fn_ctx_t *, const char *name, struct bb_fn_t *);
struct bb_fn_t *bbFnCtxLookUpFn(struct bb_fn_ctx_t *, const char *name);

error_t bbFnCtxAddFnPass(struct bb_fn_ctx_t *, bb_fn_pass_t);
error_t bbFnCtxRunPasses();

// -----------------------------------------------------------------------------
// Passes
// -----------------------------------------------------------------------------

// pass_dce.c
error_t runDCEPass(struct bb_fn_t *fn, struct bb_fn_ctx_t *,
                   _mut_ int      *changed);
// pass_math.c
error_t runMathPass(struct bb_fn_t *fn, struct bb_fn_ctx_t *,
                    _mut_ int      *changed);

#endif
