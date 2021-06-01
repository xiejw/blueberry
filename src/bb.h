#ifndef BB_H_
#define BB_H_

// eva
#include "adt/sds.h"
#include "adt/vec.h"
#include "base/error.h"

// mlvm
#include "vm.h"

// -----------------------------------------------------------------------------
// Program APIs.  // prop.c
// -----------------------------------------------------------------------------

struct vm_t *bbVmInit();  // put some preallocated tds.

struct bb_inst_t {
        struct oparg_t    op;
        struct bb_inst_t *next;
        struct bb_inst_t *prev;
};

struct bb_program_t {
        vec_t(int) inputs;
        vec_t(int) labels;
        vec_t(int) outputs;
        vec_t(int) weights;
        vec_t(int) grads;
        struct bb_inst_t *head;
        struct bb_inst_t *tail;
};

struct bb_program_t *bbProgNew();
void                 bbProgFree(struct bb_program_t *);
void                 bbProgAppend(struct bb_program_t *, struct oparg_t *);
void                 bbProgDump(struct bb_program_t *, sds_t *);

// -----------------------------------------------------------------------------
// Constants.
// -----------------------------------------------------------------------------
#define BB_FORWARD  0
#define BB_BACKWARD 1

#define BB_ACTN_NONE 0
#define BB_ACTN_RELU 1

#define BB_INIT_NULL       0
#define BB_INIT_ZERO       1
#define BB_INIT_STD_NORMAL 2
#define BB_INIT_STOPPER    3  // should not use

#define BB_REDUCTION_SUM  0
#define BB_REDUCTION_MEAN 1

#define BB_OPT_SGD 0

#define BB_FLAG_NONE  0
#define BB_FLAG_RESET 1

// -----------------------------------------------------------------------------
// Layer, Loss, Metric APIs.
// -----------------------------------------------------------------------------

struct bb_context_t {
        int is_training;
};

struct bb_layer_t;

struct bb_layer_operations_t {
        error_t (*init)(struct bb_layer_t *, const struct bb_context_t *,
                        struct srng64_t *);
        error_t (*release)(struct bb_layer_t *);

        error_t (*weights)(struct bb_layer_t *, vec_t(int) * tds);
        error_t (*grads)(struct bb_layer_t *, vec_t(int) * tds);

        error_t (*jit)(struct bb_layer_t *, const struct bb_context_t *,
                       struct bb_program_t *, int direction,
                       const vec_t(int) inputs, vec_t(int) * outputs);

        // metric only
        error_t (*summary)(struct bb_layer_t *, void *data, int flag);
};

// Same for Layer, Loss, Metric.
struct bb_layer_t {
        vec_t(int) weights;  // weights.
        vec_t(int) grads;    // grads for weights in order.
        vec_t(int) ivs;      // intermediate values.

        struct vm_t *                vm;
        struct bb_layer_operations_t ops;
};

void bbLayerFree(struct bb_layer_t *);

// The following header file provides the layer definition and factory method to
// create them.
#include "bb_layers.h"

// -----------------------------------------------------------------------------
// Optimizer APIs.  // opt.c
// -----------------------------------------------------------------------------

struct bb_opt_t {
        float32_t    lr;
        struct vm_t *vm;
        int          type;
        vec_t(int) weights;  // unowned.
        vec_t(int) grads;    // unowned
        vec_t(int) states;   // owned.
        void *private_data;
};

error_t bbOptNew(struct vm_t *vm, int type, float32_t lr, struct bb_opt_t **);
error_t bbOptInit(struct bb_opt_t *, vec_t(int) weights, vec_t(int) grads);
error_t bbOptApply(struct bb_opt_t *, struct bb_program_t *);
void    bbOptFree(struct bb_opt_t *);

// -----------------------------------------------------------------------------
// Module APIs.  // module.c
// -----------------------------------------------------------------------------

error_t bbCompileSeqModule(const struct bb_context_t *ctx,
                           struct bb_program_t *p, int x, int y,
                           vec_t(struct bb_layer_t *) layers,
                           struct bb_layer_t *loss, struct bb_opt_t *opt,
                           struct srng64_t *r);

#endif
