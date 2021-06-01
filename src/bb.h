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
// Layer APIs.
// -----------------------------------------------------------------------------

#define BB_FORWARD  0
#define BB_BACKWARD 1

struct bb_layer_t;

struct bb_context_t {
        int is_training;
};

struct bb_layer_operations_t {
        error_t (*init)(struct bb_layer_t *, const struct bb_context_t *,
                        struct srng64_t *);
        error_t (*release)(struct bb_layer_t *);

        error_t (*weights)(struct bb_layer_t *, vec_t(int) * tds);
        error_t (*grads)(struct bb_layer_t *, vec_t(int) * tds);

        error_t (*jit)(struct bb_layer_t *, const struct bb_context_t *,
                       struct bb_program_t *, int direction,
                       const vec_t(int) inputs, vec_t(int) * outputs);
};

struct bb_layer_t {
        vec_t(int) weights;  // weights.
        vec_t(int) grads;    // grads for weights in order.
        vec_t(int) ivs;      // intermediate values.

        struct vm_t *                vm;
        struct bb_layer_operations_t ops;
};

void bbLayerFree(struct bb_layer_t *);

// -----------------------------------------------------------------------------
// Module APIs.  // module.c
// -----------------------------------------------------------------------------

struct bb_opt_t;

error_t bbCompileSeqModule(const struct bb_context_t *ctx,
                           struct bb_program_t *p, int x, int y,
                           vec_t(struct bb_layer_t *) layers,
                           struct bb_layer_t *loss, struct bb_opt_t *opt,
                           struct srng64_t *r);

// -----------------------------------------------------------------------------
// Dense layer.
// -----------------------------------------------------------------------------

#define BB_ACTN_NONE 0
#define BB_ACTN_RELU 1

#define BB_INIT_NULL       0
#define BB_INIT_ZERO       1
#define BB_INIT_STD_NORMAL 2
#define BB_INIT_STOPPER    3  // should not use

struct bb_dense_config_t {
        int input_dim;
        int output_dim;
        int kernel_init;
        int bias_init;  // NULL => absent
        int actn;
};

struct bb_dense_layer_t {
        struct bb_layer_t base;

        struct bb_dense_config_t config;

        // weights
        int w;  // kernel
        int b;  // bias. 0 means absent.

        // grads.
        int d_w;
        int d_b;

        // iv
        int h, hb, y;          // forward
        int state, d_hb, d_x;  // backward
};

error_t bbDenseLayer(struct vm_t *, const struct bb_dense_config_t *,
                     struct bb_layer_t **);

// -----------------------------------------------------------------------------
// Softmax Crossentropy Loss Layer.
// -----------------------------------------------------------------------------

#define BB_REDUCTION_SUM  0
#define BB_REDUCTION_MEAN 1

struct bb_scel_config_t {
        int reduction;
};

struct bb_scel_layer_t {
        struct bb_layer_t base;

        struct bb_scel_config_t config;

        // iv
        int o, r;  // forward
        int d_x;   // backward

        // other
        int batch_size;
        int input_dim;
};

error_t bbSCELLayer(struct vm_t *, const struct bb_scel_config_t *,
                    struct bb_layer_t **);

#endif
