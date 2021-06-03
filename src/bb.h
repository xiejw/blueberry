#ifndef BB_H_
#define BB_H_

// eva
#include "adt/sds.h"
#include "adt/vec.h"
#include "base/defs.h"
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
        vec_t(int) inputs;   // ele unowned.
        vec_t(int) labels;   // ele unowned.
        vec_t(int) outputs;  // ele unowned.
        vec_t(int) weights;  // ele unowned.
        vec_t(int) grads;    // ele unowned.
        vec_t(int) states;   // ele unowned.
        int               count;
        struct bb_inst_t *head;
        struct bb_inst_t *tail;
};

struct bb_program_t *bbProgNew();
void                 bbProgFree(struct bb_program_t *);
void                 bbProgAppend(struct bb_program_t *, struct oparg_t *);
void                 bbProgDump(struct bb_program_t *, _mut_ sds_t *);
error_t bbProgCompileToBatchOps(struct bb_program_t *, _out_ int *count,
                                _out_ struct oparg_t **out);

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

#define BB_OPT_SGD     0
#define BB_OPT_RMSPROP 1
#define BB_OPT_ADAM    2

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

        error_t (*weights)(struct bb_layer_t *, _mut_ vec_t(int) * tds);
        error_t (*grads)(struct bb_layer_t *, _mut_ vec_t(int) * tds);
        error_t (*states)(struct bb_layer_t *, _mut_ vec_t(int) * tds);

        error_t (*jit)(struct bb_layer_t *, const struct bb_context_t *,
                       struct bb_program_t *, int     direction,
                       const vec_t(int) inputs, _mut_ vec_t(int) * outputs);

        // metric only
        error_t (*summary)(struct bb_layer_t *, _mut_ void *data, int flag);
};

// Same for Layer, Loss, Metric.
struct bb_layer_t {
        vec_t(int) weights;  // weights.
        vec_t(int) grads;    // grads for weights in order.
        vec_t(int) states;   // grads for weights in order.
        vec_t(int) ivs;      // intermediate values.

        struct vm_t *                vm;  // unowned.
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
        struct vm_t *vm;  // unowned.
        int          type;
        vec_t(int) weights;  // unowned.
        vec_t(int) grads;    // unowned
        vec_t(int) states;   // owned.
        vec_t(int) ivs;      // owned.
        void *config;        // owned.
        void *private_data;  // owned.
};

struct bb_opt_rmsprop_config_t {
        float rho;      // suggested 0.9
        float epsilon;  // suggested 1e-8
};

struct bb_opt_adam_config_t {
        float beta_1;   // suggested 0.9
        float beta_2;   // suggested 0.999
        float epsilon;  // suggested 1e-8
};

error_t bbOptNew(struct vm_t *vm, int type, float32_t lr, void *cfg,
                 _out_ struct bb_opt_t **);
error_t bbOptInit(struct bb_opt_t *, vec_t(int) weights, vec_t(int) grads);
error_t bbOptApply(struct bb_opt_t *, struct bb_program_t *);
void    bbOptFree(struct bb_opt_t *);

// -----------------------------------------------------------------------------
// Module APIs.  // module.c
// -----------------------------------------------------------------------------

struct bb_seq_module_t {
        int x;
        int y;
        vec_t(struct bb_layer_t *) layers;  // owned.
        struct bb_layer_t *loss;            // owned.
        struct bb_opt_t *  opt;             // owned.
        struct bb_layer_t *metric;          // owned.
        struct srng64_t *  r;               // owned.
};

struct bb_seq_module_t *bbSeqModuleNew();
void                    bbSeqModuleFree(struct bb_seq_module_t *);

error_t bbCompileSeqModule(const struct bb_context_t *ctx,
                           struct bb_program_t *p, struct bb_seq_module_t *);

// -----------------------------------------------------------------------------
// Experimental way to create Layers.
// -----------------------------------------------------------------------------
//
// Thoughts: This creates a bunch of layers. It is good, but only for sequential
// layers. For residual network, how to express?
#define BB_TAG_NULL  0
#define BB_TAG_DENSE 1
#define BB_TAG_SCEL  2

struct bb_layer_config_t {
        int   tag;
        void *config;  // unowned
};

error_t bbCreateLayers(struct vm_t *                   vm,
                       const struct bb_layer_config_t *layer_configs,
                       _out_ vec_t(struct bb_layer_t *) * layers);

#endif
