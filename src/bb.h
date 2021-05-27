#ifndef BB_H_
#define BB_H_

// eva
#include "adt/vec.h"
#include "base/error.h"

// mlvm
#include "vm.h"

// -----------------------------------------------------------------------------
// Layer APIs.
// -----------------------------------------------------------------------------

struct bb_program_t {
  vec_t(struct oparg_t) ops;
};

struct bb_context_t {
  struct vm_t *vm;
  int is_training;
};

struct bb_layer_t {
  error_t (*init)(void *, const struct bb_context_t *);
  error_t (*release)(void *, const struct bb_context_t *);
  error_t (*states)(void *, const struct bb_context_t *, vec_t(int) * tds);
  error_t (*jit)(void *, const struct bb_context_t *, struct bb_program_t *,
                 int direction, const vec_t(int) inputs, vec_t(int) * *outputs);
};

// -----------------------------------------------------------------------------
// Dense layer.
// -----------------------------------------------------------------------------

#define BB_ACTN_NONE 0
#define BB_ACTN_RELU 1

struct bb_dense_config_t {
  int input_dim;
  int output_dim;
  int has_bias;
  int actn;
};

struct bb_dense_layer_t {
  struct bb_dense_config_t config;
  int w; // kernel
  int b; // bias. -1 means absent.
};

error_t bbDenseLayer(struct vm_t *, const struct bb_dense_config_t *,
                     struct bb_layer_t **);

// -----------------------------------------------------------------------------
// Softmax Crossentropy Loss Layer.
// -----------------------------------------------------------------------------

#define BB_REDUCTION_SUM 0
#define BB_REDUCTION_MEAN 1

struct bb_scel_config_t {
  int input_dim;
  int reduction;
};

struct bb_scel_layer_t {
  struct bb_scel_config_t config;
};

error_t bbSCELLayer(struct vm_t *, const struct bb_scel_config_t *,
                    struct bb_layer_t **);

#endif
