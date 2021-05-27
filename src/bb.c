#include "bb.h"

#include <assert.h>
#include <string.h>

// -----------------------------------------------------------------------------
// Impl.
// -----------------------------------------------------------------------------
error_t bbDenseLayer(struct vm_t *vm, const struct bb_dense_config_t *cfg,
                     struct bb_layer_t **out) {
  // error checks.
  if (cfg->input_dim <= 0)
    return errNew("input dim must be positive; got %d", cfg->input_dim);
  if (cfg->output_dim <= 0)
    return errNew("output dim must be positive; got %d", cfg->output_dim);
  if (!(cfg->actn == BB_ACTN_NONE || cfg->actn == BB_ACTN_RELU))
    return errNew("acvn must be NONE or RELU; got %d", cfg->actn);

  struct bb_dense_layer_t *l = malloc(sizeof(struct bb_dense_layer_t));
  memset(l, 0, sizeof(struct bb_dense_layer_t));
  l->config = *cfg;

  //*out = (strl;
  return OK;
}

error_t bbSCELLayer(struct vm_t *vm, const struct bb_scel_config_t *cfg,
                    struct bb_layer_t **out) {
  // error checks.
  if (cfg->input_dim <= 0)
    return errNew("input dim must be positive; got %d", cfg->input_dim);
  if (!(cfg->reduction == BB_REDUCTION_SUM ||
        cfg->reduction == BB_REDUCTION_MEAN))
    return errNew("reduction must be SUM or MEAN; got %d", cfg->reduction);

  return OK;
}
