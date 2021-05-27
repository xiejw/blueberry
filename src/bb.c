#include "bb.h"

#include <assert.h>
#include <string.h>

// -----------------------------------------------------------------------------
// Impl for Dense.
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

error_t _bbDenseWeights(void *self, const struct bb_context_t *ctx,
                        vec_t(int) * tds) {
  struct bb_dense_layer_t *this = self;
  const struct bb_dense_config_t *cfg = &this->config;
  vec_t(int) weights = vecNew();
  vecReserve(weights, 2);
  vecPushBack(weights, this->w);
  if (cfg->has_bias)
    vecPushBack(weights, this->b);

  *tds = weights;
  return OK;
}

error_t _bbDenseGrads(void *self, const struct bb_context_t *ctx,
                      vec_t(int) * tds) {
  struct bb_dense_layer_t *this = self;
  const struct bb_dense_config_t *cfg = &this->config;
  vec_t(int) weights = vecNew();
  vecReserve(weights, 2);
  vecPushBack(weights, this->d_w);
  if (cfg->has_bias)
    vecPushBack(weights, this->d_b);

  *tds = weights;
  return OK;
}

error_t _bbDenseInit(void *self, const struct bb_context_t *ctx) {
  // struct bb_dense_layer_t *this = self;
  // struct vm_t *vm = ctx->vm;
  // const struct bb_dense_config_t *cfg = &this->config;
  // int has_bias = cfg->has_bias;

  // TODO init the weights and logits.

  return OK;
}
error_t _bbDenseRelease(void *self, const struct bb_context_t *ctx) {
  struct bb_dense_layer_t *this = self;
  struct vm_t *vm = ctx->vm;

  vec_t(int) tds = this->tds;
  int size = vecSize(tds);
  error_t err;
  for (int i = 0; i < size; i++) {
    err = vmTensorFree(vm, tds[i]);
    if (err) {
      return errEmitNote("failed to release internal logit tensor");
    }
  }
  vecFree(tds);
  this->tds = vecNew();
  return OK;
}

// -----------------------------------------------------------------------------
// Impl for SCEL.
// -----------------------------------------------------------------------------
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
