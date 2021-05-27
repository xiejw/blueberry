#include "bb.h"

#include <assert.h>
#include <string.h>

static error_t _bbInitTensor(struct vm_t *vm, int td, int mode,
                             struct rng64_t *rng) {
  struct opopt_t opt;
  switch (mode) {
  case BB_INIT_ZERO:
    return vmExec(vm, OP_FILL, NULL, td, -1, -1);
  case BB_INIT_STD_NORMAL:
    opt.mode = OPT_RNG_STD_NORMAL | OPT_MODE_R_BIT;
    opt.r = *rng;
    return vmExec(vm, OP_RNG, &opt, td, -1, -1);
  default:
    return errNew("init mode is not supported: %d", mode);
  }
}

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
  vecPushBack(*tds, this->w);
  if (cfg->bias_init != BB_INIT_NULL)
    vecPushBack(*tds, this->b);
  return OK;
}

error_t _bbDenseGrads(void *self, const struct bb_context_t *ctx,
                      vec_t(int) * tds) {
  struct bb_dense_layer_t *this = self;
  const struct bb_dense_config_t *cfg = &this->config;
  vecPushBack(*tds, this->d_w);
  if (cfg->bias_init != BB_INIT_NULL)
    vecPushBack(*tds, this->d_b);
  return OK;
}

error_t _bbDenseInit(void *self, const struct bb_context_t *ctx,
                     struct rng64_t *rng) {
  struct bb_dense_layer_t *this = self;
  struct vm_t *vm = ctx->vm;
  const struct bb_dense_config_t *cfg = &this->config;
  int has_bias = cfg->bias_init != BB_INIT_NULL;

  // stage 1: error check
  if (!(cfg->kernel_init > BB_INIT_NULL && cfg->kernel_init < BB_INIT_STOPPER))
    return errNew("kernel init is out of range; got %d", cfg->kernel_init);
  if (!(cfg->bias_init >= BB_INIT_NULL && cfg->bias_init < BB_INIT_STOPPER))
    return errNew("bias init is out of range; got %d", cfg->bias_init);

  // stage 2: create the shapes.
  struct shape_t *sp_w = R2S(vm, cfg->input_dim, cfg->output_dim);
  int w = vmTensorNew(vm, F32, sp_w);
  _bbInitTensor(vm, w, cfg->kernel_init, rng);
  this->w = w;
  vecPushBack(this->tds, w);

  if (has_bias) {
    struct shape_t *sp_b = R1S(vm, cfg->output_dim);
    int b = vmTensorNew(vm, F32, sp_b);
    _bbInitTensor(vm, b, cfg->bias_init, rng);
    this->b = b;
    vecPushBack(this->tds, b);
  }
  return OK;
}

// error_t _bbDenseJitAndInit(void *self, const struct bb_context_t *ctx,
//                           struct bb_program_t * p, int direction,
//                           const vec_t(int) inputs, vec_t(int) * *outputs) {
//   struct bb_dense_layer_t *this = self;
//   struct vm_t *vm = ctx->vm;
//   const struct bb_dense_config_t *cfg = &this->config;
//   int has_bias = cfg->has_bias;
//
//   // stage 1: error check
//
//   // stage 1: create the shapes.
//   struct shape_t *sp_w = R2S(vm, cfg->input_dim, cfg->output_dim);
//   struct shape_t *sp_h1 = R2S(vm, bs, h1_s);
//   struct shape_t *sp_b1 = R1S(vm, h1_s);
//   struct shape_t *sp_w2 = R2S(vm, h1_s, h2_s);
//   struct shape_t *sp_h2 = R2S(vm, bs, h2_s);
//   struct shape_t *sp_b2 = R1S(vm, h2_s);
//   struct shape_t *sp_w3 = R2S(vm, h2_s, ls);
//
//   int x = vmTensorNew(vm, F32, sp_x);
//   int y = vmTensorNew(vm, F32, sp_y);
//   int z = vmTensorNew(vm, F32, sp_scalar);
//   int w1 = vmTensorNew(vm, F32, sp_w1);
//   int h1 = vmTensorNew(vm, F32, sp_h1);
//   int b1 = vmTensorNew(vm, F32, sp_b1);
//   int h1b = vmTensorNew(vm, F32, sp_h1);
//   int z1 = vmTensorNew(vm, F32, sp_h1);
//
//   // TODO init the weights and logits.
//
//   return OK;
// }
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
