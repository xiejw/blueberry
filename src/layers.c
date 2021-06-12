#include "bb.h"

#include "vm.h"

#include <string.h>

// -----------------------------------------------------------------------------
// Helper.
// -----------------------------------------------------------------------------
static error_t
_bbInitTensor(struct vm_t *vm, int td, int mode, struct srng64_t *rng)
{
        struct srng64_t *r;
        struct opopt_t   opt;
        switch (mode) {
        case BB_INIT_ZERO:
                return vmExec(vm, OP_FILL, NULL, td, -1, -1);
        case BB_INIT_STD_NORMAL:
                r        = srng64Split(rng);
                opt.mode = OPT_RNG_STD_NORMAL | OPT_MODE_R_BIT;
                opt.r    = *(struct rng64_t *)r;
                free(r);
                return vmExec(vm, OP_RNG, &opt, td, -1, -1);
        default:
                return errNew("init mode is not supported: %d", mode);
        }
}

#define DECLARE_LAYER_METHODS(name)                                        \
        static error_t _bb##name##Init(struct bb_layer_t *,                \
                                       const struct bb_context_t *,        \
                                       struct srng64_t *);                 \
        static error_t _bb##name##Jit(                                     \
            struct bb_layer_t *, const struct bb_context_t *,              \
            struct bb_program_t *, int direction, const vec_t(int) inputs, \
            vec_t(int) * outputs)

// defs in bb.c
error_t _bbLayerGrads(struct bb_layer_t *, vec_t(int) *);
error_t _bbLayerWeights(struct bb_layer_t *, vec_t(int) *);
error_t _bbLayerStates(struct bb_layer_t *, vec_t(int) *);
error_t _bbLayerRelease(struct bb_layer_t *);

// -----------------------------------------------------------------------------
// Impl for Dense.
// -----------------------------------------------------------------------------
DECLARE_LAYER_METHODS(Dense);

error_t
bbDenseLayer(struct vm_t *vm, const struct bb_dense_config_t *cfg,
             struct bb_layer_t **out)
{
        // error checks.
        if (cfg->input_dim <= 0)
                return errNew("input dim must be positive; got %d",
                              cfg->input_dim);
        if (cfg->output_dim <= 0)
                return errNew("output dim must be positive; got %d",
                              cfg->output_dim);
        if (!(cfg->actn == BB_ACTN_NONE || cfg->actn == BB_ACTN_RELU))
                return errNew("acvn must be NONE or RELU; got %d", cfg->actn);

        if (!(cfg->kernel_init > BB_INIT_NULL &&
              cfg->kernel_init < BB_INIT_STOPPER))
                return errNew("kernel init is out of range; got %d",
                              cfg->kernel_init);
        if (!(cfg->bias_init >= BB_INIT_NULL &&
              cfg->bias_init < BB_INIT_STOPPER))
                return errNew("bias init is out of range; got %d",
                              cfg->bias_init);

        struct bb_dense_layer_t *l = malloc(sizeof(struct bb_dense_layer_t));
        memset(l, 0, sizeof(struct bb_dense_layer_t));
        l->base.vm = vm;
        l->config  = *cfg;

        struct bb_layer_operations_t *ops = &l->base.ops;
        ops->init                         = _bbDenseInit;
        ops->release                      = _bbLayerRelease;
        ops->weights                      = _bbLayerWeights;
        ops->grads                        = _bbLayerGrads;
        ops->states                       = _bbLayerStates;
        ops->jit                          = _bbDenseJit;

        *out = (struct bb_layer_t *)l;
        return OK;
}

error_t
_bbDenseInit(struct bb_layer_t *self, const struct bb_context_t *ctx,
             struct srng64_t *rng)
{
        struct bb_dense_layer_t *this       = (struct bb_dense_layer_t *)self;
        struct vm_t                    *vm  = self->vm;
        const struct bb_dense_config_t *cfg = &this->config;
        int has_bias                        = cfg->bias_init != BB_INIT_NULL;
        int is_training                     = ctx->is_training;

        // create the shapes, weights, and grads if training.
        struct shape_t *sp_w = R2S(vm, cfg->input_dim, cfg->output_dim);

#define ALLOC_STATE(name, sp, collection)         \
        int name = vmTensorNew(vm, F32, sp);      \
        vecPushBack(this->base.collection, name); \
        this->name = name;

        ALLOC_STATE(w, sp_w, weights);
        _bbInitTensor(vm, this->w, cfg->kernel_init, rng);
        if (is_training) {
                ALLOC_STATE(d_w, sp_w, grads);
        }

        if (has_bias) {
                struct shape_t *sp_b = R1S(vm, cfg->output_dim);
                ALLOC_STATE(b, sp_b, weights);
                _bbInitTensor(vm, b, cfg->bias_init, rng);

                if (is_training) {
                        ALLOC_STATE(d_b, sp_b, grads);
                }
        }

#undef ALLOC_STATE
        return OK;
}

error_t
_bbDenseJit(struct bb_layer_t *self, const struct bb_context_t *ctx,
            struct bb_program_t *p, int direction, const vec_t(int) inputs,
            vec_t(int) * outputs)
{
        error_t err;
        struct bb_dense_layer_t *this       = (struct bb_dense_layer_t *)self;
        struct vm_t                    *vm  = self->vm;
        const struct bb_dense_config_t *cfg = &this->config;
        int has_bias                        = cfg->bias_init != BB_INIT_NULL;
        int has_relu                        = cfg->actn == BB_ACTN_RELU;

        assert(cfg->actn == BB_ACTN_NONE || cfg->actn == BB_ACTN_RELU);
        assert(direction == BB_FORWARD || direction == BB_BACKWARD);

        // stage 1: error check and retrieve the batch size.
        if (vecSize(inputs) != 1)
                return errNew("expect one input for dense layer. got %d",
                              vecSize(inputs));

        int bs;
        {
                struct shape_t *sp_x;
                // checks the shape of input and gets the batch size.
                err = vmTensorInfo(vm, inputs[0], /*dtype=*/NULL, &sp_x);
                if (err) return errEmitNote("failed to grab the input shape.");
                if (sp_x->rank != 2)
                        return errNew(
                            "expect rank 2 input for dense layer. got %d",
                            sp_x->rank);

                if (direction == BB_FORWARD) {
                        if (sp_x->dims[1] != cfg->input_dim)
                                return errNew(
                                    "expect input.dims[1] == cfg.input_dim for "
                                    "dense layer. "
                                    "got %d vs %d",
                                    sp_x->dims[1], cfg->input_dim);
                } else {
                        if (sp_x->dims[1] != cfg->output_dim)
                                return errNew(
                                    "expect (grad)input.dims[1] == "
                                    "cfg.output_dim for dense "
                                    "layer. got %d vs %d",
                                    sp_x->dims[1], cfg->output_dim);
                }
                bs = sp_x->dims[0];
        }

#define ALLOC_T(name, sp)                                    \
        {                                                    \
                int name = vmTensorNew(self->vm, F32, (sp)); \
                vecPushBack(self->ivs, name);                \
                this->name = name;                           \
        }

        // stage 2: allocate intermediate values (iv).
        struct shape_t *sp_h = R2S(vm, bs, cfg->output_dim);
        if (direction == BB_FORWARD) {
                ALLOC_T(h, sp_h);
                if (has_bias) ALLOC_T(hb, sp_h);
                if (has_relu) ALLOC_T(y, sp_h);

        } else {
                if (has_relu) {
                        ALLOC_T(state, sp_h);
                        ALLOC_T(d_hb, sp_h);
                }
                struct shape_t *sp_x = R2S(vm, bs, cfg->input_dim);
                ALLOC_T(d_x, sp_x);
        }

#undef ALLOC_T

        // stage 3: jit
        if (direction == BB_FORWARD) {
                // emit forward.
                //   z[1] = zeros([1])
                //
                //   h [bs, out] = matmul(x[bs, in], w[in, out])
                //   hb[bs, out] = h[bs, out] + b[out]
                //   y [bs, out] = max(hb[bs, out], z[1])
                int x   = inputs[0];
                this->x = x;  // record for backprop

                bbProgAppend(
                    p, &(struct oparg_t){OP_MATMUL, this->h, x, this->w, 0});
                int y = this->h;
                if (has_bias) {
                        bbProgAppend(p, &(struct oparg_t){OP_ADD, this->hb, y,
                                                          this->b, 0});
                        y = this->hb;
                }
                if (has_relu) {
                        bbProgAppend(p, &(struct oparg_t){OP_MAX, this->y, y,
                                                          /*zero=*/0, 0});
                        y = this->y;
                }
                vecPushBack(*outputs, y);
                return OK;
        } else {
                //
                // emit backward
                //   state         = cmpL(hb[bs, out], z[1])
                //   d_hb[bs, out] = mul(d_z[bs, out], state)
                //
                //   d_h[bs, out]  = d_hb[bs, out]
                //   d_b[out]      = sum(d_hb[bs, out], axis=1)
                //
                //   d_w[in, out] = matmul(x[bs, in], d_h[bs, out], trans_a)
                //   d_x[bs, in]  = matmul(d_h[bs, out], w[h1, out] trans_b)
                int d_y = inputs[0];
                if (has_relu) {
                        bbProgAppend(
                            p, &(struct oparg_t){OP_CMPL, this->state, this->hb,
                                                 /*zero=*/0, 0});
                        bbProgAppend(p, &(struct oparg_t){OP_MUL, this->d_hb,
                                                          d_y, this->state, 0});
                        d_y = this->d_hb;
                }
                if (has_bias) {
                        bbProgAppend(p, &(struct oparg_t){
                                            OP_REDUCE,
                                            this->d_b,
                                            d_y,
                                            -1,
                                            1,
                                            {.mode = OPT_MODE_I_BIT, .i = 1}});
                }
                bbProgAppend(p,
                             &(struct oparg_t){OP_MATMUL,
                                               this->d_w,
                                               this->x,
                                               d_y,
                                               1,
                                               {.mode = OPT_MATMUL_TRANS_LHS}});
                bbProgAppend(p,
                             &(struct oparg_t){OP_MATMUL,
                                               this->d_x,
                                               d_y,
                                               this->w,
                                               1,
                                               {.mode = OPT_MATMUL_TRANS_RHS}});
                vecPushBack(*outputs, this->d_x);
        }
        return OK;
}

// -----------------------------------------------------------------------------
// Impl for SCEL.
// -----------------------------------------------------------------------------
DECLARE_LAYER_METHODS(SCEL);

error_t
bbSCELLayer(struct vm_t *vm, const struct bb_scel_config_t *cfg,
            struct bb_layer_t **out)
{
        // error checks.
        if (!(cfg->reduction == BB_REDUCTION_SUM ||
              cfg->reduction == BB_REDUCTION_MEAN))
                return errNew("reduction must be SUM or MEAN; got %d",
                              cfg->reduction);

        struct bb_scel_layer_t *l = malloc(sizeof(struct bb_scel_layer_t));
        memset(l, 0, sizeof(struct bb_scel_layer_t));
        l->base.vm = vm;
        l->config  = *cfg;

        struct bb_layer_operations_t *ops = &l->base.ops;
        ops->init                         = _bbSCELInit;
        ops->release                      = _bbLayerRelease;
        ops->weights                      = _bbLayerWeights;
        ops->grads                        = _bbLayerGrads;
        ops->states                       = _bbLayerStates;
        ops->jit                          = _bbSCELJit;

        *out = (struct bb_layer_t *)l;
        return OK;
}

error_t
_bbSCELInit(struct bb_layer_t *self, const struct bb_context_t *ctx,
            struct srng64_t *rng)
{
        // no weights/grads.
        return OK;
}

error_t
_bbSCELJit(struct bb_layer_t *self, const struct bb_context_t *ctx,
           struct bb_program_t *p, int direction, const vec_t(int) inputs,
           vec_t(int) * outputs)
{
        error_t err;
        struct bb_scel_layer_t *this       = (struct bb_scel_layer_t *)self;
        struct vm_t                   *vm  = self->vm;
        const struct bb_scel_config_t *cfg = &this->config;
        int                            is_training = ctx->is_training;
        int reduce_mean = cfg->reduction == BB_REDUCTION_MEAN;

        assert(cfg->reduction == BB_REDUCTION_SUM ||
               cfg->reduction == BB_REDUCTION_MEAN);
        assert(direction == BB_FORWARD || direction == BB_BACKWARD);

        // stage 1: error check. retrieve the batch size and input_dim;
        int bs;
        int input_dim;
        {
                if (direction == BB_FORWARD) {
                        if (vecSize(inputs) != 2)
                                return errNew(
                                    "expect two inputs for scel layer. got %d",
                                    vecSize(inputs));

                        struct shape_t *sp_x;
                        err =
                            vmTensorInfo(vm, inputs[1], /*dtype=*/NULL, &sp_x);
                        if (err)
                                return errEmitNote(
                                    "failed to grab the input shape.");
                        if (sp_x->rank != 2)
                                return errNew(
                                    "expect rank 2 input for scel layer. got "
                                    "%d",
                                    sp_x->rank);

                        // record.
                        bs               = sp_x->dims[0];
                        this->batch_size = bs;
                        input_dim        = sp_x->dims[1];
                } else {
                        if (vecSize(inputs) != 1)
                                return errNew(
                                    "expect one grad for scel layer. got %d",
                                    vecSize(inputs));

                        // no way to deduce.
                        bs = this->batch_size;
                }
        }

#define ALLOC_T(name, sp)                                    \
        {                                                    \
                int name = vmTensorNew(self->vm, F32, (sp)); \
                vecPushBack(self->ivs, name);                \
                this->name = name;                           \
        }

        // stage 2: allocate intermediate values (iv).
        if (direction == BB_FORWARD) {
                struct shape_t *sp_x = R2S(vm, bs, input_dim);
                struct shape_t *sp_o = R1S(vm, bs);
                struct shape_t *sp_r = R1S(vm, 1);
                ALLOC_T(o, sp_o);
                ALLOC_T(r, sp_r);
                if (is_training) {
                        // used for backprop, but need in forward pass.
                        ALLOC_T(d_x, sp_x);
                        ALLOC_T(d_o, sp_x);
                }
        } else {
                if (reduce_mean) {
                        struct shape_t *sp_r = R1S(vm, 1);
                        ALLOC_T(d_r, sp_r);
                }
        }

#undef ALLOC_T

        // stage 3: jit
        if (direction == BB_FORWARD) {
                // emit forward.
                //
                //   o = scel(y, x)
                //   r = reduce(o, axis=0)
                int y = inputs[0];
                int x = inputs[1];
                if (is_training) {
                        bbProgAppend(
                            p, &(struct oparg_t){
                                   OP_LS_SCEL,
                                   this->o,
                                   y,
                                   x,
                                   1,
                                   {.mode = OPT_MODE_I_BIT, .i = this->d_o}});
                } else {
                        bbProgAppend(
                            p, &(struct oparg_t){OP_LS_SCEL, this->o, y, x, 0});
                }

                bbProgAppend(
                    p, &(struct oparg_t){OP_REDUCE,
                                         this->r,
                                         this->o,
                                         -1,
                                         1,
                                         {.mode = OPT_MODE_I_BIT, .i = 0}});
                if (reduce_mean) {
                        bbProgAppend(
                            p, &(struct oparg_t){
                                   OP_MUL,
                                   this->r,
                                   this->r,
                                   -1,
                                   1,
                                   {.mode = OPT_MODE_F_BIT, .f = 1.0 / bs}});
                }
                vecPushBack(*outputs, this->r);
                return OK;
        } else {
                // emit backward
                //
                // o = scel(y, x, .i = d_o)
                // d_x = mul(d_o, d_r)
                int d_r = inputs[0];
                if (reduce_mean) {
                        bbProgAppend(
                            p, &(struct oparg_t){
                                   OP_MUL,
                                   this->d_r,
                                   d_r,
                                   -1,
                                   1,
                                   {.mode = OPT_MODE_F_BIT, .f = 1.0 / bs}});
                        d_r = this->d_r;
                }
                bbProgAppend(
                    p, &(struct oparg_t){OP_MUL, this->d_x, this->d_o, d_r, 0});
                vecPushBack(*outputs, this->d_x);
                return OK;
        }
}

// -----------------------------------------------------------------------------
// Impl for AUC Metric.
// -----------------------------------------------------------------------------
DECLARE_LAYER_METHODS(AUC);

static error_t _bbAUCSummary(struct bb_layer_t *, void *data, int flag);

error_t
bbAUCMetric(struct vm_t *vm, struct bb_layer_t **out)
{
        struct bb_auc_layer_t *l = malloc(sizeof(struct bb_auc_layer_t));
        memset(l, 0, sizeof(struct bb_auc_layer_t));
        l->base.vm = vm;

        struct bb_layer_operations_t *ops = &l->base.ops;
        ops->init                         = _bbAUCInit;
        ops->release                      = _bbLayerRelease;
        ops->weights                      = _bbLayerWeights;
        ops->grads                        = _bbLayerGrads;
        ops->states                       = _bbLayerStates;
        ops->jit                          = _bbAUCJit;
        ops->summary                      = _bbAUCSummary;

        *out = (struct bb_layer_t *)l;
        return OK;
}

error_t
_bbAUCInit(struct bb_layer_t *self, const struct bb_context_t *ctx,
           struct srng64_t *rng)
{
        struct bb_auc_layer_t *this = (struct bb_auc_layer_t *)self;
        struct vm_t *vm             = self->vm;

        // create the shapes, states and save into ivs
        struct shape_t *sp = R1S(vm, 1);

#define ALLOC_STATE(name, sp, collection)         \
        int name = vmTensorNew(vm, F32, sp);      \
        vecPushBack(this->base.collection, name); \
        this->name = name;

        ALLOC_STATE(total, sp, states);
        _bbInitTensor(vm, this->total, BB_INIT_ZERO, NULL);

        ALLOC_STATE(count, sp, states);
        _bbInitTensor(vm, this->count, BB_INIT_ZERO, NULL);

#undef ALLOC_STATE
        return OK;
}

error_t
_bbAUCJit(struct bb_layer_t *self, const struct bb_context_t *ctx,
          struct bb_program_t *p, int direction, const vec_t(int) inputs,
          vec_t(int) * outputs)
{
        error_t err;
        struct bb_auc_layer_t *this = (struct bb_auc_layer_t *)self;
        struct vm_t *vm             = self->vm;

        assert(direction == BB_FORWARD);

        // stage 1: error check. retrieve the batch size.
        int bs;
        {
                if (vecSize(inputs) != 2)
                        return errNew("expect two inputs for metric. got %d",
                                      vecSize(inputs));

                struct shape_t *sp_x;
                err = vmTensorInfo(vm, inputs[1], /*dtype=*/NULL, &sp_x);
                if (err) return errEmitNote("failed to grab the input shape.");
                if (sp_x->rank != 2)
                        return errNew(
                            "expect rank 2 input for scel layer. got "
                            "%d",
                            sp_x->rank);

                bs = sp_x->dims[0];
        }

#define ALLOC_T(name, sp)                                    \
        {                                                    \
                int name = vmTensorNew(self->vm, F32, (sp)); \
                vecPushBack(self->ivs, name);                \
                this->name = name;                           \
        }

        // stage 2: allocate intermediate values (iv).
        struct shape_t *sp_arg = R1S(vm, bs);
        struct shape_t *sp_r   = R1S(vm, 1);
        ALLOC_T(arg_y, sp_arg);
        ALLOC_T(arg_x, sp_arg);
        ALLOC_T(same, sp_arg);
        ALLOC_T(local_count, sp_r);
#undef ALLOC_T

        // stage 3: jit
        int y = inputs[0];
        int x = inputs[1];
        bbProgAppend(p, &(struct oparg_t){OP_ARGMAX, this->arg_y, y, -1, 0});
        bbProgAppend(p, &(struct oparg_t){OP_ARGMAX, this->arg_x, x, -1, 0});
        bbProgAppend(p, &(struct oparg_t){OP_EQ, this->same, this->arg_x,
                                          this->arg_y, 0});
        bbProgAppend(p, &(struct oparg_t){OP_REDUCE,
                                          this->local_count,
                                          this->same,
                                          -1,
                                          1,
                                          {.mode = OPT_MODE_I_BIT, .i = 0}});
        bbProgAppend(p, &(struct oparg_t){OP_ADD, this->count, this->count,
                                          this->local_count, 0});
        bbProgAppend(p, &(struct oparg_t){OP_ADD,
                                          this->total,
                                          this->total,
                                          -1,
                                          1,
                                          {.mode = OPT_MODE_F_BIT, .f = bs}});
        return OK;
}

error_t
_bbAUCSummary(struct bb_layer_t *self, void *data, int flag)
{
        struct bb_auc_layer_t *this = (struct bb_auc_layer_t *)self;
        struct vm_t *vm             = self->vm;

        error_t err;
        float   auc;
        float   count;
        float   total;

        float *buf;

        err = vmTensorData(vm, this->count, (void **)&buf);
        if (err) return errEmitNote("failed to get the data of count.");
        count = *buf;

        err = vmTensorData(vm, this->total, (void **)&buf);
        if (err) return errEmitNote("failed to get the data of total.");
        total = *buf;

        auc            = count / total;
        *(float *)data = auc;

        if (flag & BB_FLAG_RESET) {
                _bbInitTensor(vm, this->total, BB_INIT_ZERO, NULL);
                _bbInitTensor(vm, this->count, BB_INIT_ZERO, NULL);
        }
        return OK;
}
