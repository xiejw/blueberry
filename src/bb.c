#include "bb.h"

#include <assert.h>
#include <string.h>

#include "rng/srng64.h"

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

static inline int
_bbAllocateTensor(struct bb_base_layer_t *this, struct shape_t *sp)
{
        int td = vmTensorNew(this->vm, F32, sp);
        vecPushBack(this->tds, td);
        return td;
}

#define DECLARE_LAYER_METHODS(name)                                            \
        static error_t _bb##name##Weights(                                     \
            void *self, const struct bb_context_t *ctx, vec_t(int) * tds);     \
        static error_t _bb##name##Grads(                                       \
            void *self, const struct bb_context_t *ctx, vec_t(int) * tds);     \
        static error_t _bb##name##Init(                                        \
            void *self, const struct bb_context_t *ctx, struct srng64_t *rng); \
        static error_t _bb##name##Release(void *                     self,     \
                                          const struct bb_context_t *ctx);     \
        static error_t _bb##name##Jit(                                         \
            void *self, const struct bb_context_t *ctx,                        \
            struct bb_program_t *p, int direction, const vec_t(int) inputs,    \
            vec_t(int) * *outputs)

// -----------------------------------------------------------------------------
// Impl for Program.
// -----------------------------------------------------------------------------

struct bb_program_t *
bbProgNew()
{
        struct bb_program_t *p = malloc(sizeof(struct bb_program_t));
        if (p == NULL) return NULL;
        p->ops = NULL;
        return p;
}

void
bbProgFree(struct bb_program_t *p)
{
        if (p == NULL) return;
        struct bb_inst_t *next, *curr;
        curr = p->ops;
        while (curr != NULL) {
                next = curr->next;
                free(curr->op);
                free(curr);
                curr = next;
        }
}
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

        struct bb_dense_layer_t *l = malloc(sizeof(struct bb_dense_layer_t));

        memset(l, 0, sizeof(struct bb_dense_layer_t));
        l->base.tds = vecNew();
        l->base.vm  = vm;
        l->config   = *cfg;

        struct bb_layer_t *layer = malloc(sizeof(struct bb_layer_t));
        layer->init              = _bbDenseInit;
        layer->release           = _bbDenseRelease;
        layer->weights           = _bbDenseWeights;
        layer->grads             = _bbDenseGrads;
        layer->jit               = _bbDenseJit;

        *out = layer;
        return OK;
}

error_t
_bbDenseWeights(void *self, const struct bb_context_t *ctx, vec_t(int) * tds)
{
        struct bb_dense_layer_t *this       = self;
        const struct bb_dense_config_t *cfg = &this->config;
        vecPushBack(*tds, this->w);
        if (cfg->bias_init != BB_INIT_NULL) vecPushBack(*tds, this->b);
        return OK;
}

error_t
_bbDenseGrads(void *self, const struct bb_context_t *ctx, vec_t(int) * tds)
{
        struct bb_dense_layer_t *this       = self;
        const struct bb_dense_config_t *cfg = &this->config;
        vecPushBack(*tds, this->d_w);
        if (cfg->bias_init != BB_INIT_NULL) vecPushBack(*tds, this->d_b);
        return OK;
}

error_t
_bbDenseInit(void *self, const struct bb_context_t *ctx, struct srng64_t *rng)
{
        struct bb_dense_layer_t *this       = self;
        struct vm_t *                   vm  = ctx->vm;
        const struct bb_dense_config_t *cfg = &this->config;
        int has_bias                        = cfg->bias_init != BB_INIT_NULL;

        // stage 1: error check
        if (!(cfg->kernel_init > BB_INIT_NULL &&
              cfg->kernel_init < BB_INIT_STOPPER))
                return errNew("kernel init is out of range; got %d",
                              cfg->kernel_init);
        if (!(cfg->bias_init >= BB_INIT_NULL &&
              cfg->bias_init < BB_INIT_STOPPER))
                return errNew("bias init is out of range; got %d",
                              cfg->bias_init);

        // stage 2: create the shapes.
        struct shape_t *sp_w = R2S(vm, cfg->input_dim, cfg->output_dim);
        int             w    = vmTensorNew(vm, F32, sp_w);
        _bbInitTensor(vm, w, cfg->kernel_init, rng);
        this->w = w;
        vecPushBack(this->base.tds, w);

        if (has_bias) {
                struct shape_t *sp_b = R1S(vm, cfg->output_dim);
                int             b    = vmTensorNew(vm, F32, sp_b);
                _bbInitTensor(vm, b, cfg->bias_init, rng);
                this->b = b;
                vecPushBack(this->base.tds, b);
        }
        return OK;
}

error_t
_bbDenseRelease(void *self, const struct bb_context_t *ctx)
{
        struct bb_dense_layer_t *this = self;
        struct vm_t *vm               = ctx->vm;

        vec_t(int) tds = this->base.tds;
        int     size   = vecSize(tds);
        error_t err;
        for (int i = 0; i < size; i++) {
                err = vmTensorFree(vm, tds[i]);
                if (err) {
                        return errEmitNote(
                            "failed to release internal logit tensor");
                }
        }
        vecFree(tds);
        this->base.tds = vecNew();
        return OK;
}

error_t
_bbDenseJit(void *self, const struct bb_context_t *ctx, struct bb_program_t *p,
            int direction, const vec_t(int) inputs, vec_t(int) * *outputs)
{
        error_t err;
        struct bb_dense_layer_t *this       = self;
        struct vm_t *                   vm  = ctx->vm;
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

#define ALLOC_T(sp) _bbAllocateTensor((struct bb_base_layer_t *)this, (sp))

        // stage 2: allocate intermediate values (iv).
        if (direction == BB_FORWARD) {
                struct shape_t *sp_h = R2S(vm, bs, cfg->output_dim);

                this->h = ALLOC_T(sp_h);
                if (has_bias) this->hb = ALLOC_T(sp_h);
                if (has_relu) this->y = ALLOC_T(sp_h);

        } else {
                if (has_relu) {
                        struct shape_t *sp_h = R2S(vm, bs, cfg->output_dim);
                        this->state          = ALLOC_T(sp_h);
                        this->d_hb           = ALLOC_T(sp_h);
                }
                if (has_bias) {
                        struct shape_t *sp_b = R1S(vm, cfg->output_dim);
                        this->d_b            = ALLOC_T(sp_b);
                }

                struct shape_t *sp_w = R2S(vm, cfg->input_dim, cfg->output_dim);
                struct shape_t *sp_x = R2S(vm, bs, cfg->input_dim);
                this->d_w            = ALLOC_T(sp_w);
                this->d_x            = ALLOC_T(sp_x);
        }

#undef ALLOC_T

        // stage 3: jit
        //
        // emit forward.
        //   z[1] = zeros([1])
        //
        //   h [bs, out] = matmul(x[bs, in], w[in, out])
        //   hb[bs, out] = h[bs, out] + b[out]
        //   z [bs, out] = max(h1b[bs, out], z[1])
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
        return OK;
}

// -----------------------------------------------------------------------------
// Impl for SCEL.
// -----------------------------------------------------------------------------
error_t
bbSCELLayer(struct vm_t *vm, const struct bb_scel_config_t *cfg,
            struct bb_layer_t **out)
{
        // error checks.
        if (cfg->input_dim <= 0)
                return errNew("input dim must be positive; got %d",
                              cfg->input_dim);
        if (!(cfg->reduction == BB_REDUCTION_SUM ||
              cfg->reduction == BB_REDUCTION_MEAN))
                return errNew("reduction must be SUM or MEAN; got %d",
                              cfg->reduction);

        return OK;
}
