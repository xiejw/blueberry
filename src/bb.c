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
_bbAllocateIntermediaValue(struct bb_layer_t *this, struct shape_t *sp)
{
        int td = vmTensorNew(this->vm, F32, sp);
        vecPushBack(this->ivs, td);
        return td;
}

#define DECLARE_LAYER_METHODS(name)                                         \
        static error_t _bb##name##Weights(struct bb_layer_t *        self,  \
                                          const struct bb_context_t *ctx,   \
                                          vec_t(int) * tds);                \
        static error_t _bb##name##Grads(struct bb_layer_t *        self,    \
                                        const struct bb_context_t *ctx,     \
                                        vec_t(int) * tds);                  \
        static error_t _bb##name##Init(struct bb_layer_t *        self,     \
                                       const struct bb_context_t *ctx,      \
                                       struct srng64_t *          rng);               \
        static error_t _bb##name##Release(struct bb_layer_t *self);         \
        static error_t _bb##name##Jit(                                      \
            struct bb_layer_t *self, const struct bb_context_t *ctx,        \
            struct bb_program_t *p, int direction, const vec_t(int) inputs, \
            vec_t(int) * outputs)

// -----------------------------------------------------------------------------
// Impl for Program.
// -----------------------------------------------------------------------------

struct vm_t *
bbVmInit()
{
        struct vm_t *vm = vmNew();
        if (vm == NULL) return NULL;

        // td is 0 scalar.
        struct shape_t *sp = R1S(vm, 1);
        int             td = vmTensorNew(vm, F32, sp);
        assert(td == 0);

        error_t err = vmExec(vm, OP_FILL, NULL, td, -1, -1);
        if (err) {
                errFatalAndExit("init vm failed: %d", err);
        }
        return vm;
}

// -----------------------------------------------------------------------------
// Impl for Program.
// -----------------------------------------------------------------------------

struct bb_program_t *
bbProgNew()
{
        struct bb_program_t *p = malloc(sizeof(struct bb_program_t));
        if (p == NULL) return NULL;
        memset(p, 0, sizeof(struct bb_program_t));
        return p;
}

void
bbProgFree(struct bb_program_t *p)
{
        if (p == NULL) return;
        struct bb_inst_t *next, *curr;
        curr = p->head;
        while (curr != NULL) {
                next = curr->next;
                free(curr);
                curr = next;
        }
        vecFree(p->inputs);
        vecFree(p->weights);
        free(p);
}

void
bbProgAppend(struct bb_program_t *p, struct oparg_t *op)
{
        struct bb_inst_t *inst = malloc(sizeof(struct bb_inst_t));
        inst->op               = *op;
        inst->next             = NULL;

        struct bb_inst_t *tail = p->tail;

        if (p->head == NULL) {
                inst->prev = NULL;
                p->head    = inst;
                p->tail    = inst;
        } else {
                inst->prev = tail;
                tail->next = inst;
                p->tail    = inst;
        }
}

void
bbProgDump(struct bb_program_t *p, sds_t *s)
{
        sdsCatPrintf(s, "program:\n");

        {
                sdsCatPrintf(s, "{  // inputs\n  ");
                size_t size = vecSize(p->inputs);
                if (size) {
                        for (int i = 0; i < size; i++) {
                                sdsCatPrintf(s, "%3d, ", p->inputs[i]);
                        }
                        sdsCatPrintf(s, "\n");
                } else {
                        sdsCatPrintf(s, "(empty)\n");
                }
                sdsCatPrintf(s, "}\n");
        }

        {
                sdsCatPrintf(s, "{  // weights\n  ");
                size_t size = vecSize(p->weights);
                if (size) {
                        for (int i = 0; i < size; i++) {
                                sdsCatPrintf(s, "%3d, ", p->weights[i]);
                        }
                        sdsCatPrintf(s, "\n");
                } else {
                        sdsCatPrintf(s, "(empty)\n");
                }
                sdsCatPrintf(s, "}\n");
        }

        {
                sdsCatPrintf(s, "{  // ops\n");
                if (p->head == NULL) {
                        sdsCatPrintf(s, "  (empty)\n");
                        return;
                }

                struct bb_inst_t *curr;
                curr = p->head;
                while (curr != NULL) {
                        char *opname;
                        switch (curr->op.op) {
                        case OP_MATMUL:
                                opname = "OP_MATMUL";
                                break;
                        case OP_MUL:
                                opname = "OP_MUL";
                                break;
                        case OP_ADD:
                                opname = "OP_ADD";
                                break;
                        case OP_MAX:
                                opname = "OP_MAX";
                                break;
                        default:
                                opname = "UNKNOWN";
                        }
                        struct oparg_t *op = &curr->op;
                        sdsCatPrintf(s,
                                     "  {.op = %2d (%-9s)}, .dst = %3d, .t1 = "
                                     "%3d, .td = %3d\n",
                                     op->op, opname, op->dst, op->t1, op->t2);
                        curr = curr->next;
                }
                sdsCatPrintf(s, "}\n");
        }
}

// -----------------------------------------------------------------------------
// Impl for Layer.
// -----------------------------------------------------------------------------
void
bbLayerFree(struct bb_layer_t *p)
{
        p->ops.release(p);
        free(p);
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
        l->base.vm = vm;
        l->config  = *cfg;

        struct bb_layer_operations_t *ops = &l->base.ops;
        ops->init                         = _bbDenseInit;
        ops->release                      = _bbDenseRelease;
        ops->weights                      = _bbDenseWeights;
        ops->grads                        = _bbDenseGrads;
        ops->jit                          = _bbDenseJit;

        *out = (struct bb_layer_t *)l;
        return OK;
}

error_t
_bbDenseWeights(struct bb_layer_t *this, const struct bb_context_t *ctx,
                vec_t(int) * tds)
{
        int old_size = vecSize(*tds);
        int inc      = vecSize(this->weights);
        vecReserve(*tds, old_size + inc);
        memcpy((*tds) + old_size, this->weights, sizeof(int) * inc);
        vecSetSize(*tds, old_size + inc);
        return OK;
}

error_t
_bbDenseGrads(struct bb_layer_t *this, const struct bb_context_t *ctx,
              vec_t(int) * tds)
{
        int old_size = vecSize(*tds);
        int inc      = vecSize(this->grads);
        vecReserve(*tds, old_size + inc);
        memcpy(tds + old_size, this->grads, sizeof(int) * inc);
        vecSetSize(*tds, old_size + inc);
        return OK;
}

error_t
_bbDenseInit(struct bb_layer_t *self, const struct bb_context_t *ctx,
             struct srng64_t *rng)
{
        struct bb_dense_layer_t *this       = (struct bb_dense_layer_t *)self;
        struct vm_t *                   vm  = self->vm;
        const struct bb_dense_config_t *cfg = &this->config;
        int has_bias                        = cfg->bias_init != BB_INIT_NULL;
        int is_training                     = ctx->is_training;

        // stage 1: error check
        if (!(cfg->kernel_init > BB_INIT_NULL &&
              cfg->kernel_init < BB_INIT_STOPPER))
                return errNew("kernel init is out of range; got %d",
                              cfg->kernel_init);
        if (!(cfg->bias_init >= BB_INIT_NULL &&
              cfg->bias_init < BB_INIT_STOPPER))
                return errNew("bias init is out of range; got %d",
                              cfg->bias_init);

        // stage 2: create the shapes, weights, and grads if training.
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
_bbDenseRelease(struct bb_layer_t *this)
{
        struct vm_t *vm = this->vm;

#define RELEAE_TDS(tds)                                          \
        {                                                        \
                vec_t(int) handles = (tds);                      \
                int     size       = vecSize(handles);           \
                error_t err;                                     \
                for (int i = 0; i < size; i++) {                 \
                        err = vmTensorFree(vm, handles[i]);      \
                        if (err) {                               \
                                return errEmitNote(              \
                                    "failed to release tensor"); \
                        }                                        \
                }                                                \
                vecFree(handles);                                \
                (tds) = vecNew();                                \
        }

        RELEAE_TDS((this->weights));
        RELEAE_TDS((this->grads));
        RELEAE_TDS((this->ivs));
#undef RELEAE_TDS
        return OK;
}

error_t
_bbDenseJit(struct bb_layer_t *self, const struct bb_context_t *ctx,
            struct bb_program_t *p, int direction, const vec_t(int) inputs,
            vec_t(int) * outputs)
{
        error_t err;
        struct bb_dense_layer_t *this       = (struct bb_dense_layer_t *)self;
        struct vm_t *                   vm  = self->vm;
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

#define ALLOC_T(sp) _bbAllocateIntermediaValue(self, (sp))

        // stage 2: allocate intermediate values (iv).
        struct shape_t *sp_h = R2S(vm, bs, cfg->output_dim);
        if (direction == BB_FORWARD) {
                this->h = ALLOC_T(sp_h);
                if (has_bias) this->hb = ALLOC_T(sp_h);
                if (has_relu) this->y = ALLOC_T(sp_h);

        } else {
                if (has_relu) {
                        this->state = ALLOC_T(sp_h);
                        this->d_hb  = ALLOC_T(sp_h);
                }
                struct shape_t *sp_x = R2S(vm, bs, cfg->input_dim);
                this->d_x            = ALLOC_T(sp_x);
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
                int x = inputs[0];
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
        }
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
