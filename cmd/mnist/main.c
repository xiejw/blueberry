#include <stdio.h>

// bb
#include "bb.h"

// -----------------------------------------------------------------------------
// Helper Prototype.
// -----------------------------------------------------------------------------

#include "../../mlvm/cmd/mnist/mnist.h"

static error_t prepareData(float32_t* x_data, size_t x_size, float32_t* y_data,
                           size_t y_size);

#define NE(err) _NE_IMPL(err, __FILE__, __LINE__)

#define _NE_IMPL(err, file, line)                                          \
        if ((err)) {                                                       \
                errDump("unexected error at file %s line %d", file, line); \
                goto cleanup;                                              \
        }

// -----------------------------------------------------------------------------
// MNIST.
// -----------------------------------------------------------------------------
#define TOTOL_IMAGES 60000
#define BATCH_SIZE   32
#define IMAGE_SIZE   (28 * 28)
#define LABEL_SIZE   (10)

static unsigned char* images   = NULL;
static unsigned char* labels   = NULL;
static size_t         it_count = 0;

// -----------------------------------------------------------------------------
// Main.
// -----------------------------------------------------------------------------
int
main()
{
        struct vm_t*            vm  = bbVmInit();
        struct bb_context_t     ctx = {.is_training = 1};
        struct bb_program_t*    p   = bbProgNew();
        sds_t                   s   = sdsEmpty();
        struct bb_seq_module_t* m   = bbSeqModuleNew();

        int             prog_count;
        struct oparg_t* prog = NULL;
        // ---------------------------------------------------------------------
        // Compile the model.
        // ---------------------------------------------------------------------

        m->r = srng64New(123);

        struct shape_t* sp_x = R2S(vm, BATCH_SIZE, IMAGE_SIZE);
        struct shape_t* sp_y = R2S(vm, BATCH_SIZE, LABEL_SIZE);

        m->x = vmTensorNew(vm, F32, sp_x);
        m->y = vmTensorNew(vm, F32, sp_y);

        NE(bbCreateLayers(
            vm,
            (struct bb_layer_config_t[]){
                {
                    .tag = BB_TAG_DENSE,
                    .config =
                        &(struct bb_dense_config_t){
                            .input_dim   = (IMAGE_SIZE),
                            .output_dim  = 64,
                            .kernel_init = BB_INIT_STD_NORMAL,
                            .bias_init   = BB_INIT_ZERO,
                            .actn        = BB_ACTN_RELU},
                },
                {
                    .tag = BB_TAG_DENSE,
                    .config =
                        &(struct bb_dense_config_t){
                            .input_dim   = 64,
                            .output_dim  = 64,
                            .kernel_init = BB_INIT_STD_NORMAL,
                            .bias_init   = BB_INIT_ZERO,
                            .actn        = BB_ACTN_RELU},
                },
                {
                    .tag = BB_TAG_DENSE,
                    .config =
                        &(struct bb_dense_config_t){
                            .input_dim   = 64,
                            .output_dim  = LABEL_SIZE,
                            .kernel_init = BB_INIT_STD_NORMAL,
                            .bias_init   = BB_INIT_NULL,
                            .actn        = BB_ACTN_NONE},
                },
                {.tag = BB_TAG_SCEL,
                 .config =
                     &(struct bb_scel_config_t){.reduction = BB_REDUCTION_SUM}},
                {.tag = BB_TAG_NULL}},
            &m->layers));

        m->loss = vecPopBack(m->layers);

        NE(bbOptNew(vm, BB_OPT_SGD, 0.001, &m->opt));
        NE(bbAUCMetric(vm, &m->metric));
        NE(bbCompileSeqModule(&ctx, p, m));

        bbProgDump(p, &s);
        printf("%s", s);

        // ---------------------------------------------------------------------
        // Fetch inputs.
        // ---------------------------------------------------------------------
        float32_t *x_data, *y_data;
        {
                NE(vmTensorData(vm, m->x, (void**)&x_data));
                NE(vmTensorData(vm, m->y, (void**)&y_data));
        }

        // ---------------------------------------------------------------------
        // Compile to Batch Ops.
        // ---------------------------------------------------------------------
        NE(bbProgCompileToBatchOps(p, &prog_count, &prog));

        // ---------------------------------------------------------------------
        // Run.
        // ---------------------------------------------------------------------
        for (int ep = 0; ep < 12; ep++) {
                for (int i = 0; i < TOTOL_IMAGES / BATCH_SIZE; i++) {
                        NE(prepareData(x_data,
                                       /*x_size=*/sp_x->size, y_data,
                                       /*y_size=*/sp_y->size));
                        NE(vmBatch(vm, prog_count, prog));
                }
                float auc;
                NE(m->metric->ops.summary(m->metric, &auc, BB_FLAG_RESET));
                printf("epoch: %2d auc: %f\n", ep + 1, auc);
                it_count = 0;
        }

cleanup:
        if (prog) free(prog);
        if (images != NULL) free(images);
        if (labels != NULL) free(labels);
        sdsFree(s);
        bbSeqModuleFree(m);
        bbProgFree(p);
        vmFree(vm);
        return OK;
}

static error_t
prepareMnistData(float32_t* x_data, size_t x_size, float32_t* y_data,
                 size_t y_size)
{
        if (images == NULL) {
                error_t err = readMnistTrainingImages(&images);
                if (err) {
                        return err;
                }

                err = readMnistTrainingLabels(&labels);
                if (err) {
                        return err;
                }
                printf("sample label %d -- image:\n", (int)*labels);
                printMnistImage(images);
                printf("smaple label %d -- image:\n", (int)*(labels + 1));
                printMnistImage(images + 28 * 28);
        }

        size_t bs = x_size / 28 / 28;
        assert(bs * LABEL_SIZE == y_size);

        unsigned char* buf = images + it_count * IMAGE_SIZE;
        for (size_t i = 0; i < x_size; i++) {
                x_data[i] = ((float32_t)buf[i]) / 256;
        }

        buf = labels + it_count;
        for (size_t i = 0; i < bs; i++) {
                int tgt = buf[i];
                assert(tgt < LABEL_SIZE);
                size_t offset = i * LABEL_SIZE;
                for (size_t j = 0; j < LABEL_SIZE; j++) {
                        y_data[offset + j] = j == tgt ? 1 : 0;
                }
        }

        it_count += bs;
        return OK;
}

error_t
prepareData(float32_t* x_data, size_t x_size, float32_t* y_data, size_t y_size)
{
        error_t err;
        if ((err = prepareMnistData(x_data, x_size, y_data, y_size))) {
                if (images != NULL) {
                        free(images);
                        images = NULL;
                }
                if (labels != NULL) {
                        free(labels);
                        labels = NULL;
                }
                return err;
        }
        return OK;
}
