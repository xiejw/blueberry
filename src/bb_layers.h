#ifndef BB_LAYERS_H_
#define BB_LAYERS_H_

// -----------------------------------------------------------------------------
// Dense layer.
// -----------------------------------------------------------------------------

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
struct bb_scel_config_t {
        int reduction;
};

struct bb_scel_layer_t {
        struct bb_layer_t base;

        struct bb_scel_config_t config;

        // iv
        int o, r;      // forward
        int d_r, d_x;  // backward

        // other
        int batch_size;
};

error_t bbSCELLayer(struct vm_t *, const struct bb_scel_config_t *,
                    struct bb_layer_t **);

#endif
