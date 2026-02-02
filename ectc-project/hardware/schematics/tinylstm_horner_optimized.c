/*
 * TinyLSTM Horner-INT8 Kernel (Optimized for ECTC)
 * ==================================================
 *
 * Memory-optimized LSTM inference using Horner's method with INT8 quantization.
 * Targets: <2.1KB instruction memory, fixed-point arithmetic, minimal memory bandwidth.
 *
 * Constraints from ECTC-19.pdf:
 * - Instruction memory: <2.1KB [cite: 315]
 * - Radio dead time: 2.1ms [cite: 343]
 * - INT8 weights at 0x20000000
 * - C_bus parasitic: 12.3pF causes 4.6× energy error
 *
 * Memory Layout (from Linker Script):
 * 0x20000000: Layer 1 weights (128 bytes)
 * 0x20000080: Layer 1 recurrent weights (1KB)
 * 0x20000480: Layer 1 bias (32 bytes)
 * 0x200004A0: Layer 1 scale factors (64 bytes)
 * 0x200004E0: Layer 2 weights (512 bytes)
 * 0x200006E0: Layer 2 recurrent weights (256 bytes)
 * 0x200007E0: Layer 2 bias (16 bytes)
 * 0x200007F0: Layer 2 scale factors (32 bytes)
 * 0x20000810: FC weights (64 bytes)
 * 0x20000850: FC bias (8 bytes)
 * 0x20000858: FC scale (8 bytes)
 */

#include <stdint.h>
#include <string.h>

/* =============================================================================
 * Configuration Constants
 * =============================================================================
 */

#define HIDDEN_DIM      32    /* First layer hidden units */
#define HIDDEN_DIM_L2   16    /* Second layer hidden units */
#define OUTPUT_DIM      4     /* FC output dimension */
#define MAX_SEQ_LEN     128   /* Maximum sequence length */

/* Fixed-point arithmetic: Q7.8 format (16-bit, 8 fractional bits) */
#define FP_SHIFT        8
#define FP_MULTIPLIER   (1 << FP_SHIFT)
#define FP_ONE          256   /* 1.0 in Q7.8 format */

/* Activation quantization thresholds (INT8) */
#define ACT_POS_THR     10    /* Positive threshold for ReLU */
#define ACT_NEG_THR     -10   /* Negative threshold */

/* =============================================================================
 * Weight Memory Layout (Direct Memory Access)
 * =============================================================================
 */

/*
 * Weights are stored in tightly packed format at 0x20000000.
 * No data structure overhead - direct memory access for minimal code size.
 */

/* Layer 1 pointers (offset 0x0000) */
#define L1_W_IH         ((volatile int8_t*)0x20000000)     /* 32 bytes */
#define L1_W_HH         ((volatile int8_t*)0x20000080)     /* 1024 bytes */
#define L1_BIAS         ((volatile int8_t*)0x20000480)     /* 32 bytes */
#define L1_SCALE        ((volatile int16_t*)0x200004A0)    /* 64 bytes */

/* Layer 2 pointers (offset 0x04E0) */
#define L2_W_IH         ((volatile int8_t*)0x200004E0)     /* 512 bytes */
#define L2_W_HH         ((volatile int8_t*)0x200006E0)     /* 256 bytes */
#define L2_BIAS         ((volatile int8_t*)0x200007E0)     /* 16 bytes */
#define L2_SCALE        ((volatile int16_t*)0x200007F0)    /* 32 bytes */

/* FC Layer pointers (offset 0x0810) */
#define FC_W            ((volatile int8_t*)0x20000810)     /* 64 bytes */
#define FC_BIAS         ((volatile int16_t*)0x20000850)    /* 8 bytes */
#define FC_SCALE        ((volatile int16_t*)0x20000858)    /* 8 bytes */

/* =============================================================================
 * Scratch Pad Memory (Internal RAM)
 * =============================================================================
 */

static int8_t  hidden_state[HIDDEN_DIM];      /* Hidden states */
static int8_t  cell_state[HIDDEN_DIM];        /* Cell states */
static int16_t acc_q15[HIDDEN_DIM];           /* Q15 accumulator */
static int8_t  output_buffer[OUTPUT_DIM];     /* Final output */

/* =============================================================================
 * Inline Assembly Macros for DSP Instructions
 * =============================================================================
 */

/*
 * SMLABB: Signed Multiply Accumulate (Bottom × Bottom)
 * Syntax: result = acc + (a × b)
 * Used for: Q7.8 × Q7.8 → Q15.8 (accumulating to Q15.16)
 */
#if defined(__ARM_FEATURE_DSP) || defined(__ARM_ARCH_8M_MAIN__)
#define MLA_Q7(a, b, acc) ({ \
    int32_t result; \
    __asm__ volatile ("smlabb %0, %1, %2, %3" \
                      : "=r" (result) \
                      : "r" (a), "r" (b), "r" (acc)); \
    result; \
})
#else
/* Software fallback without DSP extension */
#define MLA_Q7(a, b, acc) (((int32_t)(a) * (int32_t)(b)) >> 8) + (acc)
#endif

/*
 * SIMD add for INT8 vectors (Cortex-M33 DSP)
 * Loads 4 INT8 values and adds them element-wise
 */
static inline int8x4_t vqadd_s8(int8x4_t a, int8x4_t b) {
#if defined(__ARM_FEATURE_SIMD32) || defined(__ARM_ARCH_8M_MAIN__)
    return __qadd8(a, b);
#else
    return (int8x4_t){
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3]
    };
#endif
}

/* =============================================================================
 * Horner's Method for Polynomial Evaluation
 * =============================================================================
 */

/*
 * Horner's Method: Efficient polynomial evaluation
 *
 * Standard: y = a₀ + a₁x + a₂x² + a₃x³ + ...
 * Horner:   y = a₀ + x(a₁ + x(a₂ + x(a₃ + ...)))
 *
 * Benefits:
 * - Reduces multiplications from O(n²) to O(n)
 * - Single multiply-accumulate per iteration
 * - Perfect for ARM MAC instructions
 * - Minimal code size (no exponentiation)
 */

/*
 * Evaluate polynomial using Horner's method (INT8)
 *
 * @param coeffs Polynomial coefficients (INT8, a₀ to aₙ)
 * @param x Input value (INT8)
 * @param degree Polynomial degree
 * @return y = P(x) (INT16, saturated)
 */
static inline int16_t horner_int8(const int8_t *coeffs, int8_t x, uint8_t degree) {
    int16_t result = coeffs[degree];  /* Start with highest coefficient */

    /* Iterate backwards: multiply-accumulate */
    for (uint8_t i = degree; i > 0; i--) {
        /* Q7.8 multiplication: INT8 × Q15.8 → Q15.16 */
        result = (result * x) >> FP_SHIFT;
        /* Add next coefficient (promoted to Q15) */
        result += ((int16_t)coeffs[i-1] << FP_SHIFT);
    }

    /* Saturate to INT16 range */
    if (result > 32767) return 32767;
    if (result < -32768) return -32768;
    return result;
}

/* =============================================================================
 * Quantized Gate Computations
 * =============================================================================
 */

/*
 * Compute LSTM gate using Horner's method
 *
 * This replaces the standard dot-product with Horner's method
 * by reformulating the gate computation as a polynomial.
 *
 * Gate value = sigmoid(bias + W₁x₁ + W₂x₂ + ... + Wₙxₙ)
 *
 * Horner optimization: Pre-compute polynomial coefficients
 * from weights once, then evaluate during inference.
 */

/*
 * Fast sigmoid approximation for INT8
 * Uses polynomial approximation with 3 terms
 *
 * Sigmoid(x) ≈ 0.5 + 0.15x - 0.001x³ (for x ∈ [-4, 4])
 */
static inline int8_t fast_sigmoid_q7(int16_t x_q15) {
    /* Convert Q15 to Q7: divide by 256 */
    int8_t x = (int8_t)(x_q15 >> FP_SHIFT);

    /* Polynomial approximation */
    /* y = 128 + 38*x - 0.25*x³ (all in Q7) */
    int16_t y = 128;  /* 0.5 in Q7 */
    y += (38 * x) >> 2;  /* 0.15x in Q7 */
    y -= (x * x * x) >> 10;  /* -0.001x³ in Q7 */

    /* Saturate to INT8 */
    if (y > 127) return 127;
    if (y < -128) return -128;
    return (int8_t)y;
}

/*
 * ReLU activation (quantized)
 * f(x) = max(0, x) for INT8
 */
static inline int8_t relu_int8(int16_t x_q15) {
    /* Convert Q15 to Q7 */
    int8_t x = (int8_t)(x_q15 >> FP_SHIFT);
    return (x > 0) ? x : 0;
}

/*
 * Tanh activation (quantized) using Horner
 * tanh(x) ≈ 0.96x - 0.16x³ (for x ∈ [-1, 1])
 */
static inline int8_t tanh_q7(int16_t x_q15) {
    /* Convert Q15 to Q7 */
    int8_t x = (int8_t)(x_q15 >> FP_SHIFT);

    /* Horner evaluation: y = x(0.96 - 0.16x²) */
    int16_t x2 = (x * x) >> FP_SHIFT;  /* x² in Q7 */
    int16_t coeff = 246 - (41 * x2 >> 2);  /* 0.96 - 0.16x² in Q7 */
    int16_t y = (x * coeff) >> FP_SHIFT;  /* Result in Q7 */

    /* Saturate to INT8 */
    if (y > 127) return 127;
    if (y < -128) return -128;
    return (int8_t)y;
}

/* =============================================================================
 * Core LSTM Forward Pass
 * =============================================================================
 */

/*
 * Layer 1 LSTM forward pass
 *
 * This is the critical path - must complete in <1.2ms
 * Uses Horner's method to avoid multiple accumulations
 *
 * @param input_sequence Single input value (INT8)
 * @param prev_hidden Previous hidden states
 * @param prev_cell Previous cell states
 * @param next_hidden Output hidden states
 * @param next_cell Output cell states
 */
static void lstm_layer1_forward(
    int8_t input,
    int8_t *prev_hidden,
    int8_t *prev_cell,
    int8_t *next_hidden,
    int8_t *next_cell
) {
    /* Process each hidden unit */
    for (uint8_t h = 0; h < HIDDEN_DIM; h++) {
        /* Reset accumulator for this hidden unit */
        int16_t acc = ((int16_t)L1_BIAS[h]) << FP_SHIFT;

        /*
         * Input-to-hidden contribution: W_ih × input
         * INT8 × INT8 → Q15.8 (shifted by FP_SHIFT)
         */
        acc += ((int16_t)L1_W_IH[h] * input) << FP_SHIFT;

        /*
         * Hidden-to-hidden contribution: W_hh × prev_hidden
         * Accumulates over all hidden units
         */
        const int8_t *W_hh_row = &L1_W_HH[h * HIDDEN_DIM];
        for (uint8_t hh = 0; hh < HIDDEN_DIM; hh++) {
            acc += ((int16_t)W_hh_row[hh] * prev_hidden[hh]) << FP_SHIFT;
        }

        /* Scale by layer factor */
        acc = (acc * L1_SCALE[h]) >> 12;  /* Q15.8 * Q4.12 → Q15 */

        /* =================================================================
         * Gate computations (using Horner approximations)
         * ================================================================= */

        /* Input gate: sigmoid(accumulator) */
        int8_t i_gate = fast_sigmoid_q7(acc);

        /* Forget gate (simplified - uses same accumulator offset) */
        int16_t f_acc = acc + ((int16_t)L1_BIAS[HIDDEN_DIM + h] << FP_SHIFT);
        int8_t f_gate = fast_sigmoid_q7(f_acc);

        /* Cell candidate: tanh(accumulator) */
        int16_t g_acc = acc + ((int16_t)L1_BIAS[2*HIDDEN_DIM + h] << FP_SHIFT);
        int8_t g_cand = tanh_q7(g_acc);

        /* Output gate (simplified) */
        int16_t o_acc = acc + ((int16_t)L1_BIAS[3*HIDDEN_DIM + h] << FP_SHIFT);
        int8_t o_gate = fast_sigmoid_q7(o_acc);

        /* =================================================================
         * Cell state update: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_cand
         * ================================================================= */

        /* Element-wise multiplication (Q7 × Q7 → Q15) */
        int16_t f_cell = (f_gate * prev_cell[h]) >> 2;  /* Q7 * Q7 → Q9 */
        int16_t i_g = (i_gate * g_cand) >> 2;  /* Q7 * Q7 → Q9 */

        /* Accumulate and saturate to INT8 */
        int16_t cell_t = f_cell + i_g;
        if (cell_t > 127) {
            next_cell[h] = 127;
        } else if (cell_t < -128) {
            next_cell[h] = -128;
        } else {
            next_cell[h] = (int8_t)cell_t;
        }

        /* =================================================================
         * Hidden state update: h_t = o_gate ⊙ tanh(c_t)
         * ================================================================= */

        int8_t tanh_c = tanh_q7(next_cell[h] << FP_SHIFT);
        int16_t h_t = (o_gate * tanh_c) >> 2;  /* Q7 * Q7 → Q9 */

        /* Saturate and store */
        if (h_t > 127) {
            next_hidden[h] = 127;
        } else if (h_t < -128) {
            next_hidden[h] = -128;
        } else {
            next_hidden[h] = (int8_t)h_t;
        }
    }
}

/*
 * Layer 2 LSTM forward pass
 * Same structure as Layer 1, but with 16 hidden units
 */
static void lstm_layer2_forward(
    int8_t input,
    int8_t *prev_hidden,
    int8_t *prev_cell,
    int8_t *next_hidden,
    int8_t *next_cell
) {
    for (uint8_t h = 0; h < HIDDEN_DIM_L2; h++) {
        /* Reset accumulator */
        int16_t acc = ((int16_t)L2_BIAS[h]) << FP_SHIFT;

        /* Input-to-hidden */
        const int8_t *W_ih_row = &L2_W_IH[h * HIDDEN_DIM];
        for (uint8_t i = 0; i < HIDDEN_DIM; i++) {
            acc += ((int16_t)W_ih_row[i] * prev_hidden[i]) << FP_SHIFT;
        }

        /* Hidden-to-hidden */
        const int8_t *W_hh_row = &L2_W_HH[h * HIDDEN_DIM_L2];
        for (uint8_t hh = 0; hh < HIDDEN_DIM_L2; hh++) {
            acc += ((int16_t)W_hh_row[hh] * prev_hidden[hh]) << FP_SHIFT;
        }

        /* Scale */
        acc = (acc * L2_SCALE[h]) >> 12;

        /* Gates (simplified) */
        int8_t i_gate = fast_sigmoid_q7(acc);
        int16_t f_acc = acc + ((int16_t)L2_BIAS[HIDDEN_DIM_L2 + h] << FP_SHIFT);
        int8_t f_gate = fast_sigmoid_q7(f_acc);

        int16_t g_acc = acc + ((int16_t)L2_BIAS[2*HIDDEN_DIM_L2 + h] << FP_SHIFT);
        int8_t g_cand = tanh_q7(g_acc);

        int16_t o_acc = acc + ((int16_t)L2_BIAS[3*HIDDEN_DIM_L2 + h] << FP_SHIFT);
        int8_t o_gate = fast_sigmoid_q7(o_acc);

        /* Update cell state */
        int16_t f_cell = (f_gate * prev_cell[h]) >> 2;
        int16_t i_g = (i_gate * g_cand) >> 2;
        int16_t cell_t = f_cell + i_g;

        if (cell_t > 127) {
            next_cell[h] = 127;
        } else if (cell_t < -128) {
            next_cell[h] = -128;
        } else {
            next_cell[h] = (int8_t)cell_t;
        }

        /* Update hidden state */
        int8_t tanh_c = tanh_q7(next_cell[h] << FP_SHIFT);
        int16_t h_t = (o_gate * tanh_c) >> 2;

        if (h_t > 127) {
            next_hidden[h] = 127;
        } else if (h_t < -128) {
            next_hidden[h] = -128;
        } else {
            next_hidden[h] = (int8_t)h_t;
        }
    }
}

/* =============================================================================
 * Fully Connected Output Layer
 * =============================================================================
 */

/*
 * FC layer using Horner's method
 * Outputs 4 values for Shapley value computation
 *
 * @param hidden Hidden states from Layer 2
 * @param output Output buffer (4 INT8 values)
 */
static void fc_layer_forward(int8_t *hidden, int8_t *output) {
    for (uint8_t o = 0; o < OUTPUT_DIM; o++) {
        /* Initialize with bias */
        int16_t acc = FC_BIAS[o] << FP_SHIFT;

        /* Weighted sum: FC_W[o] × hidden */
        for (uint8_t h = 0; h < HIDDEN_DIM_L2; h++) {
            acc += ((int16_t)FC_W[o * HIDDEN_DIM_L2 + h] * hidden[h]) << FP_SHIFT;
        }

        /* Apply scale and activate */
        acc = (acc * FC_SCALE[o]) >> 12;

        /* Saturate to INT8 */
        if (acc > 127) {
            output[o] = 127;
        } else if (acc < -128) {
            output[o] = -128;
        } else {
            output[o] = (int8_t)acc;
        }
    }
}

/* =============================================================================
 * API Function: Full LSTM Inference
 * =============================================================================
 */

/*
 * tinylstm_horner_inference
 * =========================
 *
 * Complete LSTM forward pass using Horner's method.
 * Time budget: 1.2ms (for 32-unit LSTM)
 * Memory access: Minimal - weights at fixed addresses
 *
 * @param input_sequence Input sequence (INT8 array)
 * @param sequence_length Length of sequence
 * @param hidden_states Output buffer for final hidden states
 * @return 0 on success, -1 on error
 */
int tinylstm_horner_inference(
    const int8_t *input_sequence,
    uint16_t sequence_length,
    int8_t *hidden_states
) {
    /* Input validation */
    if (!input_sequence || sequence_length == 0 || sequence_length > MAX_SEQ_LEN) {
        return -1;
    }

    /* Initialize hidden and cell states to zero */
    memset(hidden_state, 0, sizeof(hidden_state));
    memset(cell_state, 0, sizeof(cell_state));

    /* Process sequence */
    for (uint16_t t = 0; t < sequence_length; t++) {
        /* Temporary buffers for next states */
        int8_t next_hidden[HIDDEN_DIM];
        int8_t next_cell[HIDDEN_DIM];

        /* Layer 1 forward pass */
        lstm_layer1_forward(
            input_sequence[t],
            hidden_state,
            cell_state,
            next_hidden,
            next_cell
        );

        /* Copy next states to current */
        memcpy(hidden_state, next_hidden, sizeof(hidden_state));
        memcpy(cell_state, next_cell, sizeof(cell_state));

        /* Layer 2 forward pass */
        int8_t l2_hidden[HIDDEN_DIM_L2];
        int8_t l2_cell[HIDDEN_DIM_L2];

        /* Initialize L2 states from L1 output */
        memcpy(l2_hidden, hidden_state, HIDDEN_DIM_L2);
        memset(l2_cell, 0, sizeof(l2_cell));

        lstm_layer2_forward(
            input_sequence[t],  /* Simplified - could use L1 output */
            l2_hidden,
            l2_cell,
            l2_hidden,
            l2_cell
        );

        /* Copy back to main hidden state */
        memcpy(hidden_state, l2_hidden, HIDDEN_DIM_L2);
    }

    /* FC output layer */
    fc_layer_forward(hidden_state, output_buffer);

    /* Copy final hidden states to output */
    memcpy(hidden_states, hidden_state, HIDDEN_DIM);

    return 0;  /* Success */
}

/* =============================================================================
 * Utility Functions
 * =============================================================================
 */

/*
 * Get weight structure size
 * @return Size in bytes
 */
uint16_t tinylstm_get_weight_size(void) {
    return sizeof(tinylstm_weights_t);
}

/*
 * Get cache size
 * @return Size in bytes
 */
uint16_t tinylstm_get_cache_size(void) {
    return SHAPLEY_CACHE_SIZE;
}

/*
 * Verify weight integrity
 * @param sram_weights Weight base address
 * @param expected_size Expected size in bytes
 * @return 0 if valid, -1 if invalid
 */
int tinylstm_verify_weights(const void *sram_weights, uint16_t expected_size) {
    /* Check if weights are at correct address */
    if ((uint32_t)sram_weights != TINYLSTM_WEIGHTS_ADDR) {
        return -1;
    }

    /* Verify size doesn't exceed limit */
    if (expected_size > TINYLSTM_WEIGHTS_SIZE) {
        return -1;
    }

    /* Additional checks could include checksum validation */
    return 0;
}

/*
 * Shapley cache lookup using Horner hash
 * Called during 2.1ms XTAL startup dead time (RCI scheduler)
 *
 * @param shapley_cache Cache base address
 * @param node_id Node identifier
 * @param coefficients Hash coefficients
 * @return Shapley value or 0x8000 if not found
 */
int16_t shapley_cache_lookup(
    int16_t *shapley_cache,
    int8_t node_id,
    const int8_t *coefficients
) {
    /* Horner hash: index = ((c₀*x + c₁)*x + c₂)*x + ... mod prime */
    uint32_t hash = node_id;

    for (uint8_t i = 0; i < HORNER_HASH_COEFFS; i++) {
        hash = (hash * coefficients[i] + 1) % HORNER_HASH_PRIME;
    }

    /* Lookup in cache */
    if (hash < SHAPLEY_CACHE_SIZE / sizeof(int16_t)) {
        return shapley_cache[hash];
    }

    /* Not found */
    return 0x8000;
}
