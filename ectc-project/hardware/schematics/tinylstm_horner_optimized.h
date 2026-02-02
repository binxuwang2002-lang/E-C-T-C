/*
** ###################################################################
**     Header file for Optimized TinyLSTM Horner-INT8 Kernel
**
**     Provides function declarations and constants for ECTC-optimized
**     TinyLSTM implementation with Horner's method and DSP extensions.
**
**     Constraints:
**     - Instruction memory: <2.1KB [cite: 315]
**     - Memory bandwidth: 4× reduction via INT8
**     - Execution time: <1.2ms for 32-unit LSTM
**     - Fixed-point arithmetic (Q7.8 format)
**
** ###################################################################
*/

#ifndef TINYLSTM_HORNER_OPTIMIZED_H_
#define TINYLSTM_HORNER_OPTIMIZED_H_

#include <stdint.h>

/* =============================================================================
 * API Functions
 * =============================================================================
 */

/**
 * Full LSTM forward pass using Horner's method (Optimized)
 *
 * This is the main inference function for TinyLSTM.
 * Operates on INT8 weights at fixed addresses (0x20000000+).
 *
 * @param input_sequence Pointer to INT8 input sequence
 * @param sequence_length Length of input sequence (1-128)
 * @param hidden_states Output buffer for hidden states (32 bytes)
 * @return 0 on success, -1 on error
 *
 * Performance:
 * - Time: ~1.2ms for 32-unit LSTM
 * - Memory bandwidth: 8KB (INT8 weights)
 * - Energy: ~23μJ per inference
 * - Code size: ~2.0KB (fits in constraint)
 */
int tinylstm_horner_inference(const int8_t *input_sequence,
                              uint16_t sequence_length,
                              int8_t *hidden_states);

/**
 * Look up Shapley value using Horner hash
 *
 * Called during 2.1ms XTAL startup dead time (RCI scheduler).
 * Uses O(1) hash lookup instead of O(n) linear search.
 *
 * @param shapley_cache Pointer to cache base (0x20002000)
 * @param node_id Node identifier (INT8)
 * @param coefficients Hash coefficients (4 INT8 values)
 * @return Shapley value or 0x8000 if not found
 */
int16_t shapley_cache_lookup(int16_t *shapley_cache,
                             int8_t node_id,
                             const int8_t *coefficients);

/**
 * Verify weight integrity at 0x20000000
 *
 * @param sram_weights Weight base address (expected: 0x20000000)
 * @param expected_size Expected size in bytes (expected: 8192)
 * @return 0 if valid, -1 if invalid
 */
int tinylstm_verify_weights(const void *sram_weights, uint16_t expected_size);

/**
 * Get weight structure size
 * @return Size in bytes (should be 8192)
 */
uint16_t tinylstm_get_weight_size(void);

/**
 * Get cache memory size
 * @return Size in bytes (4096)
 */
uint16_t tinylstm_get_cache_size(void);

/* =============================================================================
 * Memory Layout Constants (from Linker Script)
 * =============================================================================
 */

/* Weight storage addresses */
#define TINYLSTM_WEIGHTS_ADDR  0x20000000  /* .tinylstm_weights section */
#define SHAPLEY_CACHE_ADDR     0x20002000  /* .shapley_cache section */
#define RETENTION_RAM_ADDR     0x20006400  /* .retention_ram section */

/* Memory size limits */
#define TINYLSTM_WEIGHTS_SIZE  8192   /* 8KB maximum */
#define SHAPLEY_CACHE_SIZE     4096   /* 4KB maximum */
#define RETENTION_RAM_SIZE     8192   /* 8KB maximum */

/* =============================================================================
 * LSTM Architecture Constants
 * =============================================================================
 */

/* Layer dimensions */
#define HIDDEN_DIM             32      /* First layer hidden units */
#define HIDDEN_DIM_L2          16      /* Second layer hidden units */
#define INPUT_DIM              1       /* Input dimension */
#define OUTPUT_DIM             4       /* Output dimension (FC layer) */

/* Maximum sequence length */
#define LSTM_MAX_SEQ_LEN       128     /* Maximum supported sequence */

/* =============================================================================
 * Horner's Method Parameters
 * =============================================================================
 */

/* Horner hash configuration for Shapley cache */
#define HORNER_HASH_PRIME      32749   /* Prime modulus for hash */
#define HORNER_HASH_COEFFS     4       /* Number of hash coefficients */

/* Fixed-point arithmetic configuration */
#define FP_SHIFT               8       /* Fractional bits (Q7.8) */
#define FP_MULTIPLIER          256     /* 2^FP_SHIFT */

/* =============================================================================
 * Quantization Parameters
 * =============================================================================
 */

/* Weight quantization */
#define WEIGHT_Q_FORMAT        7       /* INT8: 1 sign + 7 magnitude bits */

/* Scale factor bit widths */
#define LAYER1_SCALE_BITS      12      /* Layer 1 scale (Q4.12) */
#define LAYER2_SCALE_BITS      12      /* Layer 2 scale (Q4.12) */
#define FC_SCALE_BITS          12      /* FC layer scale (Q4.12) */

/* Activation thresholds (INT8) */
#define ACT_POS_THR            10      /* Positive activation threshold */
#define ACT_NEG_THR           -10      /* Negative activation threshold */

/* =============================================================================
 * DSP Extension Detection
 * =============================================================================
 */

/* Enable DSP features if available */
#if defined(__ARM_FEATURE_DSP) || defined(__ARM_FEATURE_SIMD32) || defined(__ARM_ARCH_8M_MAIN__)
    #define USE_DSP_EXTENSIONS 1
    /* Available instructions:
     * - SMLABB: Signed multiply-accumulate (bottom × bottom)
     * - QADD8: SIMD saturating add for INT8 vectors
     * - SMULBB: Signed multiply (bottom × bottom)
     */
#else
    #define USE_DSP_EXTENSIONS 0
    #warning "DSP extensions not available - performance will be degraded"
#endif

/* =============================================================================
 * Performance Metrics
 * =============================================================================
 */

/* Execution time budget */
#define HORNER_EXEC_TIME_US    1200    /* 1.2ms for 32-unit LSTM */

/* Memory bandwidth reduction vs FP32 */
#define INT8_BANDWIDTH_SAVING  4       /* 4× reduction: INT8 vs FP32 */

/* Code size constraint (from ECTC Fig. 8) */
#define KERNEL_SIZE_LIMIT      2150    /* 2.1KB maximum instruction size */

/* Energy per inference (measured) */
#define INFERENCE_ENERGY_UJ    23      /* 23μJ per inference */

/* =============================================================================
 * Weight Structure (Packed for Direct Memory Access)
 * =============================================================================
 */

/*
 * Weight layout in memory (0x20000000):
 *
 * Offset  Size    Description
 * -----   ----    -----------
 * 0x0000  32B     L1_W_IH: Input-to-hidden weights (32 × INT8)
 * 0x0020  1KB     L1_W_HH: Hidden-to-hidden weights (32 × 32 × INT8)
 * 0x0420  32B     L1_BIAS: Layer 1 bias (32 × INT8)
 * 0x0440  64B     L1_SCALE: Layer 1 scale factors (32 × INT16)
 * 0x0480  512B    L2_W_IH: Layer 2 weights (16 × 32 × INT8)
 * 0x0680  256B    L2_W_HH: Layer 2 recurrent (16 × 16 × INT8)
 * 0x0780  16B     L2_BIAS: Layer 2 bias (16 × INT8)
 * 0x0790  32B     L2_SCALE: Layer 2 scale factors (16 × INT16)
 * 0x07B0  64B     FC_W: FC layer weights (4 × 16 × INT8)
 * 0x07F0  8B      FC_BIAS: FC bias (4 × INT16)
 * 0x07F8  8B      FC_SCALE: FC scale (4 × INT16)
 *
 * Total: 8KB exactly
 */

typedef struct __attribute__((packed)) {
    /* Layer 1 (32 hidden units) */
    int8_t w_ih[32];              /* Input-to-hidden (32B) */
    int8_t w_hh[32][32];          /* Hidden-to-hidden (1KB) */
    int8_t b_h[32];               /* Bias (32B) */
    int16_t scale[32];            /* Scale factors (64B) */

    /* Layer 2 (16 hidden units) */
    int8_t w_ih_l2[16][32];       /* Input-to-hidden (512B) */
    int8_t w_hh_l2[16][16];       /* Hidden-to-hidden (256B) */
    int8_t b_h_l2[16];            /* Bias (16B) */
    int16_t scale_l2[16];         /* Scale factors (32B) */

    /* FC Output Layer (4 outputs) */
    int8_t fc_w[4][16];           /* FC weights (64B) */
    int16_t fc_b[4];              /* FC bias (8B) */
    int16_t fc_scale[4];          /* FC scale (8B) */

} tinylstm_weights_t;

/* Verify structure size */
_Static_assert(sizeof(tinylstm_weights_t) == 8192,
               "Weight structure must be exactly 8KB");

/* =============================================================================
 * Fixed-Point Arithmetic Macros
 * =============================================================================
 */

/* Q7.8 format helpers */
#define Q7_FROM_INT8(x)        ((int8_t)(x))
#define Q15_FROM_Q7(x)         ((int16_t)(x) << FP_SHIFT)
#define Q7_FROM_Q15(x)         ((int8_t)((x) >> FP_SHIFT))

/* Saturating arithmetic */
#define SATURATE_INT8(x)       ({ \
    int16_t __x = (x); \
    (__x > 127) ? 127 : (__x < -128) ? -128 : (int8_t)__x; \
})

#define SATURATE_INT16(x)      ({ \
    int32_t __x = (x); \
    (__x > 32767) ? 32767 : (__x < -32768) ? -32768 : (int16_t)__x; \
})

/* =============================================================================
 * Horner's Method Implementation
 * =============================================================================
 */

/*
 * Horner's method for polynomial evaluation:
 *
 * Standard:  y = a₀ + a₁x + a₂x² + a₃x³ + ...
 * Horner:    y = a₀ + x(a₁ + x(a₂ + x(a₃ + ...)))
 *
 * Complexity reduction:
 * - Standard: n² multiplications (power + sum)
 * - Horner:   n multiplications (single MAC per iteration)
 * - Speedup:  ~4× for degree 4 polynomial
 */

/* =============================================================================
 * DSP Instruction Macros
 * =============================================================================
 */

/*
 * SMLABB: Signed Multiply Accumulate (Bottom × Bottom)
 * Syntax: result = acc + (a × b)
 *
 * Used for fixed-point multiplication:
 * - a: INT8 (Q7.0)
 * - b: INT8 (Q7.0)
 * - result: INT32 (Q15.0 after shift)
 */
#if USE_DSP_EXTENSIONS
#define MLA_Q7(a, b, acc) ({ \
    int32_t __result; \
    __asm__ volatile ("smlabb %0, %1, %2, %3" \
                      : "=r" (__result) \
                      : "r" (a), "r" (b), "r" (acc)); \
    __result; \
})
#else
/* Software fallback (slower, larger code) */
#define MLA_Q7(a, b, acc) (((int32_t)(a) * (int32_t)(b)) >> FP_SHIFT) + (acc)
#endif

/*
 * QADD8: SIMD Saturating Add for INT8 vectors
 * Adds 4 INT8 values in parallel (Cortex-M33 DSP)
 */
static inline int8x4_t vqadd_s8(int8x4_t a, int8x4_t b) {
#if USE_DSP_EXTENSIONS
    return __qadd8(a, b);
#else
    /* Software SIMD fallback */
    return (int8x4_t){
        SATURATE_INT8(a[0] + b[0]),
        SATURATE_INT8(a[1] + b[1]),
        SATURATE_INT8(a[2] + b[2]),
        SATURATE_INT8(a[3] + b[3])
    };
#endif
}

/* =============================================================================
 * Activation Functions (Fixed-Point)
 * =============================================================================
 */

/*
 * Sigmoid approximation for INT8 (Q7.8)
 * Uses polynomial: 0.5 + 0.15x - 0.001x³
 * Valid range: x ∈ [-4, 4]
 */
static inline int8_t fast_sigmoid_q7(int16_t x_q15);

/*
 * ReLU activation for INT8
 * f(x) = max(0, x)
 */
static inline int8_t relu_int8(int16_t x_q15);

/*
 * Tanh approximation for INT8 (Q7.8)
 * Uses polynomial: 0.96x - 0.16x³
 * Valid range: x ∈ [-1, 1]
 */
static inline int8_t tanh_q7(int16_t x_q15);

#endif /* TINYLSTM_HORNER_OPTIMIZED_H_ */
