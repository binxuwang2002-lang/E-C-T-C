/*
** ###################################################################
**     Header file for TinyLSTM Horner INT8 Kernel
**
**     Provides function declarations and memory layout definitions
**
** ###################################################################
*/

#ifndef TINYLSTM_HORNER_INT8_H_
#define TINYLSTM_HORNER_INT8_H_

#include <stdint.h>

/* =============================================================================
 * API Functions
 * =============================================================================
 */

/**
 * Full LSTM forward pass (2 layers) using Horner's method
 *
 * @param input_sequence INT8 input sequence
 * @param sequence_length Length of input sequence (< 128)
 * @param hidden_states Output buffer for hidden states
 * @return 0 on success, -1 on error
 */
int tinylstm_horner_inference(const int8_t *input_sequence,
                              uint16_t sequence_length,
                              int8_t *hidden_states);

/**
 * Load quantized weights from flash to SRAM
 *
 * @param flash_weights Source address in flash
 * @param sram_weights Destination address in SRAM (0x20000000)
 * @param size Size in bytes (< 8192)
 */
void tinylstm_load_weights(const void *flash_weights,
                           void *sram_weights,
                           uint16_t size);

/**
 * Verify weight integrity
 *
 * @param sram_weights Weight base address
 * @param expected_size Expected size in bytes
 * @return 0 if valid, -1 if invalid
 */
int tinylstm_verify_weights(const void *sram_weights, uint16_t expected_size);

/**
 * Look up Shapley value using Horner hash
 *
 * @param shapley_cache Pointer to cache base (0x20002000)
 * @param node_id Node identifier (INT8)
 * @param coefficients Hash coefficients
 * @return Shapley value or 0x8000 if not found
 */
int16_t shapley_cache_lookup(int16_t *shapley_cache,
                             int8_t node_id,
                             const int8_t *coefficients);

/**
 * Get weight memory usage
 *
 * @return Size in bytes used by weights
 */
uint16_t tinylstm_get_weight_size(void);

/**
 * Get cache memory usage
 *
 * @return Size in bytes used by Shapley cache
 */
uint16_t tinylstm_get_cache_size(void);

/* =============================================================================
 * Memory Layout Constants
 * =============================================================================
 */

/* Memory addresses from linker script */
#define TINYLSTM_WEIGHTS_ADDR  0x20000000  /* .tinylstm_weights section */
#define SHAPLEY_CACHE_ADDR     0x20002000  /* .shapley_cache section */
#define RETENTION_RAM_ADDR     0x20006400  /* .retention_ram section */

/* Memory size limits (from ECTC Table I) */
#define TINYLSTM_WEIGHTS_SIZE  8192   /* 8KB limit */
#define SHAPLEY_CACHE_SIZE     4096   /* 4KB limit */
#define RETENTION_RAM_SIZE     8192   /* 8KB limit */

/* =============================================================================
 * Horner's Method Parameters
 * =============================================================================
 */

/* Horner hash parameters for Shapley cache */
#define HORNER_HASH_PRIME      32749   /* Modulus prime for hash function */
#define HORNER_HASH_COEFFS     4       /* Number of coefficients */

/* Scale factors for INT8 quantization */
#define LAYER1_SCALE_BITS      8       /* Layer 1 quantization bits */
#define LAYER2_SCALE_BITS      10      /* Layer 2 quantization bits */
#define FC_SCALE_BITS          12      /* FC layer quantization bits */

/* =============================================================================
 * LSTM Configuration
 * =============================================================================
 */

/* LSTM architecture */
#define LSTM_LAYER1_HIDDEN     32      /* First layer hidden units */
#define LSTM_LAYER2_HIDDEN     16      /* Second layer hidden units */
#define LSTM_INPUT_SIZE        1       /* Input dimension */
#define LSTM_OUTPUT_SIZE       4       /* Output dimension */

/* Max sequence length */
#define LSTM_MAX_SEQ_LEN       128     /* Maximum supported sequence */

/* =============================================================================
 * Performance Metrics
 * =============================================================================
 */

/* Execution time budget (from ECTC paper) */
#define HORNER_EXEC_TIME_US    1200    /* 1.2ms for 32-unit LSTM */

/* Memory bandwidth reduction (INT8 vs FP32) */
#define INT8_BANDWIDTH_SAVING  4       /* 4x reduction in memory access */

/* Kernel size constraint (from Fig. 8) */
#define KERNEL_SIZE_LIMIT      2150    /* 2.1KB maximum */

/* =============================================================================
 * DSP Extension Detection
 * =============================================================================
 */

#if defined(__ARM_ARCH_8M_MAIN__) || defined(__ARM_FEATURE_DSP)
#define USE_DSP_EXTENSIONS 1
#else
#define USE_DSP_EXTENSIONS 0
#endif

/* =============================================================================
 * Weight Format Structure
 * =============================================================================
 */

typedef struct {
    /* Layer 1 (32 hidden units) */
    int8_t w_ih[32];        /* Input-to-hidden weights (32 bytes) */
    int8_t w_hh[32][32];    /* Hidden-to-hidden weights (1024 bytes) */
    int8_t b_h[32];         /* Bias terms (32 bytes) */
    int16_t scale[32];      /* Scale factors (64 bytes) */

    /* Layer 2 (16 hidden units) */
    int8_t w_ih_l2[16][32]; /* Input-to-hidden weights (512 bytes) */
    int8_t w_hh_l2[16][16]; /* Hidden-to-hidden weights (256 bytes) */
    int8_t b_h_l2[16];      /* Bias terms (16 bytes) */
    int16_t scale_l2[16];   /* Scale factors (32 bytes) */

    /* FC Output Layer */
    int8_t fc_w[4][16];     /* FC weights (64 bytes) */
    int16_t fc_b[4];        /* FC bias (8 bytes) */
    int16_t fc_scale[4];    /* FC scale (8 bytes) */

} tinylstm_weights_t;

/* Ensure sizeof(tinylstm_weights_t) <= 8192 bytes */
_Static_assert(sizeof(tinylstm_weights_t) <= 8192,
               "Weight structure exceeds 8KB limit");

#endif /* TINYLSTM_HORNER_INT8_H_ */
