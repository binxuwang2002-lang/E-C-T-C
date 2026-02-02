/*
** ###################################################################
**     Header for TinyLSTM Horner-INT8 RCI Kernel
**
**     Optimized for Radio-Compute Interleaving on Cortex-M33
**
** ###################################################################
*/

#ifndef TINYLSTM_HORNER_RCI_H_
#define TINYLSTM_HORNER_RCI_H_

#include <stdint.h>
#include <stddef.h>

/* =============================================================================
 * API Functions - RCI Optimized
 * =============================================================================
 */

/**
 * Main RCI handler: Decode schedule and execute inference
 *
 * @param schedule_bitmask 32-bit schedule from gateway
 * @param horner_coeffs Hash coefficients for Horner decoding
 * @param x_t Current input (INT8)
 * @param h_prev Previous hidden state (32 INT8 values)
 * @param h_out Output hidden state (32 INT8 values)
 * @return 1 to execute task, 0 to sleep
 *
 * Time: <1.2ms (fits in 2.1ms RCI window)
 */
uint8_t rci_decode_and_infer(uint32_t schedule_bitmask,
                             const uint8_t horner_coeffs[4],
                             int8_t x_t,
                             const int8_t h_prev[32],
                             int8_t h_out[32]);

/**
 * Quick decision during RCI (fast path, Layer 1 only)
 *
 * @param x_t Input (INT8)
 * @param h_prev Previous hidden state (32 INT8)
 * @return 1 if should execute, 0 otherwise
 *
 * Time: <200μs (ultra-fast path)
 */
uint8_t rci_quick_decision(int8_t x_t,
                           const int8_t h_prev[32]);

/**
 * Horner-based O(1) schedule lookup
 *
 * @param node_id Node identifier (0-255)
 * @param timestamp Current timestamp
 * @param coeffs Horner coefficients [4]
 * @return Schedule bitmask (32-bit)
 */
uint32_t horner_schedule_lookup(uint8_t node_id,
                                uint32_t timestamp,
                                const uint8_t coeffs[4]);

/**
 * Load weights from flash to SRAM
 *
 * @param flash_weights Source address in flash
 * @param size Size in bytes (< 8192)
 * @return 0 on success, -1 on error
 */
int load_rci_weights(const void *flash_weights, uint16_t size);

/**
 * Update Horner coefficients for adaptive scheduling
 *
 * @param coeffs New coefficients [a₀, a₁, a₂, a₃]
 */
void update_horner_coeffs(const uint8_t coeffs[4]);

/**
 * Get pointer to weight structure in SRAM
 *
 * @return Pointer at 0x20000000
 */
void* get_rci_weights(void);

/* =============================================================================
 * Memory Layout Constants (from Linker Script)
 * =============================================================================
 */

#define TINYLSTM_WEIGHTS_ADDR     0x20000000  /* .tinylstm_weights section */
#define SHAPLEY_CACHE_ADDR        0x20002000  /* .shapley_cache section */
#define RETENTION_RAM_ADDR        0x20006400  /* .retention_ram section */

#define TINYLSTM_WEIGHTS_SIZE     8192        /* 8KB limit */
#define SHAPLEY_CACHE_SIZE        4096        /* 4KB limit */
#define RETENTION_RAM_SIZE        8192        /* 8KB limit */

/* =============================================================================
 * RCI Timing Constraints
 * =============================================================================
 */

#define RCI_XTAL_STARTUP_MS       2.1f       /* XTAL startup dead time */
#define RCI_MAX_INFERENCE_US      1200       /* 1.2ms inference time */
#define RCI_QUICK_DECISION_US     200        /* 200μs fast path */
#define RCI_MARGIN_US             900        /* 0.9ms margin for radio */

/* =============================================================================
 * Horner Hash Parameters
 * =============================================================================
 */

#define HORNER_PRIME              32749      /* Prime modulus */
#define HORNER_COEFFS_COUNT       4          /* Number of coefficients */
#define SCHEDULE_SLOTS            32         /* 32 execution slots */

/* =============================================================================
 * Weight Structure (fits in 8KB)
 * =============================================================================
 */

typedef struct __attribute__((packed)) {
    /* Layer 1 (32 units) - Fast path */
    int8_t  l1_w_ih[32];
    int8_t  l1_w_hh[32][32];
    int8_t  l1_bias[32];
    int16_t l1_scale[32];

    /* Layer 2 (16 units) - Deferred */
    int8_t  l2_w_ih[16][32];
    int8_t  l2_w_hh[16][16];
    int8_t  l2_bias[16];
    int16_t l2_scale[16];

    /* FC Output (4) */
    int8_t  fc_w[4][16];
    int16_t fc_b[4];
    int16_t fc_scale[4];

    /* RCI Scheduling */
    uint8_t horner_coeffs[4];
    uint8_t spare[12];

} tinylstm_rci_weights_t;

/* Ensure structure fits in 8KB */
_Static_assert(sizeof(tinylstm_rci_weights_t) <= 8192,
               "Weight structure exceeds 8KB limit");

/* =============================================================================
 * Schedule Bitmask Macros
 * =============================================================================
 */

/**
 * Create schedule bitmask for execution slot
 *
 * @param slot Execution slot (0-31)
 * @return Bitmask with bit 'slot' set
 */
#define SCHEDULE_BITMASK(slot)    (1U << ((slot) & 0x1F))

/**
 * Check if slot is scheduled
 *
 * @param mask Schedule bitmask
 * @param slot Slot to check (0-31)
 * @return 1 if scheduled, 0 otherwise
 */
#define SCHEDULE_IS_SET(mask, slot)   (((mask) >> ((slot) & 0x1F)) & 1U)

/**
 * Find first scheduled slot
 *
 * @param mask Schedule bitmask
 * @return Slot number or 32 if none
 */
#define SCHEDULE_FIRST_SLOT(mask)     __builtin_clz(mask)

/* =============================================================================
 * Assembly-Optimized Operations for Cortex-M33
 * =============================================================================
 */

#if defined(__ARM_ARCH_8M_MAIN__) || defined(__ARM_FEATURE_DSP)
#define HAVE_DSP_EXTENSIONS 1

/**
 * MLA (Multiply-Accumulate) for INT8
 *
 * @param result Current result (INT32)
 * @param a First operand (INT8)
 * @param b Second operand (INT8)
 * @return result * a + b (using SMLABB)
 */
static inline int32_t smlabb_mla(int32_t result, int8_t a, int8_t b) {
    return __SMLABB(result, a, b);
}
#else
#define HAVE_DSP_EXTENSIONS 0

static inline int32_t smlabb_mla(int32_t result, int8_t a, int8_t b) {
    return (result * a) + b;
}
#endif

/* =============================================================================
 * Example Usage
 * =============================================================================
 */

/*
#include "tinylstm_horner_rci.h"

void example_rci_usage(void) {
    // Setup
    static int8_t hidden_state[32] = {0};
    uint8_t coeffs[4] = {0x12, 0x34, 0x56, 0x78};

    // RCI event (XTAL startup)
    uint32_t schedule = 0x00000001;  // Execute in slot 0
    int8_t input = 5;
    int8_t output[32];

    // Decode and infer
    uint8_t decision = rci_decode_and_infer(schedule, coeffs,
                                            input, hidden_state, output);

    if (decision) {
        // Execute task
        execute_transmission();
    }
}
*/

#endif /* TINYLSTM_HORNER_RCI_H_ */
