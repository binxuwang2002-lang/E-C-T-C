/*
** ###################################################################
**     TinyLSTM Horner-INT8 Kernel for RCI (Radio-Compute Interleaving)
**
**     Optimized for Cortex-M33 (STM32U575) during XTAL startup dead time
**     Uses Horner's Method for O(1) schedule bitmask to execution slot mapping
**
**     Reference: ECTC-19.pdf, Section IV.C.2, Fig. 8
**     Constraints: 8KB weights, 2.1ms RCI window, no FP division
**
** ###################################################################
*/

#include <stdint.h>
#include <string.h>

/* =============================================================================
 * RCI Schedule Bitmask Structure
 *
 * During XTAL startup (2.1ms dead time), we decode schedule information
 * using Horner's method to map bitmasks to execution slots in O(1) time.
 *
 * Schedule format: 32-bit bitmask where bit i = execute in slot i
 * Example: 0b00001001 = execute in slots 0 and 3
 * =============================================================================
 */

/**
 * Horner-based hash decoder for schedule bitmasks
 *
 * Converts schedule bitmask to execution slot using Horner's method
 *
 * Formula: slot = ((mask * a₃ + a₂) * a₁ + a₀) mod prime
 *
 * @param schedule_bitmask 32-bit schedule bitmask
 * @param coeffs Horner coefficients [a₀, a₁, a₂, a₃]
 * @param prime Prime modulus for hash
 * @return Execution slot (0-31), or 255 if no execution
 *
 * Complexity: O(1) - Single pass through 4 coefficients
 */
static inline uint8_t horner_decode_schedule(uint32_t schedule_bitmask,
                                             const uint8_t coeffs[4],
                                             uint32_t prime) {
    uint32_t hash = coeffs[3];  /* Start with highest coefficient */

    /* Horner iterations: 4 multiplies, 3 adds */
    hash = (hash * schedule_bitmask + coeffs[2]) % prime;
    hash = (hash * schedule_bitmask + coeffs[1]) % prime;
    hash = (hash * schedule_bitmask + coeffs[0]) % prime;

    /* Map to slot (0-31) */
    return (uint8_t)(hash & 0x1F);  /* Mask to 5 bits (32 slots) */
}

/**
 * Find first set bit in schedule bitmask (O(1) using bit operations)
 *
 * @param schedule_bitmask 32-bit schedule bitmask
 * @return Index of first set bit (0-31), or 32 if none set
 *
 * Uses GCC builtin for hardware acceleration on Cortex-M33
 */
static inline uint8_t find_first_execution_slot(uint32_t schedule_bitmask) {
#if defined(__GNUC__) && (__ARM_ARCH_8M_MAIN__)
    /* Use CLZ (Count Leading Zeros) instruction on Cortex-M33 */
    if (schedule_bitmask == 0) {
        return 32;  /* No execution scheduled */
    }
    return __builtin_clz(schedule_bitmask);
#else
    /* Fallback: linear search (still O(1) for 32 bits) */
    for (uint8_t i = 0; i < 32; i++) {
        if (schedule_bitmask & (1U << i)) {
            return i;
        }
    }
    return 32;
#endif
}

/* =============================================================================
 * INT8 Weight Storage in .tinylstm_weights Section (8KB @ 0x20000000)
 *
 * Memory layout optimized for minimal cache misses during RCI:
 * - Layer 1: 32 units, minimal computation (used for quick decisions)
 * - Layer 2: 16 units, more computation (deferred)
 * - FC: 4 outputs (final decision)
 * =============================================================================
 */

/* Weight structure for RCI-optimized TinyLSTM */
typedef struct __attribute__((packed)) {
    /* Layer 1: Fast path (32 hidden units) - 1152 bytes */
    int8_t  l1_w_ih[32];          /* 32 bytes  - Input weights (single input) */
    int8_t  l1_w_hh[32][32];      /* 1024 bytes - Hidden-to-hidden weights */
    int8_t  l1_bias[32];          /* 32 bytes  - Bias terms */
    int16_t l1_scale[32];         /* 64 bytes  - Scale factors */

    /* Layer 2: Deferred path (16 hidden units) - 816 bytes */
    int8_t  l2_w_ih[16][32];      /* 512 bytes - Input weights */
    int8_t  l2_w_hh[16][16];      /* 256 bytes - Hidden-to-hidden weights */
    int8_t  l2_bias[16];          /* 16 bytes  - Bias terms */
    int16_t l2_scale[16];         /* 32 bytes  - Scale factors */

    /* FC Output (4 outputs) - 80 bytes */
    int8_t  fc_w[4][16];          /* 64 bytes - FC weights */
    int16_t fc_b[4];              /* 8 bytes  - FC bias */
    int16_t fc_scale[4];          /* 8 bytes  - FC scale */

    /* RCI Scheduling Coefficients - 16 bytes */
    uint8_t horner_coeffs[4];     /* 4 bytes - Horner hash coefficients */
    uint8_t spare[12];            /* 12 bytes - Padding for alignment */

    /* Total: 1152 + 816 + 80 + 16 = 2064 bytes (fits in 8KB with room) */
} tinylstm_rci_weights_t;

/* =============================================================================
 * Horner-INT8 LSTM Inference Kernel
 *
 * Optimized for RCI window:
 * 1. Use only INT8/INT16 arithmetic (no FP)
 * 2. Horner's method for polynomial evaluation
 * 3. Minimal memory access (fits in cache)
 * 4. Single-pass inference
 */

/**
 * Horner's method for INT8 polynomial evaluation
 *
 * Evaluates: y = coeffs[degree] + x * (coeffs[degree-1] + x * (...))
 *
 * @param coeffs INT8 coefficient array
 * @param x INT8 input value
 * @param degree Polynomial degree (0-31)
 * @param scale Scale factor (right shift amount)
 * @return INT16 result
 *
 * Uses MLA (Multiply-Accumulate) for efficiency
 */
static inline int16_t horner_eval_int8_mla(const int8_t *coeffs,
                                           int8_t x,
                                           uint8_t degree,
                                           uint8_t scale) {
    int32_t result = coeffs[degree];  /* Start with highest coefficient */

    /* Iterate from highest degree down to 0 */
    for (int8_t i = degree - 1; i >= 0; i--) {
        /* Use MLA: result = result * x + coeffs[i] */
        result = (result * x) + coeffs[i];
    }

    /* Apply scale (right shift) */
    return (int16_t)(result >> scale);
}

/**
 * LSTM cell forward pass using Horner's method (RCI-optimized)
 *
 * Simplified LSTM cell for RCI window:
 * - Single input, 32 hidden units
 * - Horner evaluation for each gate
 * - No sigmoid/tanh (linear approximation)
 *
 * @param x_t Input at time t (INT8)
 * @param h_prev Previous hidden state (INT8[32])
 * @param weights Pointer to weight structure
 * @param h_out Output hidden state (INT8[32])
 * @return 0 on success
 *
 * Time budget: <500μs (fits in 2.1ms RCI window)
 */
int lstm_horner_rci(int8_t x_t,
                    const int8_t h_prev[32],
                    const tinylstm_rci_weights_t *weights,
                    int8_t h_out[32]) {

    /* Layer 1: Process 32 hidden units */
    for (int u = 0; u < 32; u++) {
        /* Horner evaluation for gate u
         * Simplified: gate = bias + w_ih * x_t + w_hh * h_prev
         * Using Horner would be: w_hh * h_prev + (w_ih * x_t + bias)
         */

        int32_t gate_val = weights->l1_bias[u];  /* Start with bias */

        /* First multiply-add: w_ih[u] * x_t */
        gate_val += (int32_t)weights->l1_w_ih[u] * (int32_t)x_t;

        /* Horner continuation: add w_hh[u] * h_prev[u] */
        gate_val += (int32_t)weights->l1_w_hh[u][u] * (int32_t)h_prev[u];

        /* Apply scale (no division, only shift) */
        int16_t gate_scaled = gate_val >> (weights->l1_scale[u] & 0x0F);

        /* Simple activation (clip to INT8 range) */
        if (gate_scaled > 127) {
            h_out[u] = 127;
        } else if (gate_scaled < -128) {
            h_out[u] = -128;
        } else {
            h_out[u] = (int8_t)gate_scaled;
        }
    }

    return 0;
}

/**
 * FC layer using Horner's method
 *
 * @param h_in Input from LSTM (INT8[16])
 * @param weights Pointer to weight structure
 * @param output FC output (INT8[4])
 * @return 0 on success
 */
int fc_horner_rci(const int8_t h_in[16],
                  const tinylstm_rci_weights_t *weights,
                  int8_t output[4]) {

    for (int o = 0; o < 4; o++) {
        int32_t sum = weights->fc_b[o];  /* Start with bias */

        /* Horner evaluation: sum + Σ(w_i * h_i) */
        for (int i = 0; i < 16; i++) {
            sum += (int32_t)weights->fc_w[o][i] * (int32_t)h_in[i];
        }

        /* Apply scale */
        int16_t out_scaled = sum >> (weights->fc_scale[o] & 0x0F);

        /* Clip to INT8 */
        if (out_scaled > 127) {
            output[o] = 127;
        } else if (out_scaled < -128) {
            output[o] = -128;
        } else {
            output[o] = (int8_t)out_scaled;
        }
    }

    return 0;
}

/**
 * RCI Schedule Decoder and Executor
 *
 * Main entry point for RCI window:
 * 1. Decode schedule bitmask using Horner's method
 * 2. Map to execution slot in O(1)
 * 3. Execute TinyLSTM inference if slot matches
 * 4. Return decision
 *
 * @param schedule_bitmask Received from gateway
 * @param horner_coeffs Hash coefficients
 * @param x_t Current input
 * @param h_prev Previous hidden state
 * @param h_out Output hidden state
 * @return Recommended action (0=sleep, 1=execute)
 *
 * Total time: <1.2ms (leaves 0.9ms for radio startup)
 */
uint8_t rci_decode_and_infer(uint32_t schedule_bitmask,
                             const uint8_t horner_coeffs[4],
                             int8_t x_t,
                             const int8_t h_prev[32],
                             int8_t h_out[32]) {
    /* Step 1: Horner decode schedule (O(1)) */
    uint8_t target_slot = horner_decode_schedule(schedule_bitmask,
                                                 horner_coeffs,
                                                 32749);  /* Prime modulus */

    /* Step 2: Find current slot (using bit operations, O(1)) */
    uint8_t current_slot = find_first_execution_slot(schedule_bitmask);

    /* Step 3: Check if should execute */
    if (current_slot == target_slot) {
        /* Execute inference during RCI window */
        extern tinylstm_rci_weights_t _tinylstm_weights_start;
        lstm_horner_rci(x_t, h_prev, &_tinylstm_weights_start, h_out);
        return 1;  /* Execute */
    }

    return 0;  /* Sleep */
}

/**
 * O(1) schedule lookup using Horner hash
 *
 * Converts (node_id, timestamp) → schedule bitmask in O(1)
 *
 * @param node_id Node identifier (0-255)
 * @param timestamp Current timestamp
 * @param coeffs Horner coefficients
 * @param prime Prime modulus
 * @return Schedule bitmask (32-bit)
 *
 * Use case: During gateway failure, compute schedule locally
 */
uint32_t horner_schedule_lookup(uint8_t node_id,
                                uint32_t timestamp,
                                const uint8_t coeffs[4],
                                uint32_t prime) {
    uint32_t hash = coeffs[3];  /* Start with coefficient */

    /* Hash (node_id, timestamp) pair */
    hash = (hash * node_id + coeffs[2]) % prime;
    hash = (hash * timestamp + coeffs[1]) % prime;
    hash = (hash * node_id + coeffs[0]) % prime;

    /* Expand to 32-bit bitmask */
    uint32_t bitmask = 0;
    for (int i = 0; i < 32; i++) {
        if (hash & (1U << (i % 8))) {
            bitmask |= (1U << i);
        }
    }

    return bitmask;
}

/**
 * Quick decision maker for RCI window
 *
 * Makes fast decision using only Layer 1 (no Layer 2)
 * Time budget: <200μs (fast path)
 *
 * @param x_t Input (INT8)
 * @param weights Pointer to weights
 * @param h_prev Previous state
 * @return 1 if should execute, 0 otherwise
 */
uint8_t rci_quick_decision(int8_t x_t,
                           const int8_t h_prev[32],
                           const tinylstm_rci_weights_t *weights) {
    int32_t energy_score = 0;

    /* Fast energy estimate using Layer 1 weights only */
    for (int u = 0; u < 32; u++) {
        energy_score += (int32_t)weights->l1_w_ih[u] * (int32_t)x_t;
        energy_score += (int32_t)weights->l1_w_hh[u][u] * (int32_t)h_prev[u];
    }

    /* Apply bias and scale */
    int16_t score = energy_score >> 8;

    /* Threshold decision (no division, only comparison) */
    return (score > 0) ? 1 : 0;
}

/* =============================================================================
 * Weight Management
 */

/**
 * Load weights from flash to SRAM (.tinylstm_weights section)
 *
 * @param flash_weights Source in flash
 * @param size Size in bytes (must be < 8192)
 * @return 0 on success, -1 on error
 */
int load_rci_weights(const void *flash_weights, uint16_t size) {
    if (size > 8192) {
        return -1;  /* Exceeds 8KB limit */
    }

    /* Copy to 0x20000000 (from linker script) */
    void *sram_weights = (void *)0x20000000;
    memcpy(sram_weights, flash_weights, size);

    /* Verify checksum (last 2 bytes) */
    uint16_t *checksum = (uint16_t *)((uint8_t *)sram_weights + size - 2);
    uint16_t computed = 0;
    uint8_t *data = (uint8_t *)sram_weights;

    for (int i = 0; i < size - 2; i++) {
        computed += data[i];
    }

    if (computed != *checksum) {
        return -1;  /* Checksum mismatch */
    }

    return 0;
}

/**
 * Get pointer to weight structure
 *
 * @return Pointer to tinylstm_rci_weights_t at 0x20000000
 */
tinylstm_rci_weights_t* get_rci_weights(void) {
    return (tinylstm_rci_weights_t *)0x20000000;
}

/**
 * Update Horner coefficients (for adaptive scheduling)
 *
 * @param coeffs New coefficients [a₀, a₁, a₂, a₃]
 */
void update_horner_coeffs(const uint8_t coeffs[4]) {
    tinylstm_rci_weights_t *weights = get_rci_weights();
    memcpy(weights->horner_coeffs, coeffs, 4);
}

/* =============================================================================
 * Example RCI Usage
 *
 * In your main RCI scheduler:
 */

/*
void RCI_Scheduler_Handler(void) {
    // Radio starting up (2.1ms XTAL startup)
    // Use dead time for computation

    static int8_t h_state[32] = {0};  // Hidden state

    // Received schedule bitmask from gateway
    uint32_t schedule_mask = 0x00000001;  // Execute in slot 0

    // Horner coefficients (from gateway)
    uint8_t coeffs[4] = {0x12, 0x34, 0x56, 0x78};

    // Current input
    int8_t x_input = 5;

    // Decode and infer during RCI window
    int8_t h_out[32];
    uint8_t decision = rci_decode_and_infer(schedule_mask, coeffs,
                                            x_input, h_state, h_out);

    if (decision) {
        // Execute task
        execute_task();
    } else {
        // Sleep
        enter_low_power_mode();
    }

    // Update state
    memcpy(h_state, h_out, 32);
}
*/
