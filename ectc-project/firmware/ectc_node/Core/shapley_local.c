/*
 * Local Shapley Value Computation
 * ================================
 *
 * Computes local marginal contribution for ECTC game.
 * Uses O(1) approximation suitable for resource-constrained MCU.
 *
 * Based on paper: "Stratified Sampling for Efficient Shapley Approximation"
 */

#include "shapley_local.h"
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

// Game parameters
#define N_MAX_NEI 8       // Maximum neighbors to consider
#define EPSILON_APPROX 0.1  // Approximation parameter
#define DELTA_CONF 0.05    // Confidence parameter
#define SAMPLE_LIMIT 16    // Maximum local samples

// Lyapunov parameters (from gateway config)
static float g_Lyapunov_V = 50.0f;
static float g_Lyapunov_beta = 0.1f;
static float g_cap_energy_max = 330.0f;  // 100μF at 3.3V

// State history for temporal features
typedef struct {
    float Q_E_history[4];
    float B_i_history[4];
    uint8_t history_idx;
    float coalition_trust[N_MAX_NEI];
    float neighbor_marginals[N_MAX_NEI];
} local_state_t;

static local_state_t g_local_state;

/**
 * Initialize local Shapley calculator
 */
void shapley_local_init(void) {
    // Clear history
    g_local_state.history_idx = 0;
    for (int i = 0; i < 4; i++) {
        g_local_state.Q_E_history[i] = 0.0f;
        g_local_state.B_i_history[i] = 0.0f;
    }

    for (int i = 0; i < N_MAX_NEI; i++) {
        g_local_state.coalition_trust[i] = 1.0f;
        g_local_state.neighbor_marginals[i] = 0.0f;
    }

    trace_event(TRACE_EVENT_SHAPLEY_INIT, 0, 0);
}

/**
 * Compute local marginal contribution
 *
 * Args:
 *   Q_E: Current energy (μJ)
 *   B_i: Data queue length
 *   predicted_harvest: Predicted energy collection (μJ)
 *   output_marginal: Pointer to store computed marginal utility
 */
void shapley_local_compute(float Q_E, uint16_t B_i, float predicted_harvest,
                          float* output_marginal) {
    // Step 1: Update local state history
    update_state_history(Q_E, B_i);

    // Step 2: Estimate coalition size based on local neighbors
    uint8_t estimated_coalition_size = estimate_coalition_size();

    // Step 3: Compute energy-based marginal
    float energy_marginal = compute_energy_marginal(Q_E, predicted_harvest);

    // Step 4: Compute information utility
    float info_marginal = compute_info_marginal(B_i);

    // Step 5: Compute trust penalty
    float trust_penalty = compute_trust_penalty();

    // Step 6: Combine components
    float marginal = 0.6f * energy_marginal + 0.3f * info_marginal - 0.1f * trust_penalty;

    // Step 7: Apply sigmoid to bound utility
    marginal = sigmoid(marginal);

    // Step 8: Update neighbor marginals
    update_neighbor_marginals(estimated_coalition_size, marginal);

    *output_marginal = marginal;

    trace_event(TRACE_EVENT_SHAPLEY_COMPUTE,
                (uint32_t)(marginal * 1000),
                estimated_coalition_size);
}

/**
 * Update state history for temporal features
 */
void update_state_history(float Q_E, uint16_t B_i) {
    g_local_state.Q_E_history[g_local_state.history_idx] = Q_E;
    g_local_state.B_i_history[g_local_state.history_idx] = (float)B_i;

    g_local_state.history_idx = (g_local_state.history_idx + 1) % 4;
}

/**
 * Estimate coalition size based on radio range and neighbor detection
 */
uint8_t estimate_coalition_size(void) {
    // In practice: count active neighbors within radio range
    // For now: estimate based on signal strength distribution
    uint8_t neighbor_count = 0;

    // Sample multiple times to account for packet loss
    for (int sample = 0; sample < 3; sample++) {
        uint8_t detected = scan_neighbors();
        neighbor_count += detected;
    }
    neighbor_count /= 3;

    // Clamp to reasonable range
    if (neighbor_count < 2) neighbor_count = 2;
    if (neighbor_count > N_MAX_NEI) neighbor_count = N_MAX_NEI;

    return neighbor_count;
}

/**
 * Compute energy-based marginal contribution
 */
float compute_energy_marginal(float Q_E, float predicted_harvest) {
    // Compare predicted trajectory with and without this node
    float future_with_node = Q_E + predicted_harvest;
    float future_without_node = Q_E + (predicted_harvest * 0.9f);  // Assume slight negative impact

    // Compute Lyapunov drift
    float drift = truncated_lyapunov_drift(future_with_node, future_without_node);

    // Normalize by coalition size for per-node contribution
    float estimated_coalition = estimate_coalition_size();
    float normalized_drift = drift / estimated_coalition;

    return normalized_drift;
}

/**
 * Compute truncated Lyapunov drift
 */
float truncated_lyapunov_drift(float E_current, float E_next) {
    float L_current = 0.0f;
    float L_next = 0.0f;

    // Current energy level
    if (E_current <= 0.9f * g_cap_energy_max) {
        L_current = 0.5f * E_current * E_current;
    } else {
        float excess = E_current - 0.9f * g_cap_energy_max;
        L_current = 0.5f * (0.9f * g_cap_energy_max) * (0.9f * g_cap_energy_max) +
                   g_Lyapunov_beta * excess * excess * excess * excess;
    }

    // Next energy level
    if (E_next <= 0.9f * g_cap_energy_max) {
        L_next = 0.5f * E_next * E_next;
    } else {
        float excess = E_next - 0.9f * g_cap_energy_max;
        L_next = 0.5f * (0.9f * g_cap_energy_max) * (0.9f * g_cap_energy_max) +
                g_Lyapunov_beta * excess * excess * excess * excess;
    }

    // Drift is negative of Lyapunov increase (we want to minimize energy waste)
    return -(L_next - L_current);
}

/**
 * Compute information utility marginal
 */
float compute_info_marginal(uint16_t B_i) {
    // Information value based on queue backlog and time sensitivity
    float queue_ratio = (float)B_i / 255.0f;  // Normalize to [0,1]

    // Compute temporal trend
    float trend = compute_temporal_trend();

    // Information utility increases with queue backlog and positive trend
    float info_util = queue_ratio * (1.0f + trend);

    return info_util;
}

/**
 * Compute temporal trend in energy/data
 */
float compute_temporal_trend(void) {
    if (g_local_state.history_idx < 3) return 0.0f;

    float energy_trend = 0.0f;
    float queue_trend = 0.0f;

    // Simple linear trend estimation
    int n = 3;
    float sum_x = 0.0f, sum_y_e = 0.0f, sum_y_q = 0.0f;
    float sum_xy_e = 0.0f, sum_xy_q = 0.0f;

    for (int i = 0; i < n; i++) {
        int idx = (g_local_state.history_idx - 1 - i + 4) % 4;
        float x = (float)i;
        sum_x += x;
        sum_y_e += g_local_state.Q_E_history[idx];
        sum_y_q += g_local_state.B_i_history[idx];
        sum_xy_e += x * g_local_state.Q_E_history[idx];
        sum_xy_q += x * g_local_state.B_i_history[idx];
    }

    float x_mean = sum_x / n;
    float energy_mean = sum_y_e / n;
    float queue_mean = sum_y_q / n;

    // Calculate slopes
    float denom = 0.0f;
    for (int i = 0; i < n; i++) {
        float x = (float)i;
        denom += (x - x_mean) * (x - x_mean);
    }

    if (denom > 0.001f) {
        energy_trend = (sum_xy_e - n * x_mean * energy_mean) / denom;
        queue_trend = (sum_xy_q - n * x_mean * queue_mean) / denom;
    }

    // Combine trends
    return 0.6f * energy_trend + 0.4f * queue_trend;
}

/**
 * Compute trust penalty for malicious behavior
 */
float compute_trust_penalty(void) {
    // Check if recent behavior is suspicious
    float avg_trust = 0.0f;
    for (int i = 0; i < N_MAX_NEI; i++) {
        avg_trust += g_local_state.coalition_trust[i];
    }
    avg_trust /= N_MAX_NEI;

    // Penalize if trust score is low
    float penalty = (1.0f - avg_trust) * 0.5f;

    return penalty;
}

/**
 * Update neighbor marginal values
 */
void update_neighbor_marginals(uint8_t coalition_size, float self_marginal) {
    // Distribute marginal to neighbors (simplified model)
    float share = self_marginal / coalition_size;

    for (int i = 0; i < coalition_size - 1; i++) {
        g_local_state.neighbor_marginals[i] = 0.8f * g_local_state.neighbor_marginals[i] +
                                              0.2f * share;
    }
}

/**
 * Scan for active neighbors
 */
uint8_t scan_neighbors(void) {
    // In practice: send beacon and count ACKs
    // For now: estimate based on recent packet reception
    // This would integrate with the radio module
    return 5;  // Placeholder
}

/**
 * Sigmoid activation for bounded utility
 */
float sigmoid(float x) {
    // Clamp input to prevent overflow
    if (x > 10.0f) x = 10.0f;
    if (x < -10.0f) x = -10.0f;

    // Approximate sigmoid: 1 / (1 + exp(-x))
    float exp_x = fast_exp(x);
    return exp_x / (1.0f + exp_x);
}

/**
 * Fast exponential approximation
 */
float fast_exp(float x) {
    // Polynomial approximation: exp(x) ≈ 1 + x + x²/2 + x³/6
    // Good for |x| < 1

    if (x < 0) {
        return 1.0f / fast_exp(-x);
    }

    if (x < 1.0f) {
        return 1.0f + x + 0.5f * x * x + 0.1667f * x * x * x;
    }

    // For larger x, use iterative doubling
    float half_exp = fast_exp(x * 0.5f);
    return half_exp * half_exp;
}

/**
 * Update trust scores based on behavior
 */
void update_trust_scores(const uint8_t* neighbor_ids, float* reported_utilities,
                        uint8_t count) {
    // Compare reported utilities with actual marginals
    for (int i = 0; i < count && i < N_MAX_NEI; i++) {
        float diff = g_local_state.neighbor_marginals[i] - reported_utilities[i];
        float update = 0.1f * diff;

        // Adjust trust score
        g_local_state.coalition_trust[i] -= update;

        // Clamp trust to [0, 1]
        if (g_local_state.coalition_trust[i] < 0.0f) {
            g_local_state.coalition_trust[i] = 0.0f;
        } else if (g_local_state.coalition_trust[i] > 1.0f) {
            g_local_state.coalition_trust[i] = 1.0f;
        }
    }

    trace_event(TRACE_EVENT_TRUST_UPDATE, count, 0);
}

/**
 * Get approximation error bounds
 */
float get_shapley_error_bound(void) {
    // Theoretical error bound: ε * sqrt(ln(1/δ) / (2 * m))
    // where m is the number of local samples
    float m = (float)SAMPLE_LIMIT;
    return EPSILON_APPROX * sqrtf(logf(1.0f / DELTA_CONF) / (2.0f * m));
}

/**
 * Get computational complexity estimate
 */
uint32_t get_shapley_complexity_ops(void) {
    // O(coalition_size) operations
    return estimate_coalition_size() * 32;  // ~32 ops per node
}
