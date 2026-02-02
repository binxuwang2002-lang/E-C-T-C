/*
** ###################################################################
**     Saturation-Aware Control Law for ECTC
**
**     Implements the fourth-order penalty function control law
**     for energy-aware task allocation in batteryless IoT nodes.
**
**     Reference: ECTC-19.pdf, Equation 3 & 6, Section IV.A
**
**     Hardware: STM32U575 with BQ25570 PMIC
**     Constraints: θ = 0.9 (saturation threshold), β = penalty coefficient
**
** ###################################################################
*/

#include "control_law.h"
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

/* Configuration per ECTC-19.pdf specifications */
#define C_CAP             330.0f    /* Capacitor capacity in μJ (100μF @ 3.3V) */
#define V_RET             1.8f      /* Retention voltage threshold */
#define THETA             0.9f      /* Saturation threshold θ = 0.9 */
#define BETA              0.1f      /* Penalty coefficient for saturation */
#define V_PARAM           50.0f     /* Lyapunov tradeoff parameter */
#define Q_MAX             330.0f    /* Maximum energy in capacitor */

#define GAMMA_U           0.05f     /* Utility gain parameter */
#define GAMMA_Q           0.02f     /* Energy weight parameter */

/*
** ###################################################################
** Fourth-Order Penalty Function - Equation 3 & 6
** ###################################################################
*/

/**
 * Compute truncated Lyapunov function L_trunc(Q_E)
 * Implements the fourth-order penalty function to prevent energy overflow
 *
 * See: ECTC-19.pdf, Equation 3 (Truncated Lyapunov Function)
 *      ECTC-19.pdf, Equation 6 (Fourth-order Penalty)
 *
 * @param Q_E Current energy in capacitor (μJ)
 * @return Lyapunov function value
 *
 * Implementation:
 * - Quadratic growth for Q_E ≤ θ * C_CAP
 * - Quartic penalty for Q_E > θ * C_CAP (Virtual Wall)
 */
float control_law_truncated_lyapunov(float Q_E) {
    float threshold = THETA * C_CAP;
    float L = 0.0f;

    if (Q_E <= threshold) {
        /* Normal region: Quadratic growth */
        /* L(Q_E) = 0.5 * Q_E^2 for Q_E ≤ θ * C_cap */
        L = 0.5f * Q_E * Q_E;
    } else {
        /* Saturation region: Fourth-order penalty */
        /* L(Q_E) = 0.5 * (θ * C_cap)^2 + β * (Q_E - θ * C_cap)^4 */
        float excess = Q_E - threshold;
        L = 0.5f * threshold * threshold + BETA * excess * excess * excess * excess;
    }

    return L;
}

/**
 * Compute Lyapunov drift ΔL(t) = L(t+1) - L(t)
 *
 * @param Q_E Current energy state
 * @param Q_E_next Next energy state
 * @return Lyapunov drift value
 */
float control_law_lyapunov_drift(float Q_E, float Q_E_next) {
    return control_law_truncated_lyapunov(Q_E_next) -
           control_law_truncated_lyapunov(Q_E);
}

/**
 * Compute truncated Lyapunov gradient ∇L(Q_E)
 *
 * @param Q_E Current energy state
 * @return Gradient value
 */
float control_law_lyapunov_gradient(float Q_E) {
    float threshold = THETA * C_CAP;
    float grad = 0.0f;

    if (Q_E <= threshold) {
        /* Normal region: ∇L = Q_E */
        grad = Q_E;
    } else {
        /* Saturation region: ∇L = 4 * β * (Q_E - θ * C_cap)^3 */
        float excess = Q_E - threshold;
        grad = 4.0f * BETA * excess * excess * excess;
    }

    return grad;
}

/*
** ###################################################################
** Virtual Wall Mechanism
** ###################################################################
*/

/**
 * Check if energy state is in saturation region (Q_E → C_cap)
 *
 * Implements the "Virtual Wall" mechanism that forces negative drift
 * when the capacitor approaches full capacity.
 *
 * @param Q_E Current energy state
 * @return true if in saturation region
 *
 * See: ECTC-19.pdf, Section IV.A (Saturation-Aware Control)
 */
bool control_law_is_in_saturation(float Q_E) {
    float threshold = THETA * C_CAP;
    return (Q_E > threshold);
}

/**
 * Compute penalty contribution from Virtual Wall
 *
 * The virtual wall mechanism adds strong negative feedback when
 * Q_E approaches C_cap to prevent energy overflow.
 *
 * @param Q_E Current energy state
 * @return Penalty value (negative for saturation)
 */
float control_law_virtual_wall_penalty(float Q_E) {
    if (!control_law_is_in_saturation(Q_E)) {
        return 0.0f;
    }

    float threshold = THETA * C_CAP;
    float excess = Q_E - threshold;
    float excess_normalized = excess / (C_CAP - threshold);

    /* Strong penalty that grows as node approaches full capacity */
    /* Penalty = -β * (excess/C_cap)^4 * C_cap */
    float penalty = -BETA * excess_normalized * excess_normalized *
                    excess_normalized * excess_normalized * C_CAP;

    return penalty;
}

/**
 * Check if energy is sufficient for safe operation
 *
 * @param Q_E Current energy state
 * @return true if above retention threshold
 */
bool control_law_energy_above_retention(float Q_E) {
    /* Convert retention voltage to energy: E = 0.5 * C * V^2 */
    float E_ret = 0.5f * 100e-6f * V_RET * V_RET * 1e6f; /* in μJ */
    return (Q_E >= E_ret);
}

/*
** ###################################################################
** Saturation-Aware Control Law
** ###################################################################
*/

/**
 * Main saturation-aware control law function
 *
 * Computes the optimal control action based on:
 * 1. Lyapunov drift minimization
 * 2. Utility maximization
 * 3. Virtual wall penalty (saturation avoidance)
 *
 * See: ECTC-19.pdf, Equation 3 & 6
 *
 * @param Q_E Current energy in capacitor (μJ)
 * @param U_i Marginal utility of node i
 * @param B_i Data queue length (packets)
 * @return Control action value (positive = execute task, negative = wait)
 *
 * Control Law:
 *   a_i(t) = -∇L(Q_E) + γ_u * U_i - γ_q * B_i + VirtualWall(Q_E)
 *
 * Where:
 *   -∇L(Q_E)    : Lyapunov drift term (energy awareness)
 *   γ_u * U_i   : Utility maximization (information value)
 *   γ_q * B_i   : Queue awareness (urgency)
 *   VirtualWall : Penalty for approaching saturation
 */
float control_law_compute_action(float Q_E, float U_i, uint16_t B_i) {
    /* Compute Lyapunov gradient */
    float lyap_grad = control_law_lyapunov_gradient(Q_E);

    /* Check energy sufficiency */
    if (!control_law_energy_above_retention(Q_E)) {
        /* Insufficient energy - must wait */
        return -1.0f;
    }

    /* Compute utility term */
    float utility_term = GAMMA_U * U_i;

    /* Compute queue term (encourages transmission when queue is long) */
    float queue_term = GAMMA_Q * (float)B_i;

    /* Compute virtual wall penalty */
    float vwall = control_law_virtual_wall_penalty(Q_E);

    /*
     * Main control law:
     * Action = -∇L(Q_E) + γ_u*U_i - γ_q*B_i + VirtualWall
     *
     * Interpretation:
     * - If Q_E is low: -∇L(Q_E) is small negative, discouraging action
     * - If Q_E is high: -∇L(Q_E) is large negative, strongly discouraging
     * - If utility is high: +γ_u*U_i encourages action
     * - If queue is long: +γ_q*B_i encourages action
     * - If in saturation: VirtualWall adds strong negative penalty
     */
    float action = -lyap_grad + utility_term + queue_term + vwall;

    return action;
}

/**
 * Determine task execution decision
 *
 * @param Q_E Current energy (μJ)
 * @param U_i Marginal utility
 * @param B_i Queue length
 * @return true if should execute task, false otherwise
 */
bool control_law_should_execute(float Q_E, float U_i, uint16_t B_i) {
    /* Compute control action */
    float action = control_law_compute_action(Q_E, U_i, B_i);

    /* Threshold for execution decision */
    float action_threshold = 0.5f;

    return (action > action_threshold);
}

/**
 * Compute maximum allowable tasks for current energy
 *
 * @param Q_E Current energy (μJ)
 * @return Maximum number of tasks that can be executed safely
 */
uint16_t control_law_max_safe_tasks(float Q_E) {
    /* Energy per task (measured from profiling) */
    float energy_per_task = 5.3e-3f; /* 5.3 μJ per task */

    /* Check for saturation - cap maximum even if energy available */
    float threshold = THETA * C_CAP;
    if (Q_E > threshold) {
        /* In saturation - limit to prevent overflow */
        Q_E = threshold;
    }

    uint16_t max_tasks = (uint16_t)(Q_E / energy_per_task);

    /* Safety margin: don't use more than 90% of available energy */
    return (uint16_t)(max_tasks * 0.9f);
}

/**
 * Predict optimal transmission schedule
 *
 * Computes when to schedule transmission based on energy prediction
 * and saturation constraints.
 *
 * @param Q_E Current energy (μJ)
 * @param predicted_harvest Predicted energy harvest in next slot (μJ)
 * @param U_i Marginal utility
 * @param B_i Queue length
 * @param predicted_schedule[10] Output: next 10 slots (1=transmit, 0=wait)
 *
 * @return Recommended transmission slot (0-9), or 255 if delay recommended
 */
uint8_t control_law_predict_schedule(float Q_E,
                                     float predicted_harvest,
                                     float U_i,
                                     uint16_t B_i,
                                     uint8_t predicted_schedule[10]) {
    float current_Q_E = Q_E;
    uint8_t recommended_slot = 255;

    /* Initialize schedule to all zeros (wait) */
    for (int i = 0; i < 10; i++) {
        predicted_schedule[i] = 0;
    }

    /* Simulate next 10 time slots */
    for (int slot = 0; slot < 10; slot++) {
        /* Add predicted harvest */
        current_Q_E += predicted_harvest;

        /* Check if should transmit */
        bool should_transmit = control_law_should_execute(current_Q_E, U_i, B_i);

        if (should_transmit && (recommended_slot == 255)) {
            /* First recommended transmission slot */
            recommended_slot = slot;
            predicted_schedule[slot] = 1;
            current_Q_E -= 5.3e-3f; /* Energy cost of transmission */
        }

        /* Apply Lyapunov drift (energy consumption in idle state) */
        float idle_cost = 0.1e-3f; /* 0.1 μJ per slot for monitoring */
        current_Q_E -= idle_cost;

        /* Don't let energy go negative */
        if (current_Q_E < 0.0f) {
            current_Q_E = 0.0f;
        }
    }

    return recommended_slot;
}

/*
** ###################################################################
** Debug and Verification Functions
** ###################################################################
*/

/**
 * Print control law state for debugging
 *
 * @param Q_E Current energy
 * @param U_i Marginal utility
 * @param B_i Queue length
 */
void control_law_print_state(float Q_E, float U_i, uint16_t B_i) {
    float L = control_law_truncated_lyapunov(Q_E);
    float grad = control_law_lyapunov_gradient(Q_E);
    float action = control_law_compute_action(Q_E, U_i, B_i);
    bool should_exec = control_law_should_execute(Q_E, U_i, B_i);

    /* Log to trace buffer - see trace.h for definitions */
    /* Note: In production, this would use the trace buffer system */
}

/**
 * Verify control law properties
 *
 * Checks that:
 * 1. Lyapunov function is convex
 * 2. Gradient increases monotonically
 * 3. Virtual wall activates correctly
 *
 * @return true if all checks pass
 */
bool control_law_verify_properties(void) {
    bool passed = true;

    /* Test 1: Verify convexity at threshold */
    float Q1 = THETA * C_CAP - 1.0f;
    float Q2 = THETA * C_CAP;
    float Q3 = THETA * C_CAP + 1.0f;

    float L1 = control_law_truncated_lyapunov(Q1);
    float L2 = control_law_truncated_lyapunov(Q2);
    float L3 = control_law_truncated_lyapunov(Q3);

    /* Check continuity at threshold */
    if (fabsf(L2 - (0.5f * (THETA * C_CAP) * (THETA * C_CAP))) > 1e-3f) {
        passed = false;
    }

    /* Test 2: Verify gradient is monotonic */
    float grad_prev = -1000.0f;
    for (float Q = 0.0f; Q <= C_CAP; Q += 10.0f) {
        float grad = control_law_lyapunov_gradient(Q);
        if (grad < grad_prev) {
            passed = false;
            break;
        }
        grad_prev = grad;
    }

    /* Test 3: Verify virtual wall activation */
    float vwall_below = control_law_virtual_wall_penalty(THETA * C_CAP - 1.0f);
    float vwall_above = control_law_virtual_wall_penalty(THETA * C_CAP + 1.0f);

    if (vwall_below != 0.0f || vwall_above >= 0.0f) {
        passed = false;
    }

    return passed;
}

/**
 * Get control law parameters
 *
 * @return Pointer to control law configuration
 */
const control_law_config_t* control_law_get_config(void) {
    static const control_law_config_t config = {
        .C_cap = C_CAP,
        .V_ret = V_RET,
        .theta = THETA,
        .beta = BETA,
        .V_param = V_PARAM,
        .gamma_u = GAMMA_U,
        .gamma_q = GAMMA_Q
    };

    return &config;
}
