/*
 * Stratification Freezing Resilience Module Header
 * =================================================
 *
 * Handles gateway failure by locking game-theoretic state locally.
 * Implements graceful degradation with renormalization penalties.
 *
 * Reference: ECTC Paper - Section IV.G (Dynamic Resilience)
 *
 * Hardware: CC2650 / STM32U575 + BQ25570
 */

#ifndef RESILIENCE_H
#define RESILIENCE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Configuration Constants
 * ============================================================================ */

/** Gateway heartbeat timeout in milliseconds */
#define GATEWAY_TIMEOUT_MS              500

/** Local renormalization penalty rate per minute (10%) */
#define RENORM_PENALTY_RATE_PER_MIN     0.10f

/** Minimum transmit probability after penalties */
#define MIN_TRANSMIT_PROBABILITY        0.05f

/* ============================================================================
 * Types
 * ============================================================================ */

/**
 * Resilience State Machine States
 */
typedef enum {
    STATE_NORMAL = 0,       /**< Normal operation - gateway connected */
    STATE_FREEZE,           /**< Freeze state - gateway timeout detected */
    STATE_RECOVERY,         /**< Recovery - heartbeat restored, stabilizing */
    STATE_EMERGENCY         /**< Emergency - extended outage, minimal operation */
} resilience_state_t;

/**
 * Freeze reason codes
 */
typedef enum {
    FREEZE_REASON_NONE = 0,
    FREEZE_REASON_TIMEOUT,      /**< Heartbeat timeout */
    FREEZE_REASON_CORRUPTION,   /**< Shapley cache corruption detected */
    FREEZE_REASON_ENERGY_LOW,   /**< Energy too low for dynamic computation */
    FREEZE_REASON_MANUAL        /**< Manually triggered freeze */
} freeze_reason_t;

/**
 * Resilience Statistics Structure
 */
typedef struct {
    resilience_state_t state;       /**< Current state */
    freeze_reason_t freeze_reason;  /**< Reason for freeze (if frozen) */
    uint32_t total_freeze_events;   /**< Total number of freeze events */
    uint32_t total_freeze_duration_ms; /**< Total time spent frozen */
    uint32_t successful_recoveries; /**< Number of successful recoveries */
    float current_penalty;          /**< Current penalty level (0-1) */
    uint32_t penalty_count;         /**< Number of penalties applied */
    uint32_t time_since_heartbeat;  /**< Time since last gateway message */
} resilience_stats_t;

/* ============================================================================
 * Core API Functions
 * ============================================================================ */

/**
 * Initialize resilience subsystem
 *
 * Must be called during system startup before main control loop.
 */
void Resilience_Init(void);

/**
 * Resilience periodic tick
 *
 * Must be called regularly from the main control loop (every 100ms).
 * Checks for gateway timeout and applies renormalization penalties.
 */
void Resilience_Tick(void);

/**
 * Called when gateway heartbeat/message is received
 *
 * Updates the last message timestamp and handles state transitions.
 * Call this from the radio receive handler when a gateway message arrives.
 */
void Resilience_On_Gateway_Heartbeat(void);

/**
 * Called when a gateway Shapley update is received
 *
 * @param new_shapley Updated Shapley value from gateway
 */
void Resilience_On_Shapley_Update(float new_shapley);

/* ============================================================================
 * Main Interface: Backoff Probability
 * ============================================================================ */

/**
 * Get modified transmit probability based on resilience state
 *
 * This is the main interface function that modifies the original
 * Shapley-based probability based on the current freeze state.
 *
 * Logic:
 * - STATE_NORMAL: Return original Shapley value unchanged
 * - STATE_FREEZE: Use frozen Shapley × (1 - cumulative_penalty)
 *   - 10% penalty applied every minute
 *   - Example: After 5 min, Shapley 0.8 → 0.8 × 0.5 = 0.4
 * - STATE_RECOVERY: Gradually restore to original
 * - STATE_EMERGENCY: Use minimum probability (0.05)
 *
 * @param original_shapley Original Shapley value (0.0 to 1.0)
 * @return Modified probability for channel contention (0.05 to 1.0)
 *
 * Example Usage:
 * @code
 *   float shapley = get_my_shapley_value();
 *   float tx_prob = Resilience_Get_Backoff_Prob(shapley);
 *   if (random() < tx_prob) {
 *       transmit_data();
 *   }
 * @endcode
 */
float Resilience_Get_Backoff_Prob(float original_shapley);

/**
 * Check if transmission should proceed
 *
 * Combines energy check with resilience state for decision making.
 *
 * @param current_energy Current capacitor energy (μJ)
 * @param energy_threshold Minimum energy for transmission (μJ)
 * @param shapley_value Node's Shapley value
 * @return true if transmission should proceed
 */
bool Resilience_Should_Transmit(float current_energy,
                                 float energy_threshold,
                                 float shapley_value);

/* ============================================================================
 * Status Query Functions
 * ============================================================================ */

/**
 * Check if system is in freeze state
 *
 * @return true if currently frozen (gateway timeout)
 */
bool Resilience_Is_Frozen(void);

/**
 * Get current resilience state
 *
 * @return Current state (NORMAL, FREEZE, RECOVERY, EMERGENCY)
 */
resilience_state_t Resilience_Get_State(void);

/**
 * Get time since last gateway message
 *
 * @return Time in milliseconds since last heartbeat
 */
uint32_t Resilience_Get_Time_Since_Heartbeat(void);

/**
 * Get current penalty level
 *
 * @return Current cumulative penalty (0.0 to 0.95)
 */
float Resilience_Get_Penalty_Level(void);

/**
 * Get frozen Shapley value
 *
 * @return Last known Shapley value before freeze
 */
float Resilience_Get_Frozen_Shapley(void);

/**
 * Get resilience statistics
 *
 * @param stats Pointer to statistics structure to fill
 */
void Resilience_Get_Statistics(resilience_stats_t* stats);

/* ============================================================================
 * Control Functions
 * ============================================================================ */

/**
 * Manually trigger freeze state
 *
 * For testing or emergency situations.
 */
void Resilience_Force_Freeze(void);

/**
 * Manually trigger recovery
 *
 * For testing - simulates heartbeat reception.
 */
void Resilience_Force_Recovery(void);

/**
 * Reset all resilience state
 *
 * Use with caution - clears all history and statistics.
 */
void Resilience_Reset(void);

/**
 * Seed random number generator
 *
 * @param seed Random seed value
 */
void Resilience_Seed_Random(uint32_t seed);

#ifdef __cplusplus
}
#endif

#endif /* RESILIENCE_H */
