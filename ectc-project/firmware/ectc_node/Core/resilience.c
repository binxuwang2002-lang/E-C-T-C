/*
 * Stratification Freezing Resilience Module
 * ==========================================
 *
 * Handles gateway failure (No Heartbeat) by locking the current
 * game-theoretic state locally. Implements graceful degradation
 * with local renormalization to prevent congestion collapse.
 *
 * Reference: ECTC Paper - Section IV.G (Dynamic Resilience)
 *
 * Key Features:
 * - Gateway heartbeat timeout detection
 * - State freezing (lock last known Shapley values)
 * - Local renormalization penalty (10%/min reduction)
 * - Congestion collapse prevention
 * - Automatic recovery on heartbeat restoration
 *
 * Hardware: CC2650 / STM32U575 + BQ25570
 *
 * Copyright (c) 2024 ECTC Research Team
 */

#include "resilience.h"
#include "trace.h"
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

/* ============================================================================
 * Configuration Constants
 * ============================================================================ */

/** Gateway heartbeat timeout in milliseconds (Table I: 5 cycles × 100ms) */
#define GATEWAY_TIMEOUT_MS              500

/** Heartbeat check interval in milliseconds */
#define HEARTBEAT_CHECK_INTERVAL_MS     100

/** Local renormalization penalty rate per minute (10%) */
#define RENORM_PENALTY_RATE_PER_MIN     0.10f

/** Minimum transmit probability after penalties */
#define MIN_TRANSMIT_PROBABILITY        0.05f

/** Maximum time in freeze state before emergency action (30 minutes) */
#define MAX_FREEZE_DURATION_MS          (30 * 60 * 1000)

/** Recovery stability period (2 seconds of stable heartbeats) */
#define RECOVERY_STABILITY_MS           2000

/** One minute in milliseconds (for penalty calculation) */
#define ONE_MINUTE_MS                   60000

/* ============================================================================
 * State Machine States
 * ============================================================================ */

/**
 * Resilience State Machine
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

/* ============================================================================
 * Resilience Context Structure
 * ============================================================================ */

/**
 * Main resilience state structure
 */
typedef struct {
    /* Current state */
    resilience_state_t state;
    freeze_reason_t freeze_reason;
    
    /* Timing */
    uint32_t last_gateway_msg_time;     /**< Last heartbeat timestamp (ms) */
    uint32_t freeze_start_time;         /**< When freeze state was entered */
    uint32_t recovery_start_time;       /**< When recovery started */
    uint32_t current_time;              /**< Current system time (cached) */
    
    /* Frozen Shapley values */
    float frozen_shapley;               /**< Last known Shapley value */
    float original_shapley;             /**< Original value before freeze */
    bool shapley_frozen;                /**< Flag: Shapley value is frozen */
    
    /* Renormalization state */
    uint32_t last_penalty_time;         /**< Last penalty application time */
    float cumulative_penalty;           /**< Accumulated penalty (0.0 to 1.0) */
    uint32_t penalty_count;             /**< Number of penalties applied */
    
    /* Statistics */
    uint32_t total_freeze_events;       /**< Total freeze events */
    uint32_t total_freeze_duration_ms;  /**< Total time in freeze state */
    uint32_t total_packets_during_freeze;
    uint32_t successful_recoveries;
    
    /* Recovery */
    uint32_t consecutive_heartbeats;    /**< Consecutive HBs since recovery start */
    
} resilience_context_t;

/* Global resilience context */
static resilience_context_t g_resilience = {0};

/* Forward declarations */
static void enter_freeze_state(freeze_reason_t reason);
static void enter_recovery_state(void);
static void enter_normal_state(void);
static void apply_renormalization_penalty(void);
static uint32_t get_current_time_ms(void);

/* ============================================================================
 * External Dependencies (Platform-specific)
 * ============================================================================ */

/**
 * Get current system time in milliseconds
 * 
 * Platform-specific implementation:
 * - CC2650: Use AON RTC
 * - STM32: Use HAL_GetTick()
 */
static uint32_t get_current_time_ms(void) {
    /* Placeholder - integrate with platform HAL */
    extern uint32_t HAL_GetTick(void);
    return HAL_GetTick();
}

/* ============================================================================
 * Initialization
 * ============================================================================ */

/**
 * Initialize resilience subsystem
 * 
 * Must be called during system startup before main control loop.
 */
void Resilience_Init(void) {
    /* Initialize state */
    g_resilience.state = STATE_NORMAL;
    g_resilience.freeze_reason = FREEZE_REASON_NONE;
    
    /* Initialize timing */
    g_resilience.last_gateway_msg_time = get_current_time_ms();
    g_resilience.freeze_start_time = 0;
    g_resilience.recovery_start_time = 0;
    
    /* Initialize Shapley state */
    g_resilience.frozen_shapley = 0.5f;  /* Default neutral value */
    g_resilience.original_shapley = 0.5f;
    g_resilience.shapley_frozen = false;
    
    /* Initialize penalty state */
    g_resilience.last_penalty_time = 0;
    g_resilience.cumulative_penalty = 0.0f;
    g_resilience.penalty_count = 0;
    
    /* Clear statistics */
    g_resilience.total_freeze_events = 0;
    g_resilience.total_freeze_duration_ms = 0;
    g_resilience.total_packets_during_freeze = 0;
    g_resilience.successful_recoveries = 0;
    g_resilience.consecutive_heartbeats = 0;
    
    trace_event(TRACE_EVENT_RESILIENCE_INIT, 0, 0);
}

/* ============================================================================
 * Heartbeat Management
 * ============================================================================ */

/**
 * Called when gateway heartbeat/message is received
 * 
 * Updates the last message timestamp and handles state transitions.
 * Call this from the radio receive handler when a gateway message arrives.
 */
void Resilience_On_Gateway_Heartbeat(void) {
    g_resilience.current_time = get_current_time_ms();
    g_resilience.last_gateway_msg_time = g_resilience.current_time;
    
    switch (g_resilience.state) {
        case STATE_NORMAL:
            /* Already normal - just update timestamp */
            break;
            
        case STATE_FREEZE:
            /* Heartbeat restored - enter recovery */
            enter_recovery_state();
            break;
            
        case STATE_RECOVERY:
            /* Track consecutive heartbeats for stability */
            g_resilience.consecutive_heartbeats++;
            
            /* Check if recovery period complete */
            uint32_t recovery_duration = g_resilience.current_time - 
                                          g_resilience.recovery_start_time;
            if (recovery_duration >= RECOVERY_STABILITY_MS && 
                g_resilience.consecutive_heartbeats >= 2) {
                enter_normal_state();
            }
            break;
            
        case STATE_EMERGENCY:
            /* Emergency recovery - need more careful transition */
            enter_recovery_state();
            break;
    }
    
    trace_event(TRACE_EVENT_HEARTBEAT_RX, g_resilience.state, 0);
}

/**
 * Called when a gateway Shapley update is received
 * 
 * @param new_shapley Updated Shapley value from gateway
 */
void Resilience_On_Shapley_Update(float new_shapley) {
    /* Only update if in normal state */
    if (g_resilience.state == STATE_NORMAL) {
        g_resilience.frozen_shapley = new_shapley;
        g_resilience.original_shapley = new_shapley;
    }
    
    /* Also count as heartbeat */
    Resilience_On_Gateway_Heartbeat();
}

/* ============================================================================
 * Main Tick Function (Call from Main Loop)
 * ============================================================================ */

/**
 * Resilience periodic tick
 * 
 * Must be called regularly from the main control loop.
 * Checks for timeout and applies renormalization penalties.
 */
void Resilience_Tick(void) {
    g_resilience.current_time = get_current_time_ms();
    
    /* Calculate time since last gateway message */
    uint32_t elapsed = g_resilience.current_time - g_resilience.last_gateway_msg_time;
    
    switch (g_resilience.state) {
        case STATE_NORMAL:
            /* Check for timeout */
            if (elapsed > GATEWAY_TIMEOUT_MS) {
                enter_freeze_state(FREEZE_REASON_TIMEOUT);
            }
            break;
            
        case STATE_FREEZE:
            /* Apply renormalization penalty every minute */
            apply_renormalization_penalty();
            
            /* Check for emergency state transition */
            uint32_t freeze_duration = g_resilience.current_time - 
                                       g_resilience.freeze_start_time;
            if (freeze_duration > MAX_FREEZE_DURATION_MS) {
                g_resilience.state = STATE_EMERGENCY;
                trace_event(TRACE_EVENT_EMERGENCY_STATE, freeze_duration, 0);
            }
            
            /* Track statistics */
            g_resilience.total_freeze_duration_ms++;
            break;
            
        case STATE_RECOVERY:
            /* Check if recovery timed out (gateway went away again) */
            if (elapsed > GATEWAY_TIMEOUT_MS) {
                /* Back to freeze state */
                enter_freeze_state(FREEZE_REASON_TIMEOUT);
            }
            break;
            
        case STATE_EMERGENCY:
            /* Minimal operation - apply maximum penalty */
            g_resilience.cumulative_penalty = 0.9f;  /* 90% reduction */
            
            /* Check for heartbeat (handled in On_Gateway_Heartbeat) */
            break;
    }
}

/* ============================================================================
 * State Transitions
 * ============================================================================ */

/**
 * Enter freeze state
 */
static void enter_freeze_state(freeze_reason_t reason) {
    if (g_resilience.state == STATE_FREEZE) {
        return;  /* Already in freeze */
    }
    
    g_resilience.state = STATE_FREEZE;
    g_resilience.freeze_reason = reason;
    g_resilience.freeze_start_time = g_resilience.current_time;
    g_resilience.last_penalty_time = g_resilience.current_time;
    
    /* Freeze current Shapley value */
    g_resilience.shapley_frozen = true;
    /* Keep frozen_shapley at last known value */
    
    /* Reset penalty accumulator */
    g_resilience.cumulative_penalty = 0.0f;
    g_resilience.penalty_count = 0;
    
    /* Update statistics */
    g_resilience.total_freeze_events++;
    
    trace_event(TRACE_EVENT_FREEZE_ENTER, reason, 
                (uint32_t)(g_resilience.frozen_shapley * 1000));
}

/**
 * Enter recovery state
 */
static void enter_recovery_state(void) {
    g_resilience.state = STATE_RECOVERY;
    g_resilience.recovery_start_time = g_resilience.current_time;
    g_resilience.consecutive_heartbeats = 1;  /* Count this heartbeat */
    
    /* Update statistics */
    uint32_t freeze_duration = g_resilience.current_time - 
                               g_resilience.freeze_start_time;
    g_resilience.total_freeze_duration_ms += freeze_duration;
    
    trace_event(TRACE_EVENT_RECOVERY_START, freeze_duration, 
                g_resilience.penalty_count);
}

/**
 * Enter normal state (full recovery)
 */
static void enter_normal_state(void) {
    g_resilience.state = STATE_NORMAL;
    g_resilience.freeze_reason = FREEZE_REASON_NONE;
    
    /* Unfreeze Shapley value */
    g_resilience.shapley_frozen = false;
    
    /* Reset penalties */
    g_resilience.cumulative_penalty = 0.0f;
    g_resilience.penalty_count = 0;
    
    /* Update statistics */
    g_resilience.successful_recoveries++;
    
    trace_event(TRACE_EVENT_RECOVERY_COMPLETE, 
                g_resilience.successful_recoveries, 0);
}

/* ============================================================================
 * Renormalization Penalty
 * ============================================================================ */

/**
 * Apply local renormalization penalty
 * 
 * Reduces transmit probability by 10% every minute to prevent
 * congestion collapse when gateway is unavailable.
 */
static void apply_renormalization_penalty(void) {
    uint32_t time_since_last_penalty = g_resilience.current_time - 
                                        g_resilience.last_penalty_time;
    
    /* Check if one minute has passed */
    if (time_since_last_penalty >= ONE_MINUTE_MS) {
        /* Apply 10% penalty */
        g_resilience.cumulative_penalty += RENORM_PENALTY_RATE_PER_MIN;
        
        /* Clamp to maximum (leave MIN_TRANSMIT_PROBABILITY remaining) */
        float max_penalty = 1.0f - MIN_TRANSMIT_PROBABILITY;
        if (g_resilience.cumulative_penalty > max_penalty) {
            g_resilience.cumulative_penalty = max_penalty;
        }
        
        /* Update timing */
        g_resilience.last_penalty_time = g_resilience.current_time;
        g_resilience.penalty_count++;
        
        trace_event(TRACE_EVENT_PENALTY_APPLIED, 
                    g_resilience.penalty_count,
                    (uint32_t)(g_resilience.cumulative_penalty * 100));
    }
}

/* ============================================================================
 * Main API: Get Backoff Probability
 * ============================================================================ */

/**
 * Get modified transmit probability based on resilience state
 * 
 * This is the main interface function that modifies the original
 * Shapley-based probability based on the current freeze state.
 * 
 * @param original_shapley Original Shapley value (0.0 to 1.0)
 * @return Modified probability for channel contention
 * 
 * Logic:
 * - STATE_NORMAL: Return original Shapley value unchanged
 * - STATE_FREEZE: Use frozen Shapley × (1 - cumulative_penalty)
 * - STATE_RECOVERY: Gradually restore to original
 * - STATE_EMERGENCY: Use minimum probability
 * 
 * Example:
 *   Original Shapley: 0.8
 *   After 2 minutes in freeze: 0.8 × (1 - 0.20) = 0.64
 *   After 5 minutes in freeze: 0.8 × (1 - 0.50) = 0.40
 *   After 10 minutes in freeze: 0.8 × (1 - 0.95) = 0.05 (clamped)
 */
float Resilience_Get_Backoff_Prob(float original_shapley) {
    float probability = original_shapley;
    
    switch (g_resilience.state) {
        case STATE_NORMAL:
            /* Normal operation - use original Shapley value */
            probability = original_shapley;
            break;
            
        case STATE_FREEZE:
            /*
             * Freeze state:
             * 1. Use last known (frozen) Shapley value
             * 2. Apply cumulative renormalization penalty
             * 
             * Formula: P = frozen_shapley × (1 - cumulative_penalty)
             */
            probability = g_resilience.frozen_shapley * 
                         (1.0f - g_resilience.cumulative_penalty);
            
            /* Ensure minimum probability */
            if (probability < MIN_TRANSMIT_PROBABILITY) {
                probability = MIN_TRANSMIT_PROBABILITY;
            }
            break;
            
        case STATE_RECOVERY:
            /*
             * Recovery state:
             * Gradually interpolate between frozen (penalized) and original
             * 
             * This prevents sudden jumps that could cause congestion
             */
            {
                /* Calculate recovery progress (0.0 to 1.0) */
                uint32_t recovery_duration = g_resilience.current_time - 
                                              g_resilience.recovery_start_time;
                float progress = (float)recovery_duration / (float)RECOVERY_STABILITY_MS;
                if (progress > 1.0f) progress = 1.0f;
                
                /* Interpolate: frozen_penalized → original */
                float frozen_prob = g_resilience.frozen_shapley * 
                                   (1.0f - g_resilience.cumulative_penalty);
                probability = frozen_prob + progress * (original_shapley - frozen_prob);
            }
            break;
            
        case STATE_EMERGENCY:
            /* Emergency state - use absolute minimum */
            probability = MIN_TRANSMIT_PROBABILITY;
            break;
    }
    
    /* Final bounds check */
    if (probability < 0.0f) probability = 0.0f;
    if (probability > 1.0f) probability = 1.0f;
    
    return probability;
}

/* ============================================================================
 * Additional API Functions
 * ============================================================================ */

/**
 * Check if system is in freeze state
 * 
 * @return true if currently frozen (gateway timeout)
 */
bool Resilience_Is_Frozen(void) {
    return (g_resilience.state == STATE_FREEZE || 
            g_resilience.state == STATE_EMERGENCY);
}

/**
 * Get current resilience state
 * 
 * @return Current state (NORMAL, FREEZE, RECOVERY, EMERGENCY)
 */
resilience_state_t Resilience_Get_State(void) {
    return g_resilience.state;
}

/**
 * Get time since last gateway message
 * 
 * @return Time in milliseconds since last heartbeat
 */
uint32_t Resilience_Get_Time_Since_Heartbeat(void) {
    return get_current_time_ms() - g_resilience.last_gateway_msg_time;
}

/**
 * Get current penalty level
 * 
 * @return Current cumulative penalty (0.0 to 0.95)
 */
float Resilience_Get_Penalty_Level(void) {
    return g_resilience.cumulative_penalty;
}

/**
 * Get frozen Shapley value
 * 
 * @return Last known Shapley value before freeze
 */
float Resilience_Get_Frozen_Shapley(void) {
    return g_resilience.frozen_shapley;
}

/**
 * Get resilience statistics
 * 
 * @param stats Pointer to statistics structure to fill
 */
void Resilience_Get_Statistics(resilience_stats_t* stats) {
    if (stats == NULL) return;
    
    stats->state = g_resilience.state;
    stats->freeze_reason = g_resilience.freeze_reason;
    stats->total_freeze_events = g_resilience.total_freeze_events;
    stats->total_freeze_duration_ms = g_resilience.total_freeze_duration_ms;
    stats->successful_recoveries = g_resilience.successful_recoveries;
    stats->current_penalty = g_resilience.cumulative_penalty;
    stats->penalty_count = g_resilience.penalty_count;
    stats->time_since_heartbeat = Resilience_Get_Time_Since_Heartbeat();
}

/**
 * Manually trigger freeze state
 * 
 * For testing or emergency situations.
 */
void Resilience_Force_Freeze(void) {
    g_resilience.current_time = get_current_time_ms();
    enter_freeze_state(FREEZE_REASON_MANUAL);
}

/**
 * Manually trigger recovery
 * 
 * For testing - simulates heartbeat reception.
 */
void Resilience_Force_Recovery(void) {
    Resilience_On_Gateway_Heartbeat();
}

/**
 * Reset all resilience state
 * 
 * Use with caution - clears all history and statistics.
 */
void Resilience_Reset(void) {
    Resilience_Init();
}

/* ============================================================================
 * Integration Helpers
 * ============================================================================ */

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
                                 float shapley_value) {
    /* Check energy first */
    if (current_energy < energy_threshold) {
        return false;
    }
    
    /* Get modified probability based on resilience state */
    float prob = Resilience_Get_Backoff_Prob(shapley_value);
    
    /* Random decision based on probability */
    /* Note: In practice, use hardware RNG or LFSR */
    float random = (float)(simple_random() % 1000) / 1000.0f;
    
    bool should_transmit = (random < prob);
    
    /* Track statistics during freeze */
    if (Resilience_Is_Frozen() && should_transmit) {
        g_resilience.total_packets_during_freeze++;
    }
    
    return should_transmit;
}

/**
 * Simple pseudo-random number generator
 * 
 * LFSR-based for deterministic testing.
 */
static uint32_t g_random_state = 0x12345678;

static uint32_t simple_random(void) {
    /* Xorshift32 */
    uint32_t x = g_random_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_random_state = x;
    return x;
}

/**
 * Seed random number generator
 */
void Resilience_Seed_Random(uint32_t seed) {
    g_random_state = seed;
    if (g_random_state == 0) {
        g_random_state = 0x12345678;  /* Avoid zero state */
    }
}

/* ============================================================================
 * Trace Event Definitions
 * ============================================================================ */

#ifndef TRACE_EVENT_RESILIENCE_INIT
#define TRACE_EVENT_RESILIENCE_INIT     0x90
#define TRACE_EVENT_HEARTBEAT_RX        0x91
#define TRACE_EVENT_FREEZE_ENTER        0x92
#define TRACE_EVENT_RECOVERY_START      0x93
#define TRACE_EVENT_RECOVERY_COMPLETE   0x94
#define TRACE_EVENT_PENALTY_APPLIED     0x95
#define TRACE_EVENT_EMERGENCY_STATE     0x96
#endif
