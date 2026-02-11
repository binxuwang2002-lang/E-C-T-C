/**
 * @file drift_compensator.c
 * @brief Online Drift Compensation for ECTC Energy Model
 * 
 * Implements the online leakage current estimation algorithm from
 * ECTC Paper Section V-C (Equation 10).
 * 
 * Algorithm:
 * 1. Record V_cap_start and t_start before Deep Sleep
 * 2. Record V_cap_end and t_end after wakeup
 * 3. Calculate slope: Slope = (V_start - V_end) / (t_end - t_start)
 * 4. Estimate leakage: I_leak_new = C_eff × Slope
 * 5. Update model if |I_leak_new - I_leak_model| > delta
 * 
 * Reference: ECTC Paper Equation 10
 * Author: ECTC Research Team
 */

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "drift_compensator.h"

/* ============================================================================
   Configuration Constants
   ============================================================================ */

/** @brief Effective capacitance in Farads (100µF = 100e-6 F) */
#define C_EFF_FARADS            (100.0e-6f)

/** @brief Update threshold delta in Amperes (5nA = 5e-9 A) */
#define LEAKAGE_UPDATE_DELTA_A  (5.0e-9f)

/** @brief Minimum sleep duration for valid measurement (ms) */
#define MIN_SLEEP_DURATION_MS   (10U)

/** @brief Maximum valid slope (V/s) - sanity check */
#define MAX_VALID_SLOPE_V_S     (0.01f)

/** @brief Exponential moving average alpha for smoothing */
#define EMA_ALPHA               (0.3f)

/** @brief ADC conversion factor: ADC counts to Volts */
#define ADC_TO_VOLTS(adc)       ((float)(adc) * 3.3f / 4095.0f)

/* ============================================================================
   Data Structures
   ============================================================================ */

/**
 * @brief Drift compensation state structure
 */
typedef struct {
    /* Measurement state */
    float v_cap_start;      /**< Capacitor voltage before sleep (V) */
    float v_cap_end;        /**< Capacitor voltage after wakeup (V) */
    uint32_t t_start_ms;    /**< Timestamp before sleep (ms) */
    uint32_t t_end_ms;      /**< Timestamp after wakeup (ms) */
    
    /* Estimated parameters */
    float i_leak_estimate;  /**< Current leakage estimate (A) */
    float slope_estimate;   /**< Voltage slope estimate (V/s) */
    
    /* Model reference */
    float i_leak_model;     /**< Leakage from static model (A) */
    
    /* Statistics */
    uint32_t update_count;  /**< Number of model updates */
    uint32_t measurement_count; /**< Total measurements */
    float cumulative_drift; /**< Accumulated drift from model */
    
    /* Flags */
    bool measurement_pending; /**< Wakeup measurement needed */
    bool model_valid;        /**< Model has been calibrated */
} DriftCompensator_State_t;

/* Global state instance */
static DriftCompensator_State_t g_drift_state = {0};

/* ============================================================================
   Private Functions
   ============================================================================ */

/**
 * @brief Get current timestamp in milliseconds
 * @return Current system time in ms
 */
static uint32_t Get_Timestamp_Ms(void)
{
    /* Platform-specific: use HAL_GetTick() or similar */
    extern uint32_t HAL_GetTick(void);
    return HAL_GetTick();
}

/**
 * @brief Read capacitor voltage from ADC
 * @return Voltage in Volts
 */
static float Read_Vcap_Voltage(void)
{
    /* Platform-specific: read ADC connected to V_cap */
    extern uint16_t ADC_Read_Vcap(void);
    uint16_t adc_value = ADC_Read_Vcap();
    return ADC_TO_VOLTS(adc_value);
}

/**
 * @brief Apply exponential moving average filter
 * @param old_value Previous value
 * @param new_value New measurement
 * @param alpha Smoothing factor (0-1)
 * @return Filtered value
 */
static float Apply_EMA(float old_value, float new_value, float alpha)
{
    return alpha * new_value + (1.0f - alpha) * old_value;
}

/* ============================================================================
   Public API Implementation
   ============================================================================ */

/**
 * @brief Initialize drift compensator
 * @param initial_i_leak Initial leakage current from model (A)
 */
void DriftComp_Init(float initial_i_leak)
{
    g_drift_state.i_leak_model = initial_i_leak;
    g_drift_state.i_leak_estimate = initial_i_leak;
    g_drift_state.slope_estimate = 0.0f;
    g_drift_state.update_count = 0;
    g_drift_state.measurement_count = 0;
    g_drift_state.cumulative_drift = 0.0f;
    g_drift_state.measurement_pending = false;
    g_drift_state.model_valid = true;
}

/**
 * @brief Record state before entering Deep Sleep
 * 
 * Call this function immediately before entering low-power mode.
 * Records V_cap_start and t_start for drift calculation.
 */
void DriftComp_PreSleep_Record(void)
{
    g_drift_state.v_cap_start = Read_Vcap_Voltage();
    g_drift_state.t_start_ms = Get_Timestamp_Ms();
    g_drift_state.measurement_pending = true;
}

/**
 * @brief Record state after waking from Deep Sleep
 * 
 * Call this function immediately after waking from low-power mode.
 * Records V_cap_end and t_end for drift calculation.
 */
void DriftComp_PostWake_Record(void)
{
    if (!g_drift_state.measurement_pending) {
        return;  /* No pending measurement */
    }
    
    g_drift_state.v_cap_end = Read_Vcap_Voltage();
    g_drift_state.t_end_ms = Get_Timestamp_Ms();
    g_drift_state.measurement_pending = false;
}

/**
 * @brief Update leakage current estimate using drift compensation
 * 
 * Implements ECTC Paper Equation 10:
 *   I_leak(T) = C_eff × (ΔV_cap / Δt)|_idle
 * 
 * @return Estimated leakage current in Amperes
 */
float Update_Leakage_Estimate(void)
{
    float delta_v;
    float delta_t_s;
    float slope;
    float i_leak_new;
    float i_leak_diff;
    
    /* Validate measurement window */
    uint32_t duration_ms = g_drift_state.t_end_ms - g_drift_state.t_start_ms;
    if (duration_ms < MIN_SLEEP_DURATION_MS) {
        /* Sleep too short for accurate measurement */
        return g_drift_state.i_leak_estimate;
    }
    
    /* Calculate voltage drop */
    delta_v = g_drift_state.v_cap_start - g_drift_state.v_cap_end;
    
    /* Sanity check: voltage should decrease during idle */
    if (delta_v < 0.0f) {
        /* Unexpected: voltage increased (noise or charging) */
        return g_drift_state.i_leak_estimate;
    }
    
    /* Convert time to seconds */
    delta_t_s = (float)duration_ms / 1000.0f;
    
    /* Calculate slope: dV/dt (V/s) */
    slope = delta_v / delta_t_s;
    
    /* Sanity check on slope */
    if (slope > MAX_VALID_SLOPE_V_S) {
        /* Abnormally high slope - likely measurement error */
        return g_drift_state.i_leak_estimate;
    }
    
    /* Apply EMA filter to slope */
    g_drift_state.slope_estimate = Apply_EMA(
        g_drift_state.slope_estimate, 
        slope, 
        EMA_ALPHA
    );
    
    /* Calculate leakage current: I = C × dV/dt */
    /* Equation 10: I_leak(T) = C_eff × (ΔV_cap / Δt)|_idle */
    i_leak_new = C_EFF_FARADS * g_drift_state.slope_estimate;
    
    /* Increment measurement count */
    g_drift_state.measurement_count++;
    
    /* Check if update is needed */
    i_leak_diff = fabsf(i_leak_new - g_drift_state.i_leak_model);
    
    if (i_leak_diff > LEAKAGE_UPDATE_DELTA_A) {
        /* Significant drift detected - update model */
        g_drift_state.i_leak_estimate = i_leak_new;
        g_drift_state.cumulative_drift += i_leak_diff;
        g_drift_state.update_count++;
        
        /* Optionally update the global model */
        /* This would trigger an update to the FEMP energy model */
        #ifdef DRIFT_UPDATE_GLOBAL_MODEL
        extern void FEMP_Update_Leakage(float new_leakage);
        FEMP_Update_Leakage(i_leak_new);
        #endif
    }
    
    return g_drift_state.i_leak_estimate;
}

/**
 * @brief Get current leakage estimate
 * @return Leakage current in Amperes
 */
float DriftComp_Get_Leakage(void)
{
    return g_drift_state.i_leak_estimate;
}

/**
 * @brief Get current leakage estimate in nanoamperes
 * @return Leakage current in nA
 */
float DriftComp_Get_Leakage_nA(void)
{
    return g_drift_state.i_leak_estimate * 1.0e9f;
}

/**
 * @brief Get drift compensation statistics
 * @param update_count Output: number of model updates
 * @param measurement_count Output: total measurements
 * @param cumulative_drift Output: total accumulated drift (A)
 */
void DriftComp_Get_Stats(
    uint32_t *update_count,
    uint32_t *measurement_count,
    float *cumulative_drift
)
{
    if (update_count) {
        *update_count = g_drift_state.update_count;
    }
    if (measurement_count) {
        *measurement_count = g_drift_state.measurement_count;
    }
    if (cumulative_drift) {
        *cumulative_drift = g_drift_state.cumulative_drift;
    }
}

/**
 * @brief Check if model has significant drift
 * @return true if cumulative drift exceeds threshold
 */
bool DriftComp_Has_Significant_Drift(void)
{
    /* Significant if drift exceeds 10% of model value */
    float threshold = 0.1f * g_drift_state.i_leak_model;
    return g_drift_state.cumulative_drift > threshold;
}

/**
 * @brief Reset drift compensation state
 * 
 * Call this after recalibration or model update.
 */
void DriftComp_Reset(void)
{
    g_drift_state.cumulative_drift = 0.0f;
    g_drift_state.update_count = 0;
    g_drift_state.measurement_count = 0;
    g_drift_state.i_leak_estimate = g_drift_state.i_leak_model;
}

/**
 * @brief Full drift compensation cycle
 * 
 * Convenience function that calls PreSleep, simulates sleep,
 * calls PostWake, and updates the estimate.
 * 
 * @note For real use, call the functions separately around actual sleep.
 * 
 * @return Updated leakage estimate in Amperes
 */
float DriftComp_FullCycle(void)
{
    /* Record pre-sleep state */
    DriftComp_PreSleep_Record();
    
    /* [MCU enters Deep Sleep here] */
    /* ... sleep ... */
    /* [MCU wakes up here] */
    
    /* Record post-wake state */
    DriftComp_PostWake_Record();
    
    /* Update and return estimate */
    return Update_Leakage_Estimate();
}

/* ============================================================================
   Example Usage (for reference)
   ============================================================================ */

#ifdef DRIFT_COMPENSATOR_EXAMPLE

/**
 * @brief Example integration with ECTC power management
 */
void Example_ECTC_Sleep_Cycle(void)
{
    /* Before entering sleep */
    DriftComp_PreSleep_Record();
    
    /* Enter low-power mode */
    /* HAL_PWR_EnterSTOPMode(PWR_LOWPOWERREGULATOR_ON, PWR_STOPENTRY_WFI); */
    
    /* After wakeup */
    DriftComp_PostWake_Record();
    
    /* Update leakage estimate (Eq 10) */
    float new_leakage = Update_Leakage_Estimate();
    
    /* Check for significant drift */
    if (DriftComp_Has_Significant_Drift()) {
        /* Trigger recalibration or notify gateway */
        /* ECTC_Notify_Model_Drift(); */
    }
}

#endif /* DRIFT_COMPENSATOR_EXAMPLE */
