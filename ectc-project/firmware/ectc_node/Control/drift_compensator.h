/**
 * @file drift_compensator.h
 * @brief Online Drift Compensation for ECTC Energy Model - Header
 * 
 * Reference: ECTC Paper Section V-C (Equation 10)
 */

#ifndef DRIFT_COMPENSATOR_H
#define DRIFT_COMPENSATOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Initialize drift compensator
 * @param initial_i_leak Initial leakage current from model (A)
 */
void DriftComp_Init(float initial_i_leak);

/**
 * @brief Record state before entering Deep Sleep
 */
void DriftComp_PreSleep_Record(void);

/**
 * @brief Record state after waking from Deep Sleep
 */
void DriftComp_PostWake_Record(void);

/**
 * @brief Update leakage current estimate (Eq 10)
 * @return Estimated leakage current in Amperes
 */
float Update_Leakage_Estimate(void);

/**
 * @brief Get current leakage estimate
 * @return Leakage current in Amperes
 */
float DriftComp_Get_Leakage(void);

/**
 * @brief Get current leakage estimate in nanoamperes
 * @return Leakage current in nA
 */
float DriftComp_Get_Leakage_nA(void);

/**
 * @brief Get drift compensation statistics
 */
void DriftComp_Get_Stats(
    uint32_t *update_count,
    uint32_t *measurement_count,
    float *cumulative_drift
);

/**
 * @brief Check if model has significant drift
 * @return true if cumulative drift exceeds threshold
 */
bool DriftComp_Has_Significant_Drift(void);

/**
 * @brief Reset drift compensation state
 */
void DriftComp_Reset(void);

#ifdef __cplusplus
}
#endif

#endif /* DRIFT_COMPENSATOR_H */
