/*
 * Radio-Compute Interleaving (RCI) Driver Header
 * ===============================================
 *
 * Exploits hardware crystal startup time (~2.1ms) to hide
 * computation latencies for TinyLSTM inference.
 *
 * Reference: ECTC Paper - Section IV.C (Radio-Compute Interleaving)
 *
 * Hardware: CC2650 + BQ25570
 */

#ifndef RADIO_RCI_H
#define RADIO_RCI_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Configuration Constants
 * ============================================================================ */

/** Crystal oscillator startup time in microseconds */
#define RCI_XTAL_STARTUP_US         2100

/** TinyLSTM single step execution time in microseconds */
#define RCI_LSTM_STEP_US            150

/** Safety margin before transmission in microseconds */
#define RCI_SAFETY_MARGIN_US        100

/** Maximum usable compute window in microseconds */
#define RCI_COMPUTE_WINDOW_US       (RCI_XTAL_STARTUP_US - RCI_SAFETY_MARGIN_US)

/** Maximum number of LSTM inference slices per RCI call */
#define RCI_MAX_LSTM_SLICES         ((RCI_COMPUTE_WINDOW_US) / (RCI_LSTM_STEP_US))

/* ============================================================================
 * Types
 * ============================================================================ */

/**
 * RCI State Machine States
 */
typedef enum {
    RCI_STATE_IDLE = 0,         /**< No RCI operation active */
    RCI_STATE_OSC_WARMING,      /**< Oscillator warming up, compute in progress */
    RCI_STATE_OSC_READY,        /**< Oscillator ready, waiting for TX */
    RCI_STATE_TRANSMITTING,     /**< Radio transmission in progress */
    RCI_STATE_COMPLETE,         /**< RCI operation complete */
    RCI_STATE_ERROR             /**< Error occurred */
} rci_state_t;

/**
 * RCI Statistics Structure
 */
typedef struct {
    uint32_t total_rci_calls;       /**< Total number of RCI transmissions */
    uint32_t total_slices_executed; /**< Total LSTM slices executed */
    uint32_t total_compute_time_us; /**< Total compute time in warm-up windows */
    uint32_t total_hidden_energy_uj;/**< Total energy hidden during warm-up (μJ) */
    float avg_slices_per_call;      /**< Average slices per RCI call */
    uint32_t avg_compute_time_us;   /**< Average compute time per call */
    float efficiency_percent;        /**< Warm-up window utilization (%) */
} rci_stats_t;

/* ============================================================================
 * Core RCI Functions
 * ============================================================================ */

/**
 * Initialize RCI subsystem
 *
 * Must be called once at startup before using other RCI functions.
 * Initializes TinyLSTM and internal state.
 */
void RCI_Init(void);

/**
 * Set energy history for TinyLSTM inference
 *
 * Provides the input sequence for TinyLSTM prediction that will be
 * executed during the next Radio_Transmit_With_RCI() call.
 *
 * @param energy_history Array of 10 normalized energy values (0.0-1.0)
 *
 * @note Must be called before Radio_Transmit_With_RCI() for inference
 */
void RCI_Set_Energy_History(const float energy_history[10]);

/**
 * Get predicted energy from last RCI inference
 *
 * Returns the TinyLSTM prediction computed during the last RCI
 * transmission's warm-up period.
 *
 * @return Predicted energy value, or -1.0 if inference not complete
 */
float RCI_Get_Predicted_Energy(void);

/**
 * Radio Transmit with Compute Interleaving
 *
 * Main RCI function that exploits crystal warm-up time (~2.1ms) to
 * execute TinyLSTM inference slices while waiting for the radio.
 *
 * Timing Budget:
 * - XTAL startup: 2100 μs
 * - TinyLSTM step: ~150 μs per slice
 * - Safety margin: 100 μs
 * - Usable window: 2000 μs → ~13 inference slices
 *
 * Mechanism:
 * 1. Trigger Radio Oscillator ON
 * 2. While Radio_Is_Warming_Up() is true:
 *    - Execute one slice of TinyLSTM_Inference() (atomic task)
 *    - Check Timer_Get_Micros() to ensure we don't overrun
 * 3. Once warm-up complete, immediately trigger Radio_Transmit()
 *
 * @param data Pointer to data buffer to transmit
 * @param len Length of data in bytes (max 127 for IEEE 802.15.4)
 *
 * @note Call RCI_Set_Energy_History() before this for ML inference
 * @note Use RCI_Was_Successful() to check transmission result
 * @note Use RCI_Get_Slices_Executed() to check inference progress
 */
void Radio_Transmit_With_RCI(uint8_t* data, size_t len);

/**
 * Transmit with RCI and custom compute callback
 *
 * Generalizes RCI to execute arbitrary compute tasks during warm-up,
 * not just TinyLSTM inference.
 *
 * @param data Data to transmit
 * @param len Data length in bytes
 * @param compute_fn Function to call during warm-up (should complete in <150μs)
 * @param compute_ctx Context pointer passed to compute function
 * @param max_iterations Maximum number of compute function calls
 *
 * @note compute_fn must be atomic and complete within RCI_LSTM_STEP_US
 */
void Radio_Transmit_With_RCI_Custom(
    uint8_t* data, 
    size_t len,
    void (*compute_fn)(void* ctx),
    void* compute_ctx,
    uint8_t max_iterations
);

/* ============================================================================
 * Status Functions
 * ============================================================================ */

/**
 * Check if last RCI transmission was successful
 *
 * @return true if transmission completed successfully
 */
bool RCI_Was_Successful(void);

/**
 * Get number of LSTM slices executed in last RCI operation
 *
 * @return Number of slices executed (0-10)
 */
uint8_t RCI_Get_Slices_Executed(void);

/**
 * Get current RCI state
 *
 * @return Current state machine state
 */
rci_state_t RCI_Get_State(void);

/* ============================================================================
 * Statistics Functions
 * ============================================================================ */

/**
 * Get RCI statistics
 *
 * @param stats Pointer to statistics structure to fill
 */
void RCI_Get_Statistics(rci_stats_t* stats);

/**
 * Reset RCI statistics
 */
void RCI_Reset_Statistics(void);

/* ============================================================================
 * Control Functions
 * ============================================================================ */

/**
 * Abort current RCI operation
 *
 * Immediately stops oscillator warm-up and cancels transmission.
 * Use in emergency situations (e.g., brownout detection).
 */
void RCI_Abort(void);

/* ============================================================================
 * Trace Events (for integration with trace subsystem)
 * ============================================================================ */

#ifndef TRACE_EVENT_RCI_INIT
#define TRACE_EVENT_RCI_INIT        0x80
#define TRACE_EVENT_RCI_START       0x81
#define TRACE_EVENT_RCI_OSC_START   0x82
#define TRACE_EVENT_RCI_OSC_READY   0x83
#define TRACE_EVENT_RCI_OSC_FAIL    0x84
#define TRACE_EVENT_RCI_LSTM_DONE   0x85
#define TRACE_EVENT_RCI_COMPLETE    0x86
#define TRACE_EVENT_RCI_ERROR       0x87
#define TRACE_EVENT_RCI_OVERRUN     0x88
#define TRACE_EVENT_RCI_ABORT       0x89
#endif

#ifdef __cplusplus
}
#endif

#endif /* RADIO_RCI_H */
