/*
 * ECTC Main Control Loop
 * ======================
 *
 * Main control loop implementing the ECTC protocol for battery-free IoT nodes.
 * Handles energy sampling, TinyLSTM prediction, and local decision making.
 *
 * Author: ECTC Team
 * Date: 2024
 */

#include "ectc_main.h"
#include "tinylstm.h"
#include "shapley_local.h"
#include "bq25570.h"
#include "radio.h"
#include "trace.h"

#include <stdint.h>
#include <stdbool.h>

// Configuration
#define SLOT_DURATION_MS      100    // 100ms time slot
#define SUPERFRAME_SLOTS      10     // 10 slots = 1s superframe
#define ENERGY_THRESHOLD      50.0f  // Minimum energy to execute tasks (μJ)
#define PREDICTION_HORIZON    5      // TinyLSTM prediction steps

// Node state structure
typedef struct {
    uint8_t node_id;
    float Q_E;              // Current energy in capacitor (μJ)
    uint16_t B_i;           // Data queue length (packets)
    float predicted_harvest; // Predicted energy collection (μJ)
    float marginal_utility;  // Local marginal utility
    bool has_data;           // Has data to transmit
    float sensor_reading;    // Latest sensor value
} node_status_t;

// Global state
static node_status_t g_node_status;
static uint32_t g_slot_counter = 0;
static float g_energy_history[10];
static uint8_t g_energy_history_idx = 0;

/**
 * Initialize ECTC node
 */
void ectc_init(void) {
    // Initialize hardware
    bq25570_init();
    radio_init();
    trace_init();

    // Initialize TinyLSTM
    tinylstm_init();

    // Initialize local Shapley calculator
    shapley_local_init();

    // Initialize node ID (read from flash or generate)
    g_node_status.node_id = read_node_id();

    // Clear energy history
    for (int i = 0; i < 10; i++) {
        g_energy_history[i] = 0.0f;
    }

    trace_event(TRACE_EVENT_INIT, 0, 0);

    // Send boot message
    radio_send_boot_message(g_node_status.node_id);
}

/**
 * ECTC main control loop - executes every time slot
 */
void ectc_control_loop(void) {
    uint32_t slot_start_ms = get_system_time_ms();

    // Step 1: Sample energy and update energy history
    sample_energy();

    // Step 2: Sample sensor and update data queue
    sample_sensor_data();

    // Step 3: Predict future energy using TinyLSTM
    predict_energy();

    // Step 4: Compute local marginal utility
    compute_local_utility();

    // Step 5: Package status and send to gateway
    package_and_send_status();

    // Step 6: Receive and process gateway commands
    receive_and_process_commands();

    // Step 7: Execute local tasks if no response
    execute_local_fallback();

    // Step 8: Sleep until next slot
    uint32_t elapsed_ms = get_system_time_ms() - slot_start_ms;
    if (elapsed_ms < SLOT_DURATION_MS) {
        enter_low_power_mode(SLOT_DURATION_MS - elapsed_ms);
    }

    g_slot_counter++;

    // Periodic tasks every superframe
    if (g_slot_counter % SUPERFRAME_SLOTS == 0) {
        periodic_maintenance();
    }
}

/**
 * Sample current energy from capacitor
 */
void sample_energy(void) {
    // Read capacitor voltage via ADC
    float cap_voltage = read_capacitor_voltage();

    // Convert to energy: E = 0.5 * C * V^2
    // For 100μF capacitor
    g_node_status.Q_E = 0.5f * 100e-6f * cap_voltage * cap_voltage * 1e6f; // in μJ

    // Update energy history (circular buffer)
    g_energy_history[g_energy_history_idx] = g_node_status.Q_E;
    g_energy_history_idx = (g_energy_history_idx + 1) % 10;

    // Check if capacitor is full (energy threshold)
    if (g_node_status.Q_E > 330.0f) {
        // Capacitor full - potential energy overflow
        trace_event(TRACE_EVENT_CAP_FULL, 0, 0);
    }
}

/**
 * Sample sensor data and update queue
 */
void sample_sensor_data(void) {
    // Sample temperature sensor (on CC2650)
    float temperature = sample_temperature();

    // Store reading
    g_node_status.sensor_reading = temperature;

    // Determine if reading is significant enough to queue
    if (is_significant_reading(temperature)) {
        g_node_status.B_i++;
        trace_event(TRACE_EVENT_DATA_QUEUED, 0, 0);
    }

    // Check queue overflow
    if (g_node_status.B_i > 255) {
        g_node_status.B_i = 255;  // Saturate
        trace_event(TRACE_EVENT_QUEUE_OVERFLOW, 0, 0);
    }
}

/**
 * Predict energy using TinyLSTM
 */
void predict_energy(void) {
    // Prepare input for TinyLSTM
    float input_seq[10];
    for (int i = 0; i < 10; i++) {
        input_seq[i] = g_energy_history[i];
    }

    // Run TinyLSTM inference
    float prediction = 0.0f;
    tinylstm_predict(input_seq, &prediction);

    // Store prediction
    g_node_status.predicted_harvest = prediction;

    trace_event(TRACE_EVENT_PREDICTION, (uint32_t)(prediction * 1000), 0);
}

/**
 * Compute local marginal utility
 */
void compute_local_utility(void) {
    // Use local Shapley calculator
    shapley_local_compute(
        g_node_status.Q_E,
        g_node_status.B_i,
        g_node_status.predicted_harvest,
        &g_node_status.marginal_utility
    );

    g_node_status.has_data = (g_node_status.B_i > 0);
}

/**
 * Package node status and send to gateway
 */
void package_and_send_status(void) {
    // Create status packet
    status_packet_t packet;
    packet.node_id = g_node_status.node_id;
    packet.Q_E = g_node_status.Q_E;
    packet.B_i = g_node_status.B_i;
    packet.marginal_utility = g_node_status.marginal_utility;
    packet.predicted_harvest = g_node_status.predicted_harvest;
    packet.has_data = g_node_status.has_data;
    packet.sensor_reading = g_node_status.sensor_reading;

    // Get current Pedersen commitment
    packet.commitment = pedersen_get_current_commitment();

    // Send to gateway via radio
    radio_send(&packet, sizeof(packet));

    trace_event(TRACE_EVENT_STATUS_SENT, 0, 0);
}

/**
 * Receive and process commands from gateway
 */
void receive_and_process_commands(void) {
    command_packet_t cmd;
    bool received = radio_recv(&cmd, sizeof(cmd), 10);  // 10ms timeout

    if (received) {
        if (cmd.shapley_value > SHAPLEY_THRESHOLD) {
            // High utility - execute assigned task
            if (g_node_status.Q_E > ENERGY_THRESHOLD) {
                execute_task(cmd.assigned_task);
                trace_event(TRACE_EVENT_TASK_EXECUTED, cmd.assigned_task, 0);
            }
        }

        // Update global coalition information
        update_coalition_info(&cmd.coalition);
    }
}

/**
 * Fallback: execute local tasks if no gateway response
 */
void execute_local_fallback(void) {
    // Simple greedy policy
    if (g_node_status.Q_E > ENERGY_THRESHOLD && g_node_status.B_i > 0) {
        // Execute transmission task
        execute_task(TASK_TRANSMIT_DATA);
        trace_event(TRACE_EVENT_LOCAL_TASK, TASK_TRANSMIT_DATA, 0);
    }
}

/**
 * Periodic maintenance (runs every superframe)
 */
void periodic_maintenance(void) {
    // Update Pedersen commitment
    pedersen_commitment_update();

    // Upload trace buffer if enough samples
    if (trace_should_upload()) {
        upload_trace_buffer();
    }

    // Check for memory corruption
    if (check_memory_integrity()) {
        // Memory OK
    } else {
        // Memory corruption detected
        trace_event(TRACE_EVENT_MEM_CORRUPTION, 0, 0);
        system_reset();  // Safe fallback
    }

    trace_event(TRACE_EVENT_MAINTENANCE, g_slot_counter, 0);
}

/**
 * Execute assigned task
 */
void execute_task(task_id_t task_id) {
    switch (task_id) {
        case TASK_TRANSMIT_DATA:
            // Transmit queued data
            if (g_node_status.B_i > 0) {
                radio_send_data(g_node_status.sensor_reading);
                g_node_status.B_i--;
                g_node_status.Q_E -= ENERGY_COST_TRANSMIT;
            }
            break;

        case TASK_SENSE_DATA:
            // Perform additional sensing
            float reading = sample_temperature();
            g_node_status.Q_E -= ENERGY_COST_SENSE;
            trace_event(TRACE_EVENT_TASK_COMPLETE, TASK_SENSE_DATA, (uint32_t)(reading * 1000));
            break;

        case TASK_RELAY_DATA:
            // Relay data for other nodes
            relay_packet_t relay;
            if (radio_recv_relay(&relay)) {
                radio_send_relay(&relay);
                g_node_status.Q_E -= ENERGY_COST_RELAY;
            }
            break;

        case TASK_COMPUTE:
            // Perform local computation
            local_computation();
            break;

        default:
            // Unknown task
            break;
    }
}

/**
 * Enter low power mode
 */
void enter_low_power_mode(uint32_t sleep_duration_ms) {
    // Configure wake-up timer
    setup_wakeup_timer(sleep_duration_ms);

    // Enter sleep mode
    PowerCPUEnterSleep();

    // System resumes here after wake-up
}

/**
 * Sample temperature sensor
 */
float sample_temperature(void) {
    // Read from internal temperature sensor
    // Returns temperature in Celsius
    return read_internal_temperature();
}

/**
 * Check if reading is significant
 */
bool is_significant_reading(float reading) {
    // Simple threshold-based detection
    // In practice, use delta encoding or anomaly detection
    return (reading > TEMP_THRESHOLD_LOW && reading < TEMP_THRESHOLD_HIGH);
}

/**
 * Local computation task
 */
void local_computation(void) {
    // Simple hash computation for PoW
    uint32_t hash_input = g_node_status.node_id + g_slot_counter;
    uint32_t hash_result = simple_hash(hash_input);

    g_node_status.Q_E -= ENERGY_COST_COMPUTE;
    trace_event(TRACE_EVENT_LOCAL_COMPUTE, hash_result, 0);
}

/**
 * Simple hash function (for demonstration)
 */
uint32_t simple_hash(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

/**
 * Memory integrity check
 */
bool check_memory_integrity(void) {
    // Simple checksum verification
    return (calculate_stack_checksum() == EXPECTED_STACK_CHECKSUM);
}

/**
 * System reset (safe fallback)
 */
void system_reset(void) {
    // Log reset reason
    trace_event(TRACE_EVENT_RESET, RESET_REASON_MEM_CORRUPT, 0);

    // Perform software reset
    *(volatile uint32_t *)0xE000ED0C = 0x05FA0001;
}
