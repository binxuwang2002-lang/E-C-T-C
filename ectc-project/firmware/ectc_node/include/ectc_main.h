/*
 * Header files for ECTC firmware
 */

#ifndef ECTC_MAIN_H
#define ECTC_MAIN_H

#include <stdint.h>
#include <stdbool.h>

// Configuration parameters
#define NODE_ID_DEFAULT          0
#define SLOT_DURATION_MS         100
#define SUPERFRAME_SLOTS         10
#define ENERGY_THRESHOLD         50.0f
#define PREDICTION_HORIZON       5
#define SHAPLEY_THRESHOLD        0.5f

// Energy costs (μJ)
#define ENERGY_COST_TRANSMIT     5.3f
#define ENERGY_COST_SENSE        1.0f
#define ENERGY_COST_RELAY        3.0f
#define ENERGY_COST_COMPUTE      2.5f

// Task types
typedef enum {
    TASK_NONE = 0,
    TASK_TRANSMIT_DATA,
    TASK_SENSE_DATA,
    TASK_RELAY_DATA,
    TASK_COMPUTE
} task_id_t;

// Trace event types
typedef enum {
    TRACE_EVENT_INIT = 0,
    TRACE_EVENT_PREDICTION,
    TRACE_EVENT_DATA_QUEUED,
    TRACE_EVENT_QUEUE_OVERFLOW,
    TRACE_EVENT_STATUS_SENT,
    TRACE_EVENT_TASK_EXECUTED,
    TRACE_EVENT_LOCAL_TASK,
    TRACE_EVENT_CAP_FULL,
    TRACE_EVENT_MAINTENANCE,
    TRACE_EVENT_MEM_CORRUPTION,
    TRACE_EVENT_RESET,
    TRACE_EVENT_PEDERSEN_INIT,
    TRACE_EVENT_PEDERSEN_COMMIT,
    TRACE_EVENT_PEDERSEN_UPDATE,
    TRACE_EVENT_PEDERSEN_VERIFY,
    TRACE_EVENT_LSTM_INIT,
    TRACE_EVENT_LSTM_INFERENCE,
    TRACE_EVENT_SHAPLEY_INIT,
    TRACE_EVENT_SHAPLEY_COMPUTE,
    TRACE_EVENT_TRUST_UPDATE,
    TRACE_EVENT_LOCAL_COMPUTE
} trace_event_type_t;

// Data structures
typedef struct {
    uint8_t node_id;
    float Q_E;
    uint16_t B_i;
    float marginal_utility;
    float predicted_harvest;
    bool has_data;
    float sensor_reading;
    uint32_t commitment;
} __attribute__((packed)) status_packet_t;

typedef struct {
    uint32_t shapley_value;
    task_id_t assigned_task;
    uint8_t coalition_info[16];
} __attribute__((packed)) command_packet_t;

typedef struct {
    uint32_t dest_node;
    float data;
} __attribute__((packed)) relay_packet_t;

// Function prototypes
void ectc_init(void);
void ectc_control_loop(void);
float sample_temperature(void);
bool is_significant_reading(float reading);
void execute_task(task_id_t task_id);
void enter_low_power_mode(uint32_t sleep_duration_ms);
bool check_memory_integrity(void);
void system_reset(void);

// Hardware abstraction functions
float read_capacitor_voltage(void);
void radio_send(const void* data, size_t len);
bool radio_recv(void* buffer, size_t len, uint32_t timeout_ms);
void radio_send_data(float data);
bool radio_recv_relay(relay_packet_t* packet);
void radio_send_relay(const relay_packet_t* packet);
float read_internal_temperature(void);
uint32_t get_system_time_ms(void);
void setup_wakeup_timer(uint32_t ms);

// BQ25570 energy harvesting interface
void bq25570_init(void);
float get_current_energy(void);

// Radio configuration
void radio_init(void);
void radio_send_boot_message(uint8_t node_id);

// Shapley computation
void shapley_local_init(void);
void shapley_local_compute(float Q_E, uint16_t B_i, float predicted_harvest,
                          float* output_marginal);

// TinyLSTM
void tinylstm_init(void);
void tinylstm_predict(float* input_seq, float* output);

// Trace functions
void trace_init(void);
void trace_event(trace_event_type_t event, uint32_t param1, uint32_t param2);
bool trace_should_upload(void);
void upload_trace_buffer(void);
uint32_t calculate_stack_checksum(void);
#define EXPECTED_STACK_CHECKSUM 0x12345678

// Temperature thresholds
#define TEMP_THRESHOLD_LOW       -10.0f
#define TEMP_THRESHOLD_HIGH      50.0f

// Reset reasons
#define RESET_REASON_MEM_CORRUPT 0x01

// Memory layout (from linker.ld)
extern uint32_t _sidata;
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;
extern uint32_t _estack;

// Task execution
void local_computation(void);

// Power management
void PowerCPUEnterSleep(void);

#endif /* ECTC_MAIN_H */
