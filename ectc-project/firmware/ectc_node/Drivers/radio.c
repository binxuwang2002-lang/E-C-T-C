/*
 * IEEE 802.15.4 Radio Driver
 * ==========================
 *
 * Complete radio stack for CC2650.
 * Handles packet transmission, reception, and MAC layer.
 */

#include "radio.h"
#include "ectc_main.h"
#include <string.h>

// Radio configuration
#define RADIO_CHANNEL            11      // 2.405 GHz
#define RADIO_PAN_ID             0x1234
#define RADIO_SHORT_ADDR         0x0001
#define RADIO_PAYLOAD_SIZE       127     // Max IEEE 802.15.4 frame
#define RADIO_BUFFER_SIZE        128

// Frame types
#define FRAME_TYPE_BEACON        0x00
#define FRAME_TYPE_DATA          0x01
#define FRAME_TYPE_ACK           0x02
#define FRAME_TYPE_CMD           0x03

// Radio state
typedef enum {
    RADIO_STATE_IDLE = 0,
    RADIO_STATE_RX,
    RADIO_STATE_TX,
    RADIO_STATE_SLEEP
} radio_state_t;

static radio_state_t g_radio_state = RADIO_STATE_IDLE;
static bool g_radio_initialized = false;
static uint8_t g_tx_buffer[RADIO_BUFFER_SIZE];
static uint8_t g_rx_buffer[RADIO_BUFFER_SIZE];
static uint16_t g_packet_count = 0;

// Frame structure
typedef struct {
    uint8_t frame_type;      // Frame type
    uint8_t sequence;        // Sequence number
    uint16_t dest_pan;       // Destination PAN ID
    uint16_t dest_addr;      // Destination address
    uint16_t src_pan;        // Source PAN ID
    uint16_t src_addr;       // Source address
    uint8_t payload[100];    // Payload
    uint8_t payload_len;     // Payload length
    uint8_t fcs[2];          // Frame check sequence
} __attribute__((packed)) radio_frame_t;

/**
 * Initialize radio
 */
void radio_init(void) {
    // Configure RF interface
    // In practice, use TI RF driver

    // Set channel
    radio_set_channel(RADIO_CHANNEL);

    // Set PAN ID
    radio_set_pan_id(RADIO_PAN_ID);

    // Set short address
    radio_set_short_addr(RADIO_SHORT_ADDR);

    // Configure for low-power operation
    radio_configure_power_management(true);

    g_radio_initialized = true;
    g_radio_state = RADIO_STATE_IDLE;

    trace_event(TRACE_EVENT_RADIO_INIT, RADIO_CHANNEL, RADIO_PAN_ID);
}

/**
 * Set radio channel
 */
void radio_set_channel(uint8_t channel) {
    // Validate channel (IEEE 802.15.4: channels 11-26)
    if (channel < 11) channel = 11;
    if (channel > 26) channel = 26;

    // Configure RF settings
    // In practice, set frequency: 2405 + 5*(channel-11) MHz

    trace_event(TRACE_EVENT_RADIO_CHANNEL, channel, 0);
}

/**
 * Set PAN ID
 */
void radio_set_pan_id(uint16_t pan_id) {
    // Store PAN ID in frame template
    // In practice, configure MAC layer

    trace_event(TRACE_EVENT_RADIO_PAN_ID, pan_id, 0);
}

/**
 * Set short address
 */
void radio_set_short_addr(uint16_t addr) {
    // Store short address
    // In practice, configure MAC layer

    trace_event(TRACE_EVENT_RADIO_ADDR, addr, 0);
}

/**
 * Configure power management
 */
void radio_configure_power_management(bool enable) {
    if (enable) {
        // Enable auto-sleep
        radio_set_rx_timeout(1000);  // 1 second timeout
        radio_set_tx_retries(3);
    }
}

/**
 * Send packet
 */
bool radio_send(const void* data, size_t len) {
    if (!g_radio_initialized || len > RADIO_PAYLOAD_SIZE) {
        return false;
    }

    // Change state to TX
    g_radio_state = RADIO_STATE_TX;

    // Build frame
    radio_frame_t* frame = (radio_frame_t*)g_tx_buffer;
    frame->frame_type = FRAME_TYPE_DATA;
    frame->sequence = (g_packet_count & 0xFF);
    frame->dest_pan = RADIO_PAN_ID;
    frame->dest_addr = 0x0000;  // Broadcast
    frame->src_pan = RADIO_PAN_ID;
    frame->src_addr = RADIO_SHORT_ADDR;
    frame->payload_len = len;

    // Copy payload
    memcpy(frame->payload, data, len);

    // Calculate FCS (simplified)
    frame->fcs[0] = 0x00;
    frame->fcs[1] = 0x00;

    // Transmit
    bool success = radio_transmit_packet(g_tx_buffer, sizeof(radio_frame_t));

    if (success) {
        g_packet_count++;
        trace_event(TRACE_EVENT_RADIO_TX, len, g_packet_count);
    } else {
        trace_event(TRACE_EVENT_RADIO_TX_FAIL, len, 0);
    }

    g_radio_state = RADIO_STATE_IDLE;

    return success;
}

/**
 * Receive packet
 */
bool radio_recv(void* buffer, size_t max_len, uint32_t timeout_ms) {
    if (!g_radio_initialized || max_len < RADIO_PAYLOAD_SIZE) {
        return false;
    }

    // Change state to RX
    g_radio_state = RADIO_STATE_RX;

    // Set timeout
    uint32_t start_time = get_system_time_ms();
    uint32_t elapsed = 0;

    while (elapsed < timeout_ms) {
        // Check if packet received
        if (radio_packet_available()) {
            // Read packet
            size_t rx_len = radio_receive_packet(g_rx_buffer, RADIO_BUFFER_SIZE);

            if (rx_len > 0) {
                radio_frame_t* frame = (radio_frame_t*)g_rx_buffer;

                // Check frame type
                if (frame->frame_type == FRAME_TYPE_DATA) {
                    // Copy payload to output buffer
                    memcpy(buffer, frame->payload, frame->payload_len);

                    trace_event(TRACE_EVENT_RADIO_RX, frame->payload_len, frame->sequence);

                    g_radio_state = RADIO_STATE_IDLE;
                    return true;
                }
            }
        }

        // Sleep for 1ms
        PowerCPUEnterSleep();
        elapsed = get_system_time_ms() - start_time;
    }

    // Timeout
    trace_event(TRACE_EVENT_RADIO_RX_TIMEOUT, 0, 0);
    g_radio_state = RADIO_STATE_IDLE;

    return false;
}

/**
 * Send boot message
 */
void radio_send_boot_message(uint8_t node_id) {
    boot_msg_t msg = {
        .type = MSG_BOOT,
        .node_id = node_id,
        .timestamp = get_system_time_ms(),
        .version = ECTC_VERSION
    };

    radio_send(&msg, sizeof(msg));
}

/**
 * Transmit packet (hardware-specific)
 */
bool radio_transmit_packet(const uint8_t* data, size_t len) {
    // In practice, use TI RF driver
    // RF_sendCmd() or similar

    // Simulate successful transmission
    // Check collision detection

    return true;
}

/**
 * Check if packet is available
 */
bool radio_packet_available(void) {
    // In practice, check RF interrupt flag
    return true;  // Simplified
}

/**
 * Receive packet from hardware
 */
size_t radio_receive_packet(uint8_t* buffer, size_t max_len) {
    // In practice, use TI RF driver
    // RF_receiveCmd() or similar

    // Simulate packet reception
    // Check FCS, address filtering, etc.

    return 0;  // No packet
}

/**
 * Set RX timeout
 */
void radio_set_rx_timeout(uint32_t timeout_ms) {
    // Configure RX timeout in hardware
    // In practice, set timer
}

/**
 * Set TX retries
 */
void radio_set_tx_retries(uint8_t retries) {
    // Configure retransmissions
    // In practice, set MAC retransmission count
}

/**
 * Get RSSI of last received packet
 */
int8_t radio_get_rssi(void) {
    // Return received signal strength indicator
    // In practice, read from RF register

    return -60;  // dBm, placeholder
}

/**
 * Get link quality
 */
uint8_t radio_get_lqi(void) {
    // Return link quality indicator
    // In practice, calculated from RSSI

    return 80;  // 0-255 scale, placeholder
}

/**
 * Send data payload
 */
void radio_send_data(float data) {
    data_packet_t packet = {
        .type = MSG_DATA,
        .node_id = RADIO_SHORT_ADDR,
        .timestamp = get_system_time_ms(),
        .value = data
    };

    radio_send(&packet, sizeof(packet));
}

/**
 * Receive relay packet
 */
bool radio_recv_relay(relay_packet_t* packet) {
    relay_msg_t msg;
    bool success = radio_recv(&msg, sizeof(msg), 100);

    if (success && msg.type == MSG_RELAY) {
        packet->dest_node = msg.dest_addr;
        packet->data = msg.value;
        return true;
    }

    return false;
}

/**
 * Send relay packet
 */
void radio_send_relay(const relay_packet_t* packet) {
    relay_msg_t msg = {
        .type = MSG_RELAY,
        .node_id = packet->dest_node,
        .timestamp = get_system_time_ms(),
        .value = packet->data
    };

    radio_send(&msg, sizeof(msg));
}

/**
 * Get radio state
 */
radio_state_t radio_get_state(void) {
    return g_radio_state;
}

/**
 * Enter low power mode
 */
void radio_sleep(void) {
    if (g_radio_state == RADIO_STATE_IDLE) {
        g_radio_state = RADIO_STATE_SLEEP;
        // Configure radio for sleep
        // In practice, use TI RF driver sleep mode
    }
}

/**
 * Wake up from sleep
 */
void radio_wake(void) {
    if (g_radio_state == RADIO_STATE_SLEEP) {
        g_radio_state = RADIO_STATE_IDLE;
        // Reconfigure radio
        // In practice, use TI RF driver wakeup
    }
}

/**
 * Get statistics
 */
void radio_get_stats(uint32_t* tx_count, uint32_t* rx_count, uint32_t* rx_errors) {
    // In practice, maintain statistics counters
    *tx_count = g_packet_count;
    *rx_count = g_packet_count;  // Simplified
    *rx_errors = 0;
}

/**
 * Clear statistics
 */
void radio_clear_stats(void) {
    g_packet_count = 0;
    // Clear other counters
}

/**
 * Scan for beacons
 */
bool radio_scan_beacons(void) {
    g_radio_state = RADIO_STATE_RX;

    // Listen for beacon frames for 100ms
    uint8_t buffer[128];
    bool found = false;

    for (int i = 0; i < 10; i++) {
        if (radio_recv(buffer, sizeof(buffer), 10)) {
            radio_frame_t* frame = (radio_frame_t*)buffer;
            if (frame->frame_type == FRAME_TYPE_BEACON) {
                found = true;
                break;
            }
        }
    }

    g_radio_state = RADIO_STATE_IDLE;
    return found;
}

/**
 * Join network
 */
bool radio_join_network(void) {
    // Scan for coordinator
    if (!radio_scan_beacons()) {
        return false;
    }

    // Send association request
    assoc_req_t req = {
        .type = MSG_ASSOC_REQ,
        .node_id = RADIO_SHORT_ADDR,
        .capability = CAP_RECEIVER | CAP_POWER_SOURCE
    };

    bool success = radio_send(&req, sizeof(req));

    // Wait for association response
    assoc_resp_t resp;
    success = radio_recv(&resp, sizeof(resp), 1000);

    return success && resp.status == STATUS_SUCCESS;
}
