/*
 * Radio Driver Header
 */

#ifndef RADIO_H
#define RADIO_H

#include <stdint.h>
#include <stdbool.h>

// Capabilities
#define CAP_RECEIVER        0x01
#define CAP_TRANSMITTER     0x02
#define CAP_POWER_SOURCE    0x04

// Message types
#define MSG_BOOT            0x01
#define MSG_DATA            0x02
#define MSG_ASSOC_REQ       0x03
#define MSG_ASSOC_RESP      0x04
#define MSG_RELAY           0x05

// Status codes
#define STATUS_SUCCESS      0x00
#define STATUS_FULL         0x01
#define STATUS_DENIED       0x02

// ECTC Version
#define ECTC_VERSION        0x10

// Frame structure
typedef struct {
    uint8_t type;
    uint8_t node_id;
    uint32_t timestamp;
    float value;
} __attribute__((packed)) data_packet_t;

typedef struct {
    uint8_t type;
    uint8_t node_id;
    uint32_t timestamp;
    uint8_t version;
} __attribute__((packed)) boot_msg_t;

typedef struct {
    uint8_t type;
    uint8_t node_id;
    uint8_t capability;
} __attribute__((packed)) assoc_req_t;

typedef struct {
    uint8_t type;
    uint8_t node_id;
    uint8_t status;
    uint16_t short_addr;
} __attribute__((packed)) assoc_resp_t;

typedef struct {
    uint8_t type;
    uint16_t dest_addr;
    float value;
} __attribute__((packed)) relay_msg_t;

// Function prototypes
void radio_init(void);
void radio_set_channel(uint8_t channel);
void radio_set_pan_id(uint16_t pan_id);
void radio_set_short_addr(uint16_t addr);
void radio_configure_power_management(bool enable);
bool radio_send(const void* data, size_t len);
bool radio_recv(void* buffer, size_t max_len, uint32_t timeout_ms);
void radio_send_boot_message(uint8_t node_id);
void radio_send_data(float data);
bool radio_recv_relay(void* packet);
void radio_send_relay(const void* packet);
int8_t radio_get_rssi(void);
uint8_t radio_get_lqi(void);
void radio_sleep(void);
void radio_wake(void);
bool radio_scan_beacons(void);
bool radio_join_network(void);

// Internal functions
bool radio_transmit_packet(const uint8_t* data, size_t len);
bool radio_packet_available(void);
size_t radio_receive_packet(uint8_t* buffer, size_t max_len);
void radio_set_rx_timeout(uint32_t timeout_ms);
void radio_set_tx_retries(uint8_t retries);

#endif /* RADIO_H */
