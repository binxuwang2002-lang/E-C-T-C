/*
 * Event Tracer for ECTC
 *
 * Stores execution trace in 200-byte circular buffer
 */

#ifndef TRACE_H
#define TRACE_H

#include <stdint.h>
#include <stdbool.h>

#define TRACE_BUFFER_SIZE    64   // 64 entries
#define TRACE_MAGIC          0xABCDEF01

// Trace record structure (200 bytes total)
typedef struct {
    uint32_t magic;        // Magic number for validation
    uint32_t timestamp;    // Microsecond timestamp
    uint16_t event_type;   // Event type
    uint16_t pc;           // Program counter
    uint32_t param1;       // First parameter
    uint32_t param2;       // Second parameter
    uint32_t stack_check;  // Stack checksum
} __attribute__((packed)) trace_record_t;

// Global buffer (retain in RAM)
__attribute__((section(".retain")))
extern trace_record_t trace_buffer[TRACE_BUFFER_SIZE];

// Function prototypes
void trace_init(void);
void trace_event(uint16_t event_type, uint32_t param1, uint32_t param2);
bool trace_should_upload(void);
void upload_trace_buffer(void);
uint32_t trace_get_count(void);

// Utility macros
#define TRACE(event_type, param1, param2) \
    trace_event(event_type, param1, param2)

#endif /* TRACE_H */
