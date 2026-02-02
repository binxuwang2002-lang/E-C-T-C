/*
 * Trace Implementation for ECTC
 */

#include "trace.h"
#include <string.h>

// Global trace buffer
__attribute__((section(".retain")))
trace_record_t trace_buffer[TRACE_BUFFER_SIZE];

static uint8_t trace_head = 0;
static uint8_t trace_count = 0;

/**
 * Initialize trace system
 */
void trace_init(void) {
    // Clear buffer
    memset(trace_buffer, 0, sizeof(trace_buffer));

    // Initialize all records with magic number
    for (int i = 0; i < TRACE_BUFFER_SIZE; i++) {
        trace_buffer[i].magic = TRACE_MAGIC;
    }

    trace_head = 0;
    trace_count = 0;
}

/**
 * Record an event
 */
void trace_event(uint16_t event_type, uint32_t param1, uint32_t param2) {
    uint32_t timestamp_us = get_system_time_us();
    uint16_t pc = get_program_counter();

    trace_record_t* record = &trace_buffer[trace_head];

    // Fill record
    record->magic = TRACE_MAGIC;
    record->timestamp = timestamp_us;
    record->event_type = event_type;
    record->pc = pc;
    record->param1 = param1;
    record->param2 = param2;
    record->stack_check = calculate_stack_checksum();

    // Update head
    trace_head = (trace_head + 1) % TRACE_BUFFER_SIZE;
    if (trace_count < TRACE_BUFFER_SIZE) {
        trace_count++;
    }
}

/**
 * Check if buffer should be uploaded
 */
bool trace_should_upload(void) {
    return (trace_count >= TRACE_BUFFER_SIZE);
}

/**
 * Upload trace buffer to gateway
 */
void upload_trace_buffer(void) {
    // Send trace buffer over radio or store in external memory
    // For now, just reset buffer
    trace_head = 0;
    trace_count = 0;
}

/**
 * Get number of records in buffer
 */
uint32_t trace_get_count(void) {
    return trace_count;
}

/**
 * Get program counter (simplified)
 */
uint16_t get_program_counter(void) {
    // In ARM assembly, PC is current instruction + 4
    // Return simplified version
    return 0x2000;  // Placeholder SRAM address
}

/**
 * Get system time in microseconds
 */
uint32_t get_system_time_us(void) {
    // Simplified - in practice, use GPT timer
    return 0;  // Placeholder
}

/**
 * Calculate stack checksum for memory integrity
 */
uint32_t calculate_stack_checksum(void) {
    // Simple checksum of stack region
    // In practice, compute CRC32
    return 0x12345678;
}
