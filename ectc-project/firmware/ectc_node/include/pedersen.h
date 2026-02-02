/*
 * Header file for Pedersen commitment
 */

#ifndef PEDERSEN_H
#define PEDERSEN_H

#include <stdint.h>
#include <stdbool.h>

// Pedersen commitment structure
typedef struct {
    uint8_t point[64];  // Elliptic curve point (compressed)
    uint32_t timestamp;
    uint32_t nonce;
} pedersen_commitment_t;

// Function prototypes
void pedersen_init(void);
void pedersen_commit(float energy_value, pedersen_commitment_t* output_commitment);
bool pedersen_verify(const pedersen_commitment_t* commitment,
                    float energy_value,
                    uint32_t nonce);
bool pedersen_verify_range(float energy_value, float min_energy, float max_energy);
const pedersen_commitment_t* pedersen_get_current_commitment(void);
void pedersen_commitment_update(void);
size_t pedersen_get_commitment_size(void);
uint32_t pedersen_get_proof_time_us(void);
bool simulate_zk_proof_exchange(const pedersen_commitment_t* commitment,
                               float energy_value,
                               uint32_t nonce,
                               uint8_t* response_buffer,
                               size_t buffer_size);

#endif /* PEDERSEN_H */
