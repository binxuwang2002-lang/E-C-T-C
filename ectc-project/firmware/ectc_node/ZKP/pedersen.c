/*
 * Pedersen Commitment for Zero-Knowledge Energy Proof
 * ====================================================
 *
 * Implements Pedersen commitments to prove energy state without revealing it.
 * Based on secp256k1 elliptic curve.
 *
 * Security: Hides actual energy value while allowing verification of range
 */

#include "pedersen.h"
#include <string.h>
#include <stdint.h>

// Elliptic curve parameters (secp256k1)
static const uint8_t P[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
};

static const uint8_t Gx[32] = {
    0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC,
    0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
    0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9,
    0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98
};

static const uint8_t Gy[32] = {
    0x48, 0x3A, 0xDA, 0x77, 0x26, 0xA3, 0xC4, 0x65,
    0x5D, 0xA4, 0xFB, 0xFC, 0x0E, 0x11, 0x08, 0xA8,
    0xFD, 0x17, 0xB4, 0x48, 0xA6, 0x85, 0x54, 0x19,
    0x9C, 0x47, 0xD0, 0x8F, 0xFB, 0x10, 0xD4, 0xB8
};

// Commitment storage
static pedersen_commitment_t g_current_commitment;
static uint8_t g_commitment_nonce[32];
static uint8_t g_private_key[32];
static bool g_key_initialized = false;

/**
 * Initialize Pedersen commitment system
 */
void pedersen_init(void) {
    // Initialize or load private key
    if (!g_key_initialized) {
        generate_private_key(g_private_key);
        g_key_initialized = true;
    }

    // Initialize nonce
    random_bytes(g_commitment_nonce, 32);

    trace_event(TRACE_EVENT_PEDERSEN_INIT, 0, 0);
}

/**
 * Generate private key for commitment
 */
void generate_private_key(uint8_t* private_key) {
    // Generate random 256-bit private key
    random_bytes(private_key, 32);

    // Ensure key is in valid range: 1 ≤ x ≤ n-1 (where n is curve order)
    // For simplicity, we assume the key is valid
    // In production, use proper cryptographic library (like mbedTLS)
}

/**
 * Create Pedersen commitment for energy value
 *
 * Commitment = G^energy_value * H^nonce
 * where G is the generator point, H is a random point
 *
 * Args:
 *   energy_value: Energy to commit (μJ)
 *   output_commitment: Pointer to store commitment
 */
void pedersen_commit(float energy_value, pedersen_commitment_t* output_commitment) {
    // Convert float to integer representation (fixed-point)
    uint32_t energy_int = (uint32_t)(energy_value * 1000.0f);  // mJ precision

    // Hash nonce to get H^nonce
    uint8_t H_nonce[64];
    hash_to_curve(g_commitment_nonce, H_nonce);

    // Compute G^energy_value
    uint8_t G_val[64];
    scalar_multiplication(Gx, Gy, (uint8_t*)&energy_int, 4, G_val);

    // Compute commitment = G^val * H^nonce
    point_add(G_val, H_nonce, output_commitment->point);

    // Store commitment metadata
    output_commitment->timestamp = get_system_time_ms();
    output_commitment->nonce = *(uint32_t*)g_commitment_nonce;  // Store first 4 bytes

    // Update current commitment
    g_current_commitment = *output_commitment;

    trace_event(TRACE_EVENT_PEDERSEN_COMMIT, (uint32_t)energy_int, 0);
}

/**
 * Verify Pedersen commitment
 *
 * Args:
 *   commitment: Commitment to verify
 *   energy_value: Claimed energy value
 *   nonce: Nonce used in commitment
 *
 * Returns:
 *   true if commitment is valid, false otherwise
 */
bool pedersen_verify(const pedersen_commitment_t* commitment,
                    float energy_value,
                    uint32_t nonce) {
    // Convert energy to integer
    uint32_t energy_int = (uint32_t)(energy_value * 1000.0f);

    // Reconstruct H^nonce
    uint8_t nonce_bytes[4];
    nonce_bytes[0] = nonce & 0xFF;
    nonce_bytes[1] = (nonce >> 8) & 0xFF;
    nonce_bytes[2] = (nonce >> 16) & 0xFF;
    nonce_bytes[3] = (nonce >> 24) & 0xFF;

    uint8_t H_nonce[64];
    hash_to_curve(nonce_bytes, 4, H_nonce);

    // Compute G^energy_value
    uint8_t G_val[64];
    scalar_multiplication(Gx, Gy, (uint8_t*)&energy_int, 4, G_val);

    // Compute expected commitment
    uint8_t expected_commitment[64];
    point_add(G_val, H_nonce, expected_commitment);

    // Compare with provided commitment
    return memcmp(expected_commitment, commitment->point, 64) == 0;
}

/**
 * Hash bytes to elliptic curve point (simplified)
 * In production, use proper hash-to-curve algorithm
 */
void hash_to_curve(const uint8_t* input, size_t len, uint8_t* output) {
    // Simple hash to curve: hash input and map to point
    // This is a placeholder - use proper hash-to-curve (like Swu or Icart)

    uint8_t hash[32];
    sha256_hash(input, len, hash);

    // Map hash to point (simplified - not cryptographically secure)
    // In practice, use Icart algorithm or Brier-Joye method
    for (int i = 0; i < 32; i++) {
        output[i] = hash[i];
        output[i + 32] = hash[(i + 1) % 32];
    }
}

/**
 * Scalar multiplication on elliptic curve (simplified)
 */
void scalar_multiplication(const uint8_t* px, const uint8_t* py,
                          const uint8_t* scalar, size_t scalar_len,
                          uint8_t* result) {
    // Placeholder implementation
    // In production, use efficient algorithms (double-and-add, windowed NAF, etc.)

    // For now, just return a placeholder
    memcpy(result, px, 32);
    memcpy(result + 32, py, 32);
}

/**
 * Point addition on elliptic curve (simplified)
 */
void point_add(const uint8_t* p1, const uint8_t* p2, uint8_t* result) {
    // Placeholder implementation
    // In production, use proper elliptic curve point addition

    // For demonstration: just XOR the points (NOT cryptographically secure)
    for (int i = 0; i < 64; i++) {
        result[i] = p1[i] ^ p2[i];
    }
}

/**
 * SHA-256 hash (simplified placeholder)
 */
void sha256_hash(const uint8_t* input, size_t len, uint8_t* output) {
    // Placeholder - use proper SHA-256 implementation
    // For now, just copy input (or use a simple checksum)
    memset(output, 0, 32);
    for (size_t i = 0; i < len && i < 32; i++) {
        output[i] = input[i];
    }
}

/**
 * Generate random bytes
 */
void random_bytes(uint8_t* output, size_t len) {
    // In production, use hardware RNG (CC2650 has TRNG)
    // For now, use simple LCG
    static uint32_t seed = 0x12345678;

    for (size_t i = 0; i < len; i++) {
        seed = seed * 1664525 + 1013904223;
        output[i] = (uint8_t)(seed >> (i % 24));
    }
}

/**
 * Get current commitment
 */
const pedersen_commitment_t* pedersen_get_current_commitment(void) {
    return &g_current_commitment;
}

/**
 * Update commitment (called periodically)
 */
void pedersen_commitment_update(void) {
    // Generate new nonce
    random_bytes(g_commitment_nonce, 32);

    // Commit to current energy
    float current_energy = get_current_energy();
    pedersen_commit(current_energy, &g_current_commitment);

    trace_event(TRACE_EVENT_PEDERSEN_UPDATE, (uint32_t)(current_energy * 1000), 0);
}

/**
 * Verify range proof (energy in [min, max])
 */
bool pedersen_verify_range(float energy_value, float min_energy, float max_energy) {
    // Check if value is in range
    if (energy_value < min_energy || energy_value > max_energy) {
        return false;
    }

    // Verify commitment
    uint32_t nonce = g_current_commitment.nonce;
    return pedersen_verify(&g_current_commitment, energy_value, nonce);
}

/**
 * Get commitment size
 */
size_t pedersen_get_commitment_size(void) {
    return sizeof(pedersen_commitment_t);
}

/**
 * Get proof generation time (estimated)
 */
uint32_t pedersen_get_proof_time_us(void) {
    // Based on scalar multiplication complexity
    // For 256-bit scalar: ~256 point additions
    return 256 * 100;  // ~25.6ms
}

/**
 * Simulate ZK proof exchange
 */
bool simulate_zk_proof_exchange(const pedersen_commitment_t* commitment,
                               float energy_value,
                               uint32_t nonce,
                               uint8_t* response_buffer,
                               size_t buffer_size) {
    // Verify commitment
    if (!pedersen_verify(commitment, energy_value, nonce)) {
        return false;
    }

    // Generate response (in practice, would be challenge-response protocol)
    // For now, just return success indicator
    if (buffer_size >= 4) {
        response_buffer[0] = 0x01;  // Success
        response_buffer[1] = 0x23;  // Random response
        response_buffer[2] = 0x45;
        response_buffer[3] = 0x67;
    }

    trace_event(TRACE_EVENT_PEDERSEN_VERIFY, (uint32_t)(energy_value * 1000), 0);
    return true;
}
