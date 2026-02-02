/*
 * Header file for local Shapley computation
 */

#ifndef SHAPLEY_LOCAL_H
#define SHAPLEY_LOCAL_H

#include <stdint.h>
#include <stdbool.h>

// Function prototypes
void shapley_local_init(void);
void shapley_local_compute(float Q_E, uint16_t B_i, float predicted_harvest,
                          float* output_marginal);
void update_trust_scores(const uint8_t* neighbor_ids, float* reported_utilities,
                        uint8_t count);
float get_shapley_error_bound(void);
uint32_t get_shapley_complexity_ops(void);

#endif /* SHAPLEY_LOCAL_H */
