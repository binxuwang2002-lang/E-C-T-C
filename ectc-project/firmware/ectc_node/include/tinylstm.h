/*
 * Header file for TinyLSTM inference engine
 */

#ifndef TINYLSTM_H
#define TINYLSTM_H

#include <stdint.h>
#include <stdbool.h>

// LSTM configuration
#define HIDDEN_DIM       32
#define INPUT_DIM        1
#define OUTPUT_DIM       1
#define SEQ_LEN          10
#define WEIGHT_SIZE      128  // 4 gates * hidden_dim

// Function prototypes
void tinylstm_init(void);
void tinylstm_predict(float* input_seq, float* output);
void tinylstm_get_memory_stats(uint32_t* sram_usage, uint32_t* flash_usage);
float tinylstm_get_inference_energy(void);

#endif /* TINYLSTM_H */
