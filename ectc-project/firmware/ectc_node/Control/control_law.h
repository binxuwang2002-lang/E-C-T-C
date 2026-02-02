/*
** ###################################################################
**     Header file for Saturation-Aware Control Law Module
**
**     Reference: ECTC-19.pdf, Equation 3 & 6, Section IV.A
**
** ###################################################################
*/

#ifndef CONTROL_LAW_H
#define CONTROL_LAW_H

#include <stdint.h>
#include <stdbool.h>

/* Configuration structure */
typedef struct {
    float C_cap;      /* Capacitor capacity (μJ) */
    float V_ret;      /* Retention voltage threshold */
    float theta;      /* Saturation threshold θ */
    float beta;       /* Penalty coefficient */
    float V_param;    /* Lyapunov tradeoff parameter */
    float gamma_u;    /* Utility gain parameter */
    float gamma_q;    /* Energy weight parameter */
} control_law_config_t;

/*
** Function Prototypes
*/

/* Lyapunov function computation */
float control_law_truncated_lyapunov(float Q_E);
float control_law_lyapunov_drift(float Q_E, float Q_E_next);
float control_law_lyapunov_gradient(float Q_E);

/* Virtual wall mechanism */
bool control_law_is_in_saturation(float Q_E);
float control_law_virtual_wall_penalty(float Q_E);
bool control_law_energy_above_retention(float Q_E);

/* Main control law */
float control_law_compute_action(float Q_E, float U_i, uint16_t B_i);
bool control_law_should_execute(float Q_E, float U_i, uint16_t B_i);
uint16_t control_law_max_safe_tasks(float Q_E);
uint8_t control_law_predict_schedule(float Q_E,
                                     float predicted_harvest,
                                     float U_i,
                                     uint16_t B_i,
                                     uint8_t predicted_schedule[10]);

/* Utility functions */
void control_law_print_state(float Q_E, float U_i, uint16_t B_i);
bool control_law_verify_properties(void);
const control_law_config_t* control_law_get_config(void);

#endif /* CONTROL_LAW_H */
