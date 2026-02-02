/*
** ###################################################################
**     BQ25570 PMIC Configuration for ECTC
**
**     Calculates resistor values for BQ25570 energy harvesting chip
**     to enforce exact voltage thresholds from ECTC-19.pdf
**
**     Reference: ECTC-19.pdf, Section IV.A
**     Hardware: BQ25570 (Texas Instruments)
**     Constraints: C_cap = 100μF, V_ret = 1.8V, θ = 0.9
**
** ################################################################===
*/

#ifndef BQ25570_PMIC_CONFIG_H_
#define BQ25570_PMIC_CONFIG_H_

#include <stdint.h>
#include <math.h>

/* =============================================================================
 * Hardware Constants (from ECTC-19.pdf)
 * =============================================================================
 */

/* PMIC Configuration Parameters */
#define C_CAP_UF               100.0f   /* Capacitor: 100μF */
#define V_RETENTION            1.8f     /* Retention voltage: 1.8V */
#define THETA_SATURATION       0.9f     /* Saturation threshold: 0.9 */
#define V_DD_NOMINAL           3.3f     /* Nominal supply: 3.3V */

/* BQ25570 Internal Reference Voltages */
#define V_REF_OV               1.5f     /* Overvoltage reference: 1.5V */
#define V_REF_UV               0.5f     /* Undervoltage reference: 0.5V */
#define V_REF_OK               1.0f     /* OK threshold reference: 1.0V */

/* =============================================================================
 * Calculated Thresholds
 * =============================================================================
 */

/* Based on ECTC requirements */
#define V_OV_PROTECT           (THETA_SATURATION * V_DD_NOMINAL)  /* 2.97V */
#define V_UV_SHUTDOWN          (V_RETENTION * 0.8f)                /* 1.44V */
#define V_OK_THRESHOLD         (V_RETENTION * 1.1f)                /* 1.98V */

/* =============================================================================
 * BQ25570 Resistor Divider Calculation
 * =============================================================================
 *
 * The BQ25570 uses external resistor dividers to set programmable thresholds:
 *
 *     V_IN (from capacitor)
 *          |
 *         R_TOP
 *          |
 *          +--- V_PROG (to PMIC)
 *          |
 *         R_BOTTOM
 *          |
 *         GND
 *
 * Formula: V_PROG = V_IN * (R_BOTTOM / (R_TOP + R_BOTTOM))
 * Rearranged: V_IN = V_PROG * (R_TOP + R_BOTTOM) / R_BOTTOM
 *
 * Standard practice:
 * - Choose R_BOTTOM = 10MΩ (high impedance, low current)
 * - Calculate R_TOP for desired V_IN
 */

/**
 * Calculate R_TOP resistor for desired threshold voltage
 *
 * @param v_threshold Desired threshold voltage (V)
 * @param v_reference PMIC reference voltage (V)
 * @param r_bottom Resistor to ground (Ω) - recommended 10MΩ
 * @return Required R_TOP resistor value (Ω)
 *
 * Formula: R_TOP = R_BOTTOM * (V_THRESHOLD / V_REF - 1)
 */
float bq25570_calc_r_top(float v_threshold,
                         float v_reference,
                         float r_bottom) {
    return r_bottom * (v_threshold / v_reference - 1.0f);
}

/**
 * Calculate R_TOP for Overvoltage Protection (V_OV = 2.97V)
 *
 * Protection threshold when capacitor reaches 90% (θ = 0.9)
 * This triggers "Hardware Clipping" to prevent energy overflow
 *
 * @return R_TOP value in Ω (or -1 if invalid)
 */
float bq25570_ovp_resistor(void) {
    const float R_BOTTOM = 10e6f;  /* 10MΩ */

    /* V_PROG_OV = V_OV / 2 (from BQ25570 datasheet) */
    float v_prog_ov = V_OV_PROTECT / 2.0f;

    /* Calculate R_TOP */
    float r_top = bq25570_calc_r_top(v_prog_ov, V_REF_OV, R_BOTTOM);

    return r_top;
}

/**
 * Calculate R_TOP for Undervoltage Shutdown (V_UV = 1.44V)
 *
 * Shuts down system to preserve energy below retention threshold
 * Prevents deep discharge that could corrupt retention RAM
 *
 * @return R_TOP value in Ω (or -1 if invalid)
 */
float bq25570_uvp_resistor(void) {
    const float R_BOTTOM = 10e6f;  /* 10MΩ */

    /* V_PROG_UV = V_UV / 2 (from BQ25570 datasheet) */
    float v_prog_uv = V_UV_SHUTDOWN / 2.0f;

    /* Calculate R_TOP */
    float r_top = bq25570_calc_r_top(v_prog_uv, V_REF_UV, R_BOTTOM);

    return r_top;
}

/**
 * Calculate R_TOP for OK Threshold (V_OK = 1.98V)
 *
 * System "OK" indicator - enables operation when above retention
 * This threshold must be above V_RETENTION to ensure reliable operation
 *
 * @return R_TOP value in Ω (or -1 if invalid)
 */
float bq25570_ok_threshold_resistor(void) {
    const float R_BOTTOM = 10e6f;  /* 10MΩ */

    /* V_PROG_OK = V_OK / 2 (from BQ25570 datasheet) */
    float v_prog_ok = V_OK_THRESHOLD / 2.0f;

    /* Calculate R_TOP */
    float r_top = bq25570_calc_r_top(v_prog_ok, V_REF_OK, R_BOTTOM);

    return r_top;
}

/**
 * Calculate all three resistors and return as array
 *
 * @param resistors Output array: [R_OVP, R_UVP, R_OK]
 * @return 0 on success, -1 if any calculation failed
 */
int bq25570_calculate_all_resistors(float resistors[3]) {
    resistors[0] = bq25570_ovp_resistor();   /* Overvoltage Protection */
    resistors[1] = bq25570_uvp_resistor();   /* Undervoltage Protection */
    resistors[2] = bq25570_ok_threshold_resistor();  /* OK Threshold */

    /* Check for invalid values (negative resistors) */
    for (int i = 0; i < 3; i++) {
        if (resistors[i] < 0.0f || resistors[i] > 100e6f) {
            return -1;
        }
    }

    return 0;
}

/* =============================================================================
 * Standard Resistor Values (E96 Series - 1% Tolerance)
 * =============================================================================
 */

/**
 * Find nearest E96 standard resistor value
 *
 * @param calculated_value Calculated resistance (Ω)
 * @return Nearest E96 standard value (Ω)
 */
float bq25570_find_standard_resistor(float calculated_value) {
    /* E96 series values (multiplied by powers of 10) */
    const float e96[] = {
        1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24,
        1.27, 1.30, 1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58,
        1.62, 1.65, 1.69, 1.74, 1.78, 1.82, 1.87, 1.91, 1.96, 2.00,
        2.05, 2.10, 2.15, 2.21, 2.26, 2.32, 2.37, 2.43, 2.49, 2.55,
        2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09, 3.16, 3.24,
        3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12,
        4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23,
        5.36, 5.49, 5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65,
        6.81, 6.98, 7.15, 7.32, 7.50, 7.68, 7.87, 8.06, 8.25, 8.45,
        8.66, 8.87, 9.09, 9.31, 9.53, 9.76
    };

    /* Determine decade */
    int decade = (int)floorf(log10f(calculated_value));
    float mantissa = calculated_value / powf(10.0f, decade);

    /* Find closest mantissa in E96 */
    float best_match = e96[0];
    float min_error = fabsf(mantissa - e96[0]);

    for (int i = 1; i < 96; i++) {
        float error = fabsf(mantissa - e96[i]);
        if (error < min_error) {
            min_error = error;
            best_match = e96[i];
        }
    }

    return best_match * powf(10.0f, decade);
}

/**
 * Get recommended resistor values for BQ25570
 *
 * @param rovp Output: Overvoltage protection resistor (Ω)
 * @param ruvp Output: Undervoltage protection resistor (Ω)
 * @param rok Output: OK threshold resistor (Ω)
 * @return 0 on success
 */
int bq25570_get_recommended_resistors(float *rovp, float *ruvp, float *rok) {
    float calculated[3];

    if (bq25570_calculate_all_resistors(calculated) != 0) {
        return -1;
    }

    /* Find standard values */
    *rovp = bq25570_find_standard_resistor(calculated[0]);
    *ruvp = bq25570_find_standard_resistor(calculated[1]);
    *rok  = bq25570_find_standard_resistor(calculated[2]);

    return 0;
}

/* =============================================================================
 * Energy Storage Calculation
 * =============================================================================
 */

/**
 * Calculate energy in capacitor
 *
 * E = 0.5 * C * V^2
 *
 * @param capacitance Capacitance in Farads
 * @param voltage Voltage in Volts
 * @return Energy in Joules
 */
float bq25570_calc_stored_energy(float capacitance, float voltage) {
    return 0.5f * capacitance * voltage * voltage;
}

/**
 * Convert voltage to energy for 100μF capacitor
 *
 * @param voltage Voltage in Volts
 * @return Energy in micro-Joules (μJ)
 *
 * For C = 100μF = 100e-6 F:
 * E = 0.5 * 100e-6 * V^2 * 1e6 = 50 * V^2 μJ
 */
float bq25570_voltage_to_energy_uj(float voltage) {
    return 50.0f * voltage * voltage;  /* For 100μF capacitor */
}

/**
 * Convert energy to voltage
 *
 * @param energy Energy in Joules
 * @param capacitance Capacitance in Farads
 * @return Voltage in Volts
 */
float bq25570_energy_to_voltage(float energy, float capacitance) {
    return sqrtf(2.0f * energy / capacitance);
}

/* =============================================================================
 * Hardware-Software Interaction Analysis
 * =============================================================================
 */

/**
 * Analyze interaction between PMIC hardware thresholds and control law
 *
 * PMIC Hardware Clipping:
 * - Hard cutoff at V_OV = 2.97V (90% of 3.3V)
 * - Any excess energy dissipated as heat
 * - Passive protection, no software control
 *
 * Software Control Law (Equation 3):
 * - Virtual wall at θ = 0.9 threshold
 * - Strong negative feedback (quartic penalty)
 * - Active prevention before hitting hardware limit
 *
 * Synergy:
 * 1. PMIC provides fail-safe protection
 * 2. Control law optimizes before fail-safe triggers
 * 3. Combination prevents energy overflow + maximizes utility
 */

/**
 * Check if system is in safe operating region
 *
 * @param v_cap Capacitor voltage
 * @return 1 if safe, 0 if warning, -1 if danger
 */
int bq25570_safety_check(float v_cap) {
    if (v_cap >= V_OV_PROTECT) {
        return -1;  /* Danger: Hardware clipping will trigger */
    } else if (v_cap >= (V_OV_PROTECT * 0.95f)) {
        return 0;   /* Warning: Approaching saturation */
    } else {
        return 1;   /* Safe: Below 95% of threshold */
    }
}

/* =============================================================================
 * Configuration Structure
 * =============================================================================
 */

typedef struct {
    float rovp;      /* Overvoltage protection resistor (Ω) */
    float ruvp;      /* Undervoltage protection resistor (Ω) */
    float rok;       /* OK threshold resistor (Ω) */
    float v_ov;      /* Overvoltage threshold (V) */
    float v_uv;      /* Undervoltage threshold (V) */
    float v_ok;      /* OK threshold (V) */
    float c_cap;     /* Capacitance (F) */
    float e_cap;     /* Maximum stored energy (J) */
} bq25570_config_t;

/**
 * Get complete BQ25570 configuration
 *
 * @return Configuration structure
 */
bq25570_config_t bq25570_get_config(void) {
    bq25570_config_t config;

    /* Calculate resistors */
    bq25570_get_recommended_resistors(&config.rovp, &config.ruvp, &config.rok);

    /* Set voltages */
    config.v_ov = V_OV_PROTECT;
    config.v_uv = V_UV_SHUTDOWN;
    config.v_ok = V_OK_THRESHOLD;

    /* Set capacitance and energy */
    config.c_cap = 100e-6f;  /* 100μF */
    config.e_cap = bq25570_calc_stored_energy(config.c_cap, V_DD_NOMINAL);

    return config;
}

#endif /* BQ25570_PMIC_CONFIG_H_ */
