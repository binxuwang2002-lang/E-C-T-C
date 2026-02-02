/*
 * BQ25570 Driver Header
 */

#ifndef BQ25570_H
#define BQ25570_H

#include <stdint.h>
#include <stdbool.h>

// Configuration structure
typedef struct {
    uint16_t vbat_uv;       // Under-voltage threshold (mV)
    uint16_t vbat_ok;       // VBAT OK threshold (mV)
    uint16_t vbat_hys;      // VBAT hysteresis (mV)
    uint16_t mppt_voltage;  // MPPT voltage (mV)
    uint16_t vstor_ok;      // Storage OK threshold (mV)
    uint16_t vstor_max;     // Storage maximum (mV)
    uint8_t istor_limit;    // Storage current limit (mA)
    uint16_t vout_target;   // VOUT target (mV)
} bq25570_config_t;

// Status structure
typedef struct {
    bool vbat_ok;    // VBAT is OK
    bool charging;   // Charging is active
    bool vstor_ok;   // VSTOR is OK
    bool ovp;        // Over-voltage protection triggered
} bq25570_status_t;

// Function prototypes
void bq25570_init(void);
void bq25570_configure(const bq25570_config_t* config);
float bq25570_read_vcap(void);
float bq25570_read_vstor(void);
float bq25570_get_capacitor_energy(void);
void bq25570_enable_charging(bool enable);
bool bq25570_is_power_good(void);
bq25570_status_t bq25570_get_status(void);
uint8_t bq25570_read_reg(uint8_t reg_addr);
void bq25570_write_reg(uint8_t reg_addr, uint8_t value);

#endif /* BQ25570_H */
