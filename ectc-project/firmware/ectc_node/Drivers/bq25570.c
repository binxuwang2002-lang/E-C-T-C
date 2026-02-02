/*
 * BQ25570 Energy Harvesting PMIC Driver
 * ====================================
 *
 * Driver for BQ25570 ultra-low power energy harvesting PMIC
 * Configures MPPT, voltage thresholds, and energy monitoring
 */

#include "bq25570.h"
#include <stdint.h>
#include <stdbool.h>

// I2C configuration
#define BQ25570_I2C_ADDR         0x6A
#define BQ25570_I2C_TIMEOUT_MS   100

// BQ25570 registers
#define BQ25570_REG_VBAT_UV        0x00
#define BQ25570_REG_VBAT_OK        0x01
#define BQ25570_REG_VBAT_HYS       0x02
#define BQ25570_REG_MPPT_V_CTL     0x03
#define BQ25570_REG_V_STOR_UV      0x04
#define BQ25570_REG_V_STOR_OK      0x05
#define BQ25570_REG_V_STOR_HYS     0x06
#define BQ25570_REG_I_STOR_LIM     0x07
#define BQ25570_REG_VOUT_OK        0x08
#define BQ25570_REG_VOUT_CTL       0x09
#define BQ25570_REG_PWR_OPT        0x0A
#define BQ25570_REG_TEMP_STATUS    0x0B
#define BQ25570_REG_VBAT_STATUS    0x0C
#define BQ25570_REG_I_STOR_STATUS  0x0D
#define BQ25570_REG_V_STOR_STATUS  0x0E

// Default configuration
static const bq25570_config_t g_default_config = {
    .vbat_uv = 2300,        // 2.3V (battery under-voltage)
    .vbat_ok = 2500,        // 2.5V (battery OK threshold)
    .vbat_hys = 200,        // 200mV hysteresis
    .mppt_voltage = 2400,   // 2.4V (solar optimized)
    .vstor_ok = 3300,       // 3.3V (storage OK)
    .vstor_max = 4500,      // 4.5V (storage max)
    .istor_limit = 100,     // 100mA storage current limit
    .vout_target = 3300     // 3.3V output
};

/**
 * Initialize BQ25570
 */
void bq25570_init(void) {
    // Initialize I2C (assumes I2C0 is configured)
    I2CMasterInitExpClk(I2C0_BASE, 48000000, false);
    I2CMasterEnable(I2C0_BASE);

    // Configure default settings
    bq25570_configure(&g_default_config);

    // Enable charging
    bq25570_enable_charging(true);

    trace_event(TRACE_EVENT_BQ25570_INIT, 0, 0);
}

/**
 * Configure BQ25570 settings
 */
void bq25570_configure(const bq25570_config_t* config) {
    // Set VBAT under-voltage threshold
    uint8_t vbat_uv = (config->vbat_uv - 1800) / 25;  // 25mV steps
    bq25570_write_reg(BQ25570_REG_VBAT_UV, vbat_uv);

    // Set VBAT OK threshold
    uint8_t vbat_ok = (config->vbat_ok - 1800) / 25;
    bq25570_write_reg(BQ25570_REG_VBAT_OK, vbat_ok);

    // Set VBAT hysteresis
    uint8_t vbat_hys = config->vbat_hys / 25;
    bq25570_write_reg(BQ25570_REG_VBAT_HYS, vbat_hys);

    // Set MPPT voltage (for solar)
    uint8_t mppt_v = (config->mppt_voltage - 500) / 50;  // 50mV steps
    bq25570_write_reg(BQ25570_REG_MPPT_V_CTL, mppt_v);

    // Set VSTOR OK threshold
    uint8_t vstor_ok = (config->vstor_ok - 2600) / 50;
    bq25570_write_reg(BQ25570_REG_V_STOR_OK, vstor_ok);

    // Set storage over-voltage
    uint8_t vstor_max = (config->vstor_max - 3300) / 50;
    bq25570_write_reg(BQ25570_REG_V_STOR_HYS, vstor_max);

    // Set storage current limit
    uint8_t istor_lim = config->istor_limit / 10;  // 10mA steps
    bq25570_write_reg(BQ25570_REG_I_STOR_LIM, istor_lim);

    // Set VOUT target
    uint8_t vout = (config->vout_target - 3000) / 100;  // 100mV steps
    bq25570_write_reg(BQ25570_REG_VOUT_CTL, vout);
}

/**
 * Read capacitor voltage
 */
float bq25570_read_vcap(void) {
    uint16_t adc_raw = 0;

    // Read from external ADC (connected to voltage divider)
    // Assuming MCP3008 or similar on SPI
    adc_raw = spi_read_adc_channel(0);

    // Convert to voltage
    // Assuming 3.3V reference, 12-bit ADC
    float vcap = (adc_raw * 3300) / 4095.0;  // mV

    return vcap / 1000.0;  // Return in Volts
}

/**
 * Read storage voltage (VSTOR)
 */
float bq25570_read_vstor(void) {
    // Read from BQ25570 ADC (12-bit, internal reference)
    // For now, use external measurement
    return bq25570_read_vcap();  // Simplified
}

/**
 * Calculate energy in capacitor
 */
float bq25570_get_capacitor_energy(void) {
    float vcap = bq25570_read_vcap();

    // E = 0.5 * C * V^2
    // For 100μF capacitor
    float energy = 0.5f * 100e-6f * vcap * vcap;  // Joules

    return energy * 1e6f;  // Convert to μJ
}

/**
 * Enable/disable charging
 */
void bq25570_enable_charging(bool enable) {
    uint8_t pwr_opt = bq25570_read_reg(BQ25570_REG_PWR_OPT);

    if (enable) {
        pwr_opt |= 0x80;  // Enable charging
    } else {
        pwr_opt &= ~0x80;  // Disable charging
    }

    bq25570_write_reg(BQ25570_REG_PWR_OPT, pwr_opt);
}

/**
 * Get power good status
 */
bool bq25570_is_power_good(void) {
    uint8_t status = bq25570_read_reg(BQ25570_REG_TEMP_STATUS);

    return (status & 0x01) != 0;  // Check PGOOD bit
}

/**
 * Write register via I2C
 */
void bq25570_write_reg(uint8_t reg_addr, uint8_t value) {
    uint8_t data[2] = {reg_addr, value};

    I2CMasterDataPut(I2C0_BASE, BQ25570_I2C_ADDR << 1);
    I2CMasterCommand(I2C0_BASE, I2C_MASTER_CMD_BURST_SEND_START);
    I2CMasterDataPut(I2C0_BASE, reg_addr);
    I2CMasterCommand(I2C0_BASE, I2C_MASTER_CMD_BURST_SEND_CONT);
    I2CMasterDataPut(I2C0_BASE, value);
    I2CMasterCommand(I2C0_BASE, I2C_MASTER_CMD_BURST_SEND_FINISH);
}

/**
 * Read register via I2C
 */
uint8_t bq25570_read_reg(uint8_t reg_addr) {
    I2CMasterDataPut(I2C0_BASE, (BQ25570_I2C_ADDR << 1) | 0x00);
    I2CMasterCommand(I2C0_BASE, I2C_MASTER_CMD_BURST_SEND_START);
    I2CMasterDataPut(I2C0_BASE, reg_addr);
    I2CMasterCommand(I2C0_BASE, I2C_MASTER_CMD_BURST_SEND_CONT);

    I2CMasterDataPut(I2C0_BASE, (BQ25570_I2C_ADDR << 1) | 0x01);
    I2CMasterCommand(I2C0_BASE, I2C_MASTER_CMD_BURST_SEND_START);
    uint8_t value = I2CMasterDataGet(I2C0_BASE);
    I2CMasterCommand(I2C0_BASE, I2C_MASTER_CMD_BURST_SEND_FINISH);

    return value;
}

/**
 * Read ADC channel (SPI)
 */
uint16_t spi_read_adc_channel(uint8_t channel) {
    // Simplified - assumes MCP3008 ADC on SPI
    // In practice, configure SPI0 and read ADC

    static uint16_t mock_adc_value = 0;
    mock_adc_value += 100;  // Mock increment

    return mock_adc_value & 0xFFF;  // 12-bit value
}

/**
 * Get energy harvesting status
 */
bq25570_status_t bq25570_get_status(void) {
    bq25570_status_t status = {0};

    uint8_t vbat_stat = bq25570_read_reg(BQ25570_REG_VBAT_STATUS);
    uint8_t istor_stat = bq25570_read_reg(BQ25570_REG_I_STOR_STATUS);
    uint8_t vstor_stat = bq25570_read_reg(BQ25570_REG_V_STOR_STATUS);

    status.vbat_ok = (vbat_stat & 0x01) != 0;
    status.charging = (istor_stat & 0x80) != 0;
    status.vstor_ok = (vstor_stat & 0x01) != 0;
    status.ovp = (vstor_stat & 0x10) != 0;

    return status;
}
