/*
** ###################################################################
**     Processors:          STM32U575ZIT6Q
**                          STM32U575AIG6Q
**                          STM32U575RIT6Q
**
**     Compiler:            GNU C Compiler / ARM CC / IAR EWARM
**     Reference manual:    STM32U575/585 Reference Manual RM0456
**                          STMicroelectronics, 2023
**
**     Version:             rev. 1.0, 2024-06-15
**     Build:               ECTC-compatible build
**
**     Abstract:
**         STM32U575 power management driver for ECTC battery-free systems.
**         Provides low-power mode control and voltage monitoring functions
**         compatible with the existing CC2650 framework.
**
**     ECTC Integration:
**         - Compatible with system_CC2650.h interface
**         - Supports FEMP 2.0 energy model parameters
**         - Retention voltage (V_ret): 1.8V
**         - Rated voltage (V_rated): 3.3V
**         - C_bus parasitic: 20.0 pF (critical for energy estimation)
**
**     Copyright 2024 ECTC Research Team
**     SPDX-License-Identifier: BSD-3-Clause
**
** ###################################################################
*/

#ifndef _STM32_POWER_H_
#define _STM32_POWER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* ============================================================================
   STM32U575 Hardware Constants (from ECTC Paper Table II)
   ============================================================================ */

/** @brief Retention voltage in millivolts */
#define STM32_V_RET_MV              1800U

/** @brief Rated operating voltage in millivolts */
#define STM32_V_RATED_MV            3300U

/** @brief Minimum operating voltage in millivolts */
#define STM32_V_MIN_MV              1710U

/** @brief Maximum operating voltage in millivolts */
#define STM32_V_MAX_MV              3600U

/** @brief Active current per MHz in nanoamperes */
#define STM32_ACTIVE_CURRENT_NA_PER_MHZ  19000U

/** @brief Leakage current in deep sleep (nA) */
#define STM32_LEAKAGE_CURRENT_NA    150U

/** @brief Bus parasitic capacitance in femtofarads (20.0 pF = 20000 fF) 
    @note CRITICAL: Ignoring this causes 4.6x energy estimation error per ECTC paper */
#define STM32_C_BUS_FF              20000U

/** @brief Capacitance derating factor (0.8 = 80%) */
#define STM32_DERATING_FACTOR       80U

/** @brief Maximum clock frequency in MHz */
#define STM32_MAX_CLOCK_MHZ         160U

/* ============================================================================
   HAL-Compatible Macro Definitions
   ============================================================================ */

/** @brief Enable PWR clock */
#define __HAL_RCC_PWR_CLK_ENABLE()  \
    do { \
        __IO uint32_t tmpreg; \
        SET_BIT(RCC->AHB3ENR, RCC_AHB3ENR_PWREN); \
        tmpreg = READ_BIT(RCC->AHB3ENR, RCC_AHB3ENR_PWREN); \
        (void)tmpreg; \
    } while(0)

/** @brief Disable PWR clock */
#define __HAL_RCC_PWR_CLK_DISABLE() \
    CLEAR_BIT(RCC->AHB3ENR, RCC_AHB3ENR_PWREN)

/** @brief Enable low-power regulator in STOP mode */
#define __HAL_PWR_LOWPOWERREGULATOR_ENABLE() \
    SET_BIT(PWR->CR1, PWR_CR1_LPR)

/** @brief Disable low-power regulator */
#define __HAL_PWR_LOWPOWERREGULATOR_DISABLE() \
    CLEAR_BIT(PWR->CR1, PWR_CR1_LPR)

/** @brief Check if VDD voltage is below PVD threshold */
#define __HAL_PWR_GET_FLAG(__FLAG__) \
    ((PWR->SR1 & (__FLAG__)) == (__FLAG__))

/** @brief Clear PWR pending flags */
#define __HAL_PWR_CLEAR_FLAG(__FLAG__) \
    SET_BIT(PWR->SCR, (__FLAG__))

/* ============================================================================
   Low Power Mode Definitions
   ============================================================================ */

/** @brief Low power mode enumeration */
typedef enum {
    STM32_LPMODE_SLEEP     = 0x00U,  /**< Sleep mode - CPU stopped, peripherals running */
    STM32_LPMODE_STOP0     = 0x01U,  /**< Stop 0 - All clocks stopped, main regulator ON */
    STM32_LPMODE_STOP1     = 0x02U,  /**< Stop 1 - All clocks stopped, LP regulator ON */
    STM32_LPMODE_STOP2     = 0x03U,  /**< Stop 2 - Ultra low power, limited wakeup */
    STM32_LPMODE_STOP3     = 0x04U,  /**< Stop 3 - Lowest power with SRAM retention */
    STM32_LPMODE_STANDBY   = 0x05U,  /**< Standby - Main regulator OFF, backup domain ON */
    STM32_LPMODE_SHUTDOWN  = 0x06U   /**< Shutdown - All power OFF except backup domain */
} STM32_LowPowerMode_t;

/** @brief Power voltage detector level enumeration */
typedef enum {
    STM32_PVD_LEVEL_0 = 0x00U,  /**< PVD threshold around 2.0V */
    STM32_PVD_LEVEL_1 = 0x01U,  /**< PVD threshold around 2.2V */
    STM32_PVD_LEVEL_2 = 0x02U,  /**< PVD threshold around 2.4V */
    STM32_PVD_LEVEL_3 = 0x03U,  /**< PVD threshold around 2.5V */
    STM32_PVD_LEVEL_4 = 0x04U,  /**< PVD threshold around 2.6V */
    STM32_PVD_LEVEL_5 = 0x05U,  /**< PVD threshold around 2.8V */
    STM32_PVD_LEVEL_6 = 0x06U,  /**< PVD threshold around 2.9V */
    STM32_PVD_LEVEL_7 = 0x07U   /**< PVD threshold external input */
} STM32_PVDLevel_t;

/** @brief Voltage monitoring status */
typedef struct {
    uint16_t vdd_mv;           /**< Current VDD voltage in millivolts */
    uint16_t vbat_mv;          /**< Current VBAT voltage in millivolts */
    uint8_t  pvd_triggered;    /**< PVD threshold crossed flag */
    uint8_t  brownout_risk;    /**< Brownout risk assessment (0-100%) */
} STM32_VoltageStatus_t;

/* ============================================================================
   Function Prototypes - Compatible with system_CC2650.h interface
   ============================================================================ */

/**
 * @brief Get recharge voltage threshold for battery-free operation.
 * 
 * Returns the voltage level at which the node should begin recharging
 * its storage capacitor. This is based on the ECTC energy model.
 * 
 * @return Recharge voltage threshold in millivolts
 * 
 * @note Compatible with CC2650_GetRechargeVoltage() interface
 */
uint16_t STM32_GetRechargeVoltage(void);

/**
 * @brief Enter specified low-power mode.
 * 
 * Configures the STM32U575 for energy-efficient sleep states.
 * Supports various STOP modes for battery-free operation.
 * 
 * @param mode  Low power mode to enter (STM32_LowPowerMode_t)
 * @return 0 on success, error code on failure
 * 
 * @note Compatible with CC2650_EnterLowPowerMode() interface
 */
int32_t STM32_EnterLowPowerMode(STM32_LowPowerMode_t mode);

/**
 * @brief Get current voltage status.
 * 
 * Reads ADC to measure current supply voltages and assess
 * brownout risk for battery-free operation.
 * 
 * @param status  Pointer to voltage status structure
 * @return 0 on success, error code on failure
 */
int32_t STM32_GetVoltageStatus(STM32_VoltageStatus_t *status);

/**
 * @brief Configure Power Voltage Detector (PVD).
 * 
 * Sets up PVD to trigger interrupt when VDD falls below threshold.
 * Essential for battery-free operation to detect energy depletion.
 * 
 * @param level  PVD threshold level
 * @return 0 on success, error code on failure
 */
int32_t STM32_ConfigurePVD(STM32_PVDLevel_t level);

/**
 * @brief Estimate energy consumption for operation.
 * 
 * Uses FEMP 2.0 model parameters including C_bus parasitic
 * capacitance to estimate energy for a given operation.
 * 
 * @param clock_mhz    Operating clock frequency in MHz
 * @param duration_us  Operation duration in microseconds
 * @return Estimated energy consumption in nanojoules
 * 
 * @note Includes C_bus (20pF) parasitic overhead
 */
uint32_t STM32_EstimateEnergy_nJ(uint32_t clock_mhz, uint32_t duration_us);

/**
 * @brief Check if energy is sufficient for operation.
 * 
 * Performs feasibility check based on current voltage and
 * estimated energy consumption using FEMP 2.0 model.
 * 
 * @param required_energy_nJ  Required energy in nanojoules
 * @return 1 if sufficient energy, 0 otherwise
 */
uint8_t STM32_CheckEnergyFeasibility(uint32_t required_energy_nJ);

/**
 * @brief Initialize STM32U575 power management.
 * 
 * Configures power rails, PVD, and low-power mode settings
 * for ECTC battery-free operation.
 * 
 * @return 0 on success, error code on failure
 */
int32_t STM32_PowerInit(void);

/**
 * @brief Get parasitic capacitance for energy calculations.
 * 
 * Returns C_bus value with derating factor applied.
 * CRITICAL: This must be used in all energy calculations.
 * 
 * @return Effective C_bus in femtofarads
 */
static inline uint32_t STM32_GetEffectiveCbus_fF(void) {
    return (STM32_C_BUS_FF * STM32_DERATING_FACTOR) / 100U;
}

/**
 * @brief Calculate dynamic power component.
 * 
 * P_dyn = alpha * C_bus * V_dd^2 * f_clk
 * 
 * @param alpha_percent  Activity factor (0-100)
 * @param v_dd_mv        Supply voltage in millivolts
 * @param f_clk_mhz      Clock frequency in MHz
 * @return Dynamic power in microwatts
 */
uint32_t STM32_CalculateDynamicPower_uW(uint8_t alpha_percent, 
                                         uint16_t v_dd_mv, 
                                         uint32_t f_clk_mhz);

#ifdef __cplusplus
}
#endif

#endif /* _STM32_POWER_H_ */
