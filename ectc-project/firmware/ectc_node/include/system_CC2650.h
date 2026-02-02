/*
** ###################################################################
**     Processors:          CC2650F128RGZ
**                          CC2650F128RSM
**                          CC2650F128RHBR
**
**     Compiler:            GNU C Compiler
**     Reference manual:    CC26xx, CC13xx Simplified Technical Reference
**                          Texas Instruments, 2016
**
**     Version:             rev. 1.2, 2016-04-29
**     Build:               b180801
**
**     Abstract:
**         Provides a system configuration function and a global variable that
**         contains the system frequency. It configures the device and initializes
**         the oscillator (PLL) that is part of the microcontroller device.
**
**     Copyright 1997-2016 Freescale Semiconductor, Inc.
**     Copyright 2016-2018 NXP
**
**     SPDX-License-Identifier: BSD-3-Clause
**
**     Redistribution and use in source and binary forms, with or without
**     modification, are permitted provided that the following conditions are met:
**
**     o Redistributions of source code must retain the above copyright
**       notice, this list of conditions and the following disclaimer.
**
**     o Redistributions in binary form must reproduce the above copyright
**       notice, this list of conditions and the following disclaimer in the
**       documentation and/or other materials provided with the distribution.
**
**     o Neither the name of the NXP Semiconductor, Inc. nor the names
**       of its contributors may be used to endorse or promote products
**       derived from this software without specific prior written permission.
**
**     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
**     "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
**     LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
**     FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
**     COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
**     INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
**     BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
**     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
**     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
**     LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
**     ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
**     POSSIBILITY OF SUCH DAMAGE.
**
**     mail: support@nxp.com
**
**     Revisions:
**     - rev. 1.0 (2016-08-12)
**         Initial version.
**     - rev. 1.1 (2016-11-25)
**         Update CANFD and Classic CAN register.
**         Add MAC TIMERSTAMP
**     - rev. 1.2 (2016-04-29)
**         Remove RTC_CTRL[RTC_OSC_BYPASS].
**
** ###################################################################
*/

/*!
 * @file CC2650
 * @version 1.2
 * @date 2016-04-29
 * @brief Device specific configuration file for CC2650 (header file)
 *
 * Provides a system configuration function and a global variable that contains
 * the system frequency. It configures the device and initializes the oscillator
 * (PLL) that is part of the microcontroller device.
 */

#ifndef _SYSTEM_CC2650_H_
#define _SYSTEM_CC2650_H_                          /**< Symbol preventing repeated inclusion */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifndef _UINT64_T
#define _UINT64_T
typedef unsigned long long uint64_t;       /**< Defines unsigned 64 bit type */
#endif

#ifndef _UINT32_T
#define _UINT32_T
typedef unsigned int uint32_t;             /**< Defines unsigned 32 bit type */
#endif

#ifndef _UINT16_T
#define _UINT16_T
typedef unsigned short uint16_t;           /**< Defines unsigned 16 bit type */
#endif

#ifndef _UINT8_T
#define _UINT8_T
typedef unsigned char uint8_t;             /**< Defines unsigned 8 bit type */
#endif

#ifndef _INT32_T
#define _INT32_T
typedef signed int int32_t;                /**< Defines signed 32 bit type */
#endif

#ifndef _INT16_T
#define _INT16_T
typedef signed short int16_t;              /**< Defines signed 16 bit type */
#endif

#ifndef _INT8_T
#define _INT8_T
typedef signed char int8_t;                /**< Defines signed 8 bit type */
#endif

#ifndef _UINT64_T
#define _UINT64_T
typedef unsigned long long uint64_t;       /**< Defines unsigned 64 bit type */
#endif

/** \\brief Exception / Interrupt Handler Function Prototype */
typedef void(*VECTOR_TABLE_Type)(void);

/** \\brief System Clock Frequency (Core Clock) */
extern uint32_t SystemCoreClock;

/** \\brief Setup the microcontroller system.

    Initialize the System and update the SystemCoreClock variable.
 */
void SystemInit (void);

/** \\brief Updates the SystemCoreClock variable.

    It must be called whenever the core clock is changed during program
    execution. SystemCoreClockUpdate() evaluates the clock register settings and calculates
    the current core clock.

 */
void SystemCoreClockUpdate (void);

/** \\brief SystemInit function hook.

    This weak function allows to call specific initialization code during the
    SystemInit() execution.This can be used when an application specific code needs
    to be called as part of the SystemInit() initialization and not by using global variables.

    Note: The default weak implementation of this function is empty. If you want to
    override this function, you can define your own implementation in your application
    or use a different compile-time switch if available.
 */
extern void SystemInitHook (void);

#ifdef __cplusplus
}
#endif

#endif  /* _SYSTEM_CC2650_H_ */
