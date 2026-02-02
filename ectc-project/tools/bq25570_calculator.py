#!/usr/bin/env python3
"""
BQ25570 PMIC Resistor Calculation Script
========================================

Calculates exact resistor values for BQ25570 energy harvesting PMIC
based on ECTC-19.pdf requirements.

Usage: python3 bq25570_calculator.py

Output: Resistor values for Overvoltage, Undervoltage, and OK thresholds
"""

import math

# =============================================================================
# Hardware Constants (from ECTC-19.pdf)
# =============================================================================

C_CAP = 100e-6  # 100μF capacitor
V_RETENTION = 1.8  # Retention voltage
THETA = 0.9  # Saturation threshold
V_DD_NOMINAL = 3.3  # Nominal supply voltage

# BQ25570 Internal References (from datasheet)
V_REF_OV = 1.5  # Overvoltage reference
V_REF_UV = 0.5  # Undervoltage reference
V_REF_OK = 1.0  # OK threshold reference

# =============================================================================
# Calculated Thresholds
# =============================================================================

V_OV_PROTECT = THETA * V_DD_NOMINAL  # 2.97V (90% saturation)
V_UV_SHUTDOWN = V_RETENTION * 0.8  # 1.44V (80% retention)
V_OK_THRESHOLD = V_RETENTION * 1.1  # 1.98V (110% retention)

# Energy calculations
def calc_energy(capacitance_f, voltage_v):
    """Calculate energy in capacitor: E = 0.5 * C * V^2"""
    return 0.5 * capacitance_f * voltage_v**2

def voltage_to_energy_uj(voltage_v):
    """Convert voltage to energy for 100μF capacitor in μJ"""
    # For C = 100μF: E = 0.5 * 100e-6 * V^2 = 50 * V^2 μJ
    return 50.0 * voltage_v**2

def energy_to_voltage(energy_j, capacitance_f):
    """Convert energy to voltage: V = sqrt(2*E/C)"""
    return math.sqrt(2 * energy_j / capacitance_f)

# =============================================================================
# Resistor Calculation
# =============================================================================

def calc_r_top(v_threshold, v_reference, r_bottom=10e6):
    """
    Calculate R_TOP resistor for BQ25570 divider

    Formula: R_TOP = R_BOTTOM * (V_THRESHOLD / V_REF - 1)

    Args:
        v_threshold: Desired threshold voltage (V)
        v_reference: PMIC reference voltage (V)
        r_bottom: Resistor to ground (Ω), default 10MΩ

    Returns:
        R_TOP value in Ω
    """
    return r_bottom * (v_threshold / v_reference - 1.0)

def calculate_rovp():
    """Calculate Overvoltage Protection resistor"""
    # V_PROG_OV = V_OV / 2 (from BQ25570 datasheet)
    v_prog_ov = V_OV_PROTECT / 2.0
    r_top = calc_r_top(v_prog_ov, V_REF_OV)
    return r_top

def calculate_ruvp():
    """Calculate Undervoltage Protection resistor"""
    # V_PROG_UV = V_UV / 2 (from BQ25570 datasheet)
    v_prog_uv = V_UV_SHUTDOWN / 2.0
    r_top = calc_r_top(v_prog_uv, V_REF_UV)
    return r_top

def calculate_rok():
    """Calculate OK Threshold resistor"""
    # V_PROG_OK = V_OK / 2 (from BQ25570 datasheet)
    v_prog_ok = V_OK_THRESHOLD / 2.0
    r_top = calc_r_top(v_prog_ok, V_REF_OK)
    return r_top

# =============================================================================
# Standard Resistor Values (E96 Series - 1% tolerance)
# =============================================================================

E96_SERIES = [
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
]

def find_e96_standard(value_ohm):
    """
    Find nearest E96 standard resistor value

    Args:
        value_ohm: Calculated resistance in Ω

    Returns:
        Nearest E96 standard value in Ω
    """
    # Determine decade and mantissa
    decade = math.floor(math.log10(value_ohm))
    mantissa = value_ohm / (10**decade)

    # Find closest mantissa in E96
    best_match = E96_SERIES[0]
    min_error = abs(mantissa - E96_SERIES[0])

    for e96_val in E96_SERIES:
        error = abs(mantissa - e96_val)
        if error < min_error:
            min_error = error
            best_match = e96_val

    return best_match * (10**decade)

def format_resistor(ohm_value):
    """Format resistor value in human-readable format"""
    if ohm_value >= 1e6:
        return f"{ohm_value/1e6:.2f}MΩ"
    elif ohm_value >= 1e3:
        return f"{ohm_value/1e3:.2f}kΩ"
    else:
        return f"{ohm_value:.2f}Ω"

# =============================================================================
# Main Calculation and Display
# =============================================================================

def main():
    print("=" * 70)
    print("BQ25570 PMIC Resistor Calculator for ECTC")
    print("=" * 70)
    print()

    # Display requirements
    print("ECTC Requirements (from ECTC-19.pdf, Section IV.A):")
    print(f"  Capacitance:          {C_CAP*1e6:.0f}μF")
    print(f"  Retention Voltage:    {V_RETENTION:.2f}V")
    print(f"  Saturation Threshold: {THETA:.1f} (90%)")
    print(f"  Nominal Voltage:      {V_DD_NOMINAL:.2f}V")
    print()

    # Calculate energy
    e_max = calc_energy(C_CAP, V_DD_NOMINAL)
    e_ret = calc_energy(C_CAP, V_RETENTION)
    print(f"Energy Storage:")
    print(f"  Maximum (3.3V):       {e_max*1e6:.2f}μJ")
    print(f"  Retention (1.8V):     {e_ret*1e6:.2f}μJ")
    print(f"  Headroom:             {(e_max-e_ret)*1e6:.2f}μJ")
    print()

    # Calculate thresholds
    print("Hardware Protection Thresholds:")
    print(f"  Overvoltage (90%):    {V_OV_PROTECT:.2f}V  ({voltage_to_energy_uj(V_OV_PROTECT):.2f}μJ)")
    print(f"  Undervoltage:         {V_UV_SHUTDOWN:.2f}V  ({voltage_to_energy_uj(V_UV_SHUTDOWN):.2f}μJ)")
    print(f"  OK Threshold:         {V_OK_THRESHOLD:.2f}V  ({voltage_to_energy_uj(V_OK_THRESHOLD):.2f}μJ)")
    print()

    # Calculate resistors
    print("Calculated Resistor Values:")
    print("-" * 70)

    # Overvoltage Protection
    rovp_calc = calculate_rovp()
    rovp_std = find_e96_standard(rovp_calc)
    print(f"\n1. Overvoltage Protection (ROVP):")
    print(f"   Calculated:      {format_resistor(rovp_calc)} ({rovp_calc:.2f}Ω)")
    print(f"   E96 Standard:    {format_resistor(rovp_std)}")
    print(f"   Actual V_OV:     {2.0 * V_REF_OV * (1 + rovp_std/10e6):.3f}V")
    print(f"   Function:        Prevents overflow at {V_OV_PROTECT:.2f}V")

    # Undervoltage Protection
    ruvp_calc = calculate_ruvp()
    ruvp_std = find_e96_standard(ruvp_calc)
    print(f"\n2. Undervoltage Protection (RUVP):")
    print(f"   Calculated:      {format_resistor(ruvp_calc)} ({ruvp_calc:.2f}Ω)")
    print(f"   E96 Standard:    {format_resistor(ruvp_std)}")
    print(f"   Actual V_UV:     {2.0 * V_REF_UV * (1 + ruvp_std/10e6):.3f}V")
    print(f"   Function:        Shuts down at {V_UV_SHUTDOWN:.2f}V to preserve retention RAM")

    # OK Threshold
    rok_calc = calculate_rok()
    rok_std = find_e96_standard(rok_calc)
    print(f"\n3. OK Threshold (ROK):")
    print(f"   Calculated:      {format_resistor(rok_calc)} ({rok_calc:.2f}Ω)")
    print(f"   E96 Standard:    {format_resistor(rok_std)}")
    print(f"   Actual V_OK:     {2.0 * V_REF_OK * (1 + rok_std/10e6):.3f}V")
    print(f"   Function:        System OK indicator at {V_OK_THRESHOLD:.2f}V")

    # Summary
    print("\n" + "=" * 70)
    print("Summary - Standard E96 Values:")
    print("=" * 70)
    print(f"  R_OVP (Overvoltage):  {format_resistor(rovp_std):>12}  (1% tolerance)")
    print(f"  R_UVP (Undervoltage): {format_resistor(ruvp_std):>12}  (1% tolerance)")
    print(f"  R_OK  (OK Threshold): {format_resistor(rok_std):>12}  (1% tolerance)")
    print(f"  R_BOTTOM (all):       10MΩ ±1% (constant)")
    print()

    # PCB Layout Notes
    print("PCB Layout Recommendations:")
    print("-" * 70)
    print("  1. Place R_BOTTOM (10MΩ) close to BQ25570 PMIC pin 15 (VOC_FB)")
    print("  2. Use 0402 or 0603 resistors for minimal parasitic capacitance")
    print("  3. Route VOC_FB trace with guard traces (GND) to minimize noise")
    print("  4. Place 100μF capacitor as close as possible to PMIC pin 11 (VBAT_SENSE)")
    print("  5. Use short, wide traces for high-current paths (capacitor charging)")
    print()

    # Hardware-Software Interaction
    print("Hardware-Software Interaction:")
    print("-" * 70)
    print(f"  PMIC Hardware Clipping:")
    print(f"    - Hard cutoff at {V_OV_PROTECT:.2f}V (90% threshold)")
    print(f"    - Excess energy dissipated as heat (passive)")
    print(f"    - Fail-safe protection, no software control")
    print()
    print(f"  Software Control Law (Equation 3):")
    print(f"    - Virtual wall at θ = {THETA} threshold")
    print(f"    - Quartic penalty: β * (Q_E - θ·C_cap)⁴")
    print(f"    - Active energy management before hardware triggers")
    print()
    print(f"  Synergy:")
    print(f"    1. PMIC provides fail-safe (last resort)")
    print(f"    2. Control law optimizes usage (prevents waste)")
    print(f"    3. Combined: No overflow + Maximum utility")
    print()

    # Tolerance Analysis
    print("Tolerance Analysis (±1% E96 resistors):")
    print("-" * 70)
    rovp_tolerance = 2.0 * V_REF_OV * (1 + (rovp_std*1.01)/10e6) - 2.0 * V_REF_OV * (1 + (rovp_std*0.99)/10e6)
    print(f"  Overvoltage tolerance:  ±{rovp_tolerance:.3f}V ({rovp_tolerance/V_OV_PROTECT*100:.1f}%)")

    ruvp_tolerance = 2.0 * V_REF_UV * (1 + (ruvp_std*1.01)/10e6) - 2.0 * V_REF_UV * (1 + (ruvp_std*0.99)/10e6)
    print(f"  Undervoltage tolerance: ±{ruvp_tolerance:.3f}V ({ruvp_tolerance/V_UV_SHUTDOWN*100:.1f}%)")
    print()

    print("=" * 70)
    print("Calculator complete. Use E96 standard values for PCB assembly.")
    print("=" * 70)

if __name__ == '__main__':
    main()
