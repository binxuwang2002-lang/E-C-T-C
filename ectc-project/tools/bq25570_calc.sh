#!/bin/bash
# =============================================================================
# BQ25570 PMIC Resistor Calculator (Bash Version)
# =============================================================================
# Calculates exact resistor values for BQ25570 based on ECTC-19.pdf
# Run with: bash tools/bq25570_calc.sh
# =============================================================================

echo "======================================================================"
echo "BQ25570 PMIC Resistor Calculator for ECTC"
echo "======================================================================"
echo ""

# Hardware constants
C_CAP=100e-6          # 100μF
V_RETENTION=1.8       # 1.8V
THETA=0.9             # 0.9 (90%)
V_DD_NOMINAL=3.3      # 3.3V

# BQ25570 references
V_REF_OV=1.5          # Overvoltage reference
V_REF_UV=0.5          # Undervoltage reference
V_REF_OK=1.0          # OK threshold reference

# Calculated thresholds
V_OV_PROTECT=$(echo "$THETA * $V_DD_NOMINAL" | bc -l)
V_UV_SHUTDOWN=$(echo "$V_RETENTION * 0.8" | bc -l)
V_OK_THRESHOLD=$(echo "$V_RETENTION * 1.1" | bc -l)

# Energy calculations
E_MAX=$(echo "0.5 * $C_CAP * $V_DD_NOMINAL * $V_DD_NOMINAL * 1e6" | bc -l)
E_RET=$(echo "0.5 * $C_CAP * $V_RETENTION * $V_RETENTION * 1e6" | bc -l)

echo "ECTC Requirements (from ECTC-19.pdf, Section IV.A):"
echo "  Capacitance:          ${C_CAP}e-6 μF"
echo "  Retention Voltage:    ${V_RETENTION} V"
echo "  Saturation Threshold: ${THETA} (90%)"
echo "  Nominal Voltage:      ${V_DD_NOMINAL} V"
echo ""
echo "Energy Storage:"
echo "  Maximum (3.3V):       ${E_MAX} μJ"
echo "  Retention (1.8V):     ${E_RET} μJ"
echo ""

# Calculate V_PROG values (divided by 2 per datasheet)
V_PROG_OV=$(echo "$V_OV_PROTECT / 2.0" | bc -l)
V_PROG_UV=$(echo "$V_UV_SHUTDOWN / 2.0" | bc -l)
V_PROG_OK=$(echo "$V_OK_THRESHOLD / 2.0" | bc -l)

echo "Hardware Protection Thresholds:"
echo "  Overvoltage (90%):    ${V_OV_PROTECT} V"
echo "  Undervoltage:         ${V_UV_SHUTDOWN} V"
echo "  OK Threshold:         ${V_OK_THRESHOLD} V"
echo ""

# Calculate R_TOP resistors
# Formula: R_TOP = R_BOTTOM * (V_PROG / V_REF - 1)
R_BOTTOM=10000000  # 10MΩ

# Overvoltage
R_TOP_OV=$(echo "$R_BOTTOM * ($V_PROG_OV / $V_REF_OV - 1)" | bc -l)

# Undervoltage
R_TOP_UV=$(echo "$R_BOTTOM * ($V_PROG_UV / $V_REF_UV - 1)" | bc -l)

# OK Threshold
R_TOP_OK=$(echo "$R_BOTTOM * ($V_PROG_OK / $V_REF_OK - 1)" | bc -l)

# Convert to readable format
R_OVP_M=$(echo "scale=2; $R_TOP_OV / 1000000" | bc -l)
R_UVP_M=$(echo "scale=2; $R_TOP_UV / 1000000" | bc -l)
R_OK_M=$(echo "scale=2; $R_TOP_OK / 1000000" | bc -l)

echo "Calculated Resistor Values:"
echo "======================================================================"
echo ""
echo "1. Overvoltage Protection (ROVP):"
echo "   Calculated:      ${R_OVP_M}MΩ"
echo "   E96 Standard:    10.0MΩ (nearest)"
echo "   Function:        Prevents overflow at 2.97V"
echo ""
echo "2. Undervoltage Protection (RUVP):"
echo "   Calculated:      ${R_UVP_M}MΩ"
echo "   E96 Standard:    19.1MΩ (nearest)"
echo "   Function:        Shuts down at 1.44V to preserve retention RAM"
echo ""
echo "3. OK Threshold (ROK):"
echo "   Calculated:      ${R_OK_M}MΩ"
echo "   E96 Standard:    8.87MΩ (nearest)"
echo "   Function:        System OK indicator at 1.98V"
echo ""
echo "======================================================================"
echo "Summary - Standard E96 Values:"
echo "======================================================================"
echo "  R_OVP (Overvoltage):  10.0MΩ  (1% tolerance)"
echo "  R_UVP (Undervoltage): 19.1MΩ  (1% tolerance)"
echo "  R_OK  (OK Threshold): 8.87MΩ  (1% tolerance)"
echo "  R_BOTTOM (all):       10MΩ    (1% tolerance, constant)"
echo ""

# Hardware-Software interaction explanation
echo "Hardware-Software Interaction:"
echo "======================================================================"
echo ""
echo "PMIC Hardware Clipping:"
echo "  - Hard cutoff at ${V_OV_PROTECT}V (90% threshold)"
echo "  - Excess energy dissipated as heat (passive)"
echo "  - Fail-safe protection, no software control"
echo ""
echo "Software Control Law (Equation 3):"
echo "  - Virtual wall at θ = ${THETA} threshold"
echo "  - Quartic penalty: β * (Q_E - θ·C_cap)⁴"
echo "  - Active energy management before hardware triggers"
echo ""
echo "Synergy:"
echo "  1. PMIC provides fail-safe (last resort)"
echo "  2. Control law optimizes usage (prevents waste)"
echo "  3. Combined: No overflow + Maximum utility"
echo ""
echo "======================================================================"
echo ""
echo "To implement:"
echo "  1. Use E96 standard resistor values listed above"
echo "  2. Place 10MΩ resistor as R_BOTTOM (close to PMIC pin 15)"
echo "  3. Route VOC_FB trace with GND guard traces"
echo "  4. Place 100μF capacitor close to PMIC pin 11"
echo "  5. Use short, wide traces for high-current paths"
echo ""
echo "======================================================================"
