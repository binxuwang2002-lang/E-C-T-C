# ECTC Hardware Schematic (KiCad Text Description)
# This file describes the complete ECTC sensor node schematic

# CC2650STK Modified Board
# ========================

# Power Supply
U1 (CC2650) { VDD, GND, DCOUPL, VDDS, VDDS2, VDDS_DCDC }
  VDD: 3.3V from BQ25570

# BQ25570 Energy Harvesting PMIC
U2 (BQ25570) { VIN_DC+, VIN_DC-, VSTOR, GND, VOUT }
  VIN_DC+: Connected to solar panel positive
  VIN_DC-: Connected to solar panel negative
  VSTOR: Storage capacitor (100μF, 6.3V)
  VOUT: 3.3V regulator output

# Energy Storage
C1 (100μF, 6.3V, X7R, 0805)
  Connected between VSTOR and GND
  Polarity: + to VSTOR, - to GND

# Voltage Divider for ADC
R1 (10kΩ, 1%, 0805)
  From VSTOR to ADC_INPUT (DIO_0)

R2 (39kΩ, 1%, 0805)
  From ADC_INPUT to GND

# Solar Panel Connector
J1 (2-pin header, 2.54mm pitch)
  Pin 1: V_IN+
  Pin 2: V_IN-

# BQ25570 Configuration
# MPPT: 2.4V (71% of Voc for solar)
R_MPPT1 (71kΩ, 1%)
R_MPPT2 (29k, 1%)
  Connected to BQ25570 MPPT pin

# Gate Control Signals
# (if using external control)
J2 (4-pin header, 2.54mm)
  Pin 1: VSTOR_SENSE (to ADC)
  Pin 2: CHG_STAT (status output)
  Pin 3: PGOOD (power good)
  Pin 4: GND

# RF Energy Harvesting (Optional)
# If using RF energy source:
ANT1 (SMA connector)
  Connected to matching network
  Matching network: L1 (18nH) + C2 (1.5pF)
  Input to BQ25570 VIN_DC

# Radio Module (CC1352P1 for gateway)
U3 (CC1352P1 LaunchPad)
  Connected via USB-UART to Raspberry Pi
  IEEE 802.15.4 coordinator
  Channel 11 (2.405 GHz)
  PAN ID: 0x1234

# Connection Table
# ================
# CC2650STK Modifications:
# 1. Remove CR2032 battery holder
# 2. Cut trace from battery holder VDD to CC2650 VDD
# 3. Wire BQ25570 VOUT to CC2650 VDD
# 4. Wire CC2650 GND to BQ25570 GND
# 5. Connect DIO_0 to voltage divider (ADC_INPUT)

# Bill of Materials (BOM)
# =======================
# CC2650STK           x1    $25
# BQ25570EVM-206      x1    $100
# 100μF Capacitor     x1    $1
# 10kΩ Resistor       x1    $0.10
# 39kΩ Resistor       x1    $0.10
# Solar Panel (2V, 200mA)  x1    $10
# J1, J2 Headers      x2    $1
# Enclosure           x1    $5

# Total Component Cost: ~$142

# PCB Layout Notes
# ================
# 1. Keep solar panel connection as short as possible (<5cm)
# 2. Place 100μF capacitor close to BQ25570 VSTOR pin
# 3. Use ground plane on layer 2
# 4. Route 3.3V power with 0.5mm trace width
# 5. Place test points for VSTOR and VOUT
# 6. Add silkscreen labels for all connectors

# Assembly Instructions
# =====================
# 1. Solder BQ25570EVM-206 to PCB
# 2. Mount 100μF capacitor (observe polarity)
# 3. Solder voltage divider resistors
# 4. Install headers J1 and J2
# 5. Modify CC2650STK (remove battery, wire connections)
# 6. Connect BQ25570 output to CC2650 VDD
# 7. Connect solar panel to J1
# 8. Test: Verify 3.3V at CC2650 VDD pin

# Safety Warnings
# ===============
# - Do not exceed 4.5V on VSTOR pin
# - Ensure correct capacitor polarity
# - Use ESD-safe handling
# - Test with multimeter before powering CC2650
# - Monitor capacitor voltage during operation

# Notes for KiCad Import
# =======================
# Create this schematic in KiCad:
# 1. Create new schematic
# 2. Add symbols from library
# 3. Connect components as described
# 4. Add global labels and power symbols
# 5. Add title block with ECTC information
# 6. Run ERC (Electrical Rules Check)
# 7. Assign footprints
# 8. Generate netlist for PCB layout
