# Hardware Setup Guide

## Overview

This guide covers the complete hardware setup for the ECTC battery-free sensor network testbed.

## Hardware Components

### Bill of Materials (BOM)

#### Minimum Testbed (5 Nodes)

| Component | Part Number | Quantity | Unit Price | Total |
|-----------|------------|----------|------------|-------|
| MCU Board | CC2650STK | 5 | $25 | $125 |
| Energy Harvesting PMIC | BQ25570EVM-206 | 5 | $100 | $500 |
| Storage Capacitor | GRM155R71C104KA88D (100μF) | 5 | $1 | $5 |
| Gateway | Raspberry Pi 4B (4GB) | 1 | $75 | $75 |
| 15.4 Coordinator | CC1352P1 LaunchPad | 1 | $30 | $30 |
| Power Monitor | Monsoon HV Power Monitor | 1 | $1500 | $1500 |
| Solar Panel | 2V 200mA Polycrystalline | 5 | $10 | $50 |
| **TOTAL** | | | | **$2,285** |

#### Full Testbed (50 Nodes)

| Component | Part Number | Quantity | Unit Price | Total |
|-----------|------------|----------|------------|-------|
| MCU Board | CC2650STK | 50 | $25 | $1,250 |
| Energy Harvesting PMIC | BQ25570EVM-206 | 50 | $100 | $5,000 |
| Storage Capacitor | GRM155R71C104KA88D | 50 | $1 | $50 |
| Gateway | XPC240400B-02 | 1 | $2500 | $2,500 |
| Mobile Platform | DJI Matrice 100 | 10 | $5000 | $50,000 |
| RF Energy Source | P2110B-EVAL-01 | 10 | $50 | $500 |
| **TOTAL** | | | | **$59,300** |

## Hardware Assembly

### Step 1: Sensor Node Modification

#### 1.1 Remove Battery

**Tools Needed:**
- Soldering iron (temperature-controlled)
- Desoldering wick or pump
- Isopropyl alcohol
- Small Phillips screwdriver

**Steps:**
1. Remove screws securing CR2032 battery holder
2. Desolder battery holder terminals
3. Clean area with isopropyl alcohol

**Warning:** Be careful not to damage PCB traces.

#### 1.2 Install Energy Harvesting Circuit

**Connections:**

```
BQ25570EVM-206
┌────────────────────────────────────┐
│ VIN_DC+ ─── Solar Panel +          │
│ VIN_DC- ─── Solar Panel -          │
│ VSTOR ───┐                         │
│          ├── 100μF Capacitor       │
│ GND ─────┤ (Polarity: + to VSTOR) │
└────────────────────────────────────┘
         │
         ↓
    CC2650STK
    ┌──────────────┐
    │ VDD ──(wire)─┤
    │ GND ──(wire)─┤
    │ DIO_0 ───────┤ (Voltage monitor)
    └──────────────┘
```

**Wiring Guide:**
- Use 22AWG wire for power connections
- Use 30AWG wire-wrap wire for signal connections
- Keep power wires short (< 5cm)
- Twist power wires together

**Schematic:**
```
    Solar Cell (2V, 200mA)
         ↓
    ┌─────────────────┐
    │ BQ25570EVM      │
    │                 │
    │ MPPT: 2.4V      │
    │                 │
    │ VSTOR ─┬── 100μF│
    │        │       │
    │ GND ───┴───────┘
    └────┬────────────┘
         │
         │ VSTOR (2.4-4.5V)
         ↓
    ┌────────────────────┐
    │ CC2650STK          │
    │                    │
    │ VDD ───────────────┤
    │ GND ───────────────┤
    │                    │
    │ DIO_0 (ADC) ───────┤  Voltage Divider
    │                    │  (10kΩ : 39kΩ)
    └────────────────────┘  to 1.35V max
```

#### 1.3 Add Capacitor Voltage Monitor

**Purpose:** Monitor capacitor voltage for energy estimation

**Circuit:**
```
BQ25570 VSTOR ──[10kΩ]──┬──[39kΩ]──→ GND
                         │
                    ADC Input (DIO_0)
```

**Calculation:**
- Max VSTOR: 4.5V
- Divider ratio: 10k/(10k+39k) = 0.204
- Max ADC input: 4.5 × 0.204 = 0.918V
- Safe for CC2650 ADC (1.8V reference)

**Assembly:**
1. Solder 10kΩ resistor (0805) from VSTOR to DIO_0
2. Solder 39kΩ resistor (0805) from DIO_0 to GND
3. Verify with multimeter

### Step 2: Gateway Setup

#### Option A: XPC240400B-02 (Original)

**Configuration:**
1. Connect power (12V adapter)
2. Connect Ethernet to PC/router
3. Configure IP address (default: 192.168.1.100)
4. Install XPC240400B-02 SDK

**Services:**
- Shapley calculation server: Port 8080
- KF-GP data recovery: Port 8081
- ZK verification: Port 8082

#### Option B: Raspberry Pi 4B + CC1352P1 (Alternative)

**Assembly:**
```
Raspberry Pi 4B
┌─────────────────────────┐
│ USB 3.0 Port 0 ── CC1352P1 LaunchPad
│  (IEEE 802.15.4 Radio) │
│                         │
│ USB 3.0 Port 1 ── Optional LoRa Hat
│                         │
│ Ethernet ─── Network    │
│                         │
│ GPIO 18 ─── Monsoon Trig│
└─────────────────────────┘
```

**Software Setup:**
```bash
# Install Ubuntu Server 20.04
# Install dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip git

# Install ECTC
git clone https://github.com/ectc/ectc-project.git
cd ectc-project
./scripts/install.sh

# Start gateway
cd gateway
python3 -m ectc_gateway.main --config config/gateway.yaml
```

### Step 3: Energy Monitoring Setup (Optional)

#### Monsoon HV Power Monitor

**Connections:**
```
Monsoon HV Power Monitor
┌──────────────────────────────────────┐
│ VOUT+ (White) ─── CC2650 VDD         │
│ VOUT- (Black) ─── CC2650 GND         │
│ SENSE+ (Red) ─── Capacitor +         │
│ SENSE- (Blue) ─── Capacitor -         │
└──────────────────────────────────────┘
```

**PowerTool Configuration:**
```
Voltage: 3.3V
Sample Rate: 1,000,000 Hz
Trigger: GPIO Rising Edge
Max Time: 3600s (1 hour)
```

**Python API:**
```python
from monsoon import HVPM

mon = HVPM.Monsoon()
mon.setup_usb()
mon.setVout(3.3)
mon.setSampleRate(1000000)
mon.startSampling()
samples = mon.collectData()
```

### Step 4: Mobile Node Platform

#### DJI Matrice 100

**Assembly:**

```
DJI Matrice 100 Payload Bay
┌────────────────────────────────────────────┐
│ Power Module (12V) ─── Buck Converter ── BQ25570
│                        (12V → 3.3V)       │
│                                              │
│ UART2 ──────────── CC2650                    │
│ (GPS, Telemetry)                            │
│                                              │
│ Current Sensor ── Gateway Connection        │
│ (Monitor payload power)                     │
└────────────────────────────────────────────┘
```

**Buck Converter Design:**
- Input: 12V (from M100)
- Output: 3.3V @ 500mA
- Efficiency: >95%
- Part: TPS62170 (Texas Instruments)

**Schematic:**
```
12V (M100)
  ↓
┌─────────────────────────┐
│ TPS62170                │
│                         │
│ 12V ──[10μH]─┬── 3.3V  │
│             │         │
│ GND ─────────┴─────────┘
│             │
│          [22μF] GND
│
└─ Sense Line to Gateway
```

### Step 5: RF Energy Harvesting (Optional)

#### Powercast P2110B

**Purpose:** Provide RF energy for indoor testing

**Configuration:**
```
Powercast P2110B
┌─────────────────────────┐
│ TX Power: 3W            │
│ Frequency: 915 MHz      │
│                          │
│ Antenna ── Directed     │
│ (toward sensor nodes)    │
│                          │
│ Output ─── Matching Net  │
│ (50Ω to BQ25570 VIN)     │
└─────────────────────────┘
```

**Antenna:**
- Type: Directional Yagi
- Gain: 10 dBi
- Range: 10 meters

### Step 6: Testing and Calibration

#### 6.1 Power-On Test

**Checklist:**
- [ ] Solar panel generates >1V in bright light
- [ ] BQ25570 charges capacitor to >2.5V
- [ ] CC2650 boots successfully (LED blinking)
- [ ] Radio can scan and join network
- [ ] Gateway detects node

#### 6.2 Calibration

**Energy Measurements:**
1. Connect Monsoon to first node
2. Run TinyLSTM inference 100 times
3. Record total energy from Monsoon
4. Compare with hybrid profiler estimate
5. Adjust RTL power model if error >5%

**Calibration Script:**
```bash
cd tools
python3 co_profiler.py \
  --node-id 1 \
  --task tinylstm \
  --duration 100 \
  --monsoon /dev/ttyUSB0 \
  --output calibration_results.json
```

**Expected Results:**
- TinyLSTM: 23.1 μJ ± 0.5 μJ
- Radio TX: 5.3 μJ ± 0.2 μJ
- Total per slot: 35.2 μJ ± 0.8 μJ

#### 6.3 RF Performance Test

**Packet Error Rate:**
```python
# Test script
for distance in [1, 5, 10, 20, 50]:
    nodes = place_nodes(distance)
    packets_sent = 1000
    packets_received = 0

    for i in range(packets_sent):
        send_packet(nodes[0], nodes[1])
        if ack_received():
            packets_received += 1

    per = 1 - (packets_received / packets_sent)
    print(f"Distance: {distance}m, PER: {per:.3f}")
```

**Expected Results:**
- <1m: PER < 0.01
- <10m: PER < 0.05
- <50m: PER < 0.20

## Safety Considerations

### Electrical Safety

1. **ESD Protection:**
   - Wear anti-static wrist strap
   - Use ESD-safe mat
   - Handle components by edges

2. **Power Limits:**
   - Do not exceed 4.5V on capacitor
   - Do not exceed 100mA charging current
   - Monitor temperature (max 85°C)

3. **Firmware Safety:**
   - Implement under-voltage lockout
   - Monitor capacitor voltage continuously
   - Safe shutdown at <1.8V

### Mechanical Safety

1. **Solar Panels:**
   - Secure mounting to prevent falling
   - Sharp edges - use caution
   - UV resistant cables

2. **Drones:**
   - Follow DJI safety guidelines
   - Pilot certification required
   - Test payload attachment thoroughly
   - Never fly indoors without safety nets

## Troubleshooting

### Common Issues

#### Node Not Booting

**Symptoms:**
- No LED activity
- No radio transmission

**Diagnosis:**
1. Check power connections
2. Measure VSTOR voltage (should be >2.5V)
3. Check capacitor polarity
4. Verify BQ25570 configuration

**Solutions:**
- Re-check BQ25570 MPPT settings
- Replace capacitor if polarity reversed
- Verify solar panel connections

#### Low Energy Harvesting

**Symptoms:**
- Capacitor voltage slowly rises
- Frequent node sleep

**Diagnosis:**
1. Measure solar panel voltage
2. Check MPPT settings (should match panel V_oc)
3. Test indoor vs outdoor

**Solutions:**
- Adjust MPPT voltage
- Move to brighter location
- Use supplementary RF energy

#### High Packet Loss

**Symptoms:**
- Gateway doesn't receive node data
- PER > 20%

**Diagnosis:**
1. Check antenna orientation
2. Measure received signal strength
3. Check for interference

**Solutions:**
- Re-orient antennas
- Change channel (802.15.4)
- Move away from WiFi 2.4GHz

## Maintenance

### Monthly Checks

1. **Visual Inspection:**
   - Check all connections
   - Look for corrosion
   - Verify capacitor health

2. **Performance Monitoring:**
   - Review energy metrics
   - Check Shapley convergence
   - Verify data integrity

3. **Firmware Updates:**
   - Check for security patches
   - Update to latest version
   - Test on bench before deployment

### Annual Maintenance

1. **Capacitor Replacement:**
   - Capacitors degrade over time
   - Replace every 2-3 years
   - Monitor ESR (equivalent series resistance)

2. **Calibration:**
   - Re-calibrate Monsoon annually
   - Update RTL power models
   - Verify energy measurements

3. **Documentation:**
   - Update test records
   - Document any modifications
   - Backup configurations

## Support

### Getting Help

1. **GitHub Issues:** https://github.com/ectc/ectc-project/issues
2. **Documentation:** docs/ directory
3. **Community Forum:** https://ectc-forum.org

### Parts Reordering

**Critical Spares (keep on hand):**
- 10x CC2650STK (for replacements)
- 20x 100μF capacitors (wear items)
- 5x BQ25570EVM-206 (if using externally)
- Power cables and adapters

### Lab Equipment

**Essential:**
- Digital multimeter (4.5 digit minimum)
- Oscilloscope (100 MHz, 2 channels)
- Network analyzer (for RF testing)
- Soldering station (temperature controlled)

**Optional:**
- Spectrum analyzer (for RF debugging)
- Power analyzer (for energy profiling)
- High-speed camera (for transient analysis)
