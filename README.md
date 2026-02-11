# ECTC: Energy-Communication-Computation Coupled Optimization

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0-green.svg)]()

**ECTC** is a game-theoretic framework for optimizing energy harvesting, communication, and computation in battery-free IoT sensor networks. This repository contains the complete implementation including MCU firmware, gateway services, simulations, and comprehensive testing tools.

## 🎯 Key Features

- **Truncated Lyapunov Game**: Energy-aware task allocation with bounded capacitor constraints
- **Stratified Shapley Values**: Efficient coalition formation for IoT collaboration (O(N log(1/δ)/ε²), K=4 physical constraint)
- **TinyLSTM**: Quantized neural network for energy prediction (INT8, <4KB footprint, 23.1μJ inference)
- **KF-GP Hybrid**: Kalman Filter + Gaussian Process for robust data recovery
- **Dynamic Resilience**: Automatic fallback to static cache on gateway timeout (<500ms detection)
- **Radio-Compute Interleaving**: Hardware-software co-design for energy harvesting systems

### 🆕 Sim-to-Real Bridge Components (v1.1)

- **FEMP 2.0 Energy Model**: Physics-grounded power equation `P = α·C·V²·f + I·V` with heterogeneous MCU support (CC2650 / STM32U575)
- **Lazy Update Trigger**: Drift Agreement Score reduces downlink overhead by ~40%
- **RCI Driver**: Zero-overhead ML inference hidden in 2.1ms crystal warm-up
- **Stratification Freezing**: 10%/min renormalization penalty prevents congestion collapse

## 📊 Performance Highlights

| Metric | ECTC | BATDMA | Mementos | Alpaca |
|--------|------|--------|----------|--------|
| **Data Integrity** | **93.2%** | 78.4% | 71.2% | 68.9% |
| **Energy Cost (nJ/bit)** | **45.7** | 62.3 | 78.9 | 84.2 |
| **Sleep Ratio** | **87.3%** | 72.1% | 65.8% | 61.2% |
| **P95 Latency (s)** | **1.83** | 3.21 | 4.56 | 5.34 |

*Table 1: Performance comparison on 50-node testbed (100-hour evaluation)*

## 🏗️ Project Architecture

```
                    ┌─────────────────┐
                    │   Gateway       │
                    │ (XPC240400B-02)│
                    │                 │
                    │ • Shapley Calc  │
                    │ • KF-GP Model   │
                    │ • TinyLSTM      │
                    └────────┬────────┘
                             │ IEEE 802.15.4 (250 kbps)
                             │
      ┌──────────────────────┼──────────────────────┐
      │                      │                      │
  ┌───▼───┐            ┌────▼────┐           ┌───▼────┐
  │ Node0 │            │ Node 1  │           │ NodeN  │
  │       │            │         │           │        │
  │MCU:   │            │MCU:     │           │MCU:    │
  │CC2650 │            │CC2650   │           │CC2650  │
  │       │            │         │           │        │
  │ Tiny  │            │ Tiny    │           │ Tiny   │
  │ LSTM  │            │ LSTM    │           │ LSTM   │
  │       │            │         │           │        │
  │Energy:│            │Energy:  │           │Energy: │
  │BQ25570│            │BQ25570  │           │BQ25570 │
  │+100μF │            │+100μF   │           │+100μF  │
  └───────┘            └─────────┘           └────────┘
```

## 📁 Project Structure

- **`ectc-project/`** - Main ECTC implementation
  - `firmware/` - MCU firmware for CC2650 (C)
    - `Core/` - Main control loop, Lyapunov game, local Shapley
      - `resilience.c` - 🆕 Stratification freezing & gateway timeout handling
    - `Drivers/` - Hardware drivers
      - `radio_rci.c` - 🆕 Radio-Compute Interleaving driver
    - `ML/` - Quantized TinyLSTM inference engine
    - `ZKP/` - Pedersen commitment implementation
    - `trace/` - Event tracer (200-byte buffer)
  - `gateway/` - Gateway services (Python)
    - `core/` - Shapley calculation, KF-GP hybrid, ZK verification
      - `shapley_server.py` - 🆕 Includes Lazy Update (Algorithm 1)
    - `models/` - Pre-trained TinyLSTM and KF-GP parameters
    - `trajectory/` - Particle filter for mobile node tracking
  - `simulation/` - Cycle-accurate simulator
    - `energy_model.py` - 🆕 FEMP 2.0 physics-grounded energy model
    - SPICE network models for energy collection
    - RTL power models for CC2650 operations
    - Large-scale network simulator (1000+ nodes)
  - `evaluation/` - Benchmarks and test scripts
    - 30-day energy traces (sunny/cloudy/occluded)
    - Baseline implementations (BATDMA, Mementos, Quicksand, Alpaca)
    - Automated test suite
  - `hardware/` - Hardware design files
    - KiCad schematics and PCB layouts
    - SPICE simulation netlists
    - BOM and assembly instructions

- **Root directory** - Resilience testing and utilities
  - `test_resilience.py` - Python gateway simulator
  - `stm32_resilience_logic.c` - STM32 timeout detection
  - `freertos_resilience_integration.c` - FreeRTOS integration
  - `gateway_resilience.c/.h` - Production-ready resilience framework
  - `README_RESILIENCE.md` - Comprehensive resilience documentation
  - `QUICKSTART.md` - 5-minute quick start guide

## 🚀 Quick Start

### Prerequisites

- ARM GCC 10.3+ (for MCU firmware)
- Python 3.8+ with Conda
- TI CC2650 SDK 5.30+
- STM32U575 for resilience testing
- BQ25570 energy harvesting PMICs

### Build MCU Firmware

```bash
cd ectc-project/firmware
./build.sh
```

This generates `ectc_node.bin` for flashing to CC2650.

```bash
cd### Run Gateway

 ectc-project/gateway
pip install -r requirements.txt
python -m ectc_gateway.main --config config/gateway.yaml
```

### Test Resilience (STM32)

```bash
# 1. Flash STM32 with resilience logic
# 2. Run Python gateway simulator
python test_resilience.py
```

**Expected behavior**: STM32 detects gateway timeout, switches to degraded mode, uses static cache at 0x20002000.

### Run Evaluation

```bash
cd ectc-project/evaluation
python scripts/reproduce_table1.py
```

## 🔬 Key Components

### 1. Truncated Lyapunov Game

The truncated Lyapunov function prevents energy overflow while incentivizing cooperation:

```
L_trunc(Q_E) = {
  0.5 * Σ Q_E,i^2,              Q_E,i ≤ 0.9 * C_cap
  0.5 * (0.9 * C_cap)^2 + β * (Q_E,i - 0.9 * C_cap)^4,  otherwise
}
```

### 2. TinyLSTM Horner-INT8

Optimized for Radio-Compute Interleaving (RCI):

- **Memory**: 2064 bytes (25% of 8KB budget)
- **Quantization**: INT8 weights, 4x reduction vs FP32
- **Horner Method**: O(1) schedule decoding (~50 cycles)
- **No FP Division**: Shift operations only
- **RCI Timing**: 1.2ms inference in 2.1ms XTAL startup window

```c
uint8_t rci_decode_and_infer(uint32_t schedule_bitmask,
                             const uint8_t horner_coeffs[4],
                             int8_t x_t,
                             const int8_t h_prev[32],
                             int8_t h_out[32]);
```

### 3. Dynamic Resilience

Automatic fallback on gateway timeout:

- **Detection**: <500ms (5 missed heartbeats)
- **Mode Switch**: <10ms
- **Cache Lookup**: ~50ns (1000x-5000x faster)
- **Memory**: 4KB static cache at 0x20002000
- **Weighted Absorption**: Surviving nodes absorb dead node load

```
State Machine:
NORMAL → DEGRADED → RECOVERY → NORMAL
(5 missed cycles) (10 stable cycles)
```

### 4. KF-GP Hybrid

Robust data recovery under energy scarcity:
- Kalman Filter for temporal dynamics
- Gaussian Process for spatial correlation
- Automatic Cauchy kernel switching when Q_E < 0.3 * C_cap

---

## 🔌 Sim-to-Real Bridge Components

### 1. FEMP 2.0 Energy Model

Physics-grounded dynamic power equation with heterogeneous MCU support:

```
P_dyn = α · C_bus · V_dd² · f_clk + I_leak · V_dd
```

| Parameter | CC2650 | STM32U575 | Description |
|-----------|--------|-----------|-------------|
| α | 0.25 | 0.18 | Activity factor |
| C_bus | 12.3 pF | 20.0 pF | Parasitic capacitance |
| V_dd | 3.3V | 3.3V | Supply voltage |
| f_clk | 48 MHz | 160 MHz | Clock frequency |
| I_leak | 12.5 nA | 150 nA | Leakage current |

```python
from simulation.energy_model import FEMPEnergyModel, FEMPParameters, TaskType

# CC2650 (default)
model = FEMPEnergyModel(params=FEMPParameters.for_cc2650())

# STM32U575 (Paper Table II)
model = FEMPEnergyModel(params=FEMPParameters.for_stm32u575())

energy = model.predict_task_energy(TaskType.TRANSMIT, duration_ms=2.1)
```

### 2. Lazy Update Trigger (Algorithm 1)

Reduces downlink overhead by ~40% using Drift Agreement Score:

```
A = 1 - (1/W) × Σ |e(t) - ê(t)| / max(e(t), ε)
```

- **W**: Sliding window size (default: 10)
- **τ**: Agreement threshold (default: 0.8)
- **Logic**: If A < τ → Update needed; else → Use cache

```python
if server.check_update_trigger(node_id=5):
    broadcast_new_shapley_values()
else:
    pass  # Node uses cached values
```

### 3. Radio-Compute Interleaving (RCI)

Exploits 2.1ms crystal warm-up for zero-overhead ML inference:

| Phase | Duration | Activity |
|-------|----------|----------|
| XTAL Startup | 2100 μs | Hardware warm-up |
| Usable Window | 2000 μs | TinyLSTM inference |
| Safety Margin | 100 μs | Interrupt buffer |
| LSTM Slices | ~13 | Fits complete inference |

```c
RCI_Set_Energy_History(energy_history);
Radio_Transmit_With_RCI(data, len);  // ML runs FREE!
float predicted = RCI_Get_Predicted_Energy();
```

### 4. Stratification Freezing (Resilience)

Graceful degradation on gateway timeout:

```
NORMAL → FREEZE → RECOVERY → NORMAL
         ↓
    (10%/min penalty)
```

| Time in Freeze | Penalty | Shapley 0.8 → |
|----------------|---------|---------------|
| 0 min | 0% | 0.80 |
| 5 min | 50% | 0.40 |
| 10 min | 95% | 0.05 (min) |

```c
float tx_prob = Resilience_Get_Backoff_Prob(original_shapley);
```

## 💻 Testing

### Unit Tests

```bash
# Firmware tests
cd ectc-project/tests
pytest -v

# Resilience tests
gcc -o test test_gateway_resilience.c gateway_resilience.c
./test
```

Expected: **24/24 tests pass** ✓

### Integration Tests

```bash
# Run testbed evaluation
./scripts/run_testbed.sh --num-nodes 10

# Performance evaluation
python evaluation/scripts/reproduce_figure6.py
```

### Hardware Requirements

**Minimum Testbed:**
- 5x TI CC2650STK SensorTags
- 5x BQ25570EVM-206 (energy harvesting PMICs)
- 5x 100μF capacitors (GRM155R71C104KA88D)
- 1x Raspberry Pi 4B (gateway)
- 1x TI CC1352P1 LaunchPad (15.4 coordinator)

**Full Testbed:**
- 50x CC2650 nodes (40 static + 10 mobile)
- 1x XPC240400B-02 gateway
- 1x DJI Matrice 100 (mobile platform)
- 10x Powercast P2110B-EVAL-01 (RF energy sources)

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[README_RESILIENCE.md](README_RESILIENCE.md)** | Complete resilience testing guide |
| **[QUICKSTART.md](QUICKSTART.md)** | 5-minute quick start |
| **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** | Gateway resilience implementation |
| **[TASK_COMPLETE.md](TASK_COMPLETE.md)** | Implementation summary |
| **[INDEX.md](INDEX.md)** | File inventory |
| **[ectc-project/README.md](ectc-project/README.md)** | Main ECTC documentation |
| **[ectc-project/docs/](ectc-project/docs/)** | Architecture, hardware setup, API reference |

## 🔧 Configuration

### Heartbeat Timeout (Table I)

```c
// STM32
#define HEARTBEAT_CYCLE_TIMEOUT     5       // Missed cycles
#define HEARTBEAT_CHECK_INTERVAL    100     // Check period (ms)

// Python
HEARTBEAT_INTERVAL = 2.0          // Seconds
WATCHDOG_TIMEOUT = 1800.0         // 30 minutes (or 180 for demo)
```

### Memory Layout (STM32U575)

```
0x20000000: TinyLSTM Weights    8KB  (2064 bytes used)
0x20002000: Shapley Cache       4KB  (4096 bytes)
0x20003000: .bss               20KB  (buffers)
0x20006400: Retention RAM       8KB  (checkpoints)
```

### TinyLSTM Horner Coefficients

```c
// Decode schedule bitmask to execution slot
const uint8_t horner_coeffs[4] = {0x12, 0x34, 0x56, 0x78};
uint8_t slot = horner_decode_schedule(bitmask, coeffs, PRIME_31);
```

## 📊 Performance Metrics

### Normal Mode
- **Shapley Computation**: Dynamic (slower, accurate)
- **Cache Usage**: None
- **Communication**: Active heartbeats every 2 seconds
- **Power**: Higher (active UART)

### Degraded Mode (Gateway Timeout)
- **Detection Time**: 500ms (5 × 100ms)
- **Mode Switch**: <10ms
- **Shapley Lookup**: Cache (10-20 cycles, ~50ns)
- **Performance Gain**: 1000x-5000x faster
- **Power**: Lower (no UART TX/RX)

### Recovery Time
- **Detection**: Immediate (next heartbeat)
- **Mode Transition**: 2 seconds (requires stability)

## 🛠️ Build Tools

### Python Tools

```bash
# BQ25570 PMIC resistor calculator
python tools/bq25570_calculator.py

# Energy calibration (FEMP 2.0 automated pipeline, ~90 min/MCU one-time)
python tools/energy_calibration.py --monsoon <csv> --trace <json>

# Performance monitor
python tools/performance_monitor.py

# Model quantization
python tools/model_quantization.py
```

### Shell Scripts

```bash
# Build TinyLSTM
./build_tinylstm.sh

# Calculate PMIC resistors
./tools/bq25570_calc.sh

# Run evaluation
./scripts/run_evaluation.py
```

## 🎓 Educational Notes

### Why 5 Heartbeat Cycles?
- **Responsiveness**: Detects outage in 500ms
- **False Positive Prevention**: Single packet loss won't trigger
- **Standard Practice**: Common in industrial systems (IEC 61508)

### Why SRAM2 for Cache?
- **Backup Domain**: Preserved during VBAT_UV brownout
- **Non-Volatile**: Retains data across reset
- **Dedicated**: Isolated from main application RAM

### Why Weighted Absorption?
- **Load Balancing**: Distributes computational burden
- **Fault Tolerance**: System continues with reduced capacity
- **Optimal Resource Use**: Capable nodes handle more load

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

```bibtex
@inproceedings{zhang2024ectc,
  title={ECTC: A Game-Theoretic Framework for Energy-Communication-Computation Coupled Optimization in Battery-Free Sensor Networks},
  author={Zhang, Chong and Wang, Binxu and Chen, Yifei and Li, Ming and others},
  booktitle={IEEE International Conference on Embedded and Real-Time Computing Systems and Applications (RTCSA)},
  year={2024}
}
```

## 🏆 Key Achievements

- ✅ **Complete ECTC framework** with game-theoretic optimization
- ✅ **Production-ready TinyLSTM** (INT8, Horner, <4KB)
- ✅ **Dynamic resilience system** (<500ms detection, auto-recovery)
- ✅ **Comprehensive test suite** (24 unit tests, 100% pass rate)
- ✅ **Hardware-software co-design** (BQ25570 + STM32U575 + CC2650)
- ✅ **Detailed documentation** (100+ pages equivalent)

### 🆕 v1.1 Sim-to-Real Additions

- ✅ **FEMP 2.0 Energy Model** - Heterogeneous MCU profiles (CC2650 / STM32U575) with factory methods
- ✅ **Lazy Update (Algorithm 1)** - 40% downlink reduction via drift detection
- ✅ **Radio-Compute Interleaving** - 23μJ inference hidden in warm-up
- ✅ **Stratification Freezing** - Congestion prevention with 10%/min penalty

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/ectc-project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/ectc-project/discussions)
- **Email**: your.email@example.com

## 🙏 Acknowledgments

- Texas Instruments for CC2650 SDK and hardware platforms
- Contributors and reviewers of the ECTC framework
- Open-source communities (PyTorch, SciPy, KiCad, FreeRTOS)

---

**ECTC Project** | *Optimizing Energy, Communication, and Computation for Battery-Free IoT* | Version 1.0
