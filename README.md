# ECTC: Energy-Communication-Computation Coupled Optimization

This repository contains the complete implementation of the ECTC (Energy-Communication-Computation Coupled Optimization) framework for battery-free sensor networks, based on the research paper "ECTC: A Game-Theoretic Framework for Energy-Communication-Computation Coupled Optimization in Battery-Free Sensor Networks."

## Overview

ECTC introduces a novel game-theoretic approach to jointly optimize energy harvesting, communication, and computation in energy-harvesting IoT networks. Key contributions include:

- **Truncated Lyapunov Game**: Energy-aware task allocation with bounded capacitor constraints
- **Stratified Shapley Values**: Efficient coalition formation for IoT collaboration
- **TinyLSTM**: Quantized neural network for energy prediction
- **KF-GP Hybrid**: Kalman Filter + Gaussian Process for robust data recovery
- **Pedersen Commitments**: Zero-knowledge proof for secure energy state reporting

## Architecture

```
                    ┌─────────────────┐
                    │   Gateway       │
                    │ (XPC240400B-02)│
                    │                 │
                    │ • Shapley Calc  │
                    │ • KF-GP Model   │
                    │ • ZK Verifier   │
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

## Quick Start

### Prerequisites

- ARM GCC 10.3+ (for MCU firmware)
- Python 3.8+ with Conda
- TI CC2650 SDK 5.30+
- Docker (optional, for containerized builds)

### Build MCU Firmware

```bash
cd firmware
./build.sh
```

This generates `ectc_node.bin` for flashing to CC2650.

### Deploy Gateway

```bash
cd gateway
pip install -r requirements.txt
python -m ectc_gateway.main --config config/gateway.yaml
```

### Run Evaluation

```bash
cd evaluation
python scripts/reproduce_table1.py
```

## Project Structure

- **firmware/**: MCU firmware for CC2650 (C language)
  - **Core/**: Main control loop, Lyapunov game, local Shapley
  - **ML/**: Quantized TinyLSTM inference engine
  - **ZKP/**: Pedersen commitment implementation
  - **trace/**: Event tracer (200-byte buffer)

- **gateway/**: Gateway services (Python)
  - **core/**: Shapley calculation, KF-GP hybrid, ZK verification
  - **models/**: Pre-trained TinyLSTM and KF-GP parameters
  - **trajectory/**: Particle filter for mobile node tracking

- **simulation/**: Cycle-accurate simulator
  - SPICE network models for energy collection
  - RTL power models for CC2650 operations
  - Large-scale network simulator (1000+ nodes)

- **evaluation/**: Benchmarks and test scripts
  - 30-day energy traces (sunny/cloudy/occluded)
  - Baseline implementations (BATDMA, Mementos, Quetzal-BFSN, DINO-BFSN, etc.)
  - Table V reproduction script
  - Automated test suite

- **tools/**: Parameter identification and calibration
  - `fallback_cbus_identification.py`: No-EDA C_bus extraction (Paper V-C.1)
  - `energy_calibration.py`: Energy model calibration
  - `bq25570_calculator.py`: PMIC configuration

- **hardware/**: Hardware design files
  - KiCad schematics and PCB layouts
  - SPICE simulation netlists
  - BOM and assembly instructions

## Key Components

### 1. Truncated Lyapunov Game

The truncated Lyapunov function prevents energy overflow:

```
L_trunc(Q_E) = {
  0.5 * Σ Q_E,i^2,              Q_E,i ≤ 0.9 * C_cap
  0.5 * (0.9 * C_cap)^2 + β * (Q_E,i - 0.9 * C_cap)^4,  otherwise
}
```

This ensures bounded energy queues while incentivizing cooperation.

### 2. Stratified Shapley Values

Efficient approximation with O(N log(1/δ)/ε²) complexity:

```python
approximator = StratifiedShapleyApproximator(N=50)
phi = approximator.approximate_shapley_values(game, positions)
```

### 3. TinyLSTM

Quantized LSTM for energy prediction:
- 32 hidden units, int8 weights, 2-bit activations
- Memory footprint: <4KB in FRAM
- Inference energy: 23.1 μJ

### 4. KF-GP Hybrid

Robust data recovery under energy scarcity:
- Kalman Filter for temporal dynamics
- Gaussian Process for spatial correlation
- Automatic switching to Cauchy kernel when Q_E < 0.3 * C_cap

### 5. FEMP 2.0 Energy Model

Physics-grounded energy model with parasitic parameters (Paper Section V-C):

```python
from simulation.energy_model import FEMPEnergyModel, TaskType

model = FEMPEnergyModel()
energy = model.predict_task_energy(TaskType.TRANSMIT, duration_ms=2.0)
```

> **CRITICAL**: Ignoring C_bus (bus parasitic capacitance) causes **4.6× energy estimation error**.

### 6. Online Drift Compensation

Real-time leakage current estimation (Equation 10):

```c
// Before Deep Sleep
DriftComp_PreSleep_Record();
// [MCU sleeps]
// After wakeup
DriftComp_PostWake_Record();
float new_leakage = Update_Leakage_Estimate();
```

## Hardware Support

### Supported MCUs

| MCU | Core | Status | Power Model |
|-----|------|--------|-------------|
| CC2650 | Cortex-M3 | ✅ Tested | `cc2650_core.json` |
| STM32U575 | Cortex-M33 | ✅ Supported | `stm32u575_core.json` |

### STM32U575 Parameters (Paper Table II)

- **V_ret**: 1.8V (retention voltage)
- **V_rated**: 3.3V (rated supply)
- **I_active**: 19 µA/MHz
- **I_leak**: 150 nA (Deep Sleep)
- **C_bus**: 20.0 pF (parasitic bus capacitance)

## Baseline Implementations

Physics-grounded baselines for fair comparison:

| Baseline | Description | Location |
|----------|-------------|----------|
| **Quetzal-BFSN** | SJF scheduling + reactive IBO | `evaluation/baselines/quetzal.py` |
| **DINO-BFSN** | Volatility-adaptive checkpointing | `evaluation/baselines/dino.py` |
| BATDMA | Energy-aware TDMA | `evaluation/baselines/batdma.py` |
| Mementos | Checkpoint-based intermittent | `evaluation/baselines/mementos.py` |
| Alpaca | Federated learning baseline | `evaluation/baselines/alpaca.py` |

## Reproducibility

### Reproduce Table V (Physical-Model Parity)

```bash
python evaluation/scripts/reproduce_table_v.py --nodes 50 --duration 1000
```

Outputs comparison of ECTC vs Quetzal-BFSN vs DINO-BFSN.

### No-EDA Parameter Extraction

Extract C_bus without SPICE/Cadence (Paper V-C.1):

```bash
python tools/fallback_cbus_identification.py --demo

# From real measurements
python tools/fallback_cbus_identification.py --leakage --idle-csv data.csv
```

## Documentation

- [BHDF Schema](docs/BHDF_SCHEMA.md) - Battery-Free Hardware Description File format
- [Architecture](docs/architecture.md) - System design overview
- [Hardware Setup](docs/hardware_setup.md) - Testbed configuration
- [API Reference](docs/api_reference.md) - Code documentation

## Performance

Compared to state-of-the-art:

| Framework  | Data Integrity | Energy Cost (nJ/bit) | Sleep Ratio | P95 Latency (s) |
|------------|----------------|----------------------|-------------|-----------------|
| **ECTC**   | **93.2%**      | **45.7**             | **87.3%**   | **1.83**        |
| Quetzal-BFSN | 82.1%        | 58.2                 | 78.5%       | 2.54            |
| DINO-BFSN  | 79.8%          | 61.5                 | 75.2%       | 2.87            |
| BATDMA     | 78.4%          | 62.3                 | 72.1%       | 3.21            |
| Mementos   | 71.2%          | 78.9                 | 65.8%       | 4.56            |

*Table 1: Performance comparison on 50-node testbed (100-hour evaluation)*

## Hardware Requirements

### Minimum Testbed
- 5x TI CC2650STK SensorTags
- 5x BQ25570EVM-206 (energy harvesting PMICs)
- 5x 100μF capacitors (GRM155R71C104KA88D)
- 1x Raspberry Pi 4B (gateway)
- 1x TI CC1352P1 LaunchPad (15.4 coordinator)
- 1x Monsoon HV Power Monitor (validation)

### Full Testbed
- 50x CC2650 nodes (40 static + 10 mobile)
- 1x XPC240400B-02 gateway (or alternative setup)
- 1x DJI Matrice 100 (mobile platform)
- 10x Powercast P2110B-EVAL-01 (RF energy sources)

## Testing

### Unit Tests
```bash
pytest tests/ -v
```

### Integration Tests
```bash
./scripts/run_testbed.sh --num-nodes 10
```

### Performance Evaluation
```bash
python evaluation/scripts/reproduce_figure6.py
```

## Citation

```bibtex
@inproceedings{zhang2024ectc,
  title={ECTC: A Game-Theoretic Framework for Energy-Communication-Computation Coupled Optimization in Battery-Free Sensor Networks},
  author={Zhang, Chong and Wang, Binxu and Chen, Yifei and Li, Ming and others},
  booktitle={IEEE International Conference on Embedded and Real-Time Computing Systems and Applications (RTCSA)},
  year={2024}
}
```

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Contact

- Project Lead: Binxu Wang
- Email: 745974903@qq.com


## Acknowledgments

- Texas Instruments for CC2650 SDK and hardware platforms
- Contributors and reviewers of the ECTC framework
- Open-source communities (PyTorch, SciPy, KiCad)
