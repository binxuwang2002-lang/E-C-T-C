# Software Deployment Guide

## Overview

This guide covers deploying and configuring the ECTC software stack across MCU firmware and gateway services.

## Quick Start

### Prerequisites

1. **Hardware:**
   - 5+ TI CC2650STK SensorTags
   - BQ25570 PMIC evaluation boards
   - Raspberry Pi or x86 gateway
   - Monsoon power monitor (optional)

2. **Software:**
   - ARM GCC 10.3+
   - Python 3.8+
   - Git

### One-Command Deployment

```bash
# Clone repository
git clone https://github.com/ectc/ectc-project.git
cd ectc-project

# Run installation script
./scripts/install.sh

# Build everything
./scripts/build.sh

# Deploy to testbed
./scripts/deploy.sh --num-nodes 5 --gateway raspberry-pi
```

## Detailed Deployment

### Step 1: Build Firmware

#### Manual Build

```bash
cd firmware
./build.sh
```

**Expected Output:**
```
=== ECTC Firmware Build ===
MCU: cc2650
SDK: /opt/ti/simplelink_cc13x2_26x2_sdk_5_30_00_00

Checking dependencies...
  Compiling: ectc_main.c
  Compiling: tinylstm.c
  Compiling: shapley_local.c
  Compiling: pedersen.c
  Compiling: trace.c
  Compiling: bq25570.c
  Compiling: radio.c

Linking...
  Text (flash):     45,632 bytes
  Data (ram):       18,432 bytes
  BSS (ram):        12,288 bytes
  Total:            76,352 bytes

=== Build Complete ===
```

#### Build Options

```bash
# Build with debug symbols
./build.sh --debug

# Build for specific MCU
./build.sh --mcu cc2650 --variant f128

# Clean build
./build.sh clean

# Generate binary only (no hex/elf)
./build.sh --bin-only
```

#### Flashing to Hardware

**Using J-Link (Recommended):**

```bash
# Install J-Link software
# https://www.segger.com/downloads/jlink/

# Flash via command line
JLink -Device CC2650F128 -If SWD -Speed 4000 \
  -CommanderScript flash_node_001.jlink

# Or use TI Uniflash (GUI tool)
```

**flash_node_001.jlink:**
```
connect
loadbin build/cc2650/ectc_node.bin 0x0
r
g
qc
```

### Step 2: Deploy Gateway

#### Option A: Raspberry Pi

```bash
# On Raspberry Pi
cd ectc-project

# Create virtual environment
python3 -m venv ectc_env
source ectc_env/bin/activate

# Install dependencies
pip install -r gateway/requirements.txt

# Install ECTC gateway
cd gateway
pip install -e .

# Create configuration
cat > config/gateway.yaml << EOF
gateway:
  port: 8080
  host: 0.0.0.0

network:
  num_nodes: 50
  radio_channel: 11
  pan_id: 0x1234

energy:
  source: solar
  mppt_voltage: 2.4
  capacitor_cap: 100e-6

algorithms:
  lyapunov_v: 50.0
  lyapunov_beta: 0.1
  shapley_epsilon: 0.1
  shapley_delta: 0.05

monitoring:
  enable_prometheus: true
  prometheus_port: 9090
EOF

# Start gateway
python -m ectc_gateway.main --config config/gateway.yaml

# Or run in background
nohup python -m ectc_gateway.main --config config/gateway.yaml > /var/log/ectc.log 2>&1 &
```

#### Option B: Docker Container

```bash
# Build gateway container
docker build -f Dockerfile.gateway -t ectc-gateway:latest .

# Run container
docker run -d \
  --name ectc-gateway \
  -p 8080:8080 \
  -v /path/to/config:/app/config \
  ectc-gateway:latest

# Check logs
docker logs -f ectc-gateway
```

#### Option C: Development Mode

```bash
# For debugging
cd gateway
export PYTHONPATH=$PWD:$PYTHONPATH

# Run with auto-reload
python -m ectc_gateway.main \
  --config config/gateway.yaml \
  --reload \
  --log-level DEBUG
```

### Step 3: Configure Network

#### IEEE 802.15.4 Configuration

**CC1352P1 Coordinator (Gateway Side):**

```bash
# Flash coordinator firmware
# Using TI Uniflash or flash program

# Configure channel and PAN ID
# Default: Channel 11, PAN ID 0x1234
```

**Node Configuration (MCU Side):**

```c
// In ectc_main.h
#define IEEE802154_CHANNEL     11
#define IEEE802154_PAN_ID      0x1234
#define IEEE802154_TX_POWER    5  // dBm
```

#### Radio Parameters

```yaml
# gateway/config/network.yaml
radio:
  protocol: IEEE_802_15_4
  channel: 11  # 2.405 GHz
  pan_id: 0x1234

  # Timing
  slot_duration_ms: 100
  superframe_slots: 10

  # MAC parameters
  max_retries: 3
  cca_threshold: -75  # dBm
  rx sensitivity: -98  # dBm
```

### Step 4: Configure Energy Harvesting

#### BQ25570 Settings

**For Solar:**
```yaml
energy:
  source: solar

  bq25570:
    vbat_uv: 2300     # 2.3V (under-voltage)
    vbat_ok: 2500     # 2.5V (OK threshold)
    vbat_hys: 200     # 200mV hysteresis
    mppt_voltage: 2400  # 2.4V (solar optimized)
    vstor_ok: 3300    # 3.3V (storage OK)
    vstor_max: 4500   # 4.5V (storage max)
    istor_limit: 100  # 100mA
```

**For RF Energy:**
```yaml
energy:
  source: rf

  bq25570:
    vbat_uv: 2300
    vbat_ok: 2500
    mppt_voltage: 1800  # 1.8V (RF optimized)
    vstor_ok: 3300
    vstor_max: 4000
    istor_limit: 50   # 50mA (RF is lower power)
```

#### Capacitor Configuration

```yaml
capacitor:
  value: 100e-6  # 100 μF
  voltage_max: 6.3  # V
  esr_max: 0.5  # Ohms

  # Derived
  energy_max: 0.5 * 100e-6 * 6.3**2 * 1e6  # 1.98 mJ
```

### Step 5: Configure Algorithms

#### Lyapunov Game

```yaml
lyapunov:
  v: 50.0      # Tradeoff parameter
  beta: 0.1    # Saturation penalty
  c_cap: 330.0  # Capacitor energy max (μJ)

  thresholds:
    cap_full: 0.9  # 90% capacity considered "full"
    energy_ok: 50  # Minimum energy for tasks (μJ)
```

#### Shapley Values

```yaml
shapley:
  epsilon: 0.1    # Error bound
  delta: 0.05     # Confidence parameter

  sampling:
    min_samples: 10
    max_samples: 100
    stratum_size: 8  # Nodes per stratum

  convergence:
    max_iterations: 100
    tolerance: 1e-4
```

#### TinyLSTM

```yaml
tinylstm:
  model_path: models/tinylstm_int8.tflite

  input:
    seq_len: 10
    features: 1

  model:
    hidden_dim: 32
    output_dim: 1
    quantization: int8

  inference:
    energy_cost_uj: 23.1  # μJ per inference
```

#### KF-GP Hybrid

```yaml
kf_gp:
  # Kalman Filter
  kf:
    process_noise: 0.1
    observation_noise: 0.5

  # Gaussian Process
  gp:
    kernel: rbf  # or cauchy
    length_scale: 15.0  # meters
    signal_variance: 1.0

    inducing_points: 8  # M = log N

  # Switching
  robust_mode:
    threshold: 0.3  # 30% of nodes below this
    kernel: cauchy
```

### Step 6: Start Services

#### Start Gateway

```bash
# Start gateway service
cd gateway
python -m ectc_gateway.main --config config/gateway.yaml

# Or with systemd (Linux)
sudo systemctl start ectc-gateway
sudo systemctl enable ectc-gateway
sudo systemctl status ectc-gateway
```

#### Check Status

```bash
# Check if gateway is running
curl http://localhost:8080/status

# Response:
{
  "status": "running",
  "uptime": 1234,
  "nodes_active": 45,
  "shapley_converged": true,
  "energy_harvest_avg": 45.2
}
```

#### Monitor Logs

```bash
# View real-time logs
tail -f logs/gateway.log

# View with grep
grep "ERROR" logs/gateway.log

# View recent errors
tail -100 logs/gateway.log | grep -i error
```

### Step 7: Deploy Nodes

#### Automated Deployment

```bash
# Deploy all nodes
./scripts/deploy_nodes.sh --num-nodes 5 --firmware-dir firmware/build/cc2650/

# Deploy specific nodes
./scripts/deploy_nodes.sh --nodes 1,3,5 --flash

# List deployed nodes
./scripts/list_nodes.sh
```

#### Manual Node Deployment

```bash
# Build node-specific firmware
cd firmware/ectc_node
sed -i 's/define NODE_ID_DEFAULT 0/define NODE_ID_DEFAULT 1/' include/ectc_main.h
cd ../build/cc2650
../../../build.sh

# Flash
# (Use J-Link, Uniflash, or similar)
```

#### Verify Node Deployment

```bash
# On gateway, check node status
curl http://localhost:8080/api/nodes

# Response:
[
  {
    "node_id": 0,
    "status": "active",
    "energy_uj": 245.3,
    "queue_len": 3,
    "last_seen": "2024-01-15T10:30:45Z",
    "shapley_value": 0.85,
    "has_data": true
  }
]
```

## Configuration Reference

### Full Configuration Example

**config/gateway.yaml:**

```yaml
# ECTC Gateway Configuration
gateway:
  port: 8080
  host: 0.0.0.0
  log_level: INFO

  # API
  api:
    version: v1
    cors_enabled: true

  # Security
  security:
    enable_zk_proofs: true
    require_commitments: true

# Network Configuration
network:
  protocol: IEEE_802_15_4
  channel: 11
  pan_id: 0x1234

  # Timing
  slot_duration_ms: 100
  superframe_slots: 10

  # Nodes
  max_nodes: 100
  num_mobile: 10

  # Radio
  tx_power_dbm: 5
  rx_sensitivity_dbm: -98

# Energy Configuration
energy:
  source: solar  # or rf, vibration

  capacitor:
    value: 100e-6  # Farads
    voltage_max: 6.3
    vdd: 3.3

  bq25570:
    vbat_uv: 2300
    vbat_ok: 2500
    vbat_hys: 200
    mppt_voltage: 2400
    vstor_ok: 3300
    vstor_max: 4500
    istor_limit: 100

# Algorithm Parameters
algorithms:
  # Lyapunov Game
  lyapunov:
    v: 50.0
    beta: 0.1
    c_cap: 330.0
    cap_full_threshold: 0.9
    min_energy_threshold: 50.0

  # Shapley Values
  shapley:
    epsilon: 0.1
    delta: 0.05
    min_samples: 10
    max_samples: 100
    stratum_size: 8
    max_iterations: 100
    tolerance: 1e-4

  # TinyLSTM
  tinylstm:
    model_path: models/tinylstm_int8.tflite
    seq_len: 10
    hidden_dim: 32
    energy_cost_uj: 23.1

  # KF-GP Hybrid
  kf_gp:
    kf_process_noise: 0.1
    kf_observation_noise: 0.5
    gp_kernel: rbf  # or cauchy
    gp_length_scale: 15.0
    gp_signal_variance: 1.0
    inducing_points: 8

    robust_mode:
      threshold: 0.3
      kernel: cauchy

# Data Storage
storage:
  type: influxdb  # or mongodb, sqlite

  influxdb:
    host: localhost
    port: 8086
    database: ectc
    username: ectc
    password: ${INFLUXDB_PASSWORD}

  retention:
    raw_data: 7d  # Keep raw for 7 days
    aggregated: 30d  # Keep aggregates for 30 days

# Monitoring
monitoring:
  enable_prometheus: true
  prometheus_port: 9090

  metrics:
    - name: ectc_node_energy
      type: gauge
      description: Node energy level

    - name: ectc_data_integrity
      type: gauge
      description: Data integrity percentage

    - name: ectc_shapley_error
      type: gauge
      description: Shapley approximation error

# Logging
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR
  file: logs/gateway.log
  max_size: 100MB
  backup_count: 5

  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Environment Variables

```bash
# Optional: Use environment variables for sensitive config
export ECTC_DB_PASSWORD="your_password"
export ECTC_MONSOON_SERIAL="/dev/ttyUSB0"
export ECTC_RADIO_SERIAL="/dev/ttyACM0"

# Start gateway with env vars
python -m ectc_gateway.main --config config/gateway.yaml
```

## Troubleshooting

### Gateway Won't Start

**Error:** Port 8080 already in use

```bash
# Find process using port
sudo netstat -tlnp | grep 8080
sudo lsof -i :8080

# Kill existing process
sudo kill -9 <PID>

# Or use different port
python -m ectc_gateway.main --port 8081
```

**Error:** Module not found

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Install in development mode
cd gateway
pip install -e .

# Verify installation
python -c "import ectc_gateway; print(ectc_gateway.__version__)"
```

### Nodes Not Joining Network

**Symptoms:**
- Gateway shows 0 active nodes
- No packets received

**Diagnosis:**
```bash
# Check radio connection
dmesg | grep tty
ls /dev/tty*

# Test serial communication
echo "test" > /dev/ttyACM0

# Check gateway logs
grep "node" logs/gateway.log
```

**Solutions:**
1. Verify CC1352P1 is configured as coordinator
2. Check USB cable and connection
3. Reset radio module
4. Verify PAN ID and channel match nodes

### High Energy Consumption

**Symptoms:**
- Nodes deplete energy quickly
- Frequent sleep/wake cycles

**Diagnosis:**
```bash
# Check TinyLSTM inference energy
curl http://localhost:8080/api/nodes/0/metrics | grep tinylstm_energy

# Check radio TX power
# Should be 5 dBm max for CC2650

# Check task execution rate
# Should be ~1 task per 10 slots (1000ms)
```

**Solutions:**
1. Reduce TinyLSTM inference frequency
2. Lower radio TX power
3. Increase slot duration
4. Optimize Lyapunov parameters (lower V)

### Shapley Values Not Converging

**Symptoms:**
- Shapley error > epsilon
- Frequent recalculations

**Diagnosis:**
```bash
# Check Shapley logs
grep "shapley" logs/gateway.log | tail -20

# Check approximation parameters
# Verify epsilon, delta, sample count
```

**Solutions:**
1. Increase sample count
2. Increase max iterations
3. Increase tolerance
4. Check if network is stable (mobile nodes)

## Performance Tuning

### Gateway Performance

**Optimization for Large Networks (N > 100):**

```yaml
# Enable parallel Shapley computation
shapley:
  parallel: true
  max_workers: 4

# Use inducing points for GP
kf_gp:
  inducing_points: 16  # Increase from 8

# Batch processing
batch_size: 32
```

**Monitor Gateway Load:**

```bash
# CPU usage
top -p $(pgrep -f ectc_gateway)

# Memory usage
ps aux | grep ectc_gateway

# Network I/O
iftop -i eth0
```

### MCU Performance

**Reduce Memory Usage:**

```c
// In tinylstm.c, reduce hidden dimensions
#define HIDDEN_DIM 16  // From 32

// Or reduce trace buffer size
#define TRACE_BUFFER_SIZE 32  // From 64
```

**Reduce Energy Consumption:**

```c
// Increase slot duration
#define SLOT_DURATION_MS 200  // From 100

// Reduce radio TX power
// In radio.c
void radio_set_tx_power(int8_t power_dbm) {
    // Set to 0 dBm instead of 5 dBm
    radio_config_tx_power(0);
}
```

## Backup and Recovery

### Backup Configuration

```bash
# Backup gateway configuration
tar -czf ectc-backup-$(date +%Y%m%d).tar.gz \
  config/ \
  models/ \
  logs/ \
  gateway/ectc_gateway/

# Backup database
influx backup /path/to/influxdb/backups
```

### Restore from Backup

```bash
# Stop services
sudo systemctl stop ectc-gateway

# Restore configuration
tar -xzf ectc-backup-20240115.tar.gz

# Restore database
influx restore /path/to/influxdb/backups

# Restart services
sudo systemctl start ectc-gateway
```

## Updates

### Firmware Update

```bash
# Build new firmware
./scripts/build.sh

# Deploy to nodes (with rolling update)
./scripts/rollout_firmware.sh --version v1.1 --nodes 1-50

# Verify update
curl http://localhost:8080/api/nodes/0/info | grep firmware_version
```

### Gateway Update

```bash
# Pull latest code
git pull origin main

# Install dependencies
pip install -r gateway/requirements.txt --upgrade

# Restart service
sudo systemctl restart ectc-gateway

# Check version
curl http://localhost:8080/version
```

## Support

### Log Analysis

```bash
# Common log patterns
grep "ERROR" logs/gateway.log | wc -l  # Count errors
grep "node.*join" logs/gateway.log      # Node joins
grep "shapley.*converge" logs/gateway.log  # Shapley convergence
grep "energy.*low" logs/gateway.log     # Energy warnings
```

### Diagnostics

```bash
# Run diagnostics
python scripts/diagnostics.py --gateway localhost:8080

# Output:
# ✓ Gateway responding
# ✓ 45/50 nodes active
# ✓ Shapley converged (error: 0.08)
# ✓ Average energy: 245 μJ
# ✓ Data integrity: 93.2%
```

### Getting Help

- **GitHub Issues:** https://github.com/ectc/ectc-project/issues
- **Documentation:** https://ectc.readthedocs.io
- **Community Forum:** https://ectc-forum.org
- **Email:** support@ectc-project.org
