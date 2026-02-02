# System Architecture

## Overview

The ECTC (Energy-Communication-Computation Coupled Optimization) framework is a comprehensive system for managing battery-free sensor networks. It combines three key optimization dimensions:

1. **Energy**: Harvesting, storage, and consumption optimization
2. **Communication**: Coalition-based task allocation and data routing
3. **Computation**: Local inference and distributed coordination

## System Components

### 1. Sensor Nodes (MCU: CC2650)

```
┌─────────────────────────────────────┐
│  CC2650 (Cortex-M3 @ 48MHz)         │
├─────────────────────────────────────┤
│  Core Components:                    │
│  ├─ ECTC Main Control Loop          │
│  ├─ TinyLSTM Predictor (32 units)   │
│  ├─ Local Shapley Calculator        │
│  └─ Pedersen Commitment Engine      │
│                                     │
│  Hardware Drivers:                   │
│  ├─ BQ25570 Energy Harvesting PMIC  │
│  ├─ IEEE 802.15.4 Radio            │
│  ├─ Temperature Sensor              │
│  └─ Event Tracer (200B buffer)      │
│                                     │
│  Memory:                             │
│  ├─ Flash: 128KB (code + weights)   │
│  ├─ SRAM: 40KB (runtime buffers)    │
│  └─ FRAM: 8KB (config, future)     │
└─────────────────────────────────────┘
```

### 2. Gateway (XPC240400B-02 or Raspberry Pi 4B)

```
┌─────────────────────────────────────┐
│  Gateway (ARM Cortex-A53 @ 1.5GHz)   │
├─────────────────────────────────────┤
│  Core Services:                      │
│  ├─ Stratified Shapley Server       │
│  ├─ KF-GP Hybrid Data Recovery      │
│  ├─ ZK Proof Verifier              │
│  └─ Particle Filter (Mobile Track)  │
│                                     │
│  Communication:                      │
│  ├─ IEEE 802.15.4 Coordinator      │
│  ├─ LoRa Control Channel (50kbps)   │
│  └─ Ethernet/UART to MCU           │
│                                     │
│  Storage:                            │
│  ├─ MongoDB/InfluxDB               │
│  ├─ Model Parameters               │
│  └─ Event History                  │
└─────────────────────────────────────┘
```

## Core Algorithms

### 1. Truncated Lyapunov Game

The truncated Lyapunov function prevents energy overflow:

```
L_trunc(Q_E) = {
  0.5 × Σ Q_E,i²,                           Q_E,i ≤ 0.9 × C_cap
  0.5 × (0.9 × C_cap)² + β × (Q_E,i - 0.9 × C_cap)⁴,  otherwise
}
```

**Key Features:**
- Quadratic growth for safe energy levels
- Quartic penalty for overflow prevention
- Bounded energy queues
- Incentive for coalition participation

**MCU Implementation:**
```c
float truncated_lyapunov(float Q_E) {
    const float cap_max = 330.0f;  // 100μF at 3.3V
    const float threshold = 0.9f * cap_max;
    const float beta = 0.1f;

    if (Q_E <= threshold) {
        return 0.5f * Q_E * Q_E;
    } else {
        float excess = Q_E - threshold;
        return 0.5f * threshold * threshold + beta * excess⁴;
    }
}
```

**Gateway Python Implementation:**
```python
def truncated_lyapunov(self, Q_E):
    threshold = 0.9 * self.C_cap
    if Q_E <= threshold:
        return 0.5 * Q_E**2
    else:
        excess = Q_E - threshold
        return 0.5 * threshold**2 + self.beta * excess**4
```

### 2. Stratified Shapley Values

Efficient approximation using spatial stratification:

```
m_k = ⌈|C_k|/N × log(1/δ) / ε²⌉
```

**Complexity:** O(N log(1/δ) / ε²)

**Implementation (Gateway):**
```python
class StratifiedShapleyApproximator:
    def __init__(self, N, epsilon=0.1, delta=0.05):
        self.N = N
        self.K = math.ceil(N / math.log2(N))  # Number of strata

    def approximate_shapley_values(self, game, positions):
        strata = self.partition_into_strata(positions)
        phi = {i: 0.0 for i in range(self.N)}

        for stratum_id, nodes in strata.items():
            mk = len(nodes) / self.N * (math.log(1/self.delta) / self.epsilon**2)
            for _ in range(int(mk)):
                coalition = random.sample(nodes, random.randint(0, len(nodes)))
                for i in nodes:
                    if i not in coalition:
                        phi[i] += self.marginal_contribution(i, coalition, game)

        return {i: phi[i] / len(nodes) for i in range(self.N)}
```

### 3. TinyLSTM Energy Predictor

Quantized LSTM for energy prediction:

**Architecture:**
- Input: 10-step energy history
- Hidden: 32 units (int8 weights)
- Output: 1-step prediction
- Activation: 2-bit ReLU
- Memory: <4KB

**Quantization Scheme:**
- Weights: int8 symmetric quantization
- Activations: 2-bit (ternary: {-1, 0, +1})
- Inference energy: ~23μJ

**MCU Implementation:**
```c
void tinylstm_predict(float* input_seq, float* output) {
    int8_t q_input[10];

    // Quantize input
    for (int i = 0; i < 10; i++) {
        q_input[i] = quantize_to_int8(input_seq[i]);
    }

    // LSTM forward pass (simplified)
    for (int step = 0; step < 10; step++) {
        for (int h = 0; h < 32; h++) {
            // Compute gates with quantized weights
            // Use 2-bit activation: {-1, 0, 1}
        }
    }

    // Dequantize output
    *output = dequantize_from_float(q_output);
}
```

### 4. KF-GP Hybrid Recovery

Combines temporal (KF) and spatial (GP) modeling:

**Kalman Filter:** State = [energy, trend]
```python
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.F = [[1., 1.],    # State transition
        [0., 1.]]
kf.H = [[1., 0.]]    # Observation matrix
```

**Gaussian Process:** Spatial correlation
```python
self.gp = GaussianProcessRegressor(
    kernel=RBF(length_scale=15.0),
    alpha=1e-6
)
```

**Switching Logic:**
```python
def predict_missing_data(self, observed_nodes, values):
    # KF prediction
    kf_pred = self.get_kf_predictions()

    # GP spatial interpolation
    if len(observed_nodes) > 2:
        residuals = values - kf_pred[observed_nodes]
        self.gp.fit(observed_coords, residuals)
        gp_correction = self.gp.predict(all_coords)
        return kf_pred + gp_correction
    else:
        return kf_pred
```

**Robust Mode (Energy Scarcity):**
```python
if np.sum(current_energies < 0.3 * 330.0) / self.N > 0.1:
    self.switch_to_robust_mode()  # Use Cauchy kernel
```

### 5. Pedersen Commitment (ZK Proof)

Zero-knowledge proof of energy state:

**Commitment:** C = g^x × h^r
- x: energy value
- r: random nonce

**MCU Implementation:**
```c
void pedersen_commit(float energy_value, commitment_t* output) {
    uint32_t energy_int = energy_value * 1000;  // mJ precision

    // Generate random nonce
    trng_generate_random_bytes(g_commitment_nonce, 32);

    // Compute commitment: C = G^energy × H^nonce
    scalar_multiplication(G, energy_int, output->point);
    point_multiply_with_nonce(output->point, g_commitment_nonce);

    output->timestamp = get_system_time_ms();
}
```

**Gateway Verification:**
```python
def verify_commitment(commitment, energy_value, nonce):
    # Recompute G^energy
    G_val = scalar_multiply(G, energy_value * 1000)

    # Recompute H^nonce
    H_nonce = hash_to_curve(nonce)

    # Verify equality
    return point_equal(commitment.point, point_add(G_val, H_nonce))
```

## System Operation Flow

### Initialization Phase

1. **Node Boot:**
   - Initialize BQ25570 PMIC
   - Load TinyLSTM weights from Flash
   - Generate Pedersen key pair
   - Join network via IEEE 802.15.4

2. **Gateway Startup:**
   - Start Shapley server
   - Initialize KF-GP model
   - Listen for node advertisements
   - Build spatial map

### Operation Loop (100ms slots)

**Node Side:**
```
for each 100ms slot:
    1. Sample capacitor voltage (ADC)
    2. Update energy history buffer
    3. Run TinyLSTM prediction
    4. Compute local marginal utility
    5. Package status (energy, queue, utility)
    6. Send to gateway via radio
    7. Wait 10ms for response
    8. Execute tasks or fallback
    9. Enter sleep mode
```

**Gateway Side:**
```
for each 100ms slot:
    1. Collect node statuses
    2. Update game state (Lyapunov)
    3. Every 10 slots: compute Shapley values
    4. Update KF-GP model
    5. Broadcast coalition assignments
    6. Verify ZK proofs
    7. Route data and relay
```

### Communication Protocol

**Node → Gateway:**
- Packet type: STATUS
- Frequency: Every slot
- Size: 40 bytes
- Contents: node_id, Q_E, B_i, marginal_utility, commitment

**Gateway → Node:**
- Packet type: COMMAND
- Frequency: Every 10 slots
- Size: 20 bytes
- Contents: shapley_value, assigned_task, coalition_map

**Node ↔ Node:**
- Packet type: RELAY
- Frequency: On demand
- Size: 32 bytes
- Contents: dest_node, data_payload, signature

## Energy Flow

### Energy Harvesting (BQ25570)
```
Solar Panel (0.8-3.2V)
    ↓
BQ25570 MPPT (tracks 2.4V)
    ↓
Storage Capacitor (100μF)
    ↓
LDO (3.3V)
    ↓
CC2650 VDD
```

**Energy Costs (per operation):**
- TinyLSTM inference: 23.1 μJ
- Pedersen commitment: 45.3 μJ
- Radio transmission: 5.3 μJ
- Sense operation: 1.0 μJ
- Sleep mode: 0.05 μJ/slot

### Capacitor Management

**Voltage to Energy Conversion:**
```
E = 0.5 × C × V²
For C = 100μF, V = 3.3V:
E = 0.5 × 100×10⁻⁶ × 3.3² = 544.5 μJ

Maximum safe: 330 μJ (at 2.57V after losses)
```

## Security Model

### Pedersen Commitment Flow

1. **Node generates random nonce** (from TRNG)
2. **Compute commitment** to current energy
3. **Send commitment** to gateway
4. **Gateway stores commitment** and timestamp
5. **Node reveals energy** with proof
6. **Gateway verifies** commitment equality

**Properties:**
- **Hiding:** Energy value not revealed
- **Binding:** Cannot change commitment after sending
- **Unlinkable:** Multiple commitments unlinkable
- **Efficient:** 64-byte commitment, 32-byte verification

### Attack Scenarios

1. **Energy Falsification:**
   - Defense: Pedersen commitment + ZK proof
   - Cost: ~45μJ per proof

2. **Eclipse Attack:**
   - Defense: Multi-path routing via coalitions
   - Cost: Energy for relay tasks

3. **Denial of Service:**
   - Defense: Lyapunov-bound resource allocation
   - Cost: Energy for malicious tasks is limited

## Performance Characteristics

### Computational Complexity

**MCU (per slot):**
- TinyLSTM: O(H×D) = 32×10 = 320 ops
- Lyapunov: O(1) = 10 ops
- Shapley local: O(K) = 5 ops
- Total: ~335 ops → 0.007ms @ 48MHz

**Gateway (per superframe = 10 slots):**
- Shapley: O(N log(1/δ)/ε²) = 50×log(20)/0.01 = ~2000 ops
- KF-GP: O(N×M) = 50×log(50) = ~195 ops
- Total: ~2200 ops → 2.2ms @ 1.5GHz

### Memory Footprint

**MCU:**
```
Flash (128KB):
├─ Application code: ~80KB
├─ TinyLSTM weights: ~4KB
├─ Pedersen constants: ~1KB
└─ Vector table: ~1KB

SRAM (40KB):
├─ Runtime buffers: ~12KB
├─ TinyLSTM states: ~4KB
├─ Event trace: ~1KB
└─ Stack/heap: ~22KB
```

**Gateway:**
```
RAM (2GB):
├─ Shapley state: ~50MB
├─ KF-GP model: ~100MB
├─ Data history: ~500MB
└─ OS overhead: ~1.3GB
```

## Scalability

### Network Size (N)

**Computational Scaling:**
- MCU: O(1) per node (constant)
- Gateway: O(N log N) for Shapley
- Communication: O(N) per slot

**Recommended Limits:**
- Minimum: N = 5 nodes
- Typical: N = 50 nodes
- Maximum: N = 1000 nodes (with optimization)

**Optimization for Large N:**
- Inducing points for GP (M = log N)
- Hierarchical Shapley (O(log N) strata)
- Distributed Shapley computation
- Hierarchical routing

## Reliability

### Fault Tolerance

1. **Node Failure:**
   - Automatic detection: No status for 10 slots
   - Coalition reassignment: Redistribute tasks
   - Data recovery: KF-GP interpolation

2. **Gateway Failure:**
   - Secondary gateway: Hot standby
   - Checkpointing: Save state every hour
   - Recovery time: <30 seconds

3. **Energy Depletion:**
   - Graceful degradation: Reduce task rate
   - Sleep extension: Increase slot duration
   - Survivor mode: Coalition of last nodes

### Monitoring

**Node Metrics:**
- Capacitor voltage
- Energy harvesting rate
- Task success rate
- Memory usage
- Radio link quality

**Gateway Metrics:**
- Shapley value convergence
- Data integrity
- Energy waste
- Coalition formation
- ZK proof verification rate

## Hardware Requirements

### Minimum Testbed (N=5)

1. **5x TI CC2650STK** ($125)
2. **5x BQ25570EVM-206** ($500)
3. **5x 100μF capacitors** ($5)
4. **Raspberry Pi 4B** ($75)
5. **1x CC1352P1 LaunchPad** ($30)
6. **Monsoon HV Power Monitor** ($1500)

**Total: ~$2,235**

### Full Testbed (N=50)

1. **50x CC2650STK** ($1,250)
2. **50x BQ25570EVM-206** ($5,000)
3. **50x 100μF capacitors** ($50)
4. **XPC240400B-02 gateway** ($2,500)
5. **10x DJI Matrice 100** ($50,000)
10x **Powercast P2110B** ($500)

**Total: ~$59,300**
