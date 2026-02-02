# ECTC API Reference

## Overview

This document provides a complete API reference for the ECTC Gateway services.

## REST API

### Base URL

```
http://localhost:8080/api/v1
```

### Authentication

Currently, no authentication is required. In future versions, API keys will be supported.

```yaml
# Future authentication
Authorization: Bearer <api_key>
```

### Endpoints

#### System Status

**GET /status**

Get overall system status.

**Response:**
```json
{
  "status": "running",
  "uptime": 12345,
  "version": "1.0.0",
  "nodes_active": 45,
  "nodes_total": 50,
  "shapley_converged": true,
  "shapley_error": 0.082,
  "energy_harvest_avg": 245.3,
  "data_integrity": 0.932,
  "memory_usage_mb": 512,
  "cpu_usage_percent": 34.5
}
```

**Response Fields:**
- `status`: System status (`running`, `stopped`, `error`)
- `uptime`: System uptime in seconds
- `version`: ECTC version
- `nodes_active`: Number of active nodes
- `nodes_total`: Total number of nodes
- `shapley_converged`: Whether Shapley values converged
- `shapley_error`: Current Shapley approximation error
- `energy_harvest_avg`: Average energy harvest rate (μJ/slot)
- `data_integrity`: Percentage of data successfully delivered
- `memory_usage_mb`: Gateway memory usage
- `cpu_usage_percent`: Gateway CPU usage

---

#### Node Management

**GET /nodes**

Get all nodes.

**Response:**
```json
[
  {
    "node_id": 0,
    "status": "active",
    "energy_uj": 245.3,
    "energy_max_uj": 330.0,
    "queue_len": 3,
    "position": {
      "x": 25.4,
      "y": 48.2
    },
    "last_seen": "2024-01-15T10:30:45Z",
    "shapley_value": 0.85,
    "has_data": true,
    "task_assignments": [
      {
        "task_id": "transmit_data",
        "priority": 0.9
      }
    ],
    "metrics": {
      "lstm_inference_time_ms": 2.3,
      "radio_tx_count": 15,
      "radio_rx_count": 12,
      "task_success_rate": 0.97
    }
  }
]
```

---

**GET /nodes/{node_id}**

Get specific node information.

**Parameters:**
- `node_id`: Node identifier (0-255)

**Response:** Same as GET /nodes but for single node.

---

**GET /nodes/{node_id}/energy**

Get node energy history.

**Query Parameters:**
- `duration`: Time window (e.g., "1h", "24h", "7d")
- `granularity`: Data granularity ("1s", "10s", "1m")

**Response:**
```json
{
  "node_id": 0,
  "start_time": "2024-01-15T00:00:00Z",
  "end_time": "2024-01-15T23:59:59Z",
  "granularity": "10s",
  "data": [
    {
      "timestamp": "2024-01-15T00:00:00Z",
      "energy_uj": 245.3,
      "harvest_rate_uj_per_slot": 5.2,
      "predicted_next_uj": 248.1
    }
  ]
}
```

---

**GET /nodes/{node_id}/metrics**

Get node performance metrics.

**Response:**
```json
{
  "node_id": 0,
  "metrics": {
    "energy":
      "total_consumed_uj": 5421.5,
      "average_per_slot_uj": 35.2,
      "efficiency_percent": 87.3,
      "lstm_accuracy": 0.91,
      "lstm_inference_energy_uj": 23.1
    },
    "communication":
      "packets_sent": 150,
      "packets_received": 142,
      "packet_error_rate": 0.053,
      "average_rssi_dbm": -65.4,
      "average_latency_ms": 1.83
    ,
    "tasks":
      "tasks_assigned": 23,
      "tasks_completed": 22,
      "success_rate": 0.957,
      "average_completion_time_ms": 245
    }
  }
}
```

---

#### Shapley Values

**GET /shapley**

Get current Shapley values.

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:45Z",
  "converged": true,
  "error": 0.082,
  "epsilon": 0.1,
  "delta": 0.05,
  "iterations": 45,
  "values": {
    "0": 0.85,
    "1": 0.72,
    "2": 0.91
  },
  "coalitions": [
    {
      "coalition_id": 0,
      "members": [0, 1, 2],
      "total_utility": 2.48,
      "members_utility": {
        "0": 0.85,
        "1": 0.72,
        "2": 0.91
      }
    }
  ]
}
```

**Response Fields:**
- `timestamp`: Time when computed
- `converged`: Whether computation converged
- `error`: Current approximation error
- `epsilon`: Target error bound
- `delta`: Confidence parameter
- `iterations`: Number of iterations performed
- `values`: Shapley value for each node
- `coalitions`: Formed coalitions

---

**POST /shapley/recompute**

Trigger Shapley value recomputation.

**Request:**
```json
{
  "force": false,
  "max_iterations": 100
}
```

**Parameters:**
- `force`: Force recomputation even if converged
- `max_iterations`: Maximum iterations to perform

**Response:**
```json
{
  "status": "initiated",
  "request_id": "abc123",
  "estimated_completion": "2024-01-15T10:31:05Z"
}
```

---

#### Data Recovery

**GET /recovery/predictions**

Get KF-GP predictions for missing data.

**Query Parameters:**
- `node_ids`: Comma-separated list of node IDs
- `timestamp`: Prediction timestamp (ISO 8601)

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:45Z",
  "observed_nodes": [0, 1, 5],
  "predictions": {
    "0": 245.3,
    "1": 238.7,
    "2": 252.1,
    "3": 241.9,
    "4": 248.5
  },
  "uncertainty": {
    "0": 5.2,
    "1": 6.1,
    "2": 4.8,
    "3": 7.3,
    "4": 6.5
  },
  "gp_kernel": "rbf",
  "robust_mode": false
}
```

---

**GET /recovery/spatial**

Get spatial correlation analysis.

**Response:**
```json
{
  "num_nodes": 50,
  "analysis": {
    "mean_distance": 42.5,
    "max_distance": 98.3,
    "energy_correlation": 0.73,
    "energy_range": 145.2,
    "correlation_length": 25.8
  },
  "clusters": [
    {
      "cluster_id": 0,
      "center": {
        "x": 25.4,
        "y": 48.2
      },
      "size": 12,
      "avg_energy": 245.3,
      "members": [0, 1, 3, 5, 8, ...]
    }
  ]
}
```

---

#### Energy Monitoring

**GET /energy/harvest**

Get energy harvesting statistics.

**Query Parameters:**
- `duration`: Time window (e.g., "1h", "24h", "7d")
- `group_by`: Aggregation ("node", "hour", "day")

**Response:**
```json
{
  "duration": "24h",
  "start_time": "2024-01-15T00:00:00Z",
  "end_time": "2024-01-15T23:59:59Z",
  "total_harvested_uj": 1542000,
  "average_rate_uj_per_slot": 5.2,
  "nodes": [
    {
      "node_id": 0,
      "total_uj": 32450,
      "average_per_slot_uj": 5.4,
      "peak_per_slot_uj": 12.3,
      "min_per_slot_uj": 0.1
    }
  ]
}
```

---

**GET /energy/waste**

Get energy waste statistics.

**Response:**
```json
{
  "total_waste_uj": 12540,
  "waste_percentage": 8.13,
  "by_node": [
    {
      "node_id": 0,
      "waste_uj": 342,
      "waste_percentage": 1.05,
      "overflow_events": 3,
      "max_energy_uj": 329.8
    }
  ],
  "recommendations": [
    {
      "node_id": 0,
      "action": "reduce_harvest_rate",
      "expected_savings_uj": 150
    }
  ]
}
```

---

#### Security

**GET /zkp/verify**

Verify Pedersen commitments.

**Query Parameters:**
- `node_id`: Node to verify
- `since`: Check commits since timestamp

**Response:**
```json
{
  "verified": true,
  "total_checked": 24,
  "total_valid": 24,
  "total_invalid": 0,
  "invalid_commits": [],
  "verification_time_ms": 15.6,
  "proofs": [
    {
      "node_id": 0,
      "timestamp": "2024-01-15T10:30:00Z",
      "energy_reported": 245.3,
      "commitment_valid": true
    }
  ]
}
```

---

**POST /zkp/challenge**

Send ZK proof challenge to node.

**Request:**
```json
{
  "node_id": 0,
  "challenge": "random_32_byte_value",
  "require_proof": true
}
```

**Response:**
```json
{
  "status": "sent",
  "node_id": 0,
  "challenge_id": "xyz789",
  "timeout_ms": 5000
}
```

---

#### Configuration

**GET /config**

Get current configuration.

**Response:**
```yaml
gateway:
  port: 8080
  host: 0.0.0.0
  log_level: INFO

network:
  protocol: IEEE_802_15_4
  channel: 11
  pan_id: 0x1234

algorithms:
  lyapunov:
    v: 50.0
    beta: 0.1
  shapley:
    epsilon: 0.1
    delta: 0.05
```

---

**PUT /config**

Update configuration.

**Request:** Same format as GET /config response

**Response:**
```json
{
  "status": "updated",
  "restart_required": false,
  "updated_fields": ["shapley.epsilon"],
  "message": "Configuration updated successfully"
}
```

---

**POST /config/reload**

Reload configuration from file.

**Response:**
```json
{
  "status": "reloaded",
  "timestamp": "2024-01-15T10:35:00Z"
}
```

---

#### System Control

**POST /shutdown**

Shutdown gateway gracefully.

**Request:**
```json
{
  "delay_seconds": 10,
  "reason": "maintenance"
}
```

**Response:**
```json
{
  "status": "shutdown_scheduled",
  "shutdown_at": "2024-01-15T10:35:10Z"
}
```

---

**POST /nodes/{node_id}/reset**

Reset specific node.

**Parameters:**
- `node_id`: Node identifier

**Response:**
```json
{
  "status": "reset_sent",
  "node_id": 0,
  "reset_type": "soft",
  "message": "Reset command sent to node"
}
```

---

**GET /logs**

Get gateway logs.

**Query Parameters:**
- `level`: Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `since`: Start time (ISO 8601)
- `count`: Number of lines (max 1000)

**Response:**
```json
{
  "logs": [
    {
      "timestamp": "2024-01-15T10:30:45Z",
      "level": "INFO",
      "module": "shapley",
      "message": "Shapley values computed for 50 nodes",
      "node_id": null
    }
  ],
  "total": 1,
  "truncated": false
}
```

---

## WebSocket API

### Connection

```
ws://localhost:8080/ws
```

### Messages

#### Subscribe to Node Updates

**Client → Server:**
```json
{
  "type": "subscribe",
  "topic": "node_updates",
  "node_id": 0
}
```

**Server → Client:**
```json
{
  "type": "node_update",
  "node_id": 0,
  "timestamp": "2024-01-15T10:30:45Z",
  "data": {
    "energy_uj": 245.3,
    "queue_len": 3,
    "shapley_value": 0.85
  }
}
```

#### Subscribe to Energy Metrics

**Client → Server:**
```json
{
  "type": "subscribe",
  "topic": "energy_metrics"
}
```

**Server → Client:**
```json
{
  "type": "energy_update",
  "timestamp": "2024-01-15T10:30:45Z",
  "data": {
    "total_harvested_uj": 1542000,
    "average_rate_uj_per_slot": 5.2,
    "waste_percentage": 8.13
  }
}
```

#### Alert Notifications

**Server → Client:**
```json
{
  "type": "alert",
  "alert_id": "alert_123",
  "severity": "warning",
  "message": "Node 5 energy below threshold",
  "timestamp": "2024-01-15T10:31:00Z",
  "data": {
    "node_id": 5,
    "energy_uj": 12.3,
    "threshold_uj": 50.0
  }
}
```

---

## Python SDK

### Installation

```bash
pip install ectc-sdk
```

### Usage

```python
from ectc import ECTCClient

# Initialize client
client = ECTCClient(
    host="localhost",
    port=8080,
    api_key="your_api_key"  # optional
)

# Get system status
status = client.get_status()
print(f"Active nodes: {status['nodes_active']}")

# Get node information
node = client.get_node(0)
print(f"Node 0 energy: {node['energy_uj']:.2f} μJ")

# Get Shapley values
shapley = client.get_shapley_values()
print(f"Node 0 Shapley value: {shapley['values']['0']:.2f}")

# Get predictions
predictions = client.get_predictions(node_ids=[0, 1, 2])
print(f"Predictions: {predictions['predictions']}")

# WebSocket subscription
def on_node_update(node_id, data):
    print(f"Node {node_id} update: {data}")

client.subscribe_node_updates(0, callback=on_node_update)

# Keep connection alive
client.run_forever()
```

### API Reference

#### ECTCClient

**Constructor:**
```python
ECTCClient(
    host: str = "localhost",
    port: int = 8080,
    api_key: str = None,
    timeout: float = 10.0
)
```

**Methods:**

- `get_status() -> dict`: Get system status
- `get_nodes() -> list`: Get all nodes
- `get_node(node_id: int) -> dict`: Get specific node
- `get_node_energy(node_id: int, duration: str = "1h") -> dict`
- `get_node_metrics(node_id: int) -> dict`
- `get_shapley_values() -> dict`
- `recompute_shapley(force: bool = False) -> dict`
- `get_predictions(node_ids: List[int], timestamp: str = None) -> dict`
- `get_spatial_analysis() -> dict`
- `get_energy_harvest(duration: str = "24h") -> dict`
- `get_energy_waste() -> dict`
- `verify_zkp(node_id: int, since: str = None) -> dict`
- `get_config() -> dict`
- `update_config(config: dict) -> dict`
- `reset_node(node_id: int) -> dict`
- `shutdown(delay_seconds: int = 0, reason: str = None) -> dict`

**WebSocket Methods:**

- `subscribe_node_updates(node_id: int, callback: Callable)`: Subscribe to node updates
- `subscribe_energy_metrics(callback: Callable)`: Subscribe to energy metrics
- `subscribe_alerts(callback: Callable)`: Subscribe to alerts
- `run_forever()`: Keep WebSocket connection alive

### Example Scripts

**Monitor Node Energy:**

```python
from ectc import ECTCClient
import time

client = ECTCClient()

def on_energy_alert(node_id, energy, threshold):
    print(f"ALERT: Node {node_id} energy {energy:.2f} below threshold {threshold:.2f}")

# Subscribe to node 0
client.subscribe_node_updates(0, callback=print)

# Monitor for low energy
for node in client.get_nodes():
    if node['energy_uj'] < 50.0:
        print(f"Warning: Node {node['node_id']} low energy: {node['energy_uj']:.2f} μJ")

client.run_forever()
```

**Track Shapley Convergence:**

```python
from ectc import ECTCClient

client = ECTCClient()

# Get initial Shapley values
shapley = client.get_shapley_values()
initial_error = shapley['error']

print(f"Initial Shapley error: {initial_error:.4f}")

# Wait for convergence
max_wait = 60  # seconds
start_time = time.time()

while shapley['error'] > 0.1 and time.time() - start_time < max_wait:
    time.sleep(1)
    shapley = client.get_shapley_values()

if shapley['converged']:
    print(f"Converged after {time.time() - start_time:.1f}s")
    print(f"Final error: {shapley['error']:.4f}")
else:
    print(f"Did not converge within {max_wait}s")
```

---

## Rate Limits

### REST API

- GET requests: 1000 requests/minute per IP
- POST/PUT requests: 100 requests/minute per IP
- WebSocket connections: 10 connections per IP

### Error Responses

**429 Too Many Requests:**
```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests",
  "retry_after": 60
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request |
| 404 | Not Found |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Error Response Format

```json
{
  "error": "error_code",
  "message": "Human-readable message",
  "details": {
    "field": "Additional error details"
  }
}
```

---

## SDKs and Tools

### Available SDKs

- **Python**: `pip install ectc-sdk`
- **JavaScript/Node.js**: `npm install ectc-sdk`
- **C++**: (Contact maintainers)
- **Rust**: (Planned)

### CLI Tool

```bash
# Install CLI
pip install ectc-cli

# Check system status
ectc status

# List nodes
ectc nodes list

# Get node info
ectc nodes info 0

# Get Shapley values
ectc shapley get

# Trigger recomputation
ectc shapley recompute

# Get predictions
ectc recovery predict --nodes 0,1,2

# Monitor in real-time
ectc monitor --node 0
```

---

## Changelog

### Version 1.0.0 (2024-01-15)

**Added:**
- Initial API release
- REST API endpoints
- WebSocket support
- Python SDK
- CLI tool

**Endpoints:**
- System status
- Node management
- Shapley values
- Data recovery
- Energy monitoring
- ZK proof verification
- Configuration management
- System control

---

## Support

### API Support

- **Email**: api-support@ectc-project.org
- **GitHub**: https://github.com/ectc/ectc-project/issues
- **Documentation**: https://ectc.readthedocs.io

### Reporting Issues

Please include:
1. API endpoint
2. Request/response data
3. Error messages
4. Gateway logs
5. ECTC version

---

## License

ECTC API is licensed under Apache License 2.0.
