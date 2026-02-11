"""
DINO-BFSN Baseline Implementation
=================================

Volatility-adaptive checkpointing for battery-free sensor networks.

Key Features:
- Dynamic checkpoint interval: T_checkpoint ∝ Q_E(t)
- Formula: current_interval = base_interval * (current_energy / max_energy)
- High energy → fewer checkpoints (reduce overhead)
- Low energy → more checkpoints (protect progress)
- Physics-grounded: Uses FEMP energy model with parasitic capacitance

Reference: DINO Paper - "Intermittent Computing with Dynamic Checkpointing"
"""

import sys
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulation.energy_model import FEMPEnergyModel, TaskType, FEMPParameters


@dataclass
class DINOCheckpoint:
    """
    DINO checkpoint state.
    
    Stores complete node state for recovery after power failure.
    """
    checkpoint_id: int
    operation_count: int
    state: Dict
    energy_at_checkpoint: float
    timestamp: int
    
    def __repr__(self) -> str:
        return (f"Checkpoint(id={self.checkpoint_id}, "
                f"ops={self.operation_count}, "
                f"energy={self.energy_at_checkpoint:.1f}μJ)")


class DINONode:
    """
    DINO node with volatility-adaptive checkpointing.
    
    Dynamically adjusts checkpoint interval based on current energy level:
    - High energy: Longer intervals (fewer checkpoints, less overhead)
    - Low energy: Shorter intervals (more checkpoints, protect progress)
    """
    
    def __init__(self,
                 node_id: int,
                 capacity_uj: float = 330.0,
                 base_interval: int = 50,
                 energy_model: Optional[FEMPEnergyModel] = None):
        """
        Initialize DINO node.
        
        Args:
            node_id: Unique node identifier
            capacity_uj: Energy storage capacity in microjoules (max_energy)
            base_interval: Base checkpoint interval in operations
            energy_model: FEMP energy model for physics-grounded calculations
        """
        self.node_id = node_id
        self.capacity = capacity_uj  # max_energy
        self.energy = 0.0  # current_energy Q_E(t)
        
        # DINO-specific parameters
        self.base_interval = base_interval
        self.current_interval = base_interval  # Dynamically adjusted
        self.min_interval = 5   # Minimum checkpoint interval
        self.max_interval = 200 # Maximum checkpoint interval
        
        # Energy model with parasitic parameters
        self.energy_model = energy_model if energy_model else FEMPEnergyModel()
        
        # Checkpoint storage
        self.checkpoints: List[DINOCheckpoint] = []
        self.checkpoint_counter = 0
        self.operations_since_checkpoint = 0
        
        # Node state
        self.state: Dict = {
            'sense_count': 0,
            'compute_count': 0,
            'transmit_count': 0,
            'data_accumulated': 0.0,
        }
        
        # Data buffer
        self.data_buffer: List[float] = []
        
        # Statistics
        self.total_operations = 0
        self.checkpoints_created = 0
        self.recoveries = 0
        self.total_energy_consumed = 0.0
        self.checkpoint_overhead = 0.0  # Energy spent on checkpoints
    
    def compute_checkpoint_interval(self) -> int:
        """
        Compute adaptive checkpoint interval.
        
        Formula: current_interval = base_interval * (current_energy / max_energy)
        
        This means:
        - At full energy (100%): interval = base_interval
        - At 50% energy: interval = base_interval * 0.5
        - At 10% energy: interval = base_interval * 0.1
        
        Returns:
            Checkpoint interval in operations
        """
        energy_ratio = self.energy / self.capacity
        
        # Apply formula: T_checkpoint ∝ Q_E(t)
        dynamic_interval = int(self.base_interval * energy_ratio)
        
        # Clamp to valid range
        self.current_interval = max(self.min_interval, 
                                    min(self.max_interval, dynamic_interval))
        
        return self.current_interval
    
    def get_checkpoint_energy_cost(self) -> float:
        """
        Get energy cost for creating a checkpoint.
        
        Uses FEMP model to compute checkpoint cost including parasitic overhead.
        Checkpoint is modeled as a COMPUTE operation (writing to NVM).
        
        Returns:
            Checkpoint energy cost in microjoules
        """
        # Checkpoint involves NVM write, modeled as short compute operation
        checkpoint_duration_ms = 0.5  # 500 microseconds
        
        return self.energy_model.predict_task_energy(
            TaskType.COMPUTE,
            checkpoint_duration_ms,
            add_noise=False
        )
    
    def should_checkpoint(self) -> bool:
        """
        Check if a checkpoint should be created.
        
        Based on adaptive interval computed from current energy level.
        
        Returns:
            True if checkpoint needed
        """
        # Recompute interval based on current energy
        self.compute_checkpoint_interval()
        
        return self.operations_since_checkpoint >= self.current_interval
    
    def create_checkpoint(self, timestamp: int) -> Optional[DINOCheckpoint]:
        """
        Create a checkpoint if energy allows.
        
        Args:
            timestamp: Current simulation timestamp
            
        Returns:
            Created checkpoint, or None if insufficient energy
        """
        checkpoint_cost = self.get_checkpoint_energy_cost()
        
        if self.energy < checkpoint_cost:
            return None
        
        # Create checkpoint
        checkpoint = DINOCheckpoint(
            checkpoint_id=self.checkpoint_counter,
            operation_count=self.total_operations,
            state=self.state.copy(),
            energy_at_checkpoint=self.energy,
            timestamp=timestamp
        )
        
        # Consume checkpoint energy
        self.energy -= checkpoint_cost
        self.total_energy_consumed += checkpoint_cost
        self.checkpoint_overhead += checkpoint_cost
        
        # Store checkpoint (keep limited history)
        self.checkpoints.append(checkpoint)
        if len(self.checkpoints) > 5:
            self.checkpoints.pop(0)
        
        # Update counters
        self.checkpoint_counter += 1
        self.checkpoints_created += 1
        self.operations_since_checkpoint = 0
        
        return checkpoint
    
    def recover_from_checkpoint(self) -> bool:
        """
        Recover state from most recent checkpoint.
        
        Called after simulated power failure.
        
        Returns:
            True if recovery successful, False if no checkpoint available
        """
        if not self.checkpoints:
            # No checkpoint, reset to initial state
            self.state = {
                'sense_count': 0,
                'compute_count': 0,
                'transmit_count': 0,
                'data_accumulated': 0.0,
            }
            return False
        
        # Restore from most recent checkpoint
        latest = self.checkpoints[-1]
        self.state = latest.state.copy()
        self.operations_since_checkpoint = 0
        self.recoveries += 1
        
        return True
    
    def execute_operation(self, op_type: TaskType, duration_ms: float) -> bool:
        """
        Execute an operation with energy accounting.
        
        Args:
            op_type: Type of operation
            duration_ms: Operation duration
            
        Returns:
            True if successful, False if insufficient energy
        """
        # Get energy cost including parasitic overhead
        energy_cost = self.energy_model.predict_task_energy(
            op_type,
            duration_ms,
            add_noise=True
        )
        
        if self.energy < energy_cost:
            return False
        
        # Execute operation
        self.energy -= energy_cost
        self.total_energy_consumed += energy_cost
        self.total_operations += 1
        self.operations_since_checkpoint += 1
        
        # Update state based on operation type
        if op_type == TaskType.SENSE:
            self.state['sense_count'] += 1
            value = np.random.uniform(20, 30)
            self.data_buffer.append(value)
            self.state['data_accumulated'] += value
        elif op_type == TaskType.COMPUTE:
            self.state['compute_count'] += 1
        elif op_type == TaskType.TRANSMIT:
            self.state['transmit_count'] += 1
            if self.data_buffer:
                self.data_buffer.pop(0)
        
        return True
    
    def harvest_energy(self) -> float:
        """
        Harvest ambient energy.
        
        Returns:
            Amount of energy harvested in microjoules
        """
        # Variable energy harvesting
        harvested = np.random.uniform(0, 8)
        self.energy = min(self.energy + harvested, self.capacity)
        return harvested
    
    def simulate_power_failure(self) -> bool:
        """
        Simulate a power failure event.
        
        Returns:
            True if recovery was possible
        """
        # Clear energy
        self.energy = 0.0
        
        # Attempt recovery
        return self.recover_from_checkpoint()


class DINOController:
    """
    DINO-BFSN network controller.
    
    Manages a network of DINO nodes with volatility-adaptive checkpointing.
    """
    
    def __init__(self, 
                 num_nodes: int, 
                 base_interval: int = 50,
                 seed: Optional[int] = None):
        """
        Initialize DINO controller.
        
        Args:
            num_nodes: Number of nodes in network
            base_interval: Base checkpoint interval for all nodes
            seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self.base_interval = base_interval
        self.rng = np.random.default_rng(seed)
        
        # Shared energy model for consistent physics
        self.energy_model = FEMPEnergyModel(seed=seed)
        
        # Create nodes
        self.nodes = [
            DINONode(
                i, 
                base_interval=base_interval,
                energy_model=self.energy_model
            )
            for i in range(num_nodes)
        ]
        
        self.tick = 0
        self.power_failures = 0
    
    def schedule(self, task_queue: List, energy_state: Dict[int, float]) -> List:
        """
        Schedule operations for nodes.
        
        In DINO, scheduling is simple - execute available operations
        while managing checkpoints adaptively.
        
        Args:
            task_queue: Not used (DINO uses local task generation)
            energy_state: Dict mapping node_id to current energy
            
        Returns:
            List of scheduled operations
        """
        scheduled = []
        
        for node in self.nodes:
            if node.node_id in energy_state:
                node.energy = energy_state[node.node_id]
            
            # Check for checkpoint
            if node.should_checkpoint():
                checkpoint = node.create_checkpoint(self.tick)
                if checkpoint:
                    scheduled.append((node.node_id, 'checkpoint', checkpoint))
            
            # Schedule next operation if energy available
            if node.energy > 10.0:  # Minimum energy threshold
                scheduled.append((node.node_id, 'execute', None))
        
        return scheduled
    
    def simulate_step(self) -> Dict:
        """
        Simulate one time step.
        
        Returns:
            Dictionary with step metrics
        """
        results = {
            'operations_completed': 0,
            'checkpoints_created': 0,
            'power_failures': 0,
            'recoveries': 0,
            'total_energy': 0.0,
            'checkpoint_intervals': [],
        }
        
        # Harvest energy for all nodes
        for node in self.nodes:
            node.harvest_energy()
            results['total_energy'] += node.energy
        
        # Simulate potential power failures (rare)
        for node in self.nodes:
            if self.rng.random() < 0.01:  # 1% chance of power failure
                results['power_failures'] += 1
                self.power_failures += 1
                if node.simulate_power_failure():
                    results['recoveries'] += 1
                # After failure, node needs to recharge
                continue
            
            # Check for adaptive checkpointing
            results['checkpoint_intervals'].append(node.current_interval)
            
            if node.should_checkpoint():
                checkpoint = node.create_checkpoint(self.tick)
                if checkpoint:
                    results['checkpoints_created'] += 1
            
            # Execute operations
            if node.energy > 10.0:
                # Choose random operation
                op_types = [TaskType.SENSE, TaskType.COMPUTE, TaskType.TRANSMIT]
                weights = [0.5, 0.3, 0.2]
                op_type = self.rng.choice(op_types, p=weights)
                
                # Duration varies by operation
                durations = {
                    TaskType.SENSE: self.rng.uniform(0.5, 1.5),
                    TaskType.COMPUTE: self.rng.uniform(1.0, 3.0),
                    TaskType.TRANSMIT: self.rng.uniform(1.5, 2.5),
                }
                
                if node.execute_operation(op_type, durations[op_type]):
                    results['operations_completed'] += 1
        
        self.tick += 1
        return results
    
    def simulate(self, duration: int) -> Dict:
        """
        Simulate full duration.
        
        Args:
            duration: Number of simulation steps
            
        Returns:
            Aggregated simulation results
        """
        step_results = []
        all_intervals = []
        
        for _ in range(duration):
            result = self.simulate_step()
            step_results.append(result)
            all_intervals.extend(result['checkpoint_intervals'])
        
        # Aggregate metrics
        total_ops = sum(r['operations_completed'] for r in step_results)
        total_checkpoints = sum(r['checkpoints_created'] for r in step_results)
        total_failures = sum(r['power_failures'] for r in step_results)
        total_recoveries = sum(r['recoveries'] for r in step_results)
        
        # Calculate standard metrics
        data_integrity = total_recoveries / max(1, total_failures) if total_failures > 0 else 1.0
        
        # Energy overhead from checkpointing
        total_checkpoint_overhead = sum(n.checkpoint_overhead for n in self.nodes)
        total_energy_consumed = sum(n.total_energy_consumed for n in self.nodes)
        checkpoint_overhead_ratio = total_checkpoint_overhead / max(1, total_energy_consumed)
        
        # Average checkpoint interval (shows adaptive behavior)
        avg_interval = np.mean(all_intervals) if all_intervals else self.base_interval
        
        # Latency (checkpointing adds overhead)
        base_latency = 3.0
        latency_overhead = checkpoint_overhead_ratio * 2.0
        
        return {
            'data_integrity': min(1.0, data_integrity),
            'energy_cost_nj_per_bit': total_energy_consumed * 1000 / max(1, total_ops) / 8,
            'sleep_ratio': 1.0 - (total_ops / (duration * self.num_nodes)),
            'latency_p50': base_latency + latency_overhead,
            'latency_p95': (base_latency + latency_overhead) * 1.5,
            'latency_p99': (base_latency + latency_overhead) * 2.0,
            'framework': 'DINO-BFSN',
            'checkpointing': 'Volatility-Adaptive',
            'base_interval': self.base_interval,
            'avg_dynamic_interval': avg_interval,
            'total_operations': total_ops,
            'total_checkpoints': total_checkpoints,
            'total_power_failures': total_failures,
            'recovery_success_rate': total_recoveries / max(1, total_failures) if total_failures > 0 else 1.0,
            'checkpoint_overhead_ratio': checkpoint_overhead_ratio,
            'parasitic_model': f'C_bus={self.energy_model.params.C_bus*1e12:.1f}pF'
        }


if __name__ == '__main__':
    print("=" * 60)
    print("DINO-BFSN Baseline Simulation")
    print("=" * 60)
    
    # Create controller with reproducible seed
    controller = DINOController(num_nodes=50, base_interval=50, seed=42)
    
    # Run simulation
    result = controller.simulate(duration=1000)
    
    print("\nDINO-BFSN Results:")
    print(json.dumps(result, indent=2))
    
    # Show checkpoint interval adaptation
    print("\n--- Adaptive Checkpoint Intervals ---")
    print(f"  Base interval: {controller.base_interval}")
    print(f"  Average dynamic interval: {result['avg_dynamic_interval']:.1f}")
    
    # Show per-node statistics
    print("\n--- Sample Node Statistics ---")
    for i in [0, 10, 25]:
        node = controller.nodes[i]
        print(f"  Node {i}: ops={node.total_operations}, "
              f"checkpoints={node.checkpoints_created}, "
              f"recoveries={node.recoveries}, "
              f"current_interval={node.current_interval}")
    
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
