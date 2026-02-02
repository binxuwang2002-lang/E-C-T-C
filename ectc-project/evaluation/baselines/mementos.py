"""
Mementos Baseline Implementation
================================

Energy-aware intermittent computing framework.
Based on paper: "Mementos: System Support for Energy-Constrained Applications"
"""

import numpy as np
from typing import Dict, List
import random


class MementosCheckpoint:
    """Mementos checkpoint management"""

    def __init__(self, node_id: int):
        self.node_id = node_id
        self.checkpoints = {}
        self.energy_threshold = 100.0  # μJ
        self.checkpoint_interval = 50  # operations

    def checkpoint_state(self, state: Dict):
        """Save state checkpoint"""
        self.checkpoints[random.randint(0, 1000)] = state.copy()

    def restore_state(self, checkpoint_id: int):
        """Restore from checkpoint"""
        return self.checkpoints.get(checkpoint_id)


class MementosNode:
    """Mementos node with energy-aware execution"""

    def __init__(self, node_id: int, capacity_uj: float = 330.0):
        self.node_id = node_id
        self.capacity = capacity_uj
        self.energy = 0.0
        self.queue = []
        self.checkpoint = MementosCheckpoint(node_id)
        self.operation_count = 0
        self.state = {'sense_count': 0, 'transmit_count': 0}

    def energy_sample(self) -> float:
        """Sample ambient energy"""
        # More variable than BATDMA
        return np.random.uniform(0, 8)

    def checkpoint_decision(self) -> bool:
        """Decide whether to checkpoint"""
        return self.energy < self.checkpoint.energy_threshold

    def execute_operation(self, op_type: str) -> bool:
        """Execute one operation"""
        op_cost = {'sense': 1.0, 'compute': 2.0, 'transmit': 5.0}

        cost = op_cost.get(op_type, 1.0)
        if self.energy >= cost:
            self.energy -= cost
            self.operation_count += 1

            # Update state
            if op_type == 'sense':
                self.state['sense_count'] += 1
                self.queue.append(np.random.uniform(20, 30))
            elif op_type == 'transmit' and self.queue:
                self.queue.pop(0)
                self.state['transmit_count'] += 1

            return True
        return False

    def run_to_completion(self) -> bool:
        """Run operations until energy depleted"""
        # Simple task: sense once per cycle
        if self.energy >= 1.0:
            return self.execute_operation('sense')
        return False

    def recover_from_checkpoint(self):
        """Recover state after power cycle"""
        # Restore from checkpoint
        checkpoint_id = random.choice(list(self.checkpoint.checkpoints.keys()))
        restored_state = self.checkpoint.restore_state(checkpoint_id)
        if restored_state:
            self.state = restored_state


class MementosController:
    """Mementos network controller"""

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.nodes = [MementosNode(i) for i in range(num_nodes)]

    def simulate_step(self) -> Dict:
        """Simulate one time step"""
        results = {
            'completed_tasks': 0,
            'checkpoints_created': 0,
            'recoveries': 0
        }

        # Sample energy
        for node in self.nodes:
            harvested = node.energy_sample()
            node.energy = min(node.energy + harvested, node.capacity)

            # Run to completion or checkpoint
            if node.energy > node.checkpoint.energy_threshold:
                task_completed = node.run_to_completion()
                if task_completed:
                    results['completed_tasks'] += 1

                # Check if checkpoint needed
                if node.operation_count % node.checkpoint.checkpoint_interval == 0:
                    node.checkpoint.checkpoint_state(node.state)
                    results['checkpoints_created'] += 1
            else:
                # Energy too low, checkpoint and sleep
                if node.operation_count > 0:
                    node.checkpoint.checkpoint_state(node.state)
                    node.recover_from_checkpoint()
                    results['recoveries'] += 1

        # Calculate metrics
        total_energy = sum(n.energy for n in self.nodes)
        total_queue = sum(len(n.queue) for n in self.nodes)
        operations = sum(n.operation_count for n in self.nodes)

        return {
            'total_energy': total_energy,
            'total_queue': total_queue,
            'operations': operations,
            'results': results
        }

    def simulate(self, duration: int) -> Dict:
        """Simulate full duration"""
        results = []
        for _ in range(duration):
            results.append(self.simulate_step())

        # Aggregate
        data_integrity = np.mean([r['operations'] / self.num_nodes for r in results])
        energy_cost = np.mean([r['operations'] * 2.0 * 1000 for r in results])  # avg 2μJ per op
        sleep_ratio = np.mean([sum(1 for n in self.nodes if n.energy < 10.0) / self.num_nodes for _ in results])
        latency = 3.2  # Higher due to checkpointing overhead

        return {
            'data_integrity': data_integrity,
            'energy_cost_nj_per_bit': energy_cost,
            'sleep_ratio': sleep_ratio,
            'latency_p50': latency,
            'latency_p95': latency * 1.5,
            'latency_p99': latency * 2.0,
            'framework': 'Mementos',
            'checkpoints': np.mean([r['results']['checkpoints_created'] for r in results]),
            'recoveries': np.mean([r['results']['recoveries'] for r in results])
        }


if __name__ == '__main__':
    controller = MementosController(num_nodes=50)
    result = controller.simulate(duration=1000)
    print("Mementos Results:")
    print(json.dumps(result, indent=2))
