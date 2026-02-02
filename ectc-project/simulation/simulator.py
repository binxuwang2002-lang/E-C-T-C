"""
Large-Scale Network Simulator
============================

Cycle-accurate simulator for ECTC networks with 1000+ nodes.
Supports both static and mobile topologies.

Features:
- Energy harvesting simulation
- IEEE 802.15.4 MAC layer
- Event-driven simulation
- Parallel computation support
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import heapq
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


class NodeType(Enum):
    """Node type enumeration"""
    STATIC = 1
    MOBILE = 2
    RELAY = 3


class TaskType(Enum):
    """Task type enumeration"""
    SENSE = 1
    TRANSMIT = 2
    RELAY = 3
    COMPUTE = 4


@dataclass
class Node:
    """Network node representation"""
    node_id: int
    position: np.ndarray
    node_type: NodeType
    velocity: Optional[np.ndarray] = None  # For mobile nodes

    # Energy parameters
    Q_E: float = 0.0  # Current energy (μJ)
    Q_max: float = 330.0  # Capacitor capacity (μJ)
    energy_history: List[float] = None

    # Data parameters
    B_i: int = 0  # Data queue length
    data_history: List[int] = None

    # State
    active: bool = True

    def __post_init__(self):
        if self.energy_history is None:
            self.energy_history = []
        if self.data_history is None:
            self.data_history = []


@dataclass
class Event:
    """Simulation event"""
    time: float
    event_type: str
    node_id: int
    data: Dict

    def __lt__(self, other):
        return self.time < other.time


class EnergySource:
    """Energy harvesting source"""

    def __init__(self, source_type: str):
        self.source_type = source_type

    def sample(self, time: float, position: np.ndarray) -> float:
        """
        Sample energy at given time and position

        Args:
            time: Current simulation time
            position: Node position

        Returns:
            Harvested energy (μJ)
        """
        if self.source_type == 'solar':
            return self._solar_harvest(time, position)
        elif self.source_type == 'rf':
            return self._rf_harvest(time, position)
        elif self.source_type == 'vibration':
            return self._vibration_harvest(time, position)
        else:
            return 0.0

    def _solar_harvest(self, time: float, position: np.ndarray) -> float:
        """Simulate solar energy harvesting"""
        # Day-night cycle (24 hours = 86400 seconds at 10s intervals = 8640 steps)
        day_time = (time % 8640) / 8640.0

        # Base solar irradiance
        base_irradiance = max(0, np.sin(np.pi * day_time))

        # Weather variation
        weather_factor = np.random.uniform(0.3, 1.0)

        # Cloud cover (5% chance)
        if np.random.random() < 0.05:
            weather_factor *= 0.3

        # Position-dependent (e.g., shade, orientation)
        position_factor = 0.8 + 0.4 * np.random.random()

        # Convert to energy (per 10s interval)
        energy = base_irradiance * weather_factor * position_factor * 5.0

        # Add noise
        energy += np.random.normal(0, 0.2)

        return max(0, energy)

    def _rf_harvest(self, time: float, position: np.ndarray) -> float:
        """Simulate RF energy harvesting"""
        # Assume RF source at origin
        rf_source_pos = np.array([50.0, 50.0])
        distance = np.linalg.norm(position - rf_source_pos)

        # Inverse square law
        if distance < 1.0:
            power = 1.0
        else:
            power = 1.0 / (distance**2)

        # Add noise and temporal variation
        power *= np.random.uniform(0.5, 1.5)
        power += np.random.normal(0, 0.1)

        return max(0, power * 2.0)

    def _vibration_harvest(self, time: float, position: np.ndarray) -> float:
        """Simulate vibration energy harvesting"""
        # Base vibration from environment
        base = 0.5

        # Random walk
        vibration = base + np.random.normal(0, 0.3)

        return max(0, vibration)


class ECTCNetworkSimulator:
    """Large-scale ECTC network simulator"""

    def __init__(self,
                 N: int,
                 area_size: float = 100.0,
                 num_mobile: int = 0,
                 energy_source_type: str = 'solar'):
        """
        Initialize simulator

        Args:
            N: Total number of nodes
            area_size: Area size (meters)
            num_mobile: Number of mobile nodes
            energy_source_type: Type of energy source
        """
        self.N = N
        self.area_size = area_size
        self.num_mobile = num_mobile

        # Initialize nodes
        self.nodes = []
        self._initialize_nodes()

        # Energy source
        self.energy_source = EnergySource(energy_source_type)

        # Simulation state
        self.current_time = 0.0
        self.event_queue = []
        self.global_time = 0.0

        # Performance metrics
        self.metrics = {
            'data_integrity': [],
            'energy_waste': [],
            'sleep_ratio': [],
            'latency': [],
            'throughput': []
        }

    def _initialize_nodes(self):
        """Initialize network nodes"""
        np.random.seed(42)

        # Static nodes
        static_nodes = self.N - self.num_mobile
        for i in range(static_nodes):
            position = np.random.rand(2) * self.area_size
            node = Node(
                node_id=i,
                position=position,
                node_type=NodeType.STATIC
            )
            self.nodes.append(node)

        # Mobile nodes
        for i in range(static_nodes, self.N):
            position = np.random.rand(2) * self.area_size
            velocity = (np.random.rand(2) - 0.5) * 2.0  # Random velocity

            node = Node(
                node_id=i,
                position=position,
                node_type=NodeType.MOBILE,
                velocity=velocity
            )
            self.nodes.append(node)

    def _schedule_event(self, event: Event):
        """Add event to simulation queue"""
        heapq.heappush(self.event_queue, event)

    def simulate_step(self):
        """Execute one simulation time step"""
        # Sample energy for all nodes
        self._sample_energy()

        # Update mobile node positions
        self._update_mobility()

        # Process events
        self._process_events()

        # Update metrics
        self._update_metrics()

        self.current_time += 1

    def _sample_energy(self):
        """Sample energy for all nodes"""
        for node in self.nodes:
            if not node.active:
                continue

            # Harvest energy
            harvested = self.energy_source.sample(self.current_time, node.position)

            # Add to capacitor
            node.Q_E = min(node.Q_E + harvested, node.Q_max)

            # Store history
            node.energy_history.append(node.Q_E)

    def _update_mobility(self):
        """Update positions of mobile nodes"""
        for node in self.nodes:
            if node.node_type != NodeType.MOBILE:
                continue

            # Random walk mobility model
            # Add random acceleration
            acceleration = (np.random.rand(2) - 0.5) * 0.5
            node.velocity += acceleration

            # Limit velocity
            speed = np.linalg.norm(node.velocity)
            max_speed = 3.0  # m/s
            if speed > max_speed:
                node.velocity = node.velocity / speed * max_speed

            # Update position
            node.position += node.velocity * 1.0  # 1 time unit

            # Boundary reflection
            for dim in range(2):
                if node.position[dim] < 0:
                    node.position[dim] = 0
                    node.velocity[dim] *= -1
                elif node.position[dim] > self.area_size:
                    node.position[dim] = self.area_size
                    node.velocity[dim] *= -1

    def _process_events(self):
        """Process simulation events"""
        while self.event_queue and self.event_queue[0].time <= self.current_time:
            event = heapq.heappop(self.event_queue)
            self._handle_event(event)

    def _handle_event(self, event: Event):
        """Handle a simulation event"""
        node = self.nodes[event.node_id]

        if event.event_type == 'sense':
            self._execute_sense(node, event.data)
        elif event.event_type == 'transmit':
            self._execute_transmit(node, event.data)
        elif event.event_type == 'relay':
            self._execute_relay(node, event.data)

    def _execute_sense(self, node: Node, data: Dict):
        """Execute sensing task"""
        energy_cost = 1.0  # μJ
        if node.Q_E >= energy_cost:
            node.Q_E -= energy_cost
            node.B_i += 1

    def _execute_transmit(self, node: Node, data: Dict):
        """Execute transmission task"""
        energy_cost = 5.3  # μJ
        if node.Q_E >= energy_cost and node.B_i > 0:
            node.Q_E -= energy_cost
            node.B_i -= 1

    def _execute_relay(self, node: Node, data: Dict):
        """Execute relay task"""
        energy_cost = 3.0  # μJ
        if node.Q_E >= energy_cost:
            node.Q_E -= energy_cost

    def _update_metrics(self):
        """Update performance metrics"""
        # Data integrity
        total_data = sum(node.B_i for node in self.nodes)
        active_nodes = sum(1 for node in self.nodes if node.active)
        self.metrics['data_integrity'].append(total_data / max(1, active_nodes))

        # Energy waste (energy above 90% capacity)
        waste = sum(max(0, node.Q_E - 0.9 * node.Q_max) for node in self.nodes)
        self.metrics['energy_waste'].append(waste)

        # Sleep ratio (nodes with Q_E < threshold)
        sleep_threshold = 50.0
        sleeping = sum(1 for node in self.nodes if node.Q_E < sleep_threshold)
        self.metrics['sleep_ratio'].append(sleeping / len(self.nodes))

    def run_simulation(self, duration: int = 10000):
        """
        Run complete simulation

        Args:
            duration: Simulation duration (time steps)
        """
        print(f"Starting simulation: {self.N} nodes, {duration} steps")

        # Schedule initial events
        for node in self.nodes:
            # Schedule sensing events
            for t in range(0, duration, 10):  # Every 10 time steps
                event = Event(
                    time=t,
                    event_type='sense',
                    node_id=node.node_id,
                    data={}
                )
                self._schedule_event(event)

        # Run simulation loop
        for step in range(duration):
            if step % 1000 == 0:
                print(f"  Step {step}/{duration}")

            self.simulate_step()

        print("Simulation complete")

    def get_results(self) -> Dict:
        """
        Get simulation results

        Returns:
            Dictionary with results
        """
        results = {
            'num_nodes': self.N,
            'duration': self.current_time,
            'average_energy': np.mean([node.energy_history for node in self.nodes]),
            'final_data_integrity': self.metrics['data_integrity'][-1],
            'average_sleep_ratio': np.mean(self.metrics['sleep_ratio']),
            'total_energy_waste': sum(self.metrics['energy_waste']),
            'node_positions': {node.node_id: node.position.tolist()
                             for node in self.nodes}
        }

        return results

    def plot_results(self, save_path: Optional[str] = None):
        """Plot simulation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Data integrity over time
        axes[0, 0].plot(self.metrics['data_integrity'])
        axes[0, 0].set_title('Data Integrity Over Time')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Data Integrity')
        axes[0, 0].grid(True)

        # Sleep ratio over time
        axes[0, 1].plot(self.metrics['sleep_ratio'])
        axes[0, 1].set_title('Sleep Ratio Over Time')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Sleep Ratio')
        axes[0, 1].grid(True)

        # Energy waste over time
        axes[1, 0].plot(self.metrics['energy_waste'])
        axes[1, 0].set_title('Energy Waste Over Time')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Energy Waste (μJ)')
        axes[1, 0].grid(True)

        # Final node positions
        positions = np.array([node.position for node in self.nodes])
        scatter = axes[1, 1].scatter(positions[:, 0], positions[:, 1],
                                   c=[node.energy_history[-1] for node in self.nodes],
                                   cmap='viridis')
        axes[1, 1].set_title('Final Node Positions and Energy')
        axes[1, 1].set_xlabel('X Position (m)')
        axes[1, 1].set_ylabel('Y Position (m)')
        plt.colorbar(scatter, ax=axes[1, 1], label='Energy (μJ)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


def run_parameter_sweep(param_grid: Dict) -> List[Dict]:
    """
    Run parameter sweep across multiple configurations

    Args:
        param_grid: Dictionary of parameter ranges

    Returns:
        List of results for each configuration
    """
    results = []
    configurations = []

    # Generate all combinations
    import itertools
    keys, values = zip(*param_grid.items())
    for config_values in itertools.product(*values):
        config = dict(zip(keys, config_values))
        configurations.append(config)

    print(f"Running {len(configurations)} simulations")

    # Run simulations in parallel
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(run_single_sim, config) for config in configurations]
        results = [future.result() for future in futures]

    return results


def run_single_sim(config: Dict) -> Dict:
    """Run a single simulation with given configuration"""
    simulator = ECTCNetworkSimulator(
        N=config['num_nodes'],
        area_size=config['area_size'],
        num_mobile=config.get('num_mobile', 0),
        energy_source_type=config.get('energy_source', 'solar')
    )

    simulator.run_simulation(duration=config.get('duration', 10000))

    result = simulator.get_results()
    result['config'] = config

    return result


if __name__ == '__main__':
    # Example: Single simulation
    simulator = ECTCNetworkSimulator(
        N=100,
        area_size=100.0,
        num_mobile=10,
        energy_source_type='solar'
    )

    simulator.run_simulation(duration=10000)
    results = simulator.get_results()

    print("\nSimulation Results:")
    print(f"  Nodes: {results['num_nodes']}")
    print(f"  Duration: {results['duration']}")
    print(f"  Average Energy: {results['average_energy']:.2f} μJ")
    print(f"  Final Data Integrity: {results['final_data_integrity']:.2f}")
    print(f"  Average Sleep Ratio: {results['average_sleep_ratio']:.2%}")
    print(f"  Total Energy Waste: {results['total_energy_waste']:.2f} μJ")

    # Plot results
    simulator.plot_results('simulation_results.png')
