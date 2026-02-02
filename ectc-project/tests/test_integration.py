"""
Test configurations for ECTC components
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch


@pytest.fixture
def sample_energy_data():
    """Fixture providing sample energy data"""
    return {
        'timestamps': np.arange(100),
        'energies': np.random.uniform(0, 330, 100),
        'harvest_rates': np.random.uniform(0, 10, 100)
    }


@pytest.fixture
def sample_node_states():
    """Fixture providing sample node states"""
    return [
        {
            'node_id': i,
            'energy': np.random.uniform(0, 330),
            'queue': np.random.randint(0, 10),
            'position': (np.random.uniform(0, 100), np.random.uniform(0, 100))
        }
        for i in range(10)
    ]


@pytest.fixture
def mock_gateway():
    """Fixture providing mock gateway"""
    gateway = Mock()
    gateway.nodes = {i: Mock() for i in range(10)}
    gateway.get_status.return_value = {'status': 'running'}
    return gateway


@pytest.fixture
def mock_monsoon():
    """Fixture providing mock Monsoon power monitor"""
    monsoon = Mock()
    monsoon.read_power.return_value = np.random.uniform(0, 100, 1000)
    monsoon.read_voltage.return_value = np.random.uniform(3.0, 3.6, 1000)
    return monsoon


@pytest.fixture
def temp_config(tmp_path):
    """Fixture providing temporary configuration file"""
    config_file = tmp_path / "test_config.yaml"
    config_content = """
gateway:
  port: 8080
  host: "localhost"
network:
  num_nodes: 10
"""
    config_file.write_text(config_content)
    return str(config_file)


class TestEnergyModels:
    """Test energy harvesting models"""

    def test_solar_model(self):
        """Test solar energy model"""
        from simulation.simulator import EnergySource

        source = EnergySource('solar')
        energy = source.sample(1000, (50, 50))

        assert energy >= 0
        assert energy <= 20  # Reasonable upper bound

    def test_rf_model(self):
        """Test RF energy model"""
        from simulation.simulator import EnergySource

        source = EnergySource('rf')
        energy = source.sample(1000, (50, 50))

        assert energy >= 0
        assert energy <= 10

    def test_vibration_model(self):
        """Test vibration energy model"""
        from simulation.simulator import EnergySource

        source = EnergySource('vibration')
        energy = source.sample(1000, (50, 50))

        assert energy >= 0
        assert energy <= 5


class TestNetworking:
    """Test networking components"""

    def test_packet_creation(self):
        """Test packet creation"""
        from gateway.ectc_gateway.core.packets import DataPacket

        packet = DataPacket(node_id=1, energy=250.0, queue=3)

        assert packet.node_id == 1
        assert packet.energy == 250.0
        assert packet.queue == 3

    def test_radio_initialization(self):
        """Test radio initialization"""
        # Mock radio initialization
        assert True  # Replace with actual test


class TestAlgorithms:
    """Test algorithm implementations"""

    def test_lyapunov_stability(self, sample_energy_data):
        """Test Lyapunov function stability"""
        from gateway.ectc_gateway.core.lyapunov import TruncatedLyapunovGame

        game = TruncatedLyapunovGame(N=10)
        energies = sample_energy_data['energies']

        L = game.truncated_lyapunov(energies)

        assert L >= 0  # Lyapunov should be non-negative

    def test_shapley_convergence(self, sample_node_states):
        """Test Shapley value convergence"""
        from gateway.ectc_gateway.core.shapley import StratifiedShapleyApproximator

        approximator = StratifiedShapleyApproximator(N=10, epsilon=0.1)
        positions = {state['node_id']: state['position'] for state in sample_node_states}

        # Should converge within expected error bound
        error = approximator.get_error_bounds()['max_error']
        assert error <= 0.1


class TestUtilities:
    """Test utility functions"""

    def test_energy_to_voltage_conversion(self):
        """Test energy to voltage conversion"""
        from tools.energy_profiler import EnergyProfiler

        profiler = EnergyProfiler()

        # For 100μF capacitor
        energy_uj = 330.0  # Maximum energy
        voltage = profiler.energy_to_voltage(energy_uj, 100e-6)

        assert voltage > 0
        assert voltage <= 3.3  # Should not exceed 3.3V

    def test_voltage_to_energy_conversion(self):
        """Test voltage to energy conversion"""
        from tools.energy_profiler import EnergyProfiler

        profiler = EnergyProfiler()

        voltage = 3.3
        energy_uj = profiler.voltage_to_energy(voltage, 100e-6)

        assert energy_uj > 0
        assert energy_uj <= 545  # 0.5 * C * V^2 * 1e6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
