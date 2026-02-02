"""
Unit tests for ECTC components
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.ectc_gateway.core.shapley_server import (
    TruncatedLyapunovGame,
    StratifiedShapleyApproximator,
    ShapleyServer
)
from gateway.ectc_gateway.core.kf_gp_hybrid import KFGPHybridModel


class TestTruncatedLyapunovGame:
    """Test truncated Lyapunov game"""

    def test_initialization(self):
        game = TruncatedLyapunovGame(N=10)
        assert game.N == 10
        assert game.C_cap == 330.0
        assert game.beta == 0.1
        assert game.V == 50.0

    def test_truncated_lyapunov_safe_region(self):
        game = TruncatedLyapunovGame(N=5)
        Q_E = np.array([100.0, 150.0, 200.0, 250.0, 270.0])  # All below 90% threshold
        L = game.truncated_lyapunov(Q_E)
        # Should be 0.5 * sum(Q^2)
        expected = 0.5 * np.sum(Q_E**2)
        assert abs(L - expected) < 1e-6

    def test_truncated_lyapunov_saturation_region(self):
        game = TruncatedLyapunovGame(N=3)
        Q_E = np.array([350.0, 360.0, 370.0])  # All above 90% threshold
        L = game.truncated_lyapunov(Q_E)
        threshold = 0.9 * 330.0
        for q in Q_E:
            assert q > threshold
        # Should include quartic penalty term
        assert L > 0.5 * threshold * threshold

    def test_drift_computation(self):
        game = TruncatedLyapunovGame(N=5)
        Q_E = np.array([100.0, 150.0, 200.0, 250.0, 270.0])
        Q_E_next = np.array([95.0, 145.0, 195.0, 245.0, 265.0])  # Slight decrease
        drift = game.compute_drift(Q_E, Q_E_next)
        # Negative drift is expected (energy decreased)
        assert drift < 0

    def test_coalition_utility(self):
        game = TruncatedLyapunovGame(N=10)
        game.Q_E = np.random.uniform(0, 330, 10)
        game.info_utils = np.random.uniform(0, 10, 10)

        S = [0, 1, 2]  # Coalition of first 3 nodes
        utility = game.coalition_utility(S)

        # Utility should be a scalar
        assert isinstance(utility, float)
        # Utility should be finite
        assert np.isfinite(utility)


class TestStratifiedShapleyApproximator:
    """Test stratified Shapley approximation"""

    def test_initialization(self):
        approx = StratifiedShapleyApproximator(N=50)
        assert approx.N == 50
        assert approx.epsilon == 0.1
        assert approx.delta == 0.05
        # K should be ceil(N / log2(N))
        assert approx.K == np.ceil(50 / np.log2(50))

    def test_partition_into_strata(self):
        approx = StratifiedShapleyApproximator(N=20, K=4)
        positions = {i: (np.random.rand()*100, np.random.rand()*100) for i in range(20)}

        strata = approx.partition_into_strata(positions)

        # Should have K strata
        assert len(strata) == approx.K
        # All nodes should be in some stratum
        all_nodes = []
        for nodes in strata.values():
            all_nodes.extend(nodes)
        assert set(all_nodes) == set(range(20))

    def test_marginal_contribution(self):
        approx = StratifiedShapleyApproximator(N=10)
        game = TruncatedLyapunovGame(N=10)

        # Set specific game state
        game.Q_E = np.array([100.0] * 10)
        game.info_utils = np.array([1.0] * 10)

        i = 5
        S = [0, 1, 2]

        mc = approx.marginal_contribution(i, S, game)

        # Should be a float
        assert isinstance(mc, float)
        # Should be finite
        assert np.isfinite(mc)

    def test_error_bounds(self):
        approx = StratifiedShapleyApproximator(N=50, epsilon=0.1, delta=0.05)
        bounds = approx.get_error_bounds()

        assert 'max_error' in bounds
        assert 'confidence' in bounds
        assert 'expected_samples_per_node' in bounds
        assert 'total_samples' in bounds

        assert bounds['max_error'] == 0.1
        assert bounds['confidence'] == 0.95


class TestShapleyServer:
    """Test Shapley server"""

    def test_initialization(self):
        server = ShapleyServer(N=20)
        assert server.N == 20
        assert server.shapley_server is not None
        assert server.approximator is not None
        assert len(server.positions) == 20

    def test_update_node_status(self):
        from gateway.ectc_gateway.core.shapley_server import NodeStatus

        server = ShapleyServer(N=10)

        status = NodeStatus(
            node_id=3,
            Q_E=250.0,
            B_i=5,
            marginal_utility=0.75,
            has_data=True,
            position=(50.0, 50.0)
        )

        server.update_node_status(status)

        # Check game state was updated
        assert server.game.Q_E[3] == 250.0
        assert server.game.info_utils[3] == 0.75
        # Position was stored
        assert server.positions[3] == (50.0, 50.0)


class TestKFGPHybridModel:
    """Test KF-GP hybrid model"""

    def test_initialization(self):
        N = 20
        coords = np.random.rand(N, 2) * 100
        model = KFGPHybridModel(N, coords)

        assert model.N == N
        assert len(model.node_states) == N
        assert model.gp is not None
        assert model.M == int(np.log2(N))

    def test_kalman_filter_initialization(self):
        N = 10
        coords = np.random.rand(N, 2) * 100
        model = KFGPHybridModel(N, coords)

        # Check each node has a Kalman filter
        for node_id in range(N):
            node_state = model.node_states[node_id]
            assert node_state.node_id == node_id
            assert node_state.kf.dim_x == 2
            assert node_state.kf.dim_z == 1

    def test_update_node(self):
        N = 10
        coords = np.random.rand(N, 2) * 100
        model = KFGPHybridModel(N, coords)

        # Update a node with measurement
        model.update_node(0, 250.0, timestamp=10.0)

        # Check state was updated
        node_state = model.node_states[0]
        assert node_state.last_update == 10.0
        assert node_state.kf.x[0, 0] == 250.0

    def test_predict_missing_data(self):
        N = 10
        coords = np.random.rand(N, 2) * 100
        model = KFGPHybridModel(N, coords)

        # Observed nodes
        observed_nodes = [0, 1, 2]
        observed_values = [250.0, 245.0, 255.0]

        predictions, std = model.predict_missing_data(observed_nodes, observed_values)

        # Should return predictions for all nodes
        assert len(predictions) == N
        assert len(std) == N
        # All predictions should be finite
        assert np.all(np.isfinite(predictions))
        assert np.all(np.isfinite(std))

    def test_spatial_correlation(self):
        N = 20
        coords = np.random.rand(N, 2) * 100
        model = KFGPHybridModel(N, coords)

        # Simulate some energy states
        for i in range(N):
            model.update_node(i, np.random.uniform(0, 330), timestamp=i)

        correlation = model.analyze_spatial_correlation()

        assert 'mean_distance' in correlation
        assert 'max_distance' in correlation
        assert 'energy_correlation' in correlation
        assert 'energy_range' in correlation

        assert 0 <= correlation['energy_correlation'] <= 1

    def test_robust_mode_switching(self):
        N = 10
        coords = np.random.rand(N, 2) * 100
        model = KFGPHybridModel(N, coords)

        # Simulate low energy state
        for i in range(N):
            model.update_node(i, 50.0, timestamp=i)

        # Switch to robust mode
        model.switch_to_robust_mode(enable=True, threshold=0.3)

        # Kernel should be Cauchy
        assert isinstance(model.gp.kernel, type(model.kernel_cauchy))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
