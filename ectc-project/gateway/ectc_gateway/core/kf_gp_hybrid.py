"""
KF-GP Hybrid Model for Robust Data Recovery
===========================================

Combines Kalman Filter (temporal dynamics) with Gaussian Process (spatial correlation)
for robust energy harvesting data recovery under energy scarcity.

When energy is abundant: Use GP for spatial interpolation
When energy is scarce: Switch to Cauchy kernel for robustness
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from filterpy.kalman import KalmanFilter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, CauchyKernel, ConstantKernel
import matplotlib.pyplot as plt


@dataclass
class NodeState:
    """Kalman Filter state for a single node"""
    node_id: int
    kf: KalmanFilter
    last_update: float
    position: Tuple[float, float]


class KFGPHybridModel:
    """
    Hybrid Kalman Filter + Gaussian Process for energy data recovery.

    Combines:
    - Kalman Filter: Temporal prediction (Markovian dynamics)
    - Gaussian Process: Spatial correlation (spatio-temporal modeling)
    - Automatic switching: RBF → Cauchy when energy is scarce
    """

    def __init__(self,
                 N: int,
                 spatial_coords: np.ndarray,
                 length_scale: float = 15.0,
                 sigma_f: float = 1.0,
                 cauchy_scale: float = 5.0):
        """
        Initialize KF-GP hybrid model

        Args:
            N: Number of nodes
            spatial_coords: [N, 2] array of node positions
            length_scale: RBF kernel length scale (meters)
            sigma_f: GP signal variance
            cauchy_scale: Cauchy kernel scale parameter
        """
        self.N = N
        self.coords = spatial_coords
        self.current_time = 0.0

        # Initialize Kalman Filters for each node
        self.node_states = {}
        for i in range(N):
            self.node_states[i] = self._init_kalman_filter(i)

        # Initialize Gaussian Process
        self.kernel_rbf = ConstantKernel(constant_value=sigma_f**2) * \
                         RBF(length_scale=length_scale)
        self.kernel_cauchy = ConstantKernel(constant_value=sigma_f**2) * \
                            CauchyKernel(length_scale=cauchy_scale)

        self.gp = GaussianProcessRegressor(
            kernel=self.kernel_rbf,
            alpha=1e-6,  # Noise variance
            n_restarts_optimizer=5
        )

        # Inducing points for sparse GP (M = log N)
        self.M = int(np.log2(N))
        self._init_inducing_points()

        # Robust mode flag (activated when energy is scarce)
        self.robust_mode = False

        # History for temporal analysis
        self.time_history = []
        self.energy_history = {i: [] for i in range(N)}

    def _init_kalman_filter(self, node_id: int) -> NodeState:
        """
        Initialize Kalman Filter for a single node

        Model: x(t+1) = x(t) + v(t) + w(t)
               y(t) = x(t) + n(t)

        Where:
        - x(t): Energy state
        - v(t): Trend (optional)
        - w(t): Process noise
        - n(t): Observation noise

        Args:
            node_id: Node ID

        Returns:
            NodeState with initialized Kalman Filter
        """
        # 2D state: [energy, trend]
        dim_x = 2

        # Create Kalman Filter
        kf = KalmanFilter(dim_x=dim_x, dim_z=1)

        # State transition matrix
        # x(t+1) = [1, 1; 0, 1] * x(t) + w(t)
        kf.F = np.array([[1., 1.],
                        [0., 1.]])

        # Observation matrix
        # y(t) = [1, 0] * x(t) + n(t)
        kf.H = np.array([[1., 0.]])

        # Initial state covariance
        kf.P = np.eye(dim_x) * 10.0

        # Process noise covariance
        kf.Q = np.array([[0.1, 0.],
                        [0., 0.01]])

        # Observation noise covariance
        kf.R = 0.5  # Based on sensor noise

        # Initial state
        kf.x = np.array([[0.],  # Energy
                        [0.]])  # Trend

        return NodeState(
            node_id=node_id,
            kf=kf,
            last_update=0.0,
            position=self.coords[node_id]
        )

    def _init_inducing_points(self):
        """
        Initialize inducing points for sparse GP using K-means
        """
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=self.M, random_state=42)
        self.inducing_points = kmeans.fit(self.coords).cluster_centers_

    def update_node(self,
                   node_id: int,
                   measurement: Optional[float],
                   timestamp: Optional[float] = None):
        """
        Update node state with new measurement

        Args:
            node_id: Node ID
            measurement: Energy measurement (μJ), or None if unavailable
            timestamp: Measurement timestamp
        """
        if timestamp is None:
            timestamp = self.current_time

        node_state = self.node_states[node_id]

        # Compute time delta
        dt = timestamp - node_state.last_update
        if dt <= 0:
            dt = 1.0  # Default time step

        # Update state transition matrix with time delta
        node_state.kf.F[0, 1] = dt

        # Predict step
        node_state.kf.predict()

        # Update step if measurement available
        if measurement is not None:
            node_state.kf.update(np.array([[measurement]]))

        node_state.last_update = timestamp

        # Store in history
        self.energy_history[node_id].append(measurement)

    def predict_missing_data(self,
                           observed_nodes: List[int],
                           observed_values: List[float],
                           timestamp: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict energy values for all nodes using hybrid KF-GP approach

        Args:
            observed_nodes: List of observed node IDs
            observed_values: Corresponding energy values
            timestamp: Prediction timestamp

        Returns:
            Tuple of (predictions, prediction_std)
            - predictions: [N] array of predicted energies
            - prediction_std: [N] array of prediction uncertainties
        """
        if timestamp is None:
            timestamp = self.current_time

        # Step 1: Get KF predictions for all nodes
        kf_predictions = np.zeros(self.N)
        kf_std = np.zeros(self.N)

        for i in range(self.N):
            node_state = self.node_states[i]
            # Propagate KF to prediction time
            dt = timestamp - node_state.last_update

            if dt > 0:
                node_state.kf.F[0, 1] = dt
                node_state.kf.predict()

            kf_predictions[i] = node_state.kf.x[0, 0]
            kf_std[i] = np.sqrt(node_state.kf.P[0, 0])

        # Step 2: Use GP for spatial interpolation if enough observations
        if len(observed_nodes) >= 2:
            observed_coords = self.coords[observed_nodes]
            residuals = np.array(observed_values) - kf_predictions[observed_nodes]

            # Train GP on residuals
            self.gp.fit(observed_coords, residuals)

            # Predict residuals for all nodes
            gp_residuals, gp_std = self.gp.predict(self.coords, return_std=True)

            # Combine KF and GP
            predictions = kf_predictions + gp_residuals

            # Uncertainty: combine KF and GP uncertainties
            prediction_std = np.sqrt(kf_std**2 + gp_std**2)

        else:
            # Fall back to KF only
            predictions = kf_predictions
            prediction_std = kf_std

        return predictions, prediction_std

    def switch_to_robust_mode(self, enable: bool = True, threshold: float = 0.3):
        """
        Switch to robust Cauchy kernel when energy is scarce

        Args:
            enable: Enable robust mode
            threshold: Energy threshold (as fraction of C_cap)
        """
        if enable:
            # Get current energy levels
            current_energies = np.array([
                self.node_states[i].kf.x[0, 0] for i in range(self.N)
            ])

            # Check if below threshold
            below_threshold = np.sum(current_energies < threshold * 330.0) / self.N

            if below_threshold > 0.1:  # >10% nodes below threshold
                self.robust_mode = True
                self.gp.kernel = self.kernel_cauchy
                print("Switched to robust Cauchy kernel (energy scarce)")
            else:
                self.robust_mode = False
                self.gp.kernel = self.kernel_rbf
        else:
            self.robust_mode = False
            self.gp.kernel = self.kernel_rbf

    def get_node_prediction(self,
                          node_id: int,
                          horizon: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict future energy for a single node

        Args:
            node_id: Node ID
            horizon: Number of future steps

        Returns:
            Tuple of (predictions, std_devs)
        """
        node_state = self.node_states[node_id]
        dt = 1.0  # Time step

        predictions = np.zeros(horizon)
        std_devs = np.zeros(horizon)

        # Propagate KF
        x = node_state.kf.x.copy()
        P = node_state.kf.P.copy()

        for t in range(horizon):
            # Predict
            x = node_state.kf.F @ x
            P = node_state.kf.F @ P @ node_state.kf.F.T + node_state.kf.Q

            predictions[t] = x[0, 0]
            std_devs[t] = np.sqrt(P[0, 0])

            # Add trend
            node_state.kf.F[0, 1] = dt

        return predictions, std_devs

    def analyze_spatial_correlation(self) -> Dict[str, float]:
        """
        Analyze spatial correlation structure

        Returns:
            Dictionary with correlation metrics
        """
        # Compute pairwise distances
        distances = []
        energy_diffs = []

        for i in range(self.N):
            for j in range(i + 1, self.N):
                dist = np.linalg.norm(self.coords[i] - self.coords[j])
                energy_diff = abs(
                    self.node_states[i].kf.x[0, 0] -
                    self.node_states[j].kf.x[0, 0]
                )
                distances.append(dist)
                energy_diffs.append(energy_diff)

        distances = np.array(distances)
        energy_diffs = np.array(energy_diffs)

        # Compute correlation
        correlation = np.corrcoef(distances, energy_diffs)[0, 1]

        return {
            'mean_distance': np.mean(distances),
            'max_distance': np.max(distances),
            'energy_correlation': correlation,
            'energy_range': np.max(energy_diffs) - np.min(energy_diffs)
        }

    def get_missing_data_mask(self) -> np.ndarray:
        """
        Identify nodes with missing data

        Returns:
            Boolean array [N] where True indicates missing data
        """
        mask = np.zeros(self.N, dtype=bool)

        # Check for stale measurements
        current_time = self.current_time
        threshold = 10.0  # 10 time steps

        for i in range(self.N):
            node_state = self.node_states[i]
            if current_time - node_state.last_update > threshold:
                mask[i] = True

        return mask

    def visualize_spatial_prediction(self, save_path: Optional[str] = None):
        """
        Visualize current spatial energy distribution

        Args:
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(12, 8))

        # Get current energy estimates
        energies = np.array([
            self.node_states[i].kf.x[0, 0] for i in range(self.N)
        ])

        # Scatter plot
        scatter = plt.scatter(self.coords[:, 0], self.coords[:, 1],
                            c=energies, cmap='viridis', s=100)
        plt.colorbar(scatter, label='Energy (μJ)')

        # Mark inducing points
        plt.scatter(self.inducing_points[:, 0], self.inducing_points[:, 1],
                   marker='x', s=200, c='red', label='Inducing Points')

        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Spatial Energy Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    N = 50

    # Generate random positions
    positions = np.random.rand(N, 2) * 100  # 100m x 100m area

    # Initialize model
    model = KFGPHybridModel(N, positions)

    # Simulate observations
    time_steps = 100
    for t in range(time_steps):
        # Randomly observe 30% of nodes
        observed_count = int(0.3 * N)
        observed_nodes = np.random.choice(N, observed_count, replace=False)

        # Generate synthetic energy values
        observed_values = np.random.uniform(0, 330, observed_count)

        # Update model
        for node_id in observed_nodes:
            model.update_node(node_id, observed_values[node_id - observed_nodes[0]], t)

        # Periodically make predictions
        if t % 10 == 0:
            predictions, std = model.predict_missing_data(
                observed_nodes, observed_values, t
            )

            print(f"Time {t}: Predicted energies - Mean: {np.mean(predictions):.2f}, "
                  f"Std: {np.mean(std):.2f}")

    # Analyze correlation
    correlation = model.analyze_spatial_correlation()
    print(f"\nSpatial correlation analysis:")
    for key, value in correlation.items():
        print(f"  {key}: {value:.4f}")

    # Visualize
    model.visualize_spatial_prediction('spatial_energy.png')
