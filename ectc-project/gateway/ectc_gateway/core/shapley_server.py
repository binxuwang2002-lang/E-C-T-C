"""
Stratified Shapley Value Approximation Server
=============================================

Implements the stratified sampling algorithm for efficient Shapley value
computation in IoT networks with O(N log(1/δ)/ε²) complexity.

Based on: "Stratified Sampling for Efficient Shapley Approximation"
"""

import numpy as np
import random
import math
from typing import List, Dict, Tuple, Optional, Deque
from collections import deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


@dataclass
class NodeStatus:
    """Node status structure"""
    node_id: int
    Q_E: float  # Energy in capacitor (μJ)
    B_i: int    # Data queue length
    marginal_utility: float
    has_data: bool
    position: Tuple[float, float]


class TruncatedLyapunovGame:
    """
    Implements the truncated Lyapunov game for energy-aware task allocation.

    The truncated Lyapunov function prevents energy overflow:
    - Quadratic growth for Q_E ≤ 0.9 * C_cap
    - Quartic penalty for Q_E > 0.9 * C_cap (enforces boundedness)
    """

    def __init__(self, N: int, C_cap: float = 330.0, beta: float = 0.1, V: float = 50.0):
        """
        Initialize the truncated Lyapunov game

        Args:
            N: Number of nodes
            C_cap: Capacitor capacity (μJ) - default 100μF at 3.3V
            beta: Saturation penalty coefficient
            V: Lyapunov tradeoff parameter
        """
        self.N = N
        self.C_cap = C_cap
        self.beta = beta
        self.V = V

        # Game state
        self.Q_E = np.zeros(N)  # Energy queues
        self.info_utils = np.zeros(N)  # Information utilities

    def truncated_lyapunov(self, Q_E: np.ndarray) -> float:
        """
        Compute truncated Lyapunov function L_trunc(Q_E)

        Args:
            Q_E: Energy queue vector

        Returns:
            Lyapunov function value
        """
        L = 0.0
        threshold = 0.9 * self.C_cap

        for q in Q_E:
            if q <= threshold:
                L += 0.5 * q * q
            else:
                excess = q - threshold
                L += 0.5 * threshold * threshold + self.beta * excess**4

        return L

    def compute_drift(self, Q_E: np.ndarray, Q_E_next: np.ndarray) -> float:
        """
        Compute Lyapunov drift ΔL(t) = L(t+1) - L(t)

        Args:
            Q_E: Current energy state
            Q_E_next: Next energy state

        Returns:
            Lyapunov drift
        """
        return self.truncated_lyapunov(Q_E_next) - self.truncated_lyapunov(Q_E)

    def coalition_utility(self, S: List[int], info_utilities: Optional[np.ndarray] = None) -> float:
        """
        Compute coalition utility v(S)

        Args:
            S: List of node IDs in coalition
            info_utilities: Optional override for information utilities

        Returns:
            Coalition utility value
        """
        if info_utilities is None:
            info_utilities = self.info_utils

        # Simulate energy state after coalition execution
        Q_E_sim = self.Q_E.copy()
        energy_cost_per_task = 5.3e-3  # 5.3mJ per task (from paper)

        for i in S:
            if Q_E_sim[i] >= energy_cost_per_task:
                Q_E_sim[i] -= energy_cost_per_task

        # Compute Lyapunov drift
        drift = -self.compute_drift(self.Q_E, Q_E_sim)

        # Compute information utility
        info_sum = sum(info_utilities[i] for i in S)

        # Total utility
        return drift + self.V * info_sum


class StratifiedShapleyApproximator:
    """
    Efficient stratified sampling for Shapley values.

    Complexity: O(N log(1/δ) / ε²)
    """

    def __init__(self, N: int, epsilon: float = 0.1, delta: float = 0.05, K: Optional[int] = None):
        """
        Initialize approximator

        Args:
            N: Number of players
            epsilon: Approximation error bound
            delta: Confidence parameter
            K: Number of strata (default: ceil(N / log₂N))
        """
        self.N = N
        self.epsilon = epsilon
        self.delta = delta
        self.K = K if K is not None else math.ceil(N / math.log2(N))

        # Cache for efficiency
        self._cached_strata = None

    def partition_into_strata(self, positions: Dict[int, Tuple[float, float]]) -> Dict[int, List[int]]:
        """
        Partition nodes into spatial strata using K-means clustering

        Args:
            positions: Node ID -> (x, y) position mapping

        Returns:
            Stratum ID -> list of node IDs
        """
        from sklearn.cluster import KMeans

        # Convert to array
        pos_array = np.array([positions[i] for i in range(self.N)])

        # K-means clustering
        kmeans = KMeans(n_clusters=self.K, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pos_array)

        # Group nodes by stratum
        strata = {k: [] for k in range(self.K)}
        for i, label in enumerate(labels):
            strata[label].append(i)

        return strata

    def marginal_contribution(self, i: int, S: List[int], game: TruncatedLyapunovGame) -> float:
        """
        Compute marginal contribution of player i to coalition S

        Args:
            i: Player ID
            S: Coalition (without i)
            game: Lyapunov game instance

        Returns:
            Marginal contribution value
        """
        # Compute v(S)
        v_S = game.coalition_utility(S)

        # Compute v(S ∪ {i})
        v_S_i = game.coalition_utility(S + [i])

        return v_S_i - v_S

    def approximate_shapley_values(self, game: TruncatedLyapunovGame,
                                   positions: Dict[int, Tuple[float, float]],
                                   max_workers: int = 4) -> Dict[int, float]:
        """
        Approximate Shapley values using stratified sampling (Algorithm 1)

        Args:
            game: Truncated Lyapunov game
            positions: Node positions
            max_workers: Parallel workers for computation

        Returns:
            Estimated Shapley values for each node
        """
        # Partition nodes into strata
        strata = self.partition_into_strata(positions)

        # Initialize results
        phi = {i: 0.0 for i in range(self.N)}
        sample_counts = {i: 0 for i in range(self.N)}

        # Process each stratum
        for stratum_id, nodes_in_stratum in strata.items():
            Nk = len(nodes_in_stratum)

            # Sample count for this stratum
            # mk = ceil(|Ck|/N * log(1/δ) / ε²)
            mk = math.ceil((Nk / self.N) * (math.log(1/self.delta) / (self.epsilon**2)))

            print(f"Stratum {stratum_id}: {Nk} nodes, {mk} samples")

            # Parallel sampling within stratum
            def sample_stratum(_):
                # Random coalition size
                coalition_size = random.randint(0, Nk - 1)

                # Sample coalition without replacement
                coalition = random.sample(nodes_in_stratum, coalition_size)

                # Compute marginal contributions for nodes not in coalition
                marginals = []
                for i in nodes_in_stratum:
                    if i not in coalition:
                        mc = self.marginal_contribution(i, coalition, game)
                        marginals.append((i, mc))

                return marginals

            # Perform sampling
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                all_marginals = list(executor.map(sample_stratum, range(mk)))

            # Aggregate results
            for marginals in all_marginals:
                for i, mc in marginals:
                    phi[i] += mc
                    sample_counts[i] += 1

            # Scale by stratum weight and sample count
            stratum_weight = Nk / (self.N * mk)
            for i in nodes_in_stratum:
                if sample_counts[i] > 0:
                    phi[i] *= stratum_weight

        # Normalize by sample count
        for i in range(self.N):
            if sample_counts[i] > 0:
                phi[i] /= sample_counts[i]

        return phi

    def compute_coalition_mapping(self, phi: Dict[int, float],
                                 positions: Dict[int, Tuple[float, float]],
                                 min_coalition_size: int = 3) -> List[List[int]]:
        """
        Build coalition structure based on Shapley values

        Args:
            phi: Shapley values
            positions: Node positions
            min_coalition_size: Minimum coalition size

        Returns:
            List of coalitions (each coalition is list of node IDs)
        """
        # Sort nodes by Shapley value (descending)
        sorted_nodes = sorted(phi.items(), key=lambda x: x[1], reverse=True)

        coalitions = []
        current_coalition = []

        for node_id, _ in sorted_nodes:
            current_coalition.append(node_id)

            # Check if coalition is complete
            # In practice: check spatial connectivity, Shapley threshold, etc.
            if len(current_coalition) >= min_coalition_size:
                # Validate coalition (check if connected)
                if self._is_connected(current_coalition, positions):
                    coalitions.append(current_coalition)
                    current_coalition = []

        # Add remaining nodes to last coalition
        if current_coalition:
            coalitions.append(current_coalition)

        return coalitions

    def _is_connected(self, coalition: List[int],
                     positions: Dict[int, Tuple[float, float]],
                     max_distance: float = 50.0) -> bool:
        """
        Check if coalition nodes are spatially connected

        Args:
            coalition: List of node IDs
            positions: Node positions
            max_distance: Maximum distance for connectivity (meters)

        Returns:
            True if coalition is connected
        """
        # Simple check: all nodes within max_distance of centroid
        centroid_x = sum(positions[i][0] for i in coalition) / len(coalition)
        centroid_y = sum(positions[i][1] for i in coalition) / len(coalition)

        for i in coalition:
            x, y = positions[i]
            dist = math.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
            if dist > max_distance:
                return False

        return True

    def get_error_bounds(self) -> Dict[str, float]:
        """
        Get theoretical error bounds

        Returns:
            Dictionary with error bounds
        """
        m_total = (self.N / self.K) * (math.log(1/self.delta) / (self.epsilon**2))

        return {
            'max_error': self.epsilon,
            'confidence': 1 - self.delta,
            'expected_samples_per_node': m_total,
            'total_samples': m_total * self.N
        }


class LazyUpdateController:
    """
    Adaptive Update Trigger - "Lazy Update" Mechanism (Algorithm 1)
    
    Reduces downlink overhead by only broadcasting new Shapley values
    when model drift is significant. This implements the Drift Agreement
    Score from the ECTC paper.
    
    Math Logic:
        A = 1 - (1/W) * Σ(t=1 to W) |e(t) - ê(t)| / max(e(t), ε)
    
    Where:
        e(t)  = Measured energy reported by node
        ê(t)  = Gateway's predicted energy
        W     = Sliding window size (default: 10)
        ε     = Small constant to avoid division by zero (default: 1e-6)
        τ     = Agreement threshold (default: 0.8)
    
    Logic:
        - If A < τ (model disagrees with reality): Return True → Trigger Update
        - If A >= τ (model agrees with reality): Return False → Nodes use cache
    
    Benefits:
        - ~40% reduction in downlink traffic
        - Nodes can operate on cached Shapley values when model is accurate
        - Automatic adaptation to changing energy conditions
    
    Example:
        >>> controller = LazyUpdateController(window_size=10, tau=0.8)
        >>> controller.update_prediction(node_id=5, predicted=100.0)
        >>> should_update = controller.check_update_trigger(
        ...     node_id=5, 
        ...     measured_energy_history=[98.2, 101.5, 99.8, ...]
        ... )
        >>> if should_update:
        ...     # Broadcast new Shapley values
        ...     gateway.broadcast_shapley_values()
    """
    
    def __init__(self,
                 window_size: int = 10,
                 tau: float = 0.8,
                 epsilon: float = 1e-6):
        """
        Initialize the Lazy Update Controller.
        
        Args:
            window_size: Sliding window size W for energy history (default: 10)
            tau: Agreement threshold τ - trigger update if A < tau (default: 0.8)
            epsilon: Small constant ε to prevent division by zero (default: 1e-6)
        """
        self.window_size = window_size
        self.tau = tau
        self.epsilon = epsilon
        
        # Per-node energy predictions from gateway model
        # node_id -> deque of predicted energies (most recent W predictions)
        self._predictions: Dict[int, Deque[float]] = {}
        
        # Per-node agreement scores history (for monitoring)
        self._agreement_history: Dict[int, Deque[float]] = {}
        
        # Statistics
        self._update_triggers = 0
        self._cache_hits = 0
        self._total_checks = 0
    
    def update_prediction(self, node_id: int, predicted_energy: float) -> None:
        """
        Record gateway's energy prediction for a node.
        
        Called when gateway makes an energy prediction (e.g., from TinyLSTM).
        
        Args:
            node_id: Node identifier
            predicted_energy: Gateway's predicted energy in μJ
        """
        if node_id not in self._predictions:
            self._predictions[node_id] = deque(maxlen=self.window_size)
        
        self._predictions[node_id].append(predicted_energy)
    
    def update_predictions_batch(self, predictions: Dict[int, float]) -> None:
        """
        Record energy predictions for multiple nodes.
        
        Args:
            predictions: Dictionary of node_id -> predicted_energy
        """
        for node_id, predicted in predictions.items():
            self.update_prediction(node_id, predicted)
    
    def compute_agreement_score(self,
                                 node_id: int,
                                 measured_energy_history: List[float]) -> float:
        """
        Compute the Drift Agreement Score A for a node.
        
        Formula:
            A = 1 - (1/W) * Σ(t=1 to W) |e(t) - ê(t)| / max(e(t), ε)
        
        Args:
            node_id: Node identifier
            measured_energy_history: List of measured energies e(t) from node
                                     Should be length W (window_size)
        
        Returns:
            Agreement score A in range [0, 1]
            - A ≈ 1: Model predictions match measurements (good agreement)
            - A ≈ 0: Model predictions differ from measurements (poor agreement)
        """
        # Get predictions for this node
        if node_id not in self._predictions:
            # No predictions yet - assume poor agreement, trigger update
            return 0.0
        
        predictions = list(self._predictions[node_id])
        
        # Determine effective window size (intersection of available data)
        W = min(len(predictions), len(measured_energy_history), self.window_size)
        
        if W == 0:
            # No data to compare - assume poor agreement
            return 0.0
        
        # Use most recent W entries (align from the end)
        predictions = predictions[-W:]
        measurements = measured_energy_history[-W:]
        
        # Compute normalized drift sum
        drift_sum = 0.0
        for e_measured, e_predicted in zip(measurements, predictions):
            # |e(t) - ê(t)| / max(e(t), ε)
            error = abs(e_measured - e_predicted)
            normalizer = max(e_measured, self.epsilon)
            drift_sum += error / normalizer
        
        # Agreement score: A = 1 - (1/W) * drift_sum
        A = 1.0 - (drift_sum / W)
        
        # Clamp to [0, 1] range (can go negative if errors are very large)
        A = max(0.0, min(1.0, A))
        
        # Store in history for monitoring
        if node_id not in self._agreement_history:
            self._agreement_history[node_id] = deque(maxlen=100)
        self._agreement_history[node_id].append(A)
        
        return A
    
    def check_update_trigger(self,
                              node_id: int,
                              measured_energy_history: List[float]) -> bool:
        """
        Check if Shapley values should be updated for this node.
        
        This is the main interface method implementing Algorithm 1.
        
        Args:
            node_id: Node identifier
            measured_energy_history: List of recent measured energies from node
                                     Should be at least W entries for accuracy
        
        Returns:
            True: Trigger update - broadcast new Shapley values
            False: No update needed - node should use cached values
        
        Logic:
            - If A < τ (threshold): Model drift is significant → Trigger Update
            - If A >= τ: Model is accurate → Use cache
        
        Example:
            >>> measured = [98.2, 101.5, 99.8, 102.1, 97.5, 
            ...            103.0, 98.9, 100.5, 101.2, 99.0]
            >>> should_update = controller.check_update_trigger(5, measured)
            >>> if should_update:
            ...     new_phi = server.compute_shapley_values()
            ...     broadcast(new_phi)
        """
        self._total_checks += 1
        
        # Compute agreement score
        A = self.compute_agreement_score(node_id, measured_energy_history)
        
        # Decision: A < τ means model drift is significant
        should_update = A < self.tau
        
        # Update statistics
        if should_update:
            self._update_triggers += 1
        else:
            self._cache_hits += 1
        
        return should_update
    
    def check_batch_update_trigger(self,
                                    node_measurements: Dict[int, List[float]],
                                    aggregation: str = 'any') -> bool:
        """
        Check update trigger for multiple nodes.
        
        Args:
            node_measurements: Dictionary of node_id -> measured_energy_history
            aggregation: How to combine individual decisions:
                        - 'any': Update if ANY node triggers (conservative)
                        - 'all': Update only if ALL nodes trigger (aggressive)
                        - 'majority': Update if >50% nodes trigger (balanced)
        
        Returns:
            True if update should be triggered based on aggregation rule
        """
        triggers = []
        
        for node_id, measurements in node_measurements.items():
            trigger = self.check_update_trigger(node_id, measurements)
            triggers.append(trigger)
        
        if not triggers:
            return False
        
        if aggregation == 'any':
            return any(triggers)
        elif aggregation == 'all':
            return all(triggers)
        elif aggregation == 'majority':
            return sum(triggers) > len(triggers) / 2
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    def get_node_agreement_score(self, node_id: int) -> Optional[float]:
        """
        Get the most recent agreement score for a node.
        
        Args:
            node_id: Node identifier
        
        Returns:
            Most recent agreement score, or None if no history
        """
        if node_id not in self._agreement_history:
            return None
        
        history = self._agreement_history[node_id]
        if not history:
            return None
        
        return history[-1]
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get controller statistics.
        
        Returns:
            Dictionary with statistics:
            - total_checks: Total number of trigger checks
            - update_triggers: Number of times update was triggered
            - cache_hits: Number of times cache was used
            - cache_hit_rate: Fraction of checks that used cache
            - avg_agreement_score: Average agreement across all nodes
        """
        cache_hit_rate = 0.0
        if self._total_checks > 0:
            cache_hit_rate = self._cache_hits / self._total_checks
        
        # Compute average agreement score across all nodes
        all_scores = []
        for history in self._agreement_history.values():
            all_scores.extend(list(history))
        
        avg_agreement = np.mean(all_scores) if all_scores else 0.0
        
        return {
            'total_checks': self._total_checks,
            'update_triggers': self._update_triggers,
            'cache_hits': self._cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'avg_agreement_score': avg_agreement,
            'window_size': self.window_size,
            'threshold_tau': self.tau,
        }
    
    def reset_statistics(self) -> None:
        """Reset statistics counters (but keep prediction history)."""
        self._update_triggers = 0
        self._cache_hits = 0
        self._total_checks = 0
    
    def clear_node(self, node_id: int) -> None:
        """Clear all data for a specific node."""
        self._predictions.pop(node_id, None)
        self._agreement_history.pop(node_id, None)
    
    def clear_all(self) -> None:
        """Clear all prediction and agreement data."""
        self._predictions.clear()
        self._agreement_history.clear()
        self.reset_statistics()
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"LazyUpdateController(\n"
                f"  W={self.window_size}, τ={self.tau}, ε={self.epsilon},\n"
                f"  cache_hit_rate={stats['cache_hit_rate']:.1%},\n"
                f"  avg_agreement={stats['avg_agreement_score']:.3f}\n"
                f")")


class ShapleyServer:
    """
    Server for Shapley value computation and coalition management
    """

    def __init__(self, N: int, gateway_position: Tuple[float, float] = (50.0, 50.0),
                 lazy_update_window: int = 10, lazy_update_tau: float = 0.8):
        """
        Initialize Shapley server

        Args:
            N: Maximum number of nodes
            gateway_position: Gateway position (for distance calculations)
            lazy_update_window: Sliding window size W for Lazy Update (default: 10)
            lazy_update_tau: Agreement threshold τ for Lazy Update (default: 0.8)
        """
        self.N = N
        self.gateway_position = gateway_position

        # Initialize game
        self.game = TruncatedLyapunovGame(N)

        # Initialize approximator
        self.approximator = StratifiedShapleyApproximator(N)

        # Node positions (learned over time)
        self.positions = {i: gateway_position for i in range(N)}

        # Latest Shapley values
        self.current_phi = {}

        # Coalition mapping
        self.coalitions = []
        
        # Lazy Update Controller (Algorithm 1)
        self.update_controller = LazyUpdateController(
            window_size=lazy_update_window,
            tau=lazy_update_tau
        )
        
        # Per-node energy history for update trigger checks
        self._node_energy_history: Dict[int, Deque[float]] = {
            i: deque(maxlen=lazy_update_window) for i in range(N)
        }

    def update_node_status(self, status: NodeStatus, predicted_energy: Optional[float] = None):
        """
        Update node status

        Args:
            status: Node status update
            predicted_energy: Optional gateway prediction for this node's energy
                             (used for Lazy Update mechanism)
        """
        node_id = status.node_id

        # Update game state
        self.game.Q_E[node_id] = status.Q_E
        self.game.info_utils[node_id] = status.marginal_utility

        # Update position (from RSSI or GPS)
        self.positions[node_id] = status.position
        
        # Track measured energy for Lazy Update (Algorithm 1)
        if node_id not in self._node_energy_history:
            self._node_energy_history[node_id] = deque(
                maxlen=self.update_controller.window_size
            )
        self._node_energy_history[node_id].append(status.Q_E)
        
        # Record gateway's energy prediction if provided
        if predicted_energy is not None:
            self.update_controller.update_prediction(node_id, predicted_energy)

    def compute_shapley_values(self) -> Dict[int, float]:
        """
        Compute Shapley values for all nodes

        Returns:
            Shapley values dictionary
        """
        self.current_phi = self.approximator.approximate_shapley_values(
            self.game, self.positions
        )

        # Build coalition structure
        self.coalitions = self.approximator.compute_coalition_mapping(
            self.current_phi, self.positions
        )

        return self.current_phi

    def get_coalitions(self) -> List[List[int]]:
        """
        Get current coalition structure

        Returns:
            List of coalitions
        """
        return self.coalitions

    def get_node_recommendation(self, node_id: int) -> Dict:
        """
        Get recommendation for a specific node

        Args:
            node_id: Node ID

        Returns:
            Recommendation dictionary
        """
        if node_id not in self.current_phi:
            return {'execute_task': False, 'reason': 'No Shapley value computed'}

        shapley_value = self.current_phi[node_id]

        # Task recommendation threshold
        threshold = 0.5  # Calibrated from experiments

        if shapley_value > threshold:
            return {
                'execute_task': True,
                'shapley_value': shapley_value,
                'coalition': self._find_node_coalition(node_id),
                'assigned_task': self._select_task(node_id)
            }
        else:
            return {
                'execute_task': False,
                'shapley_value': shapley_value,
                'reason': 'Low Shapley value'
            }

    def _find_node_coalition(self, node_id: int) -> Optional[List[int]]:
        """
        Find coalition containing node_id

        Args:
            node_id: Node ID

        Returns:
            Coalition or None
        """
        for coalition in self.coalitions:
            if node_id in coalition:
                return coalition
        return None

    def _select_task(self, node_id: int) -> str:
        """
        Select task for node based on state

        Args:
            node_id: Node ID

        Returns:
            Task name
        """
        # Simple task selection logic
        if self.game.Q_E[node_id] > 100.0:  # High energy
            return 'transmit_data'
        elif self.game.Q_E[node_id] > 50.0:  # Medium energy
            return 'sense_data'
        else:
            return 'sleep'
    
    # =========================================================================
    # Lazy Update Methods (Algorithm 1)
    # =========================================================================
    
    def check_update_trigger(self, 
                              node_id: int, 
                              measured_energy_history: Optional[List[float]] = None) -> bool:
        """
        Check if Shapley values should be updated for a node.
        
        Implements Algorithm 1 (Lazy Update) from the ECTC paper.
        Only triggers broadcast when model drift is significant (A < τ).
        
        Args:
            node_id: Node identifier
            measured_energy_history: List of measured energies from node.
                                     If None, uses internally tracked history.
        
        Returns:
            True: Trigger update - broadcast new Shapley values
            False: No update needed - node should use cached values
        
        Math Logic:
            A = 1 - (1/W) * Σ(t=1 to W) |e(t) - ê(t)| / max(e(t), ε)
            
            If A < τ (threshold 0.8): Return True (significant drift)
            Else: Return False (model accurate, use cache)
        
        Example:
            >>> # Check if we need to update after receiving node status
            >>> server.update_node_status(status, predicted_energy=model.predict())
            >>> if server.check_update_trigger(node_id=5):
            ...     new_phi = server.compute_shapley_values()
            ...     broadcast(new_phi)
            ... else:
            ...     # Node uses cached Shapley values
            ...     pass
        """
        # Use internal history if not provided
        if measured_energy_history is None:
            if node_id not in self._node_energy_history:
                # No history - trigger update to be safe
                return True
            measured_energy_history = list(self._node_energy_history[node_id])
        
        return self.update_controller.check_update_trigger(
            node_id, measured_energy_history
        )
    
    def check_global_update_trigger(self, 
                                     aggregation: str = 'any') -> bool:
        """
        Check if global Shapley update should be triggered.
        
        Checks all nodes and aggregates decisions.
        
        Args:
            aggregation: How to combine individual decisions:
                        - 'any': Update if ANY node triggers (conservative)
                        - 'all': Update only if ALL nodes trigger (aggressive)
                        - 'majority': Update if >50% nodes trigger (balanced)
        
        Returns:
            True if global update should be triggered
        """
        node_measurements = {
            node_id: list(history)
            for node_id, history in self._node_energy_history.items()
            if len(history) > 0
        }
        
        if not node_measurements:
            return True  # No data, trigger update
        
        return self.update_controller.check_batch_update_trigger(
            node_measurements, aggregation
        )
    
    def set_energy_prediction(self, node_id: int, predicted_energy: float) -> None:
        """
        Record gateway's energy prediction for a node.
        
        Call this when the TinyLSTM or other prediction model
        generates an energy forecast for a node.
        
        Args:
            node_id: Node identifier
            predicted_energy: Predicted energy in μJ
        """
        self.update_controller.update_prediction(node_id, predicted_energy)
    
    def set_energy_predictions_batch(self, predictions: Dict[int, float]) -> None:
        """
        Record energy predictions for multiple nodes.
        
        Args:
            predictions: Dictionary of node_id -> predicted_energy
        """
        self.update_controller.update_predictions_batch(predictions)
    
    def get_node_agreement_score(self, node_id: int) -> Optional[float]:
        """
        Get the current model agreement score for a node.
        
        Args:
            node_id: Node identifier
        
        Returns:
            Agreement score A (0-1), or None if not available
            - A ≈ 1: Model predictions match reality
            - A ≈ 0: Model predictions differ from reality
        """
        return self.update_controller.get_node_agreement_score(node_id)
    
    def get_update_controller_statistics(self) -> Dict[str, float]:
        """
        Get Lazy Update controller statistics.
        
        Returns:
            Dictionary with:
            - cache_hit_rate: Fraction of checks that avoided updates
            - avg_agreement_score: Average model agreement
            - total_checks, update_triggers, cache_hits
        """
        return self.update_controller.get_statistics()


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("Shapley Server with Lazy Update (Algorithm 1)")
    print("=" * 60)
    
    N = 50
    server = ShapleyServer(N, lazy_update_window=10, lazy_update_tau=0.8)

    np.random.seed(42)
    
    # Simulate multiple rounds of updates
    print("\n--- Simulating 15 rounds of node updates ---")
    for round_num in range(15):
        for i in range(N):
            # Measured energy (from node)
            measured_Q_E = np.random.uniform(50, 280)
            
            # Gateway prediction (TinyLSTM would provide this)
            # Add some prediction error (5-10% in normal conditions)
            if round_num < 10:
                # Good prediction - low drift
                predicted_Q_E = measured_Q_E * np.random.uniform(0.95, 1.05)
            else:
                # Poor prediction - high drift (simulates changing conditions)
                predicted_Q_E = measured_Q_E * np.random.uniform(0.7, 1.3)
            
            status = NodeStatus(
                node_id=i,
                Q_E=measured_Q_E,
                B_i=np.random.randint(0, 10),
                marginal_utility=np.random.uniform(-1, 1),
                has_data=np.random.random() > 0.5,
                position=(np.random.uniform(0, 100), np.random.uniform(0, 100))
            )
            
            # Update with prediction for Lazy Update tracking
            server.update_node_status(status, predicted_energy=predicted_Q_E)
    
    print(f"  Completed {15 * N} node status updates")
    
    # Check update triggers for sample nodes
    print("\n--- Lazy Update Trigger Checks ---")
    test_nodes = [0, 10, 25, 40, 49]
    for node_id in test_nodes:
        should_update = server.check_update_trigger(node_id)
        agreement = server.get_node_agreement_score(node_id)
        status = "UPDATE NEEDED" if should_update else "USE CACHE"
        print(f"  Node {node_id:2d}: A={agreement:.3f} → {status}")
    
    # Check global update trigger
    global_update = server.check_global_update_trigger(aggregation='majority')
    print(f"\n  Global update (majority): {'TRIGGERED' if global_update else 'NOT NEEDED'}")
    
    # Get controller statistics
    print("\n--- Lazy Update Statistics ---")
    stats = server.get_update_controller_statistics()
    print(f"  Total checks:      {stats['total_checks']}")
    print(f"  Update triggers:   {stats['update_triggers']}")
    print(f"  Cache hits:        {stats['cache_hits']}")
    print(f"  Cache hit rate:    {stats['cache_hit_rate']:.1%}")
    print(f"  Avg agreement:     {stats['avg_agreement_score']:.3f}")
    
    # Compute Shapley values (only when needed)
    print("\n--- Computing Shapley Values ---")
    phi = server.compute_shapley_values()

    print(f"Computed Shapley values for {N} nodes")
    print(f"Top 5 nodes by Shapley value:")
    top_nodes = sorted(phi.items(), key=lambda x: x[1], reverse=True)[:5]
    for node_id, value in top_nodes:
        print(f"  Node {node_id}: {value:.4f}")

    # Get error bounds
    bounds = server.approximator.get_error_bounds()
    print(f"\nError bounds: {bounds}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

