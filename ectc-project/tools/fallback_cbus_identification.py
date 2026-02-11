#!/usr/bin/env python3
"""
No-EDA C_bus Identification Tool
=================================

Fallback script for parasitic parameter extraction without SPICE/Cadence EDA tools.
Uses physical measurement data to reverse-engineer chip parasitic parameters.

Based on ECTC Paper Section V-C.1:
- Leakage current via idle voltage slope (Eq 10)
- C_bus extraction via differential measurement

Author: ECTC Research Team
Reference: ECTC Paper - FEMP 2.0 Methodology
"""

import argparse
import csv
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
from scipy import optimize
from scipy import stats


@dataclass
class MeasurementData:
    """Container for measurement data."""
    time_s: np.ndarray      # Time in seconds
    voltage_v: np.ndarray   # Capacitor voltage in volts
    
    def __post_init__(self):
        """Validate data."""
        if len(self.time_s) != len(self.voltage_v):
            raise ValueError("Time and voltage arrays must have same length")
        if len(self.time_s) < 2:
            raise ValueError("Need at least 2 data points")


@dataclass
class TaskMeasurement:
    """Container for task energy measurement."""
    task_name: str
    measured_energy_uj: float    # Measured energy in microjoules
    instruction_count: int       # Number of instructions executed
    bus_switches: int            # Estimated number of bus switches (N_switch)
    theoretical_energy_uj: float # Theoretical energy assuming C_bus=0


@dataclass
class ExtractionResult:
    """Results from parameter extraction."""
    value: float
    unit: str
    confidence: float
    method: str
    r_squared: float


# =============================================================================
# Function 1: Leakage Current Calculation
# =============================================================================

def calculate_leakage_current(
    data: MeasurementData,
    c_eff_uF: float,
    use_curve_fit: bool = True
) -> ExtractionResult:
    """
    Calculate leakage current from idle voltage decay.
    
    Based on ECTC Paper Equation 10:
        I_leak(T) = C_eff * (dV_cap / dt)|_idle
    
    During idle, the capacitor voltage drops due to leakage current.
    The slope of this decay gives us the leakage current.
    
    Args:
        data: MeasurementData containing time and voltage arrays
        c_eff_uF: Effective capacitance in microfarads
        use_curve_fit: If True, use scipy curve_fit; else use linear regression
        
    Returns:
        ExtractionResult with leakage current in nanoamperes
    """
    t = data.time_s
    v = data.voltage_v
    
    if use_curve_fit:
        # Fit linear model: V(t) = V0 - slope * t
        def linear_model(t, v0, slope):
            return v0 - slope * t
        
        try:
            popt, pcov = optimize.curve_fit(
                linear_model, 
                t, 
                v,
                p0=[v[0], 0.001],  # Initial guess
                maxfev=5000
            )
            v0_fit, slope = popt
            slope_std = np.sqrt(np.diag(pcov))[1]
            
            # Calculate R-squared
            v_pred = linear_model(t, *popt)
            ss_res = np.sum((v - v_pred) ** 2)
            ss_tot = np.sum((v - np.mean(v)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
        except Exception as e:
            print(f"Warning: curve_fit failed ({e}), falling back to linear regression")
            use_curve_fit = False
    
    if not use_curve_fit:
        # Simple linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, v)
        slope = -slope  # We want the decay rate (positive value)
        r_squared = r_value ** 2
        slope_std = std_err
    
    # Calculate leakage current
    # I = C * dV/dt
    # I [A] = C [F] * dV/dt [V/s]
    # I [nA] = C [µF] * 1e-6 * dV/dt * 1e9 = C [µF] * dV/dt * 1e3
    c_eff_F = c_eff_uF * 1e-6
    i_leak_A = c_eff_F * slope
    i_leak_nA = i_leak_A * 1e9
    
    # Confidence based on R-squared
    confidence = min(100, r_squared * 100)
    
    return ExtractionResult(
        value=i_leak_nA,
        unit="nA",
        confidence=confidence,
        method="idle_voltage_slope",
        r_squared=r_squared
    )


# =============================================================================
# Function 2: C_bus Extraction via Differential Measurement
# =============================================================================

def extract_cbus_differential(
    task_high_bus: TaskMeasurement,
    task_low_bus: TaskMeasurement,
    v_dd: float = 3.3
) -> ExtractionResult:
    """
    Extract C_bus using differential measurement between high and low bus activity tasks.
    
    Based on ECTC Paper Section V-C.1:
    1. Compare high-bus activity task (memcpy, LDM/STM) vs low-bus activity (ADD, MUL)
    2. Calculate residual energy: ΔE = (E_meas_A - E_inst_A) - (E_meas_B - E_inst_B)
    3. Attribute residual to bus capacitance charging: ΔE ≈ N_switch * (1/2) * C_bus * V²
    4. Solve for C_bus
    
    Args:
        task_high_bus: Measurement for high bus activity task (Task A)
        task_low_bus: Measurement for low bus activity task (Task B)
        v_dd: Supply voltage in volts (default 3.3V)
        
    Returns:
        ExtractionResult with C_bus in picofarads
    """
    # Calculate residuals
    # Residual = Measured - Theoretical (assuming C_bus = 0)
    residual_A = task_high_bus.measured_energy_uj - task_high_bus.theoretical_energy_uj
    residual_B = task_low_bus.measured_energy_uj - task_low_bus.theoretical_energy_uj
    
    # Differential energy
    delta_E_uj = residual_A - residual_B
    delta_E_J = delta_E_uj * 1e-6
    
    # Differential bus switches
    delta_N_switch = task_high_bus.bus_switches - task_low_bus.bus_switches
    
    if delta_N_switch <= 0:
        raise ValueError("High-bus task must have more bus switches than low-bus task")
    
    # Solve for C_bus
    # ΔE = N_switch * (1/2) * C_bus * V²
    # C_bus = 2 * ΔE / (N_switch * V²)
    c_bus_F = (2 * delta_E_J) / (delta_N_switch * v_dd ** 2)
    c_bus_pF = c_bus_F * 1e12
    
    # Confidence estimation based on energy ratio
    # Higher confidence if residuals are significantly different
    energy_ratio = abs(delta_E_uj) / max(
        abs(residual_A), abs(residual_B), 0.001
    )
    confidence = min(100, energy_ratio * 50)
    
    return ExtractionResult(
        value=c_bus_pF,
        unit="pF",
        confidence=confidence,
        method="differential_measurement",
        r_squared=0.0  # Not applicable
    )


# =============================================================================
# I/O Functions
# =============================================================================

def load_idle_data_csv(filepath: str) -> MeasurementData:
    """
    Load idle measurement data from CSV.
    
    Expected format:
        time_s,voltage_v
        0.0,3.28
        0.001,3.279
        ...
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        MeasurementData object
    """
    times = []
    voltages = []
    
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time_s']))
            voltages.append(float(row['voltage_v']))
    
    return MeasurementData(
        time_s=np.array(times),
        voltage_v=np.array(voltages)
    )


def generate_sample_idle_data(
    duration_s: float = 1.0,
    sample_rate_hz: float = 1000,
    v_init: float = 3.3,
    c_eff_uF: float = 100,
    i_leak_nA: float = 150,
    noise_mv: float = 2.0,
    seed: int = 42
) -> MeasurementData:
    """
    Generate synthetic idle measurement data for testing.
    
    Args:
        duration_s: Measurement duration in seconds
        sample_rate_hz: Sampling rate in Hz
        v_init: Initial voltage
        c_eff_uF: Effective capacitance in µF
        i_leak_nA: Leakage current in nA (ground truth)
        noise_mv: Measurement noise standard deviation in mV
        seed: Random seed
        
    Returns:
        MeasurementData object
    """
    rng = np.random.default_rng(seed)
    
    n_samples = int(duration_s * sample_rate_hz)
    t = np.linspace(0, duration_s, n_samples)
    
    # V(t) = V0 - (I_leak / C) * t
    # I_leak [A] = i_leak_nA * 1e-9
    # C [F] = c_eff_uF * 1e-6
    i_leak_A = i_leak_nA * 1e-9
    c_F = c_eff_uF * 1e-6
    
    slope = i_leak_A / c_F  # V/s
    v = v_init - slope * t
    
    # Add measurement noise
    noise = rng.normal(0, noise_mv * 1e-3, n_samples)
    v = v + noise
    
    return MeasurementData(time_s=t, voltage_v=v)


def generate_sample_task_measurements(
    v_dd: float = 3.3,
    c_bus_pF_true: float = 20.0
) -> Tuple[TaskMeasurement, TaskMeasurement]:
    """
    Generate synthetic task measurements for testing C_bus extraction.
    
    Args:
        v_dd: Supply voltage
        c_bus_pF_true: Ground truth C_bus value
        
    Returns:
        Tuple of (high_bus_task, low_bus_task)
    """
    c_bus_F = c_bus_pF_true * 1e-12
    
    # High bus activity task (memcpy, LDM/STM)
    high_bus = TaskMeasurement(
        task_name="memcpy_1KB",
        instruction_count=2000,
        bus_switches=15000,  # Many bus transitions
        theoretical_energy_uj=50.0,  # Energy assuming C_bus=0
        measured_energy_uj=0.0  # Will be calculated
    )
    # Add C_bus energy contribution
    e_cbus_high = high_bus.bus_switches * 0.5 * c_bus_F * v_dd**2
    high_bus.measured_energy_uj = high_bus.theoretical_energy_uj + e_cbus_high * 1e6
    
    # Low bus activity task (register operations)
    low_bus = TaskMeasurement(
        task_name="multiply_loop",
        instruction_count=2000,
        bus_switches=500,  # Few bus transitions
        theoretical_energy_uj=40.0,  # Energy assuming C_bus=0
        measured_energy_uj=0.0  # Will be calculated
    )
    # Add C_bus energy contribution
    e_cbus_low = low_bus.bus_switches * 0.5 * c_bus_F * v_dd**2
    low_bus.measured_energy_uj = low_bus.theoretical_energy_uj + e_cbus_low * 1e6
    
    return high_bus, low_bus


# =============================================================================
# Main CLI Interface
# =============================================================================

def print_banner():
    """Print tool banner."""
    print("=" * 70)
    print("  ECTC No-EDA Parasitic Parameter Identification Tool")
    print("  Based on FEMP 2.0 Methodology (Paper Section V-C.1)")
    print("=" * 70)
    print()


def run_leakage_calculation(args):
    """Run leakage current calculation from idle data."""
    print("─" * 50)
    print("Function 1: Leakage Current Calculation")
    print("─" * 50)
    print(f"Formula: I_leak = C_eff × (ΔV_cap / Δt)|_idle")
    print()
    
    # Load or generate data
    if args.idle_csv:
        print(f"Loading idle data from: {args.idle_csv}")
        data = load_idle_data_csv(args.idle_csv)
    else:
        print("Generating synthetic idle data for demonstration...")
        data = generate_sample_idle_data(
            i_leak_nA=args.true_i_leak_nA,
            c_eff_uF=args.c_eff_uF
        )
        print(f"  (Ground truth I_leak = {args.true_i_leak_nA} nA)")
    
    print(f"Effective capacitance C_eff = {args.c_eff_uF} µF")
    print(f"Data points: {len(data.time_s)}")
    print(f"Time range: {data.time_s[0]:.4f}s to {data.time_s[-1]:.4f}s")
    print(f"Voltage range: {data.voltage_v[-1]:.4f}V to {data.voltage_v[0]:.4f}V")
    print()
    
    # Calculate leakage
    result = calculate_leakage_current(
        data,
        c_eff_uF=args.c_eff_uF,
        use_curve_fit=True
    )
    
    print("─── Results ───")
    print(f"  Calculated I_leak: {result.value:.2f} {result.unit}")
    print(f"  R-squared: {result.r_squared:.4f}")
    print(f"  Confidence: {result.confidence:.1f}%")
    print(f"  Method: {result.method}")
    print()
    
    return result


def run_cbus_extraction(args):
    """Run C_bus extraction from differential measurements."""
    print("─" * 50)
    print("Function 2: C_bus Extraction via Differential Measurement")
    print("─" * 50)
    print("Principle: Compare high-bus vs low-bus activity tasks")
    print("  ΔE = (E_meas_A - E_inst_A) - (E_meas_B - E_inst_B)")
    print("  ΔE ≈ N_switch × (1/2) × C_bus × V²")
    print()
    
    # Load or generate data
    if args.task_csv:
        print(f"Loading task data from: {args.task_csv}")
        # Future: implement CSV loading for task measurements
        raise NotImplementedError("Task CSV loading not yet implemented")
    else:
        print("Generating synthetic task data for demonstration...")
        high_bus, low_bus = generate_sample_task_measurements(
            v_dd=args.v_dd,
            c_bus_pF_true=args.true_c_bus_pF
        )
        print(f"  (Ground truth C_bus = {args.true_c_bus_pF} pF)")
    
    print()
    print("Task A (High Bus Activity):")
    print(f"  Name: {high_bus.task_name}")
    print(f"  Instructions: {high_bus.instruction_count}")
    print(f"  Bus switches: {high_bus.bus_switches}")
    print(f"  Theoretical energy (C_bus=0): {high_bus.theoretical_energy_uj:.2f} µJ")
    print(f"  Measured energy: {high_bus.measured_energy_uj:.2f} µJ")
    print()
    print("Task B (Low Bus Activity):")
    print(f"  Name: {low_bus.task_name}")
    print(f"  Instructions: {low_bus.instruction_count}")
    print(f"  Bus switches: {low_bus.bus_switches}")
    print(f"  Theoretical energy (C_bus=0): {low_bus.theoretical_energy_uj:.2f} µJ")
    print(f"  Measured energy: {low_bus.measured_energy_uj:.2f} µJ")
    print()
    
    # Extract C_bus
    result = extract_cbus_differential(
        high_bus,
        low_bus,
        v_dd=args.v_dd
    )
    
    print("─── Results ───")
    print(f"  Calculated C_bus: {result.value:.2f} {result.unit}")
    print(f"  Confidence: {result.confidence:.1f}%")
    print(f"  Method: {result.method}")
    print(f"  Supply voltage: {args.v_dd} V")
    print()
    
    # Calculate error if ground truth available
    if args.true_c_bus_pF:
        error_percent = abs(result.value - args.true_c_bus_pF) / args.true_c_bus_pF * 100
        print(f"  Error vs ground truth: {error_percent:.2f}%")
        print()
    
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ECTC No-EDA Parasitic Parameter Identification Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic demo data
  python fallback_cbus_identification.py --demo
  
  # Calculate leakage from idle measurement CSV
  python fallback_cbus_identification.py --leakage --idle-csv idle_data.csv --c-eff 100
  
  # Extract C_bus from task measurements
  python fallback_cbus_identification.py --cbus --task-csv tasks.csv
  
  # Run both with custom parameters
  python fallback_cbus_identification.py --all --c-eff 100 --v-dd 3.3
"""
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--demo', action='store_true',
        help='Run demonstration with synthetic data'
    )
    mode_group.add_argument(
        '--leakage', action='store_true',
        help='Calculate leakage current only'
    )
    mode_group.add_argument(
        '--cbus', action='store_true',
        help='Extract C_bus only'
    )
    mode_group.add_argument(
        '--all', action='store_true',
        help='Run all analyses'
    )
    
    # Input files
    parser.add_argument(
        '--idle-csv', type=str, default=None,
        help='CSV file with idle voltage measurements (time_s,voltage_v)'
    )
    parser.add_argument(
        '--task-csv', type=str, default=None,
        help='CSV file with task energy measurements'
    )
    
    # Parameters
    parser.add_argument(
        '--c-eff', dest='c_eff_uF', type=float, default=100.0,
        help='Effective capacitance in µF (default: 100)'
    )
    parser.add_argument(
        '--v-dd', dest='v_dd', type=float, default=3.3,
        help='Supply voltage in V (default: 3.3)'
    )
    
    # Ground truth for validation
    parser.add_argument(
        '--true-i-leak', dest='true_i_leak_nA', type=float, default=150.0,
        help='Ground truth leakage current in nA for demo (default: 150)'
    )
    parser.add_argument(
        '--true-c-bus', dest='true_c_bus_pF', type=float, default=20.0,
        help='Ground truth C_bus in pF for demo (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Default to demo mode if no mode specified
    if not any([args.demo, args.leakage, args.cbus, args.all]):
        args.demo = True
    
    print_banner()
    
    results = {}
    
    # Run requested analyses
    if args.demo or args.all or args.leakage:
        results['leakage'] = run_leakage_calculation(args)
    
    if args.demo or args.all or args.cbus:
        results['cbus'] = run_cbus_extraction(args)
    
    # Summary
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    if 'leakage' in results:
        r = results['leakage']
        print(f"  I_leak = {r.value:.2f} {r.unit} (confidence: {r.confidence:.0f}%)")
    
    if 'cbus' in results:
        r = results['cbus']
        print(f"  C_bus  = {r.value:.2f} {r.unit} (confidence: {r.confidence:.0f}%)")
    
    print()
    print("CRITICAL: These parasitic values MUST be included in energy models.")
    print("          Ignoring C_bus can cause up to 4.6x energy estimation error!")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
