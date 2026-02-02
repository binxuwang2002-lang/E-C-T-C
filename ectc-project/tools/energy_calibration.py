#!/usr/bin/env python3
"""
Energy Calibration Tool
======================

Calibrate energy models against real hardware measurements.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


class EnergyCalibrator:
    """Energy model calibration tool"""

    def __init__(self, monsoon_log_path: str, trace_log_path: str):
        self.monsoon_log = pd.read_csv(monsoon_log_path)
        self.trace_log = pd.read_json(trace_log_path)
        self.calibrated_opcodes = {}
        self.calibration_errors = []

    def analyze_monsoon_data(self) -> Dict:
        """Analyze Monsoon power measurement data"""
        # Calculate total energy
        time = self.monsoon_log['timestamp'].values
        power = self.monsoon_log['power'].values

        # Trapezoidal integration for energy
        energy = np.trapz(power, time) * 1e6  # Convert to μJ

        # Calculate energy per operation
        num_operations = len(self.trace_log)

        energy_per_op = energy / num_operations

        return {
            'total_energy_uj': energy,
            'num_operations': num_operations,
            'energy_per_operation_uj': energy_per_op,
            'average_power_mw': np.mean(power) * 1000,
            'peak_power_mw': np.max(power) * 1000
        }

    def calibrate_opcode_energies(self) -> Dict[str, float]:
        """Calibrate energy cost for each opcode"""
        opcode_counts = {}

        # Count opcode frequencies
        for _, row in self.trace_log.iterrows():
            opcode = row.get('opcode', 'unknown')
            opcode_counts[opcode] = opcode_counts.get(opcode, 0) + 1

        # Get total energy
        monsoon_stats = self.analyze_monsoon_data()
        total_energy = monsoon_stats['total_energy_uj']

        # Allocate energy proportionally
        calibrated = {}
        for opcode, count in opcode_counts.items():
            energy_fraction = count / sum(opcode_counts.values())
            calibrated[opcode] = total_energy * energy_fraction / count

        self.calibrated_opcodes = calibrated

        return calibrated

    def compare_models(self) -> Dict[str, float]:
        """Compare original vs calibrated models"""
        # Load original model
        with open('simulation/rtl_power_models/cc2650_core.json', 'r') as f:
            original = json.load(f)

        original_opcodes = original['opcode_energy_pJ']
        calibrated_opcodes = self.calibrated_opcodes

        # Calculate errors
        errors = {}
        for opcode in calibrated_opcodes:
            if opcode in original_opcodes:
                orig = original_opcodes[opcode]
                cal = calibrated_opcodes[opcode] * 1000  # Convert μJ to pJ
                error = abs(cal - orig) / orig * 100
                errors[opcode] = error

        return errors

    def generate_report(self, output_path: str):
        """Generate calibration report"""
        monsoon_stats = self.analyze_monsoon_data()
        calibrated = self.calibrate_opcode_energies()
        errors = self.compare_models()

        # Create report
        report = {
            'calibration_date': pd.Timestamp.now().isoformat(),
            'monsoon_measurements': {
                'total_operations': monsoon_stats['num_operations'],
                'total_energy_uj': monsoon_stats['total_energy_uj'],
                'energy_per_op_uj': monsoon_stats['energy_per_operation_uj'],
                'average_power_mw': monsoon_stats['average_power_mw']
            },
            'calibrated_opcodes': {k: v * 1000 for k, v in calibrated.items()},  # Convert to pJ
            'calibration_errors_percent': errors,
            'summary': {
                'mean_error_percent': np.mean(list(errors.values())),
                'max_error_percent': np.max(list(errors.values())),
                'num_opcodes': len(calibrated)
            }
        }

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Calibration report saved to {output_path}")
        print(f"\nCalibration Summary:")
        print(f"  Total operations: {monsoon_stats['num_operations']}")
        print(f"  Total energy: {monsoon_stats['total_energy_uj']:.2f} μJ")
        print(f"  Energy per operation: {monsoon_stats['energy_per_operation_uj']:.2f} μJ")
        print(f"  Mean calibration error: {report['summary']['mean_error_percent']:.2f}%")

        return report

    def plot_comparison(self, output_path: str):
        """Plot original vs calibrated energies"""
        errors = self.compare_models()

        if not errors:
            print("No errors to plot")
            return

        opcodes = list(errors.keys())
        error_values = list(errors.values())

        plt.figure(figsize=(12, 8))

        # Bar plot
        plt.bar(opcodes, error_values)
        plt.xlabel('Opcode')
        plt.ylabel('Calibration Error (%)')
        plt.title('Energy Model Calibration Errors')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Add horizontal line at 10%
        plt.axhline(y=10, color='r', linestyle='--', label='10% Threshold')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Comparison plot saved to {output_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Calibrate energy models')
    parser.add_argument('--monsoon', required=True,
                       help='Monsoon measurement log (CSV)')
    parser.add_argument('--trace', required=True,
                       help='Execution trace log (JSON)')
    parser.add_argument('--output', default='calibration_report.json',
                       help='Output report path')
    parser.add_argument('--plot', default='calibration_comparison.png',
                       help='Output plot path')

    args = parser.parse_args()

    if not Path(args.monsoon).exists():
        print(f"Error: Monsoon log not found: {args.monsoon}")
        return 1

    if not Path(args.trace).exists():
        print(f"Error: Trace log not found: {args.trace}")
        return 1

    # Run calibration
    calibrator = EnergyCalibrator(args.monsoon, args.trace)
    calibrator.generate_report(args.output)
    calibrator.plot_comparison(args.plot)

    return 0


if __name__ == '__main__':
    exit(main())
