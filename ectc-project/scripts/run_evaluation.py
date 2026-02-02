#!/usr/bin/env python3
"""
Run Evaluation Suite
====================

Execute complete evaluation and generate reports.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.scripts.evaluator import ECTCEvaluator


def main():
    """Run evaluation suite"""
    print("="*70)
    print("ECTC Evaluation Suite")
    print("="*70)
    print()

    # Initialize evaluator
    evaluator = ECTCEvaluator('evaluation/results')

    # Run baseline comparison
    print("Running baseline comparison...")
    frameworks = ['ECTC', 'BATDMA', 'Mementos', 'Alpaca']
    df = evaluator.run_baseline_comparison(frameworks)

    # Generate plots
    print("\nGenerating performance plots...")
    evaluator.plot_results(df, Path('evaluation/plots'))

    # Run ablation study
    print("\nRunning ablation study...")
    ablation_df = evaluator.run_ablation_study()

    # Reproduce Figure 6
    print("\nReproducing Figure 6...")
    evaluator.reproduce_figure6(Path('evaluation/plots/figure6_dynamic_response.png'))

    print()
    print("="*70)
    print("Evaluation Complete!")
    print("="*70)
    print()
    print("Results:")
    print(f"  - Baseline comparison: evaluation/results/baseline_comparison.csv")
    print(f"  - Summary report: evaluation/results/baseline_summary.txt")
    print(f"  - Ablation study: evaluation/results/ablation_study.csv")
    print(f"  - Plots: evaluation/plots/")
    print()

    # Display summary
    print("Performance Summary:")
    print("-" * 70)

    import pandas as pd
    df_grouped = df.groupby('framework').mean()

    for framework in ['ECTC', 'BATDMA', 'Mementos', 'Alpaca']:
        if framework in df_grouped.index:
            row = df_grouped.loc[framework]
            print(f"\n{framework}:")
            print(f"  Data Integrity: {row['data_integrity']*100:.1f}%")
            print(f"  Energy Cost:    {row['energy_cost_nj_per_bit']:.1f} nJ/bit")
            print(f"  Sleep Ratio:    {row['sleep_ratio']*100:.1f}%")
            print(f"  P95 Latency:    {row['latency_p95']:.2f} s")


if __name__ == '__main__':
    main()
