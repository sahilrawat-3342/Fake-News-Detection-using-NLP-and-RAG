"""
Performance Evaluation Script
Runs comprehensive evaluation using the MetricsCollector
Generates performance metrics, comparisons, and visualizations
"""

import os
import sys
import json
import argparse
from pathlib import Path

from src.metrics_collector import MetricsCollector


def main(args):
    """
    Main evaluation function.
    
    Args:
        args: Command line arguments
    """
    # Initialize metrics collector
    metrics_collector = MetricsCollector(metrics_file=args.metrics_file)

    # Run full evaluation
    eval_results = metrics_collector.run_full_evaluation(
        fake_path=args.fake_data,
        true_path=args.true_data
    )

    # Export results
    if args.export_json:
        metrics_collector.export_metrics_json(output_path=args.export_json)
    
    if args.export_csv:
        metrics_collector.export_metrics_csv(output_path=args.export_csv)

    # Generate visualization if requested
    if args.visualize:
        try:
            from src.metrics_visualizer import MetricsVisualizer
            visualizer = MetricsVisualizer(metrics_collector.metrics_data)
            visualizer.generate_all_visualizations(output_dir=args.viz_output)
            print(f"\n📊 Visualizations saved to: {args.viz_output}")
        except ImportError:
            print("⚠️  Visualization requires matplotlib. Install: pip install matplotlib seaborn")
        except Exception as e:
            print(f"⚠️  Error during visualization: {e}")

    print("\n✅ Evaluation Complete!")
    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive model performance evaluation for TruthLens"
    )
    
    parser.add_argument(
        "--fake-data",
        default="data/Fake.csv",
        help="Path to fake news dataset (default: data/Fake.csv)"
    )
    
    parser.add_argument(
        "--true-data",
        default="data/True.csv",
        help="Path to true news dataset (default: data/True.csv)"
    )
    
    parser.add_argument(
        "--metrics-file",
        default="performance_metrics.json",
        help="Path to store metrics data (default: performance_metrics.json)"
    )
    
    parser.add_argument(
        "--export-json",
        default="performance_metrics.json",
        help="Export metrics as JSON (default: performance_metrics.json)"
    )
    
    parser.add_argument(
        "--export-csv",
        default="performance_metrics.csv",
        help="Export metrics as CSV (default: performance_metrics.csv)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization charts"
    )
    
    parser.add_argument(
        "--viz-output",
        default="visualizations",
        help="Output directory for visualizations (default: visualizations)"
    )

    args = parser.parse_args()

    # Run evaluation
    results = main(args)
    sys.exit(0)
