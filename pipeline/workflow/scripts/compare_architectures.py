#!/usr/bin/env python3
"""
Compare multiple model architectures on the same task.

This script aggregates metrics across architectures and generates:
1. A TSV table comparing all architectures
2. A text summary with key findings
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np


def load_architecture_metrics(metrics_dir, architectures):
    """
    Load metrics for all architectures.

    Args:
        metrics_dir: Directory containing architecture subdirectories
        architectures: List of architecture names

    Returns:
        DataFrame with metrics per architecture per sample
    """
    metrics_dir = Path(metrics_dir)
    all_metrics = []

    for arch in architectures:
        arch_dir = metrics_dir / arch
        if not arch_dir.exists():
            print(f"Warning: No metrics found for {arch} at {arch_dir}")
            continue

        # Find all metrics JSON files
        for json_file in arch_dir.rglob("*_metrics.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Add metadata
                data["architecture"] = arch
                data["sample"] = json_file.stem.replace("_metrics", "")

                all_metrics.append(data)

            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")

    return pd.DataFrame(all_metrics)


def aggregate_metrics(df):
    """
    Aggregate metrics across samples for each architecture.

    Args:
        df: DataFrame with per-sample metrics

    Returns:
        DataFrame with aggregated statistics per architecture
    """
    # Identify numeric metric columns (exclude metadata)
    meta_cols = ["architecture", "sample"]
    metric_cols = [
        c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Aggregate statistics
    agg_funcs = {col: ["mean", "std", "min", "max"] for col in metric_cols}
    agg_funcs["sample"] = "count"  # Count number of samples

    aggregated = df.groupby("architecture").agg(agg_funcs)

    # Flatten column names
    aggregated.columns = [
        f"{col}_{stat}" if stat != "count" else "n_samples" for col, stat in aggregated.columns
    ]

    return aggregated.reset_index()


def create_comparison_table(df_agg):
    """
    Create a formatted comparison table.

    Args:
        df_agg: Aggregated metrics DataFrame

    Returns:
        Formatted DataFrame for output
    """
    # Select key metrics for comparison
    # Common metrics: accuracy, precision, recall, f1, auroc, auprc, loss
    key_metrics = []
    for base_metric in ["accuracy", "f1", "auroc", "auprc", "precision", "recall"]:
        mean_col = f"{base_metric}_mean"
        std_col = f"{base_metric}_std"
        if mean_col in df_agg.columns:
            key_metrics.extend([mean_col, std_col])

    if not key_metrics:
        print("Warning: No standard metrics found, using all columns")
        key_metrics = [c for c in df_agg.columns if c != "architecture"]

    # Create comparison table
    comparison = df_agg[["architecture", "n_samples"] + key_metrics].copy()

    # Round numeric columns
    numeric_cols = comparison.select_dtypes(include=[np.number]).columns
    comparison[numeric_cols] = comparison[numeric_cols].round(4)

    # Sort by primary metric (usually f1 or accuracy)
    if "f1_mean" in comparison.columns:
        comparison = comparison.sort_values("f1_mean", ascending=False)
    elif "accuracy_mean" in comparison.columns:
        comparison = comparison.sort_values("accuracy_mean", ascending=False)

    return comparison


def create_summary_text(df_agg, comparison):
    """
    Create a human-readable summary.

    Args:
        df_agg: Aggregated metrics DataFrame
        comparison: Comparison table DataFrame

    Returns:
        Summary text string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("MODEL ARCHITECTURE COMPARISON SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    # Overall statistics
    lines.append(f"Total architectures compared: {len(df_agg)}")
    lines.append(f"Architectures: {', '.join(df_agg['architecture'].tolist())}")
    lines.append("")

    # Identify best model for each metric
    lines.append("BEST MODELS PER METRIC:")
    lines.append("-" * 80)

    for base_metric in ["accuracy", "f1", "auroc", "auprc", "precision", "recall"]:
        mean_col = f"{base_metric}_mean"
        if mean_col in df_agg.columns:
            best_idx = df_agg[mean_col].idxmax()
            best_arch = df_agg.loc[best_idx, "architecture"]
            best_value = df_agg.loc[best_idx, mean_col]
            best_std = df_agg.loc[best_idx, f"{base_metric}_std"]

            lines.append(
                f"  {base_metric.upper()}: {best_arch} ({best_value:.4f} ± {best_std:.4f})"
            )

    lines.append("")

    # Rankings
    lines.append("OVERALL RANKINGS (by F1 score):")
    lines.append("-" * 80)

    if "f1_mean" in comparison.columns:
        for rank, row in enumerate(comparison.itertuples(), 1):
            f1_mean = row.f1_mean if hasattr(row, "f1_mean") else None
            f1_std = row.f1_std if hasattr(row, "f1_std") else None

            if f1_mean is not None and f1_std is not None:
                lines.append(f"  {rank}. {row.architecture}: F1 = {f1_mean:.4f} ± {f1_std:.4f}")
            else:
                lines.append(f"  {rank}. {row.architecture}")
    else:
        lines.append("  (F1 scores not available)")

    lines.append("")
    lines.append("=" * 80)
    lines.append("")

    # Detailed table
    lines.append("DETAILED COMPARISON TABLE:")
    lines.append("-" * 80)
    lines.append(comparison.to_string(index=False))
    lines.append("")

    return "\n".join(lines)


def main():
    # Get parameters from Snakemake
    metrics_dir = snakemake.params.metrics_dir
    architectures = snakemake.params.architectures
    output_comparison = snakemake.output.comparison
    output_summary = snakemake.output.summary

    print(f"Comparing architectures: {', '.join(architectures)}")
    print(f"Loading metrics from: {metrics_dir}")

    # Load metrics
    df = load_architecture_metrics(metrics_dir, architectures)

    if df.empty:
        print("ERROR: No metrics found!")
        # Create empty outputs
        Path(output_comparison).write_text("No metrics found\n")
        Path(output_summary).write_text("No metrics found\n")
        return

    print(f"Loaded {len(df)} metric entries")

    # Aggregate across samples
    df_agg = aggregate_metrics(df)

    # Create comparison table
    comparison = create_comparison_table(df_agg)

    # Write comparison table
    comparison.to_csv(output_comparison, sep="\t", index=False)
    print(f"Wrote comparison table to {output_comparison}")

    # Create and write summary
    summary = create_summary_text(df_agg, comparison)
    with open(output_summary, "w") as f:
        f.write(summary)
    print(f"Wrote summary to {output_summary}")

    # Print summary to console
    print("\n" + summary)


if __name__ == "__main__":
    main()
