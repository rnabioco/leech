#!/usr/bin/env python3
"""
Aggregate metrics from multiple JSON files into a summary TSV.

This script reads all metrics JSON files from a directory tree and
creates a tab-separated summary table.
"""

import json
from pathlib import Path

import pandas as pd

# Snakemake injects the 'snakemake' object at runtime
snakemake = snakemake  # type: ignore # noqa: F821


def load_metrics(metrics_dir):
    """
    Load all metrics JSON files from a directory tree.

    Args:
        metrics_dir: Directory containing metrics JSON files

    Returns:
        List of dicts with metrics
    """
    metrics_dir = Path(metrics_dir)
    metrics_list = []

    for json_file in metrics_dir.rglob("*_metrics.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Add metadata from file path
            rel_path = json_file.relative_to(metrics_dir)
            parts = rel_path.parts

            # Extract sample name (filename without _metrics.json)
            sample = json_file.stem.replace("_metrics", "")
            data["sample"] = sample

            # Extract pair name if in pairwise subdirectory
            if len(parts) > 1:
                data["pair"] = parts[0]

            metrics_list.append(data)

        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return metrics_list


def main():
    # Get parameters from Snakemake
    metrics_dir = snakemake.params.metrics_dir
    output_file = snakemake.output.summary

    # Load all metrics
    metrics = load_metrics(metrics_dir)

    if not metrics:
        print(f"Warning: No metrics found in {metrics_dir}")
        # Create empty file
        Path(output_file).touch()
        return

    # Convert to DataFrame
    df = pd.DataFrame(metrics)

    # Sort columns for better readability
    # Put metadata columns first
    meta_cols = ["sample", "pair"] if "pair" in df.columns else ["sample"]
    other_cols = sorted([c for c in df.columns if c not in meta_cols])
    df = df[meta_cols + other_cols]

    # Sort by sample (and pair if present)
    sort_cols = ["pair", "sample"] if "pair" in df.columns else ["sample"]
    df = df.sort_values(sort_cols)

    # Write to TSV
    df.to_csv(output_file, sep="\t", index=False)

    print(f"Wrote {len(df)} metric rows to {output_file}")


if __name__ == "__main__":
    main()
