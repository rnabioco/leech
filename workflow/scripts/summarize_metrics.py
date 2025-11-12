#!/usr/bin/env python3
"""
Aggregate metrics from multiple JSON files into a summary TSV.

This script reads all metrics JSON files from a directory tree and
creates a tab-separated summary table.
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Snakemake injects the 'snakemake' object at runtime
snakemake = snakemake  # type: ignore # noqa: F821


def discover_expected_combinations(metrics_dir):
    """
    Discover all expected pair×model combinations from directory structure.

    Args:
        metrics_dir: Directory containing metrics files

    Returns:
        tuple: (set of pairs, set of models, bool indicating if pairwise structure exists)
    """
    metrics_dir = Path(metrics_dir)
    pairs = set()
    models = set()
    has_pairwise = False

    # Check for pairwise structure: metrics_dir/pair/model/
    for pair_dir in metrics_dir.iterdir():
        if pair_dir.is_dir() and "_" in pair_dir.name:
            # Looks like a pair directory (e.g., Ala_Arg)
            has_pairwise = True
            pairs.add(pair_dir.name)
            for model_dir in pair_dir.iterdir():
                if model_dir.is_dir():
                    models.add(model_dir.name)

    return pairs, models, has_pairwise


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

    # Match both patterns: *_metrics.json (flat) and metrics.json (pairwise)
    for pattern in ["*_metrics.json", "metrics.json"]:
        for json_file in metrics_dir.rglob(pattern):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Add metadata from file path
                rel_path = json_file.relative_to(metrics_dir)
                parts = rel_path.parts

                # Determine structure and extract metadata
                if len(parts) == 3 and json_file.name == "metrics.json":
                    # Pairwise structure: pair/model/metrics.json
                    data["pair"] = parts[0]
                    data["model"] = parts[1]
                    data["sample"] = parts[1]  # For backward compatibility
                elif len(parts) == 1:
                    # Flat structure: sample_metrics.json
                    sample = json_file.stem.replace("_metrics", "")
                    data["sample"] = sample
                else:
                    # Unknown structure, use filename
                    sample = json_file.stem.replace("_metrics", "")
                    data["sample"] = sample

                metrics_list.append(data)

            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}", file=sys.stderr)

    return metrics_list


def main():
    # Get parameters from Snakemake
    metrics_dir = snakemake.params.metrics_dir
    output_file = snakemake.output.summary

    # Discover expected combinations
    pairs, models, has_pairwise = discover_expected_combinations(metrics_dir)

    # Load all metrics
    metrics = load_metrics(metrics_dir)

    if not metrics:
        print(f"Warning: No metrics found in {metrics_dir}", file=sys.stderr)
        # Create empty file
        Path(output_file).touch()
        return

    # Convert to DataFrame
    df = pd.DataFrame(metrics)

    # Handle missing combinations for pairwise structure
    if has_pairwise and pairs and models:
        # Create a complete index of all pair×model combinations
        print(f"Found {len(pairs)} pairs and {len(models)} models", file=sys.stderr)

        # Track which combinations we have
        if "pair" in df.columns and "model" in df.columns:
            found_combinations = set(zip(df["pair"], df["model"], strict=False))
        else:
            found_combinations = set()

        # Find missing combinations
        expected_combinations = {(pair, model) for pair in pairs for model in models}
        missing_combinations = expected_combinations - found_combinations

        if missing_combinations:
            print(
                f"\nWarning: {len(missing_combinations)} missing pair×model combinations:",
                file=sys.stderr,
            )
            for pair, model in sorted(missing_combinations):
                print(f"  - {pair}/{model}", file=sys.stderr)

            # Create placeholder rows for missing combinations
            missing_rows = []
            for pair, model in missing_combinations:
                placeholder = {
                    "pair": pair,
                    "model": model,
                    "sample": model,
                }
                # Add NaN for all metric columns
                for col in df.columns:
                    if col not in ["pair", "model", "sample"]:
                        placeholder[col] = float("nan")
                missing_rows.append(placeholder)

            # Append missing rows to dataframe
            if missing_rows:
                df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)
                print(
                    f"\nAdded {len(missing_rows)} placeholder rows for failed jobs", file=sys.stderr
                )

    # Sort columns for better readability
    # Put metadata columns first
    if "pair" in df.columns and "model" in df.columns:
        meta_cols = ["pair", "model"]
    elif "pair" in df.columns:
        meta_cols = ["sample", "pair"]
    else:
        meta_cols = ["sample"]

    other_cols = sorted([c for c in df.columns if c not in meta_cols])
    df = df[meta_cols + other_cols]

    # Sort by pair and model (or sample if not pairwise)
    if "pair" in df.columns and "model" in df.columns:
        sort_cols = ["pair", "model"]
    elif "pair" in df.columns:
        sort_cols = ["pair", "sample"]
    else:
        sort_cols = ["sample"]
    df = df.sort_values(sort_cols)

    # Write to TSV
    df.to_csv(output_file, sep="\t", index=False)

    print(f"\nWrote {len(df)} metric rows to {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
