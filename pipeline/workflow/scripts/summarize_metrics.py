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
snakemake = globals().get("snakemake")  # type: ignore # injected by Snakemake at runtime


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


def flatten_junction_strata(data):
    """Lift the junction strata of one metrics dict into flat columns (issue #395).

    ``leech eval test`` writes ``junction_stratified: {exact: {...},
    disrupted: {...}}`` whenever the test corpus carries ``junction_indels``.
    The ``disrupted`` stratum is the population the junction gate used to
    discard, so it is reported as ``junction_disrupted_*`` columns beside
    ``junction_exact_*``, and ``junction_status`` says whether it can be
    trusted: ``ok``, ``absent`` (no stratification in the metrics, e.g. an
    older corpus) or ``disrupted_empty`` (stratified, but no disrupted reads
    were scored). Absence is a status, never a silent blank.
    """
    strata = data.pop("junction_stratified", None)
    if not isinstance(strata, dict) or not strata:
        data["junction_status"] = "absent"
        return data
    for name in ("exact", "disrupted"):
        stratum = strata.get(name) or {}
        for key, value in stratum.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                data[f"junction_{name}_{key}"] = value
    disrupted = strata.get("disrupted") or {}
    data["junction_status"] = "ok" if disrupted.get("n", 0) > 0 else "disrupted_empty"
    return data


def flatten_leakage(data):
    """Lift the label-divergence verdict of one metrics dict into flat columns (#402).

    ``leech eval test`` writes ``leakage: {status, position, focus_offset}``
    when the model was trained on a corpus that declared where its label-defining
    sequence starts. ``leakage_status`` is ``clear``, ``override`` (the window
    reached that point and was allowed through) or ``undeclared``, and is never
    blank: a metrics file without the block is ``undeclared`` explicitly, so an
    unguarded run cannot pass for a guarded one in a table.
    """
    leakage = data.pop("leakage", None)
    if isinstance(leakage, dict) and leakage:
        data["leakage_status"] = leakage.get("status") or "undeclared"
        data["leakage_focus_offset"] = leakage.get("focus_offset")
    else:
        data["leakage_status"] = "undeclared"
        data["leakage_focus_offset"] = None
    return data


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

                metrics_list.append(flatten_leakage(flatten_junction_strata(data)))

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
                # A failed job has no verdict; it is not a guarded run.
                placeholder["leakage_status"] = "undeclared"
                missing_rows.append(placeholder)

            # Append missing rows to dataframe
            if missing_rows:
                df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)
                print(
                    f"\nAdded {len(missing_rows)} placeholder rows for failed jobs", file=sys.stderr
                )

    # Flag runs without a usable disrupted stratum. The gate that used to hide
    # these reads is off by default now, so a missing stratum means the score
    # is for an unstratified (or all-exact) test set -- say so.
    if "junction_status" in df.columns:
        # Placeholder rows for failed jobs have no status; that is their own story.
        flagged = df[df["junction_status"].notna() & (df["junction_status"] != "ok")]
        if len(flagged):
            print(
                f"\nWarning: {len(flagged)} metric row(s) have no usable 'disrupted' "
                "junction stratum (see junction_status column):",
                file=sys.stderr,
            )
            for _, row in flagged.iterrows():
                label = row.get("model", row.get("sample"))
                print(f"  - {label}: {row['junction_status']}", file=sys.stderr)

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

    # Write to compressed TSV
    df.to_csv(output_file, sep="\t", index=False, compression="gzip")

    print(f"\nWrote {len(df)} metric rows to {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
