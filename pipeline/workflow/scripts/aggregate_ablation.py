#!/usr/bin/env python3
"""
Aggregate channel-ablation metrics across arms, seeds, and pairwise tasks.

Each `test_metrics.json` under `{metrics_dir}/{pair}/{arm}/seed{seed}/` is
one training run's held-out evaluation -- `leech eval test`'s own output,
the same file `summarize_metrics.py`/`compare_architectures.py` read
elsewhere in this pipeline. This script does not compute anything `eval
test` doesn't already report: it reshapes per-(pair, arm, seed) rows into
one table and adds, per (pair, arm), the mean/sd across seeds and the delta
against that pair's `baseline` arm.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from leech.seed_stats import paired_seed_differences

# Snakemake injects the 'snakemake' object at runtime
snakemake = snakemake  # type: ignore # noqa: F821

# Scalar metrics carried through from `eval test`'s output (skip nested
# fields like confusion_matrix/threshold_sweep).
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]


def load_rows(metrics_dir, pairs, arms, seeds, arm_channels):
    """Load one row per (pair, arm, seed) that has a test_metrics.json."""
    metrics_dir = Path(metrics_dir)
    rows = []
    for pair in pairs:
        for arm in arms:
            channels = arm_channels.get(arm, [])
            for seed in seeds:
                json_path = metrics_dir / pair / arm / f"seed{seed}" / "test_metrics.json"
                if not json_path.exists():
                    print(f"Warning: missing {json_path}", file=sys.stderr)
                    continue
                with open(json_path) as f:
                    data = json.load(f)
                row = {
                    "pair": pair,
                    "arm": arm,
                    "channels": ",".join(channels),
                    "n_channels": len(channels),
                    "seed": seed,
                }
                for metric in METRIC_COLUMNS:
                    if metric in data:
                        row[metric] = data[metric]
                rows.append(row)
    return pd.DataFrame(rows)


def add_arm_stats(df, baseline_arm):
    """Add per-(pair, arm) mean/sd across seeds and delta vs. baseline.

    Returns (per_row_df_with_stats_merged_in, per_arm_summary_df).
    """
    metric_cols = [c for c in METRIC_COLUMNS if c in df.columns]

    grouped = df.groupby(["pair", "arm"])[metric_cols].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()

    baseline = grouped[grouped["arm"] == baseline_arm].set_index("pair")

    for metric in metric_cols:
        mean_col = f"{metric}_mean"
        delta_col = f"{metric}_delta_vs_baseline"

        def _delta(row, mean_col=mean_col):
            if row["pair"] not in baseline.index:
                return float("nan")
            return row[mean_col] - baseline.loc[row["pair"], mean_col]

        grouped[delta_col] = grouped.apply(_delta, axis=1)

    # Paired-seed difference vs. baseline (same --seed in both arms), with SD:
    # the quantity to judge an effect by; the mean-of-means delta above
    # discards the pairing.
    paired = pd.DataFrame(paired_seed_differences(df.to_dict("records"), baseline_arm, metric_cols))
    if not paired.empty:
        grouped = grouped.merge(paired, on=["pair", "arm"], how="left")

    merged = df.merge(grouped, on=["pair", "arm"], how="left")
    return merged, grouped


def create_summary_text(grouped, metric_cols, baseline_arm):
    lines = []
    lines.append("=" * 80)
    lines.append("CHANNEL ABLATION SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Baseline arm: {baseline_arm}")
    lines.append(f"Arms compared: {grouped['arm'].nunique()}")
    lines.append(f"Pairs evaluated: {grouped['pair'].nunique()}")
    lines.append("")
    lines.append(
        "NOTE: a delta smaller than that arm's seed-to-seed sd (the "
        "<metric>_sd column) is not evidence the channel changed the "
        "outcome -- it is within run-to-run noise. See "
        "docs/pipeline.md#channel-ablation."
    )
    lines.append("")

    sort_col = None
    for candidate in ("f1_mean", "accuracy_mean"):
        if candidate in grouped.columns:
            sort_col = candidate
            break

    for pair, pair_df in grouped.groupby("pair"):
        lines.append(f"--- {pair} ---")
        ordered = pair_df.sort_values(sort_col, ascending=False) if sort_col else pair_df
        for _, row in ordered.iterrows():
            parts = [f"{row['arm']:24s}"]
            for metric in metric_cols:
                mean_v = row.get(f"{metric}_mean")
                sd_v = row.get(f"{metric}_sd")
                delta_v = row.get(f"{metric}_delta_vs_baseline")
                if mean_v is None or pd.isna(mean_v):
                    continue
                sd_v = 0.0 if pd.isna(sd_v) else sd_v
                delta_v = 0.0 if pd.isna(delta_v) else delta_v
                part = f"{metric}={mean_v:.4f}±{sd_v:.4f} (delta {delta_v:+.4f}"
                pd_mean = row.get(f"{metric}_paired_diff_mean")
                pd_sd = row.get(f"{metric}_paired_diff_sd")
                if pd_mean is not None and not pd.isna(pd_mean):
                    sd_txt = "n/a" if pd_sd is None or pd.isna(pd_sd) else f"{pd_sd:.4f}"
                    part += f"; paired {pd_mean:+.4f}±{sd_txt}"
                parts.append(part + ")")
            lines.append("  " + " ".join(parts))
        lines.append("")

    return "\n".join(lines)


def main():
    metrics_dir = snakemake.params.metrics_dir
    pairs = snakemake.params.pairs
    arms = snakemake.params.arms
    seeds = snakemake.params.seeds
    arm_channels = snakemake.params.arm_channels
    baseline_arm = snakemake.params.baseline_arm
    output_comparison = snakemake.output.comparison
    output_summary = snakemake.output.summary

    if not pairs:
        Path(output_comparison).write_text("")
        Path(output_summary).write_text("No pairwise comparisons configured.\n")
        return

    df = load_rows(metrics_dir, pairs, arms, seeds, arm_channels)

    if df.empty:
        print("ERROR: no channel-ablation metrics found!", file=sys.stderr)
        empty_cols = ["pair", "arm", "channels", "n_channels", "seed", *METRIC_COLUMNS]
        pd.DataFrame(columns=empty_cols).to_csv(
            output_comparison, sep="\t", index=False, compression="gzip"
        )
        Path(output_summary).write_text("No channel-ablation metrics found.\n")
        return

    merged, grouped = add_arm_stats(df, baseline_arm)
    merged = merged.sort_values(["pair", "arm", "seed"]).reset_index(drop=True)

    merged.to_csv(output_comparison, sep="\t", index=False, compression="gzip")
    print(f"Wrote {len(merged)} rows to {output_comparison}")

    metric_cols = [c for c in METRIC_COLUMNS if c in df.columns]
    summary = create_summary_text(grouped, metric_cols, baseline_arm)
    Path(output_summary).write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
