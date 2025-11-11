#!/usr/bin/env python3
"""
Diagnostic script to identify data leakage from batch effects.

This script analyzes training chunks to detect if the model might be learning
sample-specific artifacts instead of biological signal. It computes statistics
to compare variance between samples vs. variance between labels.

Usage:
    python scripts/diagnose_leakage.py --chunks-dir path/to/chunks/ --sample-info samples.json

Sample info format (samples.json):
    {
        "sample1": {"label": "charged", "run": "run1"},
        "sample2": {"label": "uncharged", "run": "run2"},
        ...
    }
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_chunks_from_npz(npz_file):
    """Load chunks from a single .npz file."""
    data = np.load(npz_file, allow_pickle=True)
    chunks = []

    # Extract chunk data
    n_chunks = len(data["signals"])
    for i in range(n_chunks):
        chunk = {
            "signal": data["signals"][i],
            "sequence": str(data["sequences"][i]),
            "dwell": data["dwells"][i] if "dwells" in data else None,
            "features": data["features"][i] if "features" in data else None,
            "label": int(data["labels"][i]),
            "read_id": str(data["read_ids"][i]),
            "base_idx": int(data["base_idxs"][i]),
        }
        chunks.append(chunk)

    return chunks


def extract_sample_from_path(npz_path):
    """Extract sample name from file path (assumes format: sample_name/all.npz or sample_name.npz)."""
    # Handle both "sample/all.npz" and "sample.npz" formats
    if npz_path.parent.name != "chunks":
        return npz_path.parent.name
    else:
        return npz_path.stem.replace("_all", "").replace("all", "")


def compute_signal_statistics(chunks_by_sample):
    """
    Compute per-sample and per-label signal statistics.

    Returns:
        dict with keys:
            - sample_means: mean signal per sample
            - sample_stds: std signal per sample
            - label_means: mean signal per label
            - label_stds: std signal per label
            - between_sample_variance: variance between samples
            - between_label_variance: variance between labels
    """
    sample_stats = {}
    label_stats = {0: [], 1: []}

    for sample, chunks in chunks_by_sample.items():
        # Collect all signals for this sample
        signals = [chunk["signal"] for chunk in chunks]
        all_signals = np.concatenate(signals)

        sample_stats[sample] = {
            "mean": np.mean(all_signals),
            "std": np.std(all_signals),
            "n_chunks": len(chunks),
        }

        # Group by label
        for chunk in chunks:
            label = chunk["label"]
            label_stats[label].append(chunk["signal"])

    # Compute between-sample variance
    sample_means = [stats["mean"] for stats in sample_stats.values()]
    between_sample_var = np.var(sample_means)

    # Compute between-label variance
    label_0_signals = np.concatenate(label_stats[0]) if label_stats[0] else np.array([])
    label_1_signals = np.concatenate(label_stats[1]) if label_stats[1] else np.array([])

    label_0_mean = np.mean(label_0_signals) if len(label_0_signals) > 0 else 0
    label_1_mean = np.mean(label_1_signals) if len(label_1_signals) > 0 else 0
    between_label_var = np.var([label_0_mean, label_1_mean])

    return {
        "sample_stats": sample_stats,
        "between_sample_variance": between_sample_var,
        "between_label_variance": between_label_var,
        "label_0_mean": label_0_mean,
        "label_1_mean": label_1_mean,
    }


def visualize_embeddings(chunks_by_sample, output_dir):
    """
    Create PCA and t-SNE visualizations colored by sample and label.
    """
    # Collect all chunk features
    all_features = []
    all_labels = []
    all_samples = []
    sample_to_idx = {}

    for sample_idx, (sample, chunks) in enumerate(chunks_by_sample.items()):
        sample_to_idx[sample] = sample_idx
        for chunk in chunks:
            # Use signal mean, std, and dwell features if available
            signal_mean = np.mean(chunk["signal"])
            signal_std = np.std(chunk["signal"])
            signal_median = np.median(chunk["signal"])

            if chunk["dwell"] is not None:
                dwell_mean = np.mean(chunk["dwell"])
                dwell_std = np.std(chunk["dwell"])
                features = [signal_mean, signal_std, signal_median, dwell_mean, dwell_std]
            else:
                features = [signal_mean, signal_std, signal_median]

            all_features.append(features)
            all_labels.append(chunk["label"])
            all_samples.append(sample_idx)

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    all_samples = np.array(all_samples)

    # PCA
    logger.info("Computing PCA...")
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(all_features)

    # t-SNE
    logger.info("Computing t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) // 4))
    tsne_features = tsne.fit_transform(all_features)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # PCA colored by label
    scatter = axes[0, 0].scatter(
        pca_features[:, 0], pca_features[:, 1], c=all_labels, cmap="viridis", alpha=0.6
    )
    axes[0, 0].set_title("PCA colored by Label")
    axes[0, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    axes[0, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.colorbar(scatter, ax=axes[0, 0], label="Label")

    # PCA colored by sample
    scatter = axes[0, 1].scatter(
        pca_features[:, 0], pca_features[:, 1], c=all_samples, cmap="tab20", alpha=0.6
    )
    axes[0, 1].set_title("PCA colored by Sample")
    axes[0, 1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    axes[0, 1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.colorbar(scatter, ax=axes[0, 1], label="Sample")

    # t-SNE colored by label
    scatter = axes[1, 0].scatter(
        tsne_features[:, 0], tsne_features[:, 1], c=all_labels, cmap="viridis", alpha=0.6
    )
    axes[1, 0].set_title("t-SNE colored by Label")
    axes[1, 0].set_xlabel("t-SNE 1")
    axes[1, 0].set_ylabel("t-SNE 2")
    plt.colorbar(scatter, ax=axes[1, 0], label="Label")

    # t-SNE colored by sample
    scatter = axes[1, 1].scatter(
        tsne_features[:, 0], tsne_features[:, 1], c=all_samples, cmap="tab20", alpha=0.6
    )
    axes[1, 1].set_title("t-SNE colored by Sample")
    axes[1, 1].set_xlabel("t-SNE 1")
    axes[1, 1].set_ylabel("t-SNE 2")
    plt.colorbar(scatter, ax=axes[1, 1], label="Sample")

    plt.tight_layout()
    output_file = output_dir / "batch_effect_visualization.png"
    plt.savefig(output_file, dpi=300)
    logger.info(f"Saved visualization to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose data leakage from batch effects")
    parser.add_argument(
        "--chunks-dir", type=Path, required=True, help="Directory containing chunk .npz files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for diagnostic plots and reports",
    )
    parser.add_argument(
        "--sample-pattern",
        type=str,
        default="*/all.npz",
        help="Glob pattern to find sample chunk files (default: */all.npz)",
    )

    args = parser.parse_args()

    chunks_dir = args.chunks_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all chunk files
    chunk_files = list(chunks_dir.glob(args.sample_pattern))
    if not chunk_files:
        logger.error(f"No chunk files found matching {chunks_dir / args.sample_pattern}")
        return

    logger.info(f"Found {len(chunk_files)} chunk files")

    # Load chunks grouped by sample
    chunks_by_sample = {}
    for npz_file in chunk_files:
        sample = extract_sample_from_path(npz_file)
        logger.info(f"Loading {sample} from {npz_file}")
        chunks = load_chunks_from_npz(npz_file)
        chunks_by_sample[sample] = chunks

    # Compute statistics
    logger.info("Computing signal statistics...")
    stats = compute_signal_statistics(chunks_by_sample)

    # Print report
    print("\n" + "=" * 80)
    print("BATCH EFFECT DIAGNOSTIC REPORT")
    print("=" * 80)
    print(f"\nSamples analyzed: {len(chunks_by_sample)}")
    print("\nPer-Sample Statistics:")
    for sample, sample_stats in stats["sample_stats"].items():
        print(f"  {sample}:")
        print(f"    Mean signal: {sample_stats['mean']:.2f}")
        print(f"    Std signal: {sample_stats['std']:.2f}")
        print(f"    Chunks: {sample_stats['n_chunks']}")

    print("\nVariance Analysis:")
    print(f"  Between-sample variance: {stats['between_sample_variance']:.4f}")
    print(f"  Between-label variance: {stats['between_label_variance']:.4f}")

    # Calculate variance ratio as a batch effect indicator
    if stats["between_label_variance"] > 0:
        variance_ratio = stats["between_sample_variance"] / stats["between_label_variance"]
        print(f"  Variance ratio (sample/label): {variance_ratio:.2f}")

        if variance_ratio > 2.0:
            print(
                "\n⚠️  WARNING: Between-sample variance is much larger than between-label variance!"
            )
            print(
                "     This suggests the model may learn sample-specific artifacts (batch effects)"
            )
            print("     instead of biological signal.")
            print("\n     Recommendations:")
            print("     1. Use --global-normalization flag when preparing data")
            print("     2. Ensure charged and uncharged samples are multiplexed in the same run")
            print("     3. Perform batch effect correction before training")
        elif variance_ratio > 1.0:
            print("\n⚠️  CAUTION: Between-sample variance is larger than between-label variance.")
            print("     Monitor training carefully for signs of batch effects.")
        else:
            print("\n✓ Between-label variance is larger than between-sample variance.")
            print("  This is expected for well-controlled experiments.")

    print("\nLabel Statistics:")
    print(f"  Label 0 (uncharged) mean: {stats['label_0_mean']:.2f}")
    print(f"  Label 1 (charged) mean: {stats['label_1_mean']:.2f}")
    print(f"  Difference: {abs(stats['label_1_mean'] - stats['label_0_mean']):.2f}")

    # Save report to file
    report_file = output_dir / "batch_effect_report.txt"
    with open(report_file, "w") as f:
        f.write("BATCH EFFECT DIAGNOSTIC REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Samples analyzed: {len(chunks_by_sample)}\n\n")
        f.write("Per-Sample Statistics:\n")
        for sample, sample_stats in stats["sample_stats"].items():
            f.write(f"  {sample}:\n")
            f.write(f"    Mean signal: {sample_stats['mean']:.2f}\n")
            f.write(f"    Std signal: {sample_stats['std']:.2f}\n")
            f.write(f"    Chunks: {sample_stats['n_chunks']}\n")
        f.write("\nVariance Analysis:\n")
        f.write(f"  Between-sample variance: {stats['between_sample_variance']:.4f}\n")
        f.write(f"  Between-label variance: {stats['between_label_variance']:.4f}\n")
        if stats["between_label_variance"] > 0:
            variance_ratio = stats["between_sample_variance"] / stats["between_label_variance"]
            f.write(f"  Variance ratio (sample/label): {variance_ratio:.2f}\n")

    logger.info(f"Saved report to {report_file}")

    # Create visualizations
    logger.info("Creating visualizations...")
    try:
        visualize_embeddings(chunks_by_sample, output_dir)
    except Exception as e:
        logger.error(f"Failed to create visualizations: {e}")

    print("\n" + "=" * 80)
    logger.info("Diagnostic complete!")


if __name__ == "__main__":
    main()
