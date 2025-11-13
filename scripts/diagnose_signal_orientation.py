#!/usr/bin/env python3
"""
Diagnose signal orientation for RNA sequencing data.

This script checks if signal and sequence are properly aligned by:
1. Checking if move table positions increase monotonically (they should)
2. Extracting chunks around motifs and visualizing signal vs sequence
3. Computing signal-to-base correlations to detect reversal
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pysam

from leech.features import MoveTable
from leech.io.bam_reader import BamReader
from leech.io.pod5_reader import POD5Reader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_move_table_monotonicity(bam_path: str, n_reads: int = 100) -> dict:
    """
    Check if move table mappings are monotonically increasing.

    If signal is 3'→5' but sequence is 5'→3' without correction,
    the seq_to_sig_map should DECREASE, not increase.
    """
    reader = BamReader(bam_path)
    results = {"increasing": 0, "decreasing": 0, "mixed": 0, "total": 0}

    for i, aln in enumerate(reader):
        if i >= n_reads:
            break
        if aln.is_unmapped or not aln.has_tag("mv"):
            continue

        try:
            move_table = MoveTable.from_bam_tag(
                mv_tag=aln.get_tag("mv"),
                ns_tag=aln.get_tag("ns"),
                seq_len=aln.query_length,
            )
            seq_to_sig = move_table.to_seq_to_sig_map()

            # Check if monotonically increasing or decreasing
            diffs = np.diff(seq_to_sig)
            if np.all(diffs >= 0):
                results["increasing"] += 1
            elif np.all(diffs <= 0):
                results["decreasing"] += 1
            else:
                results["mixed"] += 1
            results["total"] += 1

        except Exception as e:
            logger.warning(f"Failed to process read {aln.query_name}: {e}")
            continue

    return results


def extract_motif_chunk_with_signal(
    bam_path: str,
    pod5_path: str,
    motif: str = "CCAGGC",
    n_reads: int = 5,
) -> list[dict]:
    """
    Extract chunks around motifs and their corresponding signal.

    Returns list of dicts with:
    - sequence: the motif + context
    - signal: raw signal for that region
    - signal_means: per-base mean signal
    - seq_to_sig_map: mapping from bases to signal indices
    """
    bam_reader = BamReader(bam_path)
    pod5_reader = POD5Reader(pod5_path)
    chunks = []

    for aln in bam_reader:
        if len(chunks) >= n_reads:
            break
        if aln.is_unmapped or not aln.has_tag("mv"):
            continue

        # Find motif in sequence
        seq = aln.query_sequence
        if seq is None:
            continue
        motif_pos = seq.find(motif)
        if motif_pos == -1:
            continue

        try:
            # Get move table and signal
            move_table = MoveTable.from_bam_tag(
                mv_tag=aln.get_tag("mv"),
                ns_tag=aln.get_tag("ns"),
                seq_len=aln.query_length,
            )
            seq_to_sig = move_table.to_seq_to_sig_map()
            signal = pod5_reader.get_signal(aln.query_name)

            # Extract context around motif
            context = 10
            start = max(0, motif_pos - context)
            end = min(len(seq), motif_pos + len(motif) + context)

            chunk_seq = seq[start:end]
            chunk_sig_start = seq_to_sig[start]
            chunk_sig_end = seq_to_sig[end - 1] if end < len(seq_to_sig) else len(signal)

            chunk_signal = signal[chunk_sig_start:chunk_sig_end]
            chunk_seq_to_sig = seq_to_sig[start:end] - chunk_sig_start

            # Compute per-base mean signal
            per_base_means = []
            for i in range(len(chunk_seq)):
                sig_start = chunk_seq_to_sig[i]
                sig_end = chunk_seq_to_sig[i + 1] if i + 1 < len(chunk_seq_to_sig) else len(chunk_signal)
                base_signal = chunk_signal[sig_start:sig_end]
                per_base_means.append(np.mean(base_signal) if len(base_signal) > 0 else 0)

            chunks.append(
                {
                    "read_id": aln.query_name,
                    "sequence": chunk_seq,
                    "signal": chunk_signal,
                    "per_base_means": np.array(per_base_means),
                    "seq_to_sig_map": chunk_seq_to_sig,
                    "motif_start": motif_pos - start,
                }
            )

        except Exception as e:
            logger.warning(f"Failed to extract chunk from {aln.query_name}: {e}")
            continue

    return chunks


def plot_signal_alignment(chunks: list[dict], output_path: str):
    """
    Plot signal traces with sequence annotations to visually check alignment.

    If signal is reversed, you'll see signal features don't match the bases.
    """
    n_chunks = min(len(chunks), 5)
    fig, axes = plt.subplots(n_chunks, 1, figsize=(12, 3 * n_chunks))
    if n_chunks == 1:
        axes = [axes]

    for i, chunk in enumerate(chunks[:n_chunks]):
        ax = axes[i]

        # Plot raw signal
        ax.plot(chunk["signal"], alpha=0.7, linewidth=0.5)

        # Mark base boundaries
        for j, sig_pos in enumerate(chunk["seq_to_sig_map"]):
            ax.axvline(sig_pos, color="gray", alpha=0.3, linewidth=0.5)
            # Annotate with base
            if j < len(chunk["sequence"]):
                ax.text(
                    sig_pos + 5,
                    ax.get_ylim()[1] * 0.9,
                    chunk["sequence"][j],
                    fontsize=8,
                    color="red" if j >= chunk["motif_start"] and j < chunk["motif_start"] + 6 else "black",
                )

        ax.set_title(f"Read: {chunk['read_id']}")
        ax.set_xlabel("Signal index")
        ax.set_ylabel("Signal (pA)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved signal alignment plot to {output_path}")


def compute_base_signal_correlation(chunks: list[dict]) -> dict:
    """
    Compute correlation between base identity and signal level.

    If reversed, correlations will be weaker or inverted.
    """
    base_signals = {"A": [], "C": [], "G": [], "T": [], "U": []}

    for chunk in chunks:
        for base, mean_sig in zip(chunk["sequence"], chunk["per_base_means"]):
            if base in base_signals:
                base_signals[base].append(mean_sig)

    # Compute mean signal per base
    base_means = {}
    for base, signals in base_signals.items():
        if len(signals) > 0:
            base_means[base] = np.mean(signals)

    return base_means


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose signal orientation in RNA nanopore data"
    )
    parser.add_argument("--bam", required=True, help="BAM file with move tables")
    parser.add_argument("--pod5", required=True, help="POD5 file with raw signal")
    parser.add_argument("--motif", default="CCAGGC", help="Motif to search for")
    parser.add_argument("--output", default="signal_diagnosis.png", help="Output plot")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("SIGNAL ORIENTATION DIAGNOSTIC")
    logger.info("=" * 80)

    # Test 1: Check move table monotonicity
    logger.info("\n1. Checking move table monotonicity...")
    monotonicity = check_move_table_monotonicity(args.bam, n_reads=100)
    logger.info(f"   Results from {monotonicity['total']} reads:")
    logger.info(f"   - Increasing: {monotonicity['increasing']} reads")
    logger.info(f"   - Decreasing: {monotonicity['decreasing']} reads")
    logger.info(f"   - Mixed: {monotonicity['mixed']} reads")

    if monotonicity["decreasing"] > 0:
        logger.warning(
            "   ⚠️  FOUND DECREASING MAPPINGS! This suggests signal reversal issue."
        )
    elif monotonicity["increasing"] == monotonicity["total"]:
        logger.info("   ✓ All mappings are increasing (expected)")

    # Test 2: Extract and visualize chunks
    logger.info(f"\n2. Extracting chunks around motif '{args.motif}'...")
    chunks = extract_motif_chunk_with_signal(args.bam, args.pod5, motif=args.motif, n_reads=5)
    logger.info(f"   Extracted {len(chunks)} chunks")

    if len(chunks) > 0:
        # Test 3: Plot signal alignment
        logger.info("\n3. Plotting signal alignment...")
        plot_signal_alignment(chunks, args.output)

        # Test 4: Base-signal correlations
        logger.info("\n4. Computing base-signal correlations...")
        base_means = compute_base_signal_correlation(chunks)
        logger.info("   Mean signal per base:")
        for base in sorted(base_means.keys()):
            logger.info(f"   - {base}: {base_means[base]:.2f} pA")

    logger.info("\n" + "=" * 80)
    logger.info("INTERPRETATION:")
    logger.info("=" * 80)
    logger.info(
        "• If move table mappings are DECREASING, signal and sequence are reversed"
    )
    logger.info("• Check the plot: bases should align with signal features")
    logger.info(
        "• If bases appear 'backwards' relative to signal transitions, need reversal"
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
