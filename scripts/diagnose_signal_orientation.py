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

from leech.features import extract_move_table
from leech.io.bam_reader import BAMReader
from leech.io.pod5_reader import POD5Reader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_move_table_monotonicity(bam_path: str, n_reads: int = 100, min_mapq: int = 10) -> dict:
    """
    Check if move table mappings are monotonically increasing.

    The move table creates a mapping from base positions to signal indices.
    For correctly oriented data:
    - Base 0 should map to early signal indices (e.g., 0-100)
    - Base 1 should map to later signal indices (e.g., 100-200)
    - The mapping should be monotonically INCREASING

    If signal and sequence are in opposite orientations (signal reversed):
    - Base 0 would map to late signal indices (e.g., 9900-10000)
    - Base 1 would map to earlier signal indices (e.g., 9800-9900)
    - The mapping would be monotonically DECREASING

    This diagnostic checks whether the seq-to-sig mapping is increasing or decreasing
    to detect potential signal reversal issues.
    """
    reader = BAMReader(Path(bam_path), min_mapq=min_mapq)
    results = {"increasing": 0, "decreasing": 0, "mixed": 0, "total": 0}

    with reader:
        for i, aln in enumerate(reader.iter_alignments()):
            if i >= n_reads:
                break
            if aln.is_unmapped or not aln.has_tag("mv"):
                continue

            try:
                move_table = extract_move_table(aln)
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
    min_mapq: int = 10,
) -> list[dict]:
    """
    Extract chunks around motifs and their corresponding signal.

    Returns list of dicts with:
    - sequence: the motif + context
    - signal: raw signal for that region
    - signal_means: per-base mean signal
    - seq_to_sig_map: mapping from bases to signal indices
    """
    bam_reader = BAMReader(Path(bam_path), min_mapq=min_mapq)
    pod5_reader = POD5Reader(Path(pod5_path))
    chunks = []

    with bam_reader, pod5_reader:
        for aln in bam_reader.iter_alignments():
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
                move_table = extract_move_table(aln)
                seq_to_sig = move_table.to_seq_to_sig_map()

                # Skip if query_name is None
                if aln.query_name is None:
                    continue

                signal, _ = pod5_reader.get_signal(aln.query_name)

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
                    sig_end = (
                        chunk_seq_to_sig[i + 1]
                        if i + 1 < len(chunk_seq_to_sig)
                        else len(chunk_signal)
                    )
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
                    color="red"
                    if j >= chunk["motif_start"] and j < chunk["motif_start"] + 6
                    else "black",
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
        for base, mean_sig in zip(chunk["sequence"], chunk["per_base_means"], strict=False):
            if base in base_signals:
                base_signals[base].append(mean_sig)

    # Compute mean signal per base
    base_means = {}
    for base, signals in base_signals.items():
        if len(signals) > 0:
            base_means[base] = np.mean(signals)

    return base_means


def main():
    parser = argparse.ArgumentParser(description="Diagnose signal orientation in RNA nanopore data")
    parser.add_argument("--bam", required=True, help="BAM file with move tables")
    parser.add_argument("--pod5", required=True, help="POD5 file with raw signal")
    parser.add_argument("--motif", default="CCAGGC", help="Motif to search for")
    parser.add_argument("--output", default="signal_diagnosis.png", help="Output plot")
    parser.add_argument(
        "--min-mapq", type=int, default=10, help="Minimum mapping quality (default: 10)"
    )
    parser.add_argument(
        "--n-reads",
        type=int,
        default=500,
        help="Number of reads to analyze for monotonicity check (default: 500)",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("SIGNAL ORIENTATION DIAGNOSTIC")
    logger.info("=" * 80)

    # Test 1: Check move table monotonicity
    logger.info("")
    logger.info("1. Checking move table monotonicity...")
    logger.info(f"   Filtering reads with MAPQ >= {args.min_mapq}")
    logger.info(f"   Analyzing up to {args.n_reads} reads")
    monotonicity = check_move_table_monotonicity(
        args.bam, n_reads=args.n_reads, min_mapq=args.min_mapq
    )
    logger.info(f"   Results from {monotonicity['total']} reads:")
    logger.info(f"   - Increasing: {monotonicity['increasing']} reads")
    logger.info(f"   - Decreasing: {monotonicity['decreasing']} reads")
    logger.info(f"   - Mixed: {monotonicity['mixed']} reads")

    if monotonicity["decreasing"] > 0:
        logger.warning("   ⚠️  FOUND DECREASING MAPPINGS! This suggests signal reversal issue.")
    elif monotonicity["increasing"] == monotonicity["total"]:
        logger.info("   ✓ All mappings are increasing (expected)")

    # Test 2: Extract and visualize chunks
    logger.info("")
    logger.info(f"2. Extracting chunks around motif '{args.motif}'...")
    chunks = extract_motif_chunk_with_signal(
        args.bam, args.pod5, motif=args.motif, n_reads=5, min_mapq=args.min_mapq
    )
    logger.info(f"   Extracted {len(chunks)} chunks")

    if len(chunks) > 0:
        # Test 3: Plot signal alignment
        logger.info("")
        logger.info("3. Plotting signal alignment...")
        plot_signal_alignment(chunks, args.output)

        # Test 4: Base-signal correlations
        logger.info("")
        logger.info("4. Computing base-signal correlations...")
        base_means = compute_base_signal_correlation(chunks)
        logger.info("   Mean signal per base:")
        for base in sorted(base_means.keys()):
            logger.info(f"   - {base}: {base_means[base]:.2f} pA")

    logger.info("")
    logger.info("=" * 80)
    logger.info("INTERPRETATION:")
    logger.info("=" * 80)
    logger.info("• Move table mappings should be INCREASING:")
    logger.info("  - The seq-to-sig map should map base 0→signal_start, base 1→signal_mid, etc.")
    logger.info("  - If mappings DECREASE (base 0→signal_end, base 1→signal_start), this indicates")
    logger.info("    that signal and sequence are in opposite orientations (signal reversed)")
    logger.info("")
    logger.info("• Visual check:")
    logger.info("  - Bases should align with corresponding signal features in the plot")
    logger.info(
        "  - If bases appear 'backwards' relative to signal transitions, signal may be reversed"
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
