"""
High-level orchestration functions for data preparation pipeline.

This module provides the main entry points for preparing training data,
coordinating the extraction, splitting, and saving of training chunks.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from leech.chunking import extract_training_chunks, save_chunks
from leech.io import get_motif_searcher, get_reference_sequences
from leech.preparation.reader import iter_bam_with_pod5
from leech.splitting import split_chunks_by_read

logger = logging.getLogger("leech.preparation.orchestrator")


def prepare_training_data(
    bam_path: Path,
    pod5_path: Path,
    motif: str | None = None,
    motif_offset: int = 0,
    label: str | None = None,
    label_int: int | None = None,
    min_mapq: int = 0,
    base_justify: str = "center",
    dwell_margin: int = 0,
    reverse_signal: bool = True,
) -> tuple[list[dict[str, np.ndarray | str | int | None]], dict[str, int]]:
    """
    Prepare training data from BAM and POD5 files with statistics tracking.

    This is the basic sequential preparation function. For large datasets,
    consider using prepare_training_data_parallel from leech.preparation.parallel
    for significant performance improvements.

    Args:
        bam_path: Path to BAM file with alignments
        pod5_path: Path to POD5 file with signal
        motif: Optional sequence motif to filter
        motif_offset: Offset within motif for focus base
        label: String label identifier (e.g., "Ala", "Gly", "charged", "uncharged")
        label_int: Optional numeric label (0, 1) - assigned during merge for pairwise comparisons
        min_mapq: Minimum mapping quality
        reverse_signal: Reverse raw signal for RNA (POD5 3'→5' vs basecaller 5'→3')

    Returns:
        Tuple of (chunks, statistics) where statistics is a dict with:
        - total_reads: Total number of reads processed
        - reads_with_motif: Number of reads containing motif
        - reads_without_motif: Number of reads without motif
        - total_chunks: Total number of chunks extracted

    Examples:
        >>> chunks, stats = prepare_training_data(
        ...     bam_path=Path("alignments.bam"),
        ...     pod5_path=Path("reads.pod5"),
        ...     motif="CCAGGC",
        ...     min_mapq=10
        ... )
        >>> print(f"Extracted {len(chunks)} chunks from {stats['total_reads']} reads")
    """
    total_reads = 0
    reads_with_motif = 0
    reads_without_motif = 0
    chunks = []

    # Get motif searcher if motif is provided
    motif_searcher = None
    if motif is not None:
        motif_searcher = get_motif_searcher(mode="bam")

    for read in iter_bam_with_pod5(
        bam_path, pod5_path, min_mapq=min_mapq, reverse_signal=reverse_signal
    ):
        total_reads += 1
        read_chunks = extract_training_chunks(
            read,
            motif=motif,
            motif_offset=motif_offset,
            label=label,
            label_int=label_int,
            motif_searcher=motif_searcher,
            base_justify=base_justify,
            dwell_margin=dwell_margin,
        )

        # Track whether this read had motif matches
        if len(read_chunks) > 0:
            reads_with_motif += 1
        else:
            reads_without_motif += 1

        chunks.extend(read_chunks)

    # Compile statistics
    stats = {
        "total_reads": total_reads,
        "reads_with_motif": reads_with_motif,
        "reads_without_motif": reads_without_motif,
        "total_chunks": len(chunks),
    }

    # Log statistics
    logger.info(f"Processed {total_reads} reads")
    if motif is not None:
        logger.info(
            f"Reads with motif '{motif}': {reads_with_motif} "
            f"({100.0 * reads_with_motif / total_reads:.1f}%)"
        )
        logger.info(
            f"Reads without motif: {reads_without_motif} "
            f"({100.0 * reads_without_motif / total_reads:.1f}%)"
        )
    logger.info(f"Extracted {len(chunks)} training chunks")

    return chunks, stats


def prepare_training_data_with_split(
    pod5_path: Path,
    bam_path: Path,
    output_dir: Path,
    motif: str | None = None,
    motif_offset: int = 0,
    motif_reference: str = "fasta",
    reference_fasta: Path | None = None,
    skip_motif_indels: bool = True,
    label: str | None = None,
    label_int: int | None = None,
    min_mapq: int = 10,
    feature_set: str = "signal+dwell+levels",
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42,
    no_split: bool = False,
    progress_callback: Callable[[int], None] | None = None,
    base_justify: str = "center",
    dwell_margin: int = 0,
    reverse_signal: bool = True,
    anchor: str = "basecall",
    norm_method: str = "median_mad",
    pa_mean: float | None = None,
    pa_stdev: float | None = None,
    refine_signal_map: bool = False,
    signal_refiner: object | None = None,
) -> dict[str, Any]:
    """
    Prepare training data from POD5/BAM files with optional splitting.

    This function handles the full pipeline:
    1. Setup random seed for reproducibility
    2. Load reference sequences if needed
    3. Extract training chunks from reads
    4. Split at read level (if requested)
    5. Save to disk

    Args:
        pod5_path: Path to POD5 file
        bam_path: Path to BAM file
        output_dir: Directory for output files
        motif: Sequence motif to extract
        motif_offset: Offset within motif for focus base
        motif_reference: Where to search for motif ("fasta" or "bam")
        reference_fasta: External reference FASTA file
        skip_motif_indels: Skip reads with indels in motif region
        label: String label identifier (e.g., "Ala", "Gly", "charged", "uncharged")
        label_int: Optional numeric label (0, 1) - assigned during merge for pairwise comparisons
        min_mapq: Minimum mapping quality
        feature_set: Feature set to extract (not currently used, reserved for future)
        train_split: Fraction for training
        val_split: Fraction for validation
        seed: Random seed
        no_split: If True, save all chunks without splitting
        progress_callback: Optional callback(n_chunks) for progress updates

    Returns:
        Dictionary with statistics:
        - n_chunks: Total number of chunks extracted
        - n_train: Number of training chunks (0 if no_split=True)
        - n_val: Number of validation chunks (0 if no_split=True)
        - n_test: Number of test chunks (0 if no_split=True)
        - output_files: Dict mapping split names to file paths

    Examples:
        >>> result = prepare_training_data_with_split(
        ...     pod5_path=Path("reads.pod5"),
        ...     bam_path=Path("alignments.bam"),
        ...     output_dir=Path("chunks/"),
        ...     motif="CCAGGC",
        ...     train_split=0.7,
        ...     val_split=0.15,
        ...     seed=42
        ... )
        >>> print(f"Saved {result['n_train']} train chunks to {result['output_files']['train']}")
    """
    from leech.util import setup_random_seed

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup seed
    setup_random_seed(seed, output_dir)

    # Load reference sequences if needed
    reference_sequences = None
    if motif_reference == "fasta":
        logger.info("Loading reference sequences for reference-based motif search")
        reference_sequences = get_reference_sequences(bam_path, reference_fasta)

    # Get motif searcher
    motif_searcher = None
    if motif is not None:
        motif_searcher = get_motif_searcher(
            mode=motif_reference,
            reference_sequences=reference_sequences,
            skip_indels=skip_motif_indels,
        )

    # Extract chunks
    logger.info("Extracting chunks...")
    chunks = []
    for read in iter_bam_with_pod5(
        bam_path,
        pod5_path,
        min_mapq=min_mapq,
        reverse_signal=reverse_signal,
        anchor=anchor,
        norm_method=norm_method,
        pa_mean=pa_mean,
        pa_stdev=pa_stdev,
        refine_signal_map=refine_signal_map,
        signal_refiner=signal_refiner,
    ):
        read_chunks = extract_training_chunks(
            read,
            motif=motif,
            motif_offset=motif_offset,
            label=label,
            label_int=label_int,
            motif_searcher=motif_searcher,
            base_justify=base_justify,
            dwell_margin=dwell_margin,
        )
        chunks.extend(read_chunks)

        # Progress callback
        if progress_callback:
            progress_callback(len(chunks))

    logger.info(f"Extracted {len(chunks)} chunks")

    # Save or split
    if no_split:
        all_file = output_dir / "all.npz"
        save_chunks(chunks, all_file)
        logger.info(f"Saved all chunks to {all_file}")
        return {
            "n_chunks": len(chunks),
            "n_train": 0,
            "n_val": 0,
            "n_test": 0,
            "output_files": {"all": all_file},
        }
    else:
        # Split at read level
        train_chunks, val_chunks, test_chunks = split_chunks_by_read(
            chunks, train_frac=train_split, val_frac=val_split, seed=seed
        )

        # Save splits
        output_files = {}
        if train_chunks:
            train_file = output_dir / "train.npz"
            save_chunks(train_chunks, train_file)
            output_files["train"] = train_file
            logger.info(f"Saved {len(train_chunks)} train chunks to {train_file}")

        if val_chunks:
            val_file = output_dir / "val.npz"
            save_chunks(val_chunks, val_file)
            output_files["val"] = val_file
            logger.info(f"Saved {len(val_chunks)} val chunks to {val_file}")

        if test_chunks:
            test_file = output_dir / "test.npz"
            save_chunks(test_chunks, test_file)
            output_files["test"] = test_file
            logger.info(f"Saved {len(test_chunks)} test chunks to {test_file}")

        return {
            "n_chunks": len(chunks),
            "n_train": len(train_chunks),
            "n_val": len(val_chunks),
            "n_test": len(test_chunks),
            "output_files": output_files,
        }
