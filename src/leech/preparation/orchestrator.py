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
from leech.configs import PrepareConfig
from leech.io import get_motif_searcher, get_reference_sequences
from leech.preparation.reader import iter_bam_with_pod5
from leech.splitting import split_chunks_by_read

logger = logging.getLogger("leech.preparation.orchestrator")


def prepare_training_data(
    bam_path: Path,
    config: PrepareConfig,
    min_mapq: int = 0,
) -> tuple[list[dict[str, np.ndarray | str | int | None]], dict[str, int]]:
    """
    Prepare training data from BAM and POD5 files with statistics tracking.

    Args:
        bam_path: Path to BAM file with alignments
        config: Preparation configuration
        min_mapq: Minimum mapping quality

    Returns:
        Tuple of (chunks, statistics)
    """
    total_reads = 0
    reads_with_motif = 0
    reads_without_motif = 0
    chunks = []

    # Get motif searcher if motif is provided
    motif_searcher = None
    if config.motif.motif is not None:
        motif_searcher = get_motif_searcher(
            mode=config.motif.motif_reference,
            reference_sequences=config.motif.reference_sequences,
            skip_indels=config.motif.skip_motif_indels,
            require_query_mapping=config.motif.require_query_mapping,
            anchor=config.signal.anchor,
        )

    for read in iter_bam_with_pod5(
        bam_path,
        config.pod5_path,
        signal_config=config.signal,
        min_mapq=min_mapq,
        reference_sequences=config.motif.reference_sequences,
    ):
        total_reads += 1
        read_chunks = extract_training_chunks(
            read,
            motif_config=config.motif,
            chunk_config=config.chunk,
            labeling=config.labeling,
            motif_searcher=motif_searcher,
        )

        if len(read_chunks) > 0:
            reads_with_motif += 1
        else:
            reads_without_motif += 1

        chunks.extend(read_chunks)

    stats = {
        "total_reads": total_reads,
        "reads_with_motif": reads_with_motif,
        "reads_without_motif": reads_without_motif,
        "total_chunks": len(chunks),
    }

    logger.info(f"Processed {total_reads} reads")
    if config.motif.motif is not None:
        logger.info(
            f"Reads with motif '{config.motif.motif}': {reads_with_motif} "
            f"({100.0 * reads_with_motif / total_reads:.1f}%)"
        )
        logger.info(
            f"Reads without motif: {reads_without_motif} "
            f"({100.0 * reads_without_motif / total_reads:.1f}%)"
        )
    logger.info(f"Extracted {len(chunks)} training chunks")

    return chunks, stats


def prepare_training_data_with_split(
    bam_path: Path,
    config: PrepareConfig,
    output_dir: Path,
    reference_fasta: Path | None = None,
    min_mapq: int = 10,
    feature_set: str = "signal+dwell+levels",
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42,
    no_split: bool = False,
    progress_callback: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """
    Prepare training data from POD5/BAM files with optional splitting.

    Args:
        bam_path: Path to BAM file
        config: Preparation configuration
        output_dir: Directory for output files
        reference_fasta: External reference FASTA file
        min_mapq: Minimum mapping quality
        feature_set: Feature set to extract (reserved for future)
        train_split: Fraction for training
        val_split: Fraction for validation
        seed: Random seed
        no_split: If True, save all chunks without splitting
        progress_callback: Optional callback(n_chunks) for progress updates

    Returns:
        Dictionary with statistics
    """
    from leech.model_loading import setup_random_seed

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup seed
    setup_random_seed(seed, output_dir)

    # Load reference sequences if needed and not already loaded
    if config.motif.motif_reference == "fasta" and config.motif.reference_sequences is None:
        logger.info("Loading reference sequences for reference-based motif search")
        config.motif.reference_sequences = get_reference_sequences(bam_path, reference_fasta)

    # Get motif searcher
    motif_searcher = None
    if config.motif.motif is not None:
        motif_searcher = get_motif_searcher(
            mode=config.motif.motif_reference,
            reference_sequences=config.motif.reference_sequences,
            skip_indels=config.motif.skip_motif_indels,
            require_query_mapping=config.motif.require_query_mapping,
            anchor=config.signal.anchor,
        )

    # Extract chunks
    logger.info("Extracting chunks...")
    chunks = []
    for read in iter_bam_with_pod5(
        bam_path,
        config.pod5_path,
        signal_config=config.signal,
        min_mapq=min_mapq,
        reference_sequences=config.motif.reference_sequences,
    ):
        read_chunks = extract_training_chunks(
            read,
            motif_config=config.motif,
            chunk_config=config.chunk,
            labeling=config.labeling,
            motif_searcher=motif_searcher,
        )
        chunks.extend(read_chunks)

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
        train_chunks, val_chunks, test_chunks = split_chunks_by_read(
            chunks, train_frac=train_split, val_frac=val_split, seed=seed
        )

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
