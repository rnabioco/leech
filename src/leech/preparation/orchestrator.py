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

from leech.chunking import ChunkSpool, extract_training_chunks
from leech.configs import PrepareConfig
from leech.io import get_motif_searcher, get_reference_sequences
from leech.preparation.reader import iter_bam_with_pod5
from leech.splitting import split_chunks_by_read

logger = logging.getLogger("leech.preparation.orchestrator")


def split_rows_by_read(
    read_ids: np.ndarray,
    train_frac: float,
    val_frac: float,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row indices per split, from the read-level assignment in ``leech.splitting``.

    When a corpus is spooled to disk (:class:`~leech.chunking.ChunkSpool`)
    there is no chunk list to hand :func:`split_chunks_by_read`. Feeding it one
    stand-in per row keeps the split rule — read-level assignment, stratified
    shuffle, and the grouped output order — in the one place that owns it
    rather than growing a second copy here, and the rows come back in exactly
    the order the chunk-level split would have produced.

    Args:
        read_ids: One read id per chunk, in corpus order.
        train_frac: Fraction of reads for training.
        val_frac: Fraction of reads for validation.
        seed: Random seed, passed through unchanged.

    Returns:
        ``(train_rows, val_rows, test_rows)`` as int64 index arrays.
    """
    stand_ins = [{"read_id": read_id, "row": row} for row, read_id in enumerate(read_ids.tolist())]
    train, val, test = split_chunks_by_read(
        stand_ins, train_frac=train_frac, val_frac=val_frac, seed=seed
    )
    del stand_ins

    def rows_of(split: list[dict]) -> np.ndarray:
        return np.fromiter((row["row"] for row in split), dtype=np.int64, count=len(split))

    return rows_of(train), rows_of(val), rows_of(test)


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

    # Extract chunks. They go straight into a spool, which holds the corpus on
    # disk: the whole corpus as chunk dicts, plus the stacked copy `save_chunks`
    # then builds, is what made this peak at several times the file it writes
    # (#211).
    logger.info("Extracting chunks...")
    logger.info(
        f"Chunks are spooled to temporary files in {output_dir} while they are "
        f"extracted; that directory needs room for the corpus twice over"
    )
    n_chunks = 0
    with ChunkSpool(output_dir) as spool:
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
            spool.append(read_chunks)
            n_chunks += len(read_chunks)
            del read_chunks

            if progress_callback:
                progress_callback(n_chunks)

        logger.info(f"Extracted {n_chunks} chunks")
        if n_chunks == 0:
            raise ValueError("No chunks to save")

        # Save or split
        if no_split:
            all_file = output_dir / "all.npz"
            spool.write_npz(all_file)
            logger.info(f"Saved all chunks to {all_file}")
            return {
                "n_chunks": n_chunks,
                "n_train": 0,
                "n_val": 0,
                "n_test": 0,
                "output_files": {"all": all_file},
            }

        train_rows, val_rows, test_rows = split_rows_by_read(
            spool.read_ids(), train_split, val_split, seed
        )

        output_files = {}
        for split, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
            if len(rows) == 0:
                continue
            split_file = output_dir / f"{split}.npz"
            spool.write_npz(split_file, rows=rows)
            output_files[split] = split_file
            logger.info(f"Saved {len(rows)} {split} chunks to {split_file}")

        return {
            "n_chunks": n_chunks,
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "n_test": len(test_rows),
            "output_files": output_files,
        }
