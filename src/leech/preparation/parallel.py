"""
Parallel data preparation using multiprocessing.

This module provides parallel processing capabilities for extracting
training chunks from large datasets using multiple worker processes.
"""

import logging
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from pod5 import DatasetReader
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from leech.chunking import extract_training_chunks
from leech.configs import PrepareConfig
from leech.io import ReadInfo, collect_read_infos, get_motif_searcher
from leech.io.pod5_reader import _extract_pod5_metadata
from leech.preparation.reader import build_leech_read

logger = logging.getLogger("leech.preparation.parallel")


def _process_read_chunk_worker(
    args: tuple[list[ReadInfo], PrepareConfig],
) -> list[dict[str, np.ndarray | str | int | None]]:
    """
    Worker function to process a chunk of reads in parallel.

    Args:
        args: Tuple of (read_infos, config)

    Returns:
        List of extracted chunks from all reads in this chunk
    """
    read_infos, config = args

    # Get motif searcher
    if config.motif.motif is not None:
        motif_searcher = get_motif_searcher(
            mode=config.motif.motif_reference,
            reference_sequences=config.motif.reference_sequences,
            skip_indels=config.motif.skip_motif_indels,
            anchor=config.signal.anchor,
        )
    else:
        motif_searcher = None

    all_chunks: list[dict[str, np.ndarray | str | int | None]] = []

    # Batch-read all POD5 signals in one traversal (avoids per-read seeks on large files)
    read_info_by_id = {ri.read_id: ri for ri in read_infos}
    pod5_cache: dict[str, tuple] = {}  # read_id -> (signal, metadata)

    with DatasetReader(config.pod5_path) as pod5_reader:
        for read in pod5_reader.reads(list(read_info_by_id.keys())):
            rid = str(read.read_id)
            pod5_cache[rid] = (read.signal, _extract_pod5_metadata(read))

        for read_info in read_infos:
            try:
                cached = pod5_cache.get(read_info.read_id)
                if cached is None:
                    continue
                raw_signal, pod5_metadata = cached

                # Build metadata
                metadata = {
                    **pod5_metadata,
                    "mapping_quality": read_info.mapping_quality,
                    "reference_name": read_info.reference_name,
                    "reference_start": read_info.reference_start,
                    "reference_end": read_info.reference_end,
                    "is_reverse": read_info.is_reverse,
                }

                # For reference-based motif search, add mock alignment
                if (
                    config.motif.motif_reference == "fasta"
                    and config.motif.reference_sequences is not None
                ):
                    metadata["alignment"] = read_info.to_mock_alignment()

                # Build LeechRead via shared helper
                leech_read = build_leech_read(
                    read_id=read_info.read_id,
                    sequence=read_info.sequence,
                    raw_signal=raw_signal,
                    move_table=read_info.to_move_table(),
                    signal_config=config.signal,
                    metadata=metadata,
                    reference_sequence=read_info.reference_sequence,
                    cigar_tuples=read_info.cigar_tuples,
                    cal_offset=pod5_metadata.get("calibration_offset"),
                    cal_scale=pod5_metadata.get("calibration_scale"),
                )

                # Extract training chunks
                read_chunks = extract_training_chunks(
                    leech_read,
                    motif_config=config.motif,
                    chunk_config=config.chunk,
                    labeling=config.labeling,
                    motif_searcher=motif_searcher,
                )

                all_chunks.extend(read_chunks)

            except Exception as e:
                logger.warning(f"Worker failed to process read {read_info.read_id}: {e}")
                continue

    return all_chunks


def prepare_training_data_parallel(
    bam_path: Path,
    config: PrepareConfig,
    num_workers: int = 8,
    chunk_size: int = 100,
    min_mapq: int = 0,
) -> tuple[list[dict[str, np.ndarray | str | int | None]], dict[str, int]]:
    """
    Prepare training data from BAM and POD5 files using multiprocessing.

    Args:
        bam_path: Path to BAM file with alignments
        config: Preparation configuration
        num_workers: Number of parallel workers
        chunk_size: Number of reads to process per worker batch
        min_mapq: Minimum mapping quality

    Returns:
        Tuple of (chunks, statistics)
    """
    logger.info(f"Starting parallel data preparation with {num_workers} workers")

    # First pass: collect read info from BAM (lightweight, sequential)
    logger.info("Pass 1: Collecting read info from BAM...")
    read_infos = collect_read_infos(bam_path, min_mapq=min_mapq)
    total_reads = len(read_infos)
    logger.info(f"Found {total_reads} reads to process")

    if total_reads == 0:
        return [], {
            "total_reads": 0,
            "reads_with_motif": 0,
            "reads_without_motif": 0,
            "total_chunks": 0,
        }

    # Split read_infos into chunks for workers
    read_chunks = [read_infos[i : i + chunk_size] for i in range(0, len(read_infos), chunk_size)]
    logger.info(f"Split into {len(read_chunks)} chunks of up to {chunk_size} reads each")

    # Prepare worker arguments
    worker_args = [(chunk, config) for chunk in read_chunks]

    # Second pass: parallel processing with progress bar
    logger.info("Pass 2: Processing reads in parallel...")
    all_chunks = []

    use_progress_bar = sys.stdout.isatty()

    if use_progress_bar:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.fields[chunks_extracted]} chunks extracted"),
        ) as progress:
            task = progress.add_task(
                "Processing chunks", total=len(read_chunks), chunks_extracted=0
            )

            with mp.Pool(processes=num_workers) as pool:
                for chunk_results in pool.imap_unordered(_process_read_chunk_worker, worker_args):
                    all_chunks.extend(chunk_results)
                    progress.update(task, advance=1, chunks_extracted=len(all_chunks))
    else:
        log_interval = max(1, len(read_chunks) // 20)
        with mp.Pool(processes=num_workers) as pool:
            for i, chunk_results in enumerate(
                pool.imap_unordered(_process_read_chunk_worker, worker_args), 1
            ):
                all_chunks.extend(chunk_results)
                if i % log_interval == 0 or i == len(read_chunks):
                    pct = (i / len(read_chunks)) * 100
                    logger.info(
                        f"Progress: {i}/{len(read_chunks)} batches ({pct:.1f}%) | "
                        f"{len(all_chunks)} chunks extracted"
                    )

    stats = {
        "total_reads": total_reads,
        "reads_with_motif": len(all_chunks),
        "reads_without_motif": total_reads - len(all_chunks),
        "total_chunks": len(all_chunks),
    }

    logger.info(
        f"Parallel processing complete: extracted {len(all_chunks)} chunks from {total_reads} reads"
    )

    return all_chunks, stats
