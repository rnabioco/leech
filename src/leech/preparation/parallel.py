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
from escapepod import Reader
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from leech.chunking import extract_training_chunks
from leech.configs import PrepareConfig
from leech.io import ReadInfo, get_motif_searcher, iter_read_info_batches
from leech.io.bam_reader import count_bam_reads
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

    reader = Reader(str(config.pod5_path))
    run_infos = reader.run_infos()
    reads = reader.get_reads(list(read_info_by_id.keys()))
    signals_list = reader.get_signals(reads)
    sig_by_id = dict(signals_list)
    for read_data in reads:
        rid = read_data.read_id
        signal = sig_by_id.get(rid)
        if signal is not None:
            pod5_cache[rid] = (signal, _extract_pod5_metadata(read_data, run_infos))

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
                "cl_value": getattr(read_info, "cl_value", None),
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

    Streams BAM reads in mega-batches so worker processing overlaps with
    BAM iteration rather than waiting for the entire BAM to be read first.

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

    # Estimate total reads from BAM index for progress bar (O(1), may be None)
    try:
        estimated_reads = count_bam_reads(bam_path)
        estimated_batches = max(1, estimated_reads // chunk_size)
        logger.info(f"BAM index reports ~{estimated_reads} mapped reads")
    except Exception:
        estimated_reads = None
        estimated_batches = None

    all_chunks: list[dict[str, np.ndarray | str | int | None]] = []
    total_reads = 0
    batches_submitted = 0
    batches_completed = 0

    use_progress_bar = sys.stdout.isatty()

    # Stream BAM reads in worker-sized batches and feed them lazily to
    # imap_unordered. The pool pulls batches from the generator as workers
    # become free, so BAM iteration overlaps with chunk extraction.
    logger.info("Streaming BAM reads and processing in parallel...")

    def _worker_arg_stream():
        nonlocal total_reads, batches_submitted
        for read_batch in iter_read_info_batches(
            bam_path, batch_size=chunk_size, min_mapq=min_mapq
        ):
            total_reads += len(read_batch)
            batches_submitted += 1
            yield (read_batch, config)

    with mp.Pool(processes=num_workers) as pool:
        if use_progress_bar:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[cyan]{task.fields[chunks_extracted]} chunks extracted"),
            ) as progress:
                task = progress.add_task(
                    "Processing reads",
                    total=estimated_batches,
                    chunks_extracted=0,
                )

                for chunk_results in pool.imap_unordered(
                    _process_read_chunk_worker, _worker_arg_stream()
                ):
                    all_chunks.extend(chunk_results)
                    batches_completed += 1
                    progress.update(
                        task, completed=batches_completed, chunks_extracted=len(all_chunks)
                    )

                # Fix total if estimate was off
                progress.update(task, total=batches_completed, completed=batches_completed)
        else:
            log_interval = max(1, (estimated_batches or 50) // 20)
            for chunk_results in pool.imap_unordered(
                _process_read_chunk_worker, _worker_arg_stream()
            ):
                all_chunks.extend(chunk_results)
                batches_completed += 1
                if batches_completed % log_interval == 0:
                    logger.info(
                        f"Progress: {batches_completed} batches, "
                        f"{total_reads} reads | {len(all_chunks)} chunks extracted"
                    )

    if total_reads == 0:
        return [], {
            "total_reads": 0,
            "reads_with_motif": 0,
            "reads_without_motif": 0,
            "total_chunks": 0,
        }

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
