"""
Parallel data preparation using multiprocessing.

This module provides parallel processing capabilities for extracting
training chunks from large datasets using multiple worker processes.
"""

import logging
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pod5 import DatasetReader
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from leech.chunking import extract_training_chunks
from leech.io import ReadInfo, collect_read_infos, get_motif_searcher
from leech.io.pod5_reader import _extract_pod5_metadata
from leech.preparation.reader import build_leech_read

if TYPE_CHECKING:
    from leech.signal_refine import SigMapRefiner

logger = logging.getLogger("leech.preparation.parallel")


@dataclass
class WorkerConfig:
    """Configuration shared by all parallel workers.

    Groups the non-data parameters that are constant across worker
    invocations so the worker receives ``(read_infos, config)``
    instead of an 18-element tuple.  Standard ``@dataclass`` is
    picklable by default.
    """

    pod5_path: Path
    motif: str | None
    motif_offset: int
    label: str | None
    label_int: int | None
    motif_reference: str
    reference_sequences: dict[str, str] | None
    skip_motif_indels: bool
    base_justify: str
    feature_start: int | None
    feature_end: int | None
    reverse_signal: bool
    anchor: str
    norm_method: str
    pa_mean: float | None
    pa_stdev: float | None
    refine_signal_map: bool
    signal_refiner: "SigMapRefiner | None"


def _process_read_chunk_worker(
    args: tuple[list[ReadInfo], WorkerConfig],
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
    if config.motif is not None:
        motif_searcher = get_motif_searcher(
            mode=config.motif_reference,
            reference_sequences=config.reference_sequences,
            skip_indels=config.skip_motif_indels,
            anchor=config.anchor,
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
                if config.motif_reference == "fasta" and config.reference_sequences is not None:
                    metadata["alignment"] = read_info.to_mock_alignment()

                # Build LeechRead via shared helper
                leech_read = build_leech_read(
                    read_id=read_info.read_id,
                    sequence=read_info.sequence,
                    raw_signal=raw_signal,
                    move_table=read_info.to_move_table(),
                    reverse_signal=config.reverse_signal,
                    metadata=metadata,
                    anchor=config.anchor,
                    reference_sequence=read_info.reference_sequence,
                    cigar_tuples=read_info.cigar_tuples,
                    norm_method=config.norm_method,
                    pa_mean=config.pa_mean,
                    pa_stdev=config.pa_stdev,
                    cal_offset=pod5_metadata.get("calibration_offset"),
                    cal_scale=pod5_metadata.get("calibration_scale"),
                    refine_signal_map=config.refine_signal_map,
                    signal_refiner=config.signal_refiner,
                )

                # Extract training chunks
                read_chunks = extract_training_chunks(
                    leech_read,
                    motif=config.motif,
                    motif_offset=config.motif_offset,
                    label=config.label,
                    label_int=config.label_int,
                    motif_searcher=motif_searcher,
                    base_justify=config.base_justify,
                    feature_start=config.feature_start,
                    feature_end=config.feature_end,
                )

                all_chunks.extend(read_chunks)

            except Exception as e:
                logger.warning(f"Worker failed to process read {read_info.read_id}: {e}")
                continue

    return all_chunks


def prepare_training_data_parallel(
    bam_path: Path,
    pod5_path: Path,
    motif: str | None = None,
    motif_offset: int = 0,
    label: str | None = None,
    label_int: int | None = None,
    min_mapq: int = 0,
    motif_reference: str = "bam",
    reference_sequences: dict[str, str] | None = None,
    skip_motif_indels: bool = True,
    num_workers: int = 8,
    chunk_size: int = 100,
    base_justify: str = "center",
    feature_start: int | None = None,
    feature_end: int | None = None,
    reverse_signal: bool = True,
    anchor: str = "basecall",
    norm_method: str = "median_mad",
    pa_mean: float | None = None,
    pa_stdev: float | None = None,
    refine_signal_map: bool = False,
    signal_refiner: "SigMapRefiner | None" = None,
) -> tuple[list[dict[str, np.ndarray | str | int | None]], dict[str, int]]:
    """
    Prepare training data from BAM and POD5 files using multiprocessing.

    This function provides significant speedup for large datasets by processing
    reads in parallel across multiple CPU cores. Expected speedup is 3-6x on
    typical multi-core machines for CPU-bound feature extraction tasks.

    Args:
        bam_path: Path to BAM file with alignments
        pod5_path: Path to POD5 file with signal
        motif: Optional sequence motif to filter
        motif_offset: Offset within motif for focus base
        label: String label identifier (e.g., "Ala", "Gly", "charged", "uncharged")
        label_int: Optional numeric label (0, 1) - assigned during merge for pairwise comparisons
        min_mapq: Minimum mapping quality
        motif_reference: Where to search for motif ("bam" or "fasta")
        reference_sequences: Dict of reference sequences (for motif_reference="fasta")
        skip_motif_indels: Skip reads with indels in motif region
        num_workers: Number of parallel workers
        chunk_size: Number of reads to process per worker batch
        reverse_signal: Reverse raw signal for RNA (POD5 3'→5' vs basecaller 5'→3')

    Returns:
        Tuple of (chunks, statistics) where statistics contains:
        - total_reads: Total number of reads found in BAM
        - reads_with_motif: Approximate count of reads with motif matches
        - reads_without_motif: Approximate count without matches
        - total_chunks: Total number of chunks extracted

    Examples:
        >>> chunks, stats = prepare_training_data_parallel(
        ...     bam_path=Path("alignments.bam"),
        ...     pod5_path=Path("reads.pod5"),
        ...     motif="CCAGGC",
        ...     num_workers=8,
        ...     chunk_size=100
        ... )
        >>> print(f"Extracted {stats['total_chunks']} chunks from {stats['total_reads']} reads")
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

    # Build shared config
    config = WorkerConfig(
        pod5_path=pod5_path,
        motif=motif,
        motif_offset=motif_offset,
        label=label,
        label_int=label_int,
        motif_reference=motif_reference,
        reference_sequences=reference_sequences,
        skip_motif_indels=skip_motif_indels,
        base_justify=base_justify,
        feature_start=feature_start,
        feature_end=feature_end,
        reverse_signal=reverse_signal,
        anchor=anchor,
        norm_method=norm_method,
        pa_mean=pa_mean,
        pa_stdev=pa_stdev,
        refine_signal_map=refine_signal_map,
        signal_refiner=signal_refiner,
    )

    # Prepare worker arguments
    worker_args = [(chunk, config) for chunk in read_chunks]

    # Second pass: parallel processing with progress bar
    logger.info("Pass 2: Processing reads in parallel...")
    all_chunks = []

    # Check if we're in a TTY (interactive terminal) or redirected (e.g., log file)
    use_progress_bar = sys.stdout.isatty()

    if use_progress_bar:
        # Interactive terminal - use rich progress bar
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
                # Use imap_unordered for progress tracking
                for chunk_results in pool.imap_unordered(_process_read_chunk_worker, worker_args):
                    all_chunks.extend(chunk_results)
                    progress.update(task, advance=1, chunks_extracted=len(all_chunks))
    else:
        # Redirected output (log file) - use periodic logging
        log_interval = max(1, len(read_chunks) // 20)  # Log every 5%
        with mp.Pool(processes=num_workers) as pool:
            for i, chunk_results in enumerate(
                pool.imap_unordered(_process_read_chunk_worker, worker_args), 1
            ):
                all_chunks.extend(chunk_results)
                # Log progress periodically
                if i % log_interval == 0 or i == len(read_chunks):
                    pct = (i / len(read_chunks)) * 100
                    logger.info(
                        f"Progress: {i}/{len(read_chunks)} batches ({pct:.1f}%) | "
                        f"{len(all_chunks)} chunks extracted"
                    )

    # Compile statistics (approximate - we don't track individual read success)
    stats = {
        "total_reads": total_reads,
        "reads_with_motif": len(all_chunks),  # Approximate
        "reads_without_motif": total_reads - len(all_chunks),  # Approximate
        "total_chunks": len(all_chunks),
    }

    logger.info(
        f"Parallel processing complete: extracted {len(all_chunks)} chunks from {total_reads} reads"
    )

    return all_chunks, stats
