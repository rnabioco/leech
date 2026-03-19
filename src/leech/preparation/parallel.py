"""
Parallel data preparation using multiprocessing.

This module provides parallel processing capabilities for extracting
training chunks from large datasets using multiple worker processes.

Supports two backends:
- Rust (default when available): Uses rayon parallelism via leech_core,
  eliminating pickle serialization overhead.
- Python: Falls back to multiprocessing.Pool.
"""

import logging
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from pod5 import DatasetReader
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from leech._rust_accel import HAS_RUST, _rs_extract_training_chunks_batch
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


def _prepare_with_rust(
    read_infos: list[ReadInfo],
    config: PrepareConfig,
) -> tuple[list[dict[str, np.ndarray | str | int | None]], dict[str, int]]:
    """
    Prepare training data using the Rust pipeline (rayon parallelism).

    Unpacks ReadInfo objects into parallel arrays, calls the Rust function,
    then converts the columnar TrainingBatchResult into list-of-dicts format.
    """
    assert _rs_extract_training_chunks_batch is not None

    total_reads = len(read_infos)

    # Unpack ReadInfo objects into parallel arrays
    read_ids = [ri.read_id for ri in read_infos]
    sequences = [ri.sequence for ri in read_infos]
    mv_strides = [ri.stride for ri in read_infos]
    mv_arrays = [list(ri.moves) for ri in read_infos]
    num_samples_list = [ri.num_samples for ri in read_infos]
    trim_offsets = [ri.trim_offset for ri in read_infos]

    # Reference anchoring data
    cigar_tuples = None
    reference_sequences = None
    ref_names: list[str | None] | None = None
    ref_starts: list[int] | None = None
    ref_ends: list[int] | None = None

    if config.signal.anchor == "reference":
        cigar_tuples = [
            [(op, length) for op, length in (ri.cigar_tuples or [])]
            for ri in read_infos
        ]
        reference_sequences = [ri.reference_sequence for ri in read_infos]
        ref_names = [ri.reference_name for ri in read_infos]
        ref_starts = [ri.reference_start or 0 for ri in read_infos]
        ref_ends = [ri.reference_end or 0 for ri in read_infos]

    # Signal refinement kmer table
    kmer_table = None
    kmer_len = 9
    if config.signal.refine_signal_map and config.signal.signal_refiner is not None:
        refiner = config.signal.signal_refiner
        kmer_table = refiner.kmer_to_level
        kmer_len = refiner.kmer_len

    # Signal context
    sig_left, sig_right = config.chunk.signal_context

    logger.info(
        f"Rust pipeline: processing {total_reads} reads "
        f"(signal context {sig_left}/{sig_right}, "
        f"kmer context {config.chunk.kmer_context})"
    )

    result = _rs_extract_training_chunks_batch(
        pod5_path=str(config.pod5_path),
        read_ids=read_ids,
        sequences=sequences,
        mv_strides=mv_strides,
        mv_arrays=mv_arrays,
        num_samples_list=num_samples_list,
        trim_offsets=trim_offsets,
        signal_context_left=sig_left,
        signal_context_right=sig_right,
        kmer_context=config.chunk.kmer_context,
        signal_len=sig_left + sig_right,
        compute_features=config.signal.compute_features,
        reverse_signal=config.signal.reverse_signal,
        feature_start=config.chunk.feature_start,
        feature_end=config.chunk.feature_end,
        anchor=config.signal.anchor,
        cigar_tuples=cigar_tuples,
        reference_sequences=reference_sequences,
        motif=config.motif.motif,
        motif_offset=config.motif.motif_offset,
        motif_reference=config.motif.motif_reference,
        ref_seq_dict=config.motif.reference_sequences,
        ref_names=ref_names,
        ref_starts=ref_starts,
        ref_ends=ref_ends,
        skip_motif_indels=config.motif.skip_motif_indels,
        norm_method=config.signal.norm_method,
        base_justify=config.chunk.base_justify,
        refine_signal_map=config.signal.refine_signal_map,
        kmer_table=kmer_table,
        kmer_len=kmer_len,
        kmer_center_idx=config.signal.refine_kmer_center_idx,
        refine_half_bandwidth=config.signal.refine_half_bandwidth,
        refine_scale_iters=config.signal.refine_scale_iters,
    )

    n_chunks = result.num_chunks
    logger.info(f"Rust pipeline: extracted {n_chunks} chunks")

    if n_chunks == 0:
        return [], {
            "total_reads": total_reads,
            "reads_with_motif": 0,
            "reads_without_motif": total_reads,
            "total_chunks": 0,
        }

    # Convert columnar TrainingBatchResult to list-of-dicts
    signals = np.asarray(result.signals)
    dwells = np.asarray(result.dwells)
    features = np.asarray(result.features)
    base_indices = np.asarray(result.base_indices)
    feature_starts = np.asarray(result.feature_starts)
    feature_ends = np.asarray(result.feature_ends)

    # Optional residuals
    residuals = np.asarray(result.signal_residuals) if result.signal_residuals is not None else None

    all_chunks: list[dict[str, np.ndarray | str | int | None]] = []
    for i in range(n_chunks):
        chunk: dict[str, np.ndarray | str | int | None] = {
            "signal": signals[i],
            "sequence": result.sequences[i],
            "dwell": dwells[i],
            "features": features[i],
            "feature_start": int(feature_starts[i]),
            "feature_end": int(feature_ends[i]),
            "base_idx": int(base_indices[i]),
            "read_id": result.read_ids[i],
            "seq_to_sig_map": np.asarray(result.seq_to_sig_maps[i]),
            "sequence_with_kmer_context": result.sequences_with_kmer_context[i],
            # Labels are applied by the caller (not available in Rust pipeline)
            "label_int": config.labeling.label_int,
            "label": config.labeling.label,
            "cl_value": None,
        }
        if residuals is not None:
            chunk["signal_residual"] = residuals[i]
        all_chunks.append(chunk)

    stats = {
        "total_reads": total_reads,
        "reads_with_motif": n_chunks,
        "reads_without_motif": total_reads - n_chunks,
        "total_chunks": n_chunks,
    }

    return all_chunks, stats


def prepare_training_data_parallel(
    bam_path: Path,
    config: PrepareConfig,
    num_workers: int = 8,
    chunk_size: int = 100,
    min_mapq: int = 0,
) -> tuple[list[dict[str, np.ndarray | str | int | None]], dict[str, int]]:
    """
    Prepare training data from BAM and POD5 files using multiprocessing.

    Uses the Rust pipeline when available (rayon parallelism, no pickle
    overhead). Falls back to Python multiprocessing.Pool otherwise.

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

    # Try Rust fast path
    if HAS_RUST and _rs_extract_training_chunks_batch is not None:
        logger.info("Using Rust pipeline (rayon parallelism)")
        return _prepare_with_rust(read_infos, config)

    # Fallback: Python multiprocessing
    logger.info("Using Python multiprocessing fallback")
    return _prepare_with_python_mp(read_infos, config, num_workers, chunk_size)


def _prepare_with_python_mp(
    read_infos: list[ReadInfo],
    config: PrepareConfig,
    num_workers: int,
    chunk_size: int,
) -> tuple[list[dict[str, np.ndarray | str | int | None]], dict[str, int]]:
    """Python multiprocessing fallback for data preparation."""
    total_reads = len(read_infos)

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
