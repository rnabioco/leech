"""
Parallel data preparation using multiprocessing.

This module provides parallel processing capabilities for extracting
training chunks from large datasets using multiple worker processes.

When leech_core Rust acceleration is available, uses a single-call Rust
pipeline (POD5 I/O + normalize + anchor + refine + features + chunk
extraction) with rayon parallelism. Falls back to Python multiprocessing
workers otherwise.
"""

import logging
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from leech._rust_accel import (
    HAS_RUST,
    RUST_NORM_METHOD,
    _rs_extract_training_chunks,
    rust_supports_norm_method,
    rust_supports_softclip_recovery,
)
from leech.chunking import extract_training_chunks
from leech.configs import PrepareConfig
from leech.io import ReadInfo, get_motif_searcher, iter_read_info_batches
from leech.io.bam_reader import count_bam_reads
from leech.io.motif_search import MotifSearcher
from leech.io.pod5_reader import read_pod5_signals_batch_cached
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

    # Focus-map early-skip: drop reads not in the map BEFORE the POD5
    # signal fetch and move-table parse. Without this, an 800k-read BAM
    # pays full POD5 I/O even when the focus map only targets 100k reads —
    # the saved work per worker is enormous.
    if config.labeling.focus_map is not None:
        read_infos = [ri for ri in read_infos if ri.read_id in config.labeling.focus_map]
        if not read_infos:
            return all_chunks

    # Batch-read all POD5 signals via the process-local reader cache.
    read_info_by_id = {ri.read_id: ri for ri in read_infos}
    pod5_cache = read_pod5_signals_batch_cached(config.pod5_path, list(read_info_by_id.keys()))

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


# ---------------------------------------------------------------------------
# Rust-accelerated batch preparation
# ---------------------------------------------------------------------------


def _find_motif_positions(
    read_info: ReadInfo,
    motif_searcher: MotifSearcher | None,
    config: PrepareConfig,
) -> list[int]:
    """Find motif positions for a single read, returning focus base indices."""
    if motif_searcher is None:
        # All bases mode (skip edges)
        num_bases = len(read_info.sequence)
        return list(range(5, max(5, num_bases - 5)))

    alignment = None
    if config.motif.motif_reference == "fasta" and config.motif.reference_sequences is not None:
        alignment = read_info.to_mock_alignment()

    positions = motif_searcher.find_motif_positions(
        read_id=read_info.read_id,
        sequence=read_info.sequence,
        alignment=alignment,
        motif=config.motif.motif,
    )
    return [p + config.motif.motif_offset for p in positions]


def _prepare_batch_rust(
    read_infos: list[ReadInfo],
    config: PrepareConfig,
    motif_searcher: MotifSearcher | None,
) -> list[dict[str, np.ndarray | str | int | None]]:
    """
    Process a batch of reads using the Rust pipeline.

    Collects BAM metadata, finds motif positions in Python, then delegates
    all signal processing + chunk extraction to Rust (rayon-parallel, GIL
    released).  Attaches Python-side labels/metadata to the returned chunks.
    """
    # Collect per-read BAM metadata arrays for the Rust call
    read_ids: list[str] = []
    sequences: list[str] = []
    mv_strides: list[int] = []
    mv_arrays: list[list[int]] = []
    num_samples_list: list[int] = []
    trim_offsets: list[int] = []
    motif_positions: list[list[int]] = []
    cigar_tuples: list[list[tuple[int, int]]] | None = None
    reference_sequences: list[str | None] | None = None

    if config.signal.anchor == "reference":
        cigar_tuples = []
        reference_sequences = []

    # Per-read metadata for label attachment after Rust extraction
    read_meta: dict[str, dict] = {}

    for ri in read_infos:
        mt = ri.to_move_table()
        positions = _find_motif_positions(ri, motif_searcher, config)
        if not positions:
            continue

        read_ids.append(ri.read_id)
        sequences.append(ri.sequence)
        mv_strides.append(mt.stride)
        mv_arrays.append(mt.moves.tolist())
        num_samples_list.append(mt.num_samples)
        trim_offsets.append(mt.trim_offset)
        motif_positions.append(positions)

        if cigar_tuples is not None:
            cigar_tuples.append(ri.cigar_tuples or [])
        if reference_sequences is not None:
            reference_sequences.append(ri.reference_sequence)

        read_meta[ri.read_id] = {
            "reference_name": ri.reference_name or "",
            "cl_value": getattr(ri, "cl_value", None),
        }

    if not read_ids:
        return []

    # Resolve signal context
    from leech.constants import DEFAULT_SIGNAL_CONTEXT

    sig_ctx = config.chunk.signal_context or DEFAULT_SIGNAL_CONTEXT
    signal_len = sig_ctx[0] + sig_ctx[1]

    # Resolve kmer table for signal refinement
    kmer_table_dict: dict[str, float] | None = None
    kmer_len = 9
    kmer_center_idx = -1
    if config.signal.refine_signal_map and config.signal.signal_refiner is not None:
        refiner = config.signal.signal_refiner
        kmer_table_dict = refiner.kmer_to_level
        kmer_len = refiner.kmer_len
        kmer_center_idx = getattr(refiner, "kmer_center_idx", -1)

    # Call Rust: POD5 I/O + normalize + anchor + refine + features + chunk extraction
    rust_chunks = _rs_extract_training_chunks(
        pod5_path=str(config.pod5_path),
        read_ids=read_ids,
        sequences=sequences,
        mv_strides=mv_strides,
        mv_arrays=mv_arrays,
        num_samples_list=num_samples_list,
        trim_offsets=trim_offsets,
        signal_context_left=sig_ctx[0],
        signal_context_right=sig_ctx[1],
        kmer_context=config.chunk.kmer_context,
        motif_positions=motif_positions,
        signal_len=signal_len,
        compute_features=config.signal.compute_features,
        reverse_signal=config.signal.reverse_signal,
        feature_start=config.chunk.feature_start,
        feature_end=config.chunk.feature_end,
        anchor=config.signal.anchor,
        cigar_tuples=cigar_tuples,
        reference_sequences=reference_sequences,
        refine_signal_map=config.signal.refine_signal_map,
        kmer_table=kmer_table_dict,
        kmer_len=kmer_len,
        kmer_center_idx=kmer_center_idx,
        refine_half_bandwidth=config.signal.refine_half_bandwidth,
        refine_scale_iters=config.signal.refine_scale_iters,
        signal_in_channels=2 if (config.signal.refine_signal_map and kmer_table_dict) else 1,
        base_justify=config.chunk.base_justify,
    )

    # Attach Python-side labels/metadata
    all_chunks: list[dict] = []
    for chunk_dict in rust_chunks:
        rid = chunk_dict["read_id"]
        meta = read_meta.get(rid, {})

        # Add labeling
        chunk_dict["label"] = config.labeling.label
        chunk_dict["label_int"] = config.labeling.label_int
        chunk_dict["source_group"] = ""
        chunk_dict["reference_name"] = meta.get("reference_name", "")
        chunk_dict["cl_value"] = meta.get("cl_value")
        chunk_dict["feature_start"] = config.chunk.feature_start or -5
        chunk_dict["feature_end"] = config.chunk.feature_end or 5

        all_chunks.append(chunk_dict)

    return all_chunks


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def rust_prepare_unsupported_reason(config: PrepareConfig) -> str | None:
    """Why the Rust prepare pipeline cannot serve ``config``, or ``None``.

    Kept as a pure function (no I/O, no ``HAS_RUST`` check) so the capability
    rules can be tested without running a preparation pass. The caller ANDs the
    result with actual Rust availability.

    The Rust path is bypassed when:

    - ``labeling.focus_map`` is set. ``_prepare_batch_rust`` skips
      ``extract_training_chunks`` entirely and stamps the file-level
      ``label_int`` on every chunk, and it hands Rust a single POD5 path
      string, which fails on a directory source with os error 19. Focus mode
      needs per-read labels and typically a directory of POD5s.
    - ``signal.norm_method`` is anything but median-MAD.
      ``rust/src/inference_pipeline/processing.rs`` normalizes
      unconditionally and its ``PipelineConfig`` has no normalization field,
      so another method would be silently ignored.
    - ``chunk.recover_softclip_signal`` is set. Recovery reads from the full
      pre-crop signal, which the Rust ``ProcessedRead`` discards when it crops
      to the aligned region, so the flag would silently degrade to zero-padding.
    """
    if config.labeling.focus_map is not None:
        return "focus_map is set (no per-read label or multi-POD5 support in Rust yet)"
    if not rust_supports_norm_method(config.signal.norm_method):
        return (
            f"signal normalization {config.signal.norm_method!r} is not implemented "
            f"in the Rust pipeline, which always applies {RUST_NORM_METHOD!r}"
        )
    if not rust_supports_softclip_recovery(config.chunk.recover_softclip_signal):
        return (
            "recover_softclip_signal is not implemented in the Rust pipeline "
            "(it discards the pre-crop signal the recovery reads from)"
        )
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


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

    When leech_core Rust acceleration is available, uses the Rust pipeline
    for signal processing + chunk extraction (rayon-parallel, GIL released).
    Falls back to Python multiprocessing workers otherwise.

    Args:
        bam_path: Path to BAM file with alignments
        config: Preparation configuration
        num_workers: Number of parallel workers
        chunk_size: Number of reads to process per worker batch
        min_mapq: Minimum mapping quality

    Returns:
        Tuple of (chunks, statistics)
    """
    reason = rust_prepare_unsupported_reason(config)
    use_rust = HAS_RUST and _rs_extract_training_chunks is not None and reason is None
    if reason is not None and HAS_RUST:
        logger.warning(f"Using Python workers instead of the Rust pipeline: {reason}")
    backend = "Rust (rayon)" if use_rust else "Python (multiprocessing)"
    logger.info(f"Starting parallel data preparation with {num_workers} workers [{backend}]")

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

    if use_rust:
        # Rust path: collect BAM metadata in Python, delegate heavy work to Rust.
        # Each mega-batch is processed in a single Rust call with rayon parallelism.
        logger.info("Streaming BAM reads with Rust-accelerated chunk extraction...")

        # Setup motif searcher (Python-side, needed for position finding)
        if config.motif.motif is not None:
            motif_searcher = get_motif_searcher(
                mode=config.motif.motif_reference,
                reference_sequences=config.motif.reference_sequences,
                skip_indels=config.motif.skip_motif_indels,
                anchor=config.signal.anchor,
            )
        else:
            motif_searcher = None

        if use_progress_bar:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[cyan]{task.fields[chunks_extracted]} chunks extracted"),
            ) as progress:
                task = progress.add_task(
                    "Processing reads (Rust)",
                    total=estimated_batches,
                    chunks_extracted=0,
                )

                for read_batch in iter_read_info_batches(
                    bam_path, batch_size=chunk_size, min_mapq=min_mapq
                ):
                    total_reads += len(read_batch)
                    batches_completed += 1
                    try:
                        batch_chunks = _prepare_batch_rust(read_batch, config, motif_searcher)
                        all_chunks.extend(batch_chunks)
                    except Exception as e:
                        logger.warning(f"Rust batch failed, skipping: {e}")
                    progress.update(
                        task, completed=batches_completed, chunks_extracted=len(all_chunks)
                    )

                progress.update(task, total=batches_completed, completed=batches_completed)
        else:
            log_interval = max(1, (estimated_batches or 50) // 20)
            for read_batch in iter_read_info_batches(
                bam_path, batch_size=chunk_size, min_mapq=min_mapq
            ):
                total_reads += len(read_batch)
                batches_completed += 1
                try:
                    batch_chunks = _prepare_batch_rust(read_batch, config, motif_searcher)
                    all_chunks.extend(batch_chunks)
                except Exception as e:
                    logger.warning(f"Rust batch failed, skipping: {e}")
                if batches_completed % log_interval == 0:
                    logger.info(
                        f"Progress: {batches_completed} batches, "
                        f"{total_reads} reads | {len(all_chunks)} chunks extracted"
                    )
    else:
        # Python multiprocessing fallback
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
