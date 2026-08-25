"""
Parallel data preparation.

Extracts training chunks from a BAM + POD5 pair by streaming the BAM in
batches and processing several batches at once. Two interchangeable backends:

- **Rust** (``leech_core`` installed and the config is supported): each batch
  is one call into a pipeline that does POD5 I/O, normalization, anchoring,
  refinement, feature computation, and chunk extraction. Batches are dispatched
  across a ``ThreadPoolExecutor``; within a batch, per-read work runs on rayon.
- **Python** (fallback): each batch goes to a worker process from an
  ``mp.Pool``.

**Both backends run batches concurrently.** The Rust one is threads rather than
processes only because it releases the GIL for the whole call, so it does not
need separate address spaces to get real parallelism.

If you are changing this module, the trap to know about: the Rust call *looks*
self-parallelizing, because per-read work inside it is rayon-parallel and the
docstrings used to say so. It is not. Roughly all of the wall clock on a large
POD5 is phase 1 — resolving read IDs and faulting signal in through an mmap on
a network filesystem — which is one sequential stream of page faults per call.
Driving those calls from a serial ``for`` loop leaves exactly one read
outstanding at a time and gives up an order of magnitude against the process
pool, which was issue #176. Concurrency across batches is the whole ballgame;
do not collapse ``_iter_rust_batches`` back into a loop.
"""

import logging
import multiprocessing as mp
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
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
from leech.chunking import (
    extract_training_chunks,
    extraction_sequence,
    find_focus_bases,
    resolve_feature_window,
)
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
            require_query_mapping=config.motif.require_query_mapping,
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


def _process_read_chunk_worker_seq(
    args: tuple[int, list[ReadInfo], PrepareConfig],
) -> tuple[int, list[dict[str, np.ndarray | str | int | None]]]:
    """``_process_read_chunk_worker`` tagged with its batch sequence number.

    ``imap_unordered`` returns results out of order, so the caller needs the tag
    to attribute a result back to the batch it came from. Module-level (not a
    closure or lambda) because it has to be picklable for ``mp.Pool``.
    """
    seq, read_infos, config = args
    return seq, _process_read_chunk_worker((read_infos, config))


# ---------------------------------------------------------------------------
# Rust-accelerated batch preparation
# ---------------------------------------------------------------------------


def _extraction_sequence(read_info: ReadInfo, config: PrepareConfig) -> str:
    """The sequence Rust will cut chunks from, and index focus bases into.

    A thin adapter around :func:`leech.chunking.extraction_sequence`, which
    inference goes through too -- the rule itself lives in one place.
    """
    return extraction_sequence(
        anchor=config.signal.anchor,
        basecall=read_info.sequence,
        reference_sequence=read_info.reference_sequence,
        cigar_tuples=read_info.cigar_tuples,
    )


def _find_motif_positions(
    read_info: ReadInfo,
    motif_searcher: MotifSearcher | None,
    config: PrepareConfig,
) -> list[int]:
    """Find focus base indices for a single read.

    A thin adapter around :func:`leech.chunking.find_focus_bases`, which is
    also what the Python backend goes through — the rule itself lives in one
    place. Do not reimplement it here.
    """
    alignment = None
    if config.motif.motif_reference == "fasta" and config.motif.reference_sequences is not None:
        alignment = read_info.to_mock_alignment()

    return find_focus_bases(
        read_id=read_info.read_id,
        sequence=_extraction_sequence(read_info, config),
        alignment=alignment,
        motif_config=config.motif,
        motif_searcher=motif_searcher,
    )


def _prepare_batch_rust(
    read_infos: list[ReadInfo],
    config: PrepareConfig,
    motif_searcher: MotifSearcher | None,
) -> list[dict[str, np.ndarray | str | int | None]]:
    """
    Process ONE batch of reads using the Rust pipeline.

    Collects BAM metadata, finds motif positions in Python, then delegates
    signal processing + chunk extraction to Rust. Attaches Python-side
    labels/metadata to the returned chunks.

    This handles a single batch and does not parallelize across batches. Rust
    releases the GIL for the whole call, so callers get real parallelism from
    plain threads — call it from ``_iter_rust_batches``, which does exactly
    that, rather than in a loop of your own.

    Inside the call, per-read work is rayon-parallel, but the POD5 I/O that
    dominates it is a single sequential stream of reads. Do not read
    "rayon-parallel" as "this call already saturates the machine".
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

    # Resolve kmer table for signal refinement.
    #
    # Read every setting off the refiner object, which is what the Python
    # backend actually runs — not off SignalConfig, whose refine_* fields are
    # provenance for the sidecar and can only agree with the refiner by
    # convention. The attribute is `center_idx`; `kmer_center_idx` does not
    # exist on SigMapRefiner, so the old getattr default silently pinned the
    # Rust path to -1 (escapepod's "use kmer_len / 2") no matter what was
    # configured. inference/helpers.py had this right.
    kmer_table_dict: dict[str, float] | None = None
    kmer_len = 9
    kmer_center_idx = -1
    half_bandwidth = config.signal.refine_half_bandwidth
    scale_iters = config.signal.refine_scale_iters
    if config.signal.refine_signal_map and config.signal.signal_refiner is not None:
        refiner = config.signal.signal_refiner
        kmer_table_dict = refiner.kmer_to_level
        kmer_len = refiner.kmer_len
        kmer_center_idx = refiner.center_idx
        half_bandwidth = refiner.half_bandwidth
        scale_iters = refiner.scale_iters

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
        refine_half_bandwidth=half_bandwidth,
        refine_scale_iters=scale_iters,
        signal_in_channels=2 if (config.signal.refine_signal_map and kmer_table_dict) else 1,
        base_justify=config.chunk.base_justify,
    )

    # Attach Python-side labels/metadata.
    # The window Rust actually used, resolved by the same rule Python's
    # `get_chunk` applies -- `feature_start=0` is a real window, not a missing
    # one, and `or -kmer_context` silently rewrote it (issue #189). The value
    # lands in the chunk file and is what `dataset.py` slices the k-mer window
    # out of the feature array with, so getting it wrong misaligns training
    # input without changing any shape.
    feat_start, feat_end, _ = resolve_feature_window(
        config.chunk.feature_start, config.chunk.feature_end, config.chunk.kmer_context
    )

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
        chunk_dict["feature_start"] = feat_start
        chunk_dict["feature_end"] = feat_end

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
# Batch dispatch
#
# Both backends expose the same shape: an iterator of ``(n_reads, chunks)``,
# one item per BAM batch, with several batches in flight at once. Neither is a
# serial loop, and neither should be turned back into one — see the module
# docstring for why the Rust path in particular looks like it could be.
# ---------------------------------------------------------------------------


def _iter_rust_batches(
    bam_path: Path,
    config: PrepareConfig,
    motif_searcher: MotifSearcher | None,
    chunk_size: int,
    min_mapq: int,
    num_workers: int,
) -> Iterator[tuple[int, list[dict]]]:
    """Yield ``(n_reads, chunks)`` per batch, ``num_workers`` batches in flight.

    Threads, not processes: ``_prepare_batch_rust`` releases the GIL for the
    entire Rust call — POD5 I/O included — so ``num_workers`` OS threads give
    ``num_workers`` genuinely concurrent readers, the same concurrency the
    multiprocessing fallback gets from its pool, without paying to pickle
    chunks back across a process boundary.

    Concurrency is the point, not a bonus. Most of the wall clock is spent in
    uninterruptible sleep waiting on page faults against an mmapped POD5 on a
    network filesystem, and the only lever on fault latency is having more of
    them outstanding at once.

    Batches are drained in submission order, so chunks come out in BAM order.
    At most ``2 * num_workers`` batches are queued, bounding resident signal to
    roughly ``2 * num_workers * chunk_size`` reads' worth.
    """
    batches = iter_read_info_batches(bam_path, batch_size=chunk_size, min_mapq=min_mapq)
    max_in_flight = max(1, num_workers)
    window = 2 * max_in_flight

    with ThreadPoolExecutor(max_workers=max_in_flight, thread_name_prefix="leech-rust") as pool:
        pending: deque[tuple[int, Future]] = deque()
        exhausted = False

        while True:
            while not exhausted and len(pending) < window:
                try:
                    read_batch = next(batches)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(
                    (
                        len(read_batch),
                        pool.submit(_prepare_batch_rust, read_batch, config, motif_searcher),
                    )
                )

            if not pending:
                return

            n_reads, future = pending.popleft()
            try:
                yield n_reads, future.result()
            except Exception as e:
                logger.warning(f"Rust batch failed, skipping: {e}")
                yield n_reads, []


def _iter_python_batches(
    bam_path: Path,
    config: PrepareConfig,
    chunk_size: int,
    min_mapq: int,
    num_workers: int,
) -> Iterator[tuple[int, list[dict]]]:
    """Yield ``(n_reads, chunks)`` per batch from a pool of worker processes.

    Each worker keeps its own cached POD5 dataset reader (see
    ``leech.io.pod5_reader``), so the pool is also ``num_workers`` concurrent
    readers. Results arrive as they complete, not in BAM order.
    """
    batch_sizes: dict[int, int] = {}

    def _worker_arg_stream():
        for seq, read_batch in enumerate(
            iter_read_info_batches(bam_path, batch_size=chunk_size, min_mapq=min_mapq)
        ):
            batch_sizes[seq] = len(read_batch)
            yield (seq, read_batch, config)

    with mp.Pool(processes=num_workers) as pool:
        for seq, chunk_results in pool.imap_unordered(
            _process_read_chunk_worker_seq, _worker_arg_stream()
        ):
            yield batch_sizes.pop(seq, 0), chunk_results


def _count_reads_with_chunks(batch_chunks: list[dict]) -> int:
    """How many distinct reads in one batch produced at least one chunk.

    Distinct read ids rather than ``len(batch_chunks)``: all-bases mode emits
    many chunks per read, and the figure this feeds is a read yield. Counted
    per batch, so a read split across two batches by a supplementary alignment
    counts once per batch — close enough for a yield, and the alternative is
    holding every read id of the run in memory.
    """
    return len({c["read_id"] for c in batch_chunks})


class _ThroughputMonitor:
    """Reports achieved reads/s so a backend regression is visible immediately.

    Issue #176 cost a 12-hour cluster allocation to a backend that was ~10-80x
    slower than the one it replaced, because nothing in the log reported a rate
    to compare against.
    """

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.start = time.monotonic()

    def elapsed(self) -> float:
        return max(time.monotonic() - self.start, 1e-9)

    def reads_per_second(self, total_reads: int) -> float:
        return total_reads / self.elapsed()

    def log_progress(self, batches: int, total_reads: int, total_chunks: int) -> None:
        logger.info(
            f"Progress [{self.backend}]: {batches} batches, {total_reads} reads | "
            f"{total_chunks} chunks extracted | "
            f"{self.reads_per_second(total_reads):.0f} reads/s"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def prepare_training_data_parallel(
    bam_path: Path,
    config: PrepareConfig,
    num_workers: int = 8,
    chunk_size: int = 100,
    min_mapq: int = 0,
    chunk_sink: Callable[[list[dict]], None] | None = None,
) -> tuple[list[dict[str, np.ndarray | str | int | None]], dict[str, int]]:
    """
    Prepare training data from BAM and POD5 files using multiprocessing.

    Streams BAM reads in mega-batches so processing overlaps with BAM
    iteration rather than waiting for the entire BAM to be read first.

    ``num_workers`` sets the number of batches in flight on either backend:
    threads for Rust, worker processes for the Python fallback. It is not
    advisory on either path.

    Logs achieved reads/s as it goes, so a backend that is slower than the one
    it replaced shows up in the first progress line rather than at the end of
    the allocation.

    Args:
        bam_path: Path to BAM file with alignments
        config: Preparation configuration
        num_workers: Number of parallel workers
        chunk_size: Number of reads to process per worker batch
        min_mapq: Minimum mapping quality
        chunk_sink: Optional callback handed each batch's chunks as it
            completes. When given, chunks are NOT accumulated and the returned
            list is empty — the sink owns them. This is how ``data prepare``
            writes a corpus without ever holding it (#211); see
            :class:`~leech.chunking.ChunkSpool`. The statistics are the same
            either way.

    Returns:
        Tuple of (chunks, statistics). ``chunks`` is empty when ``chunk_sink``
        is given.
    """
    reason = rust_prepare_unsupported_reason(config)
    use_rust = HAS_RUST and _rs_extract_training_chunks is not None and reason is None
    if reason is not None and HAS_RUST:
        logger.warning(f"Using Python workers instead of the Rust pipeline: {reason}")
    backend = "Rust (rayon)" if use_rust else "Python (multiprocessing)"
    # Name the dispatch, not just the backend. `leech_core` is a separate
    # package from `leech`, so a freshly built extension can sit alongside a
    # stale `leech` — new Rust, old serial driver — and the only symptom is
    # being slow. A build that does not print "batches in flight" is that
    # pairing (#176).
    dispatch = "threads" if use_rust else "processes"
    logger.info(
        f"Starting parallel data preparation with {num_workers} workers "
        f"[{backend}, {num_workers} batches in flight via {dispatch}]"
    )

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
    total_chunks = 0
    reads_with_chunks = 0
    batches_completed = 0

    use_progress_bar = sys.stdout.isatty()
    monitor = _ThroughputMonitor(backend)

    if use_rust:
        # Rust path. Batches are dispatched CONCURRENTLY (see _iter_rust_batches);
        # the Rust call releases the GIL for POD5 I/O and for per-read work.
        logger.info("Streaming BAM reads with Rust-accelerated chunk extraction...")

        # Setup motif searcher (Python-side, needed for position finding)
        if config.motif.motif is not None:
            motif_searcher = get_motif_searcher(
                mode=config.motif.motif_reference,
                reference_sequences=config.motif.reference_sequences,
                skip_indels=config.motif.skip_motif_indels,
                require_query_mapping=config.motif.require_query_mapping,
                anchor=config.signal.anchor,
            )
        else:
            motif_searcher = None

        results = _iter_rust_batches(
            bam_path,
            config,
            motif_searcher,
            chunk_size=chunk_size,
            min_mapq=min_mapq,
            num_workers=num_workers,
        )
    else:
        # Python multiprocessing fallback: one worker process per batch, each
        # holding its own cached POD5 dataset reader.
        logger.info("Streaming BAM reads and processing in parallel...")
        results = _iter_python_batches(
            bam_path,
            config,
            chunk_size=chunk_size,
            min_mapq=min_mapq,
            num_workers=num_workers,
        )

    if use_progress_bar:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.fields[chunks_extracted]} chunks extracted"),
            TextColumn("[magenta]{task.fields[rate]}"),
        ) as progress:
            task = progress.add_task(
                f"Processing reads [{backend}]",
                total=estimated_batches,
                chunks_extracted=0,
                rate="",
            )

            for n_reads, batch_chunks in results:
                total_reads += n_reads
                batches_completed += 1
                reads_with_chunks += _count_reads_with_chunks(batch_chunks)
                total_chunks += len(batch_chunks)
                if chunk_sink is None:
                    all_chunks.extend(batch_chunks)
                else:
                    chunk_sink(batch_chunks)
                del batch_chunks
                progress.update(
                    task,
                    completed=batches_completed,
                    chunks_extracted=total_chunks,
                    rate=f"{monitor.reads_per_second(total_reads):.0f} reads/s",
                )

            # Fix total if estimate was off
            progress.update(task, total=batches_completed, completed=batches_completed)
    else:
        log_interval = max(1, (estimated_batches or 50) // 20)
        for n_reads, batch_chunks in results:
            total_reads += n_reads
            batches_completed += 1
            reads_with_chunks += _count_reads_with_chunks(batch_chunks)
            total_chunks += len(batch_chunks)
            if chunk_sink is None:
                all_chunks.extend(batch_chunks)
            else:
                chunk_sink(batch_chunks)
            del batch_chunks
            if batches_completed % log_interval == 0:
                monitor.log_progress(batches_completed, total_reads, total_chunks)

    if total_reads == 0:
        return [], {
            "total_reads": 0,
            "reads_with_motif": 0,
            "reads_without_motif": 0,
            "total_chunks": 0,
        }

    # `reads_with_motif` counts READS that yielded at least one chunk, the same
    # thing it counts in the sequential orchestrator. It used to be
    # `len(all_chunks)`, which is chunks, and which therefore reported a yield
    # of 100% no matter how many reads the backend had dropped.
    stats = {
        "total_reads": total_reads,
        "reads_with_motif": reads_with_chunks,
        "reads_without_motif": total_reads - reads_with_chunks,
        "total_chunks": total_chunks,
    }

    logger.info(
        f"Parallel processing complete [{backend}]: extracted {total_chunks} chunks "
        f"from {total_reads} reads in {monitor.elapsed():.1f}s "
        f"({monitor.reads_per_second(total_reads):.0f} reads/s)"
    )
    # Read yield, always, on both backends. The backends must agree on this
    # number; when they did not, the Rust one silently dropped ~1% of reads and
    # it took a performance comparison to notice (issue #185). A figure in the
    # log is what makes the next such divergence a one-line diff.
    logger.info(
        f"Read yield [{backend}]: {reads_with_chunks}/{total_reads} reads produced chunks "
        f"({100.0 * reads_with_chunks / total_reads:.2f}%); "
        f"{total_reads - reads_with_chunks} produced none"
    )

    return all_chunks, stats
