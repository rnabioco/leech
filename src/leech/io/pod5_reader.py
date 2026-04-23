"""
POD5 file reading utilities with batched access support.

Provides efficient reading of nanopore signal data from POD5 files,
with support for batched reading to improve I/O performance.
Uses escapepod-rs Python bindings for fast Rust-based POD5 access.

POD5 sources may be either a single ``.pod5`` file or a directory
containing one or more ``.pod5`` files (the "sequencing run" layout
emitted by MinKNOW). Directory sources are expanded lazily and
dispatched per file internally — each underlying ``escapepod.Reader``
is cached separately so a fresh batch call over the same directory
doesn't re-open every file.
"""

import logging
from pathlib import Path

import numpy as np
from escapepod import Reader

logger = logging.getLogger("leech.io.pod5_reader")


# Process-local cache of opened POD5 readers. Keyed by the single-file str
# path so directory-backed sources reuse per-file handles across batches.
# Workers keep files open until process exit; the OS reclaims handles then.
_READER_CACHE: dict[str, tuple[Reader, list]] = {}


def _resolve_pod5_paths(pod5_path: Path | str) -> list[Path]:
    """Expand a POD5 source to a sorted list of concrete ``.pod5`` files.

    - Regular file: returns ``[path]``.
    - Directory: returns all ``*.pod5`` files inside, sorted by name so
      the iteration order is reproducible across runs.

    Raises ``FileNotFoundError`` if the directory has no ``.pod5`` files.
    """
    p = Path(pod5_path)
    if p.is_dir():
        files = sorted(p.glob("*.pod5"))
        if not files:
            raise FileNotFoundError(f"No .pod5 files in directory: {p}")
        return files
    return [p]


def _get_cached_reader_single(pod5_file: str) -> tuple[Reader, list]:
    """Open (or reuse) a Reader for one concrete POD5 file."""
    cached = _READER_CACHE.get(pod5_file)
    if cached is not None:
        return cached
    reader = Reader(pod5_file)
    run_infos = reader.run_infos
    _READER_CACHE[pod5_file] = (reader, run_infos)
    return reader, run_infos


def get_cached_reader(pod5_path: Path | str) -> tuple[Reader, list]:
    """Return ``(reader, run_infos)`` for a single POD5 file.

    Legacy single-file API preserved for existing callers. For multi-file
    sources (a directory of POD5s), use
    :func:`read_pod5_signals_batch_cached` or :class:`POD5Reader`, both of
    which transparently dispatch across files.
    """
    paths = _resolve_pod5_paths(pod5_path)
    if len(paths) > 1:
        raise ValueError(
            f"get_cached_reader expects a single POD5 file, but {pod5_path} "
            f"resolved to {len(paths)} files. Use read_pod5_signals_batch_cached "
            "or POD5Reader for directory sources."
        )
    return _get_cached_reader_single(str(paths[0]))


def read_pod5_signals_batch_cached(
    pod5_path: Path | str, read_ids: list[str]
) -> dict[str, tuple[np.ndarray, dict]]:
    """Batch POD5 read using cached Reader handles.

    Accepts either a single ``.pod5`` file or a directory containing many.
    For directory sources, iterates files and dispatches each read to the
    first file that contains it. Readers are cached per file across calls,
    so a later batch over the same directory is free of file-open cost.

    Same result shape as :func:`read_pod5_signals_batch`; use this when
    the caller issues many batches against the same source (workers,
    multi-shard inference, etc.).
    """
    paths = _resolve_pod5_paths(pod5_path)
    remaining = set(read_ids)
    results: dict[str, tuple[np.ndarray, dict]] = {}
    for p in paths:
        if not remaining:
            break
        reader, run_infos = _get_cached_reader_single(str(p))
        # `get_reads` silently drops ids not present in this file, so
        # iterating all files is O(files × batch_size) lookups but fetches
        # each read from exactly one reader.
        reads = reader.get_reads(list(remaining))
        if not reads:
            continue
        signals_list = reader.get_signals(reads)
        sig_by_id = dict(signals_list)
        for read_data in reads:
            rid = read_data.read_id
            signal = sig_by_id.get(rid)
            if signal is not None:
                results[rid] = (signal, _extract_pod5_metadata(read_data, run_infos))
                remaining.discard(rid)
    return results


def _extract_pod5_metadata(read, run_infos: list) -> dict:
    """
    Extract standard metadata dict from an escapepod ReadData object.

    Args:
        read: An escapepod ReadData object
        run_infos: List of RunInfo objects from reader.run_infos()

    Returns:
        Dictionary with read_id, channel, well, pore_type,
        calibration_offset, calibration_scale, and sample_rate.
    """
    return {
        "read_id": read.read_id,
        "channel": read.channel,
        "well": read.well,
        "pore_type": read.pore_type,
        "calibration_offset": read.calibration_offset,
        "calibration_scale": read.calibration_scale,
        "sample_rate": run_infos[read.run_info_index].sample_rate,
    }


def read_pod5_signal(pod5_path: Path, read_id: str) -> tuple[np.ndarray, dict]:
    """
    Read raw signal from a POD5 source for a specific read.

    Args:
        pod5_path: Path to a ``.pod5`` file or a directory of ``.pod5`` files.
        read_id: Read identifier

    Returns:
        Tuple of (signal_array, metadata_dict)

    Raises:
        ValueError: If read_id not found in any file under ``pod5_path``.

    Examples:
        >>> signal, meta = read_pod5_signal(Path("reads.pod5"), "read_001")
        >>> print(f"Signal length: {len(signal)}")
        >>> print(f"Sample rate: {meta['sample_rate']}")
    """
    for p in _resolve_pod5_paths(pod5_path):
        reader = Reader(str(p))
        run_infos = reader.run_infos
        try:
            read_data = reader.get_read(read_id)
        except Exception:
            # Reader raises when the id isn't in this file — fall through.
            continue
        signal = reader.get_signal(read_data)
        return signal, _extract_pod5_metadata(read_data, run_infos)
    raise ValueError(f"read_id {read_id!r} not found under {pod5_path}")


def read_pod5_signals_batch(
    pod5_path: Path, read_ids: list[str]
) -> dict[str, tuple[np.ndarray, dict]]:
    """
    Read multiple signals from a POD5 source in a single batch.

    Accepts a single ``.pod5`` file or a directory of ``.pod5`` files;
    iterates files until every requested read is found (or none of the
    remaining sources contain it). More efficient than reading one-by-one
    for large batches — each file's get_signals uses parallel VBZ
    decompression via rayon.

    Args:
        pod5_path: Path to a ``.pod5`` file or a directory of ``.pod5`` files.
        read_ids: List of read identifiers

    Returns:
        Dictionary mapping read_id to (signal, metadata) tuples.
        Missing reads are not included in the output.

    Examples:
        >>> read_ids = ["read_001", "read_002", "read_003"]
        >>> signals = read_pod5_signals_batch(Path("reads.pod5"), read_ids)
        >>> for read_id, (signal, meta) in signals.items():
        ...     print(f"{read_id}: {len(signal)} samples")
    """
    remaining = set(read_ids)
    results: dict[str, tuple[np.ndarray, dict]] = {}
    for p in _resolve_pod5_paths(pod5_path):
        if not remaining:
            break
        reader = Reader(str(p))
        run_infos = reader.run_infos
        reads = reader.get_reads(list(remaining))
        if not reads:
            continue
        signals_list = reader.get_signals(reads)
        sig_by_id = dict(signals_list)
        for read_data in reads:
            rid = read_data.read_id
            signal = sig_by_id.get(rid)
            if signal is not None:
                results[rid] = (signal, _extract_pod5_metadata(read_data, run_infos))
                remaining.discard(rid)

    if remaining:
        logger.warning(
            f"Could not find {len(remaining)} reads in POD5 source: {list(remaining)[:5]}..."
        )

    return results


class POD5Reader:
    """
    Context manager for efficient POD5 reading.

    Provides a high-level interface for reading signals from POD5 sources,
    with support for batched access and caching. Accepts either a single
    ``.pod5`` file or a directory containing many; directory sources open
    one underlying ``escapepod.Reader`` per file and dispatch per read.

    Examples:
        >>> with POD5Reader(Path("reads.pod5")) as reader:
        ...     for read_id in ["read_001", "read_002"]:
        ...         signal, meta = reader.get_signal(read_id)
        ...         print(f"{read_id}: {len(signal)} samples")

        >>> # Directory of POD5s (one MinKNOW run)
        >>> with POD5Reader(Path("run_42/pod5/")) as reader:
        ...     reader.preload(all_read_ids)  # populates across all files
        ...     signal, meta = reader.get_signal("read_001")
    """

    def __init__(self, pod5_path: Path, batch_size: int = 100, backend: str = "auto"):
        """
        Initialize POD5 reader.

        Args:
            pod5_path: Path to a ``.pod5`` file or a directory of ``.pod5`` files.
            batch_size: Number of reads to fetch in each batch (for batch mode)
            backend: "auto" (Rust if available), "rust" (force), or "python" (force)
        """
        self.pod5_path = pod5_path
        self.batch_size = batch_size
        self.backend = backend
        # _readers is a list of (reader, run_infos, path_str) — one entry
        # per underlying .pod5 file. Single-file sources produce a one-
        # element list so the rest of the class doesn't branch.
        self._readers: list[tuple[Reader, list, str]] = []
        self._cache: dict[str, tuple[np.ndarray, dict]] = {}

    def __enter__(self):
        """Open the POD5 source (single file or directory of files)."""
        self._readers = []
        for p in _resolve_pod5_paths(self.pod5_path):
            reader = Reader(str(p))
            self._readers.append((reader, reader.run_infos, str(p)))
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Close POD5 handles."""
        self._readers = []
        self._cache.clear()

    def preload(self, read_ids: list[str]) -> None:
        """
        Pre-load signals for a batch of reads into the internal cache.

        When Rust acceleration (leech_core) is available, uses its
        read_pod5_batch for slightly lower overhead. Otherwise uses
        escapepod.Reader.get_signals() for parallel VBZ decompression.
        For directory sources, iterates each underlying file in turn and
        stops once every requested read has been located.

        Args:
            read_ids: List of read identifiers to preload
        """
        if not self._readers:
            raise RuntimeError("POD5Reader must be used as a context manager")

        self._cache.clear()
        remaining = set(read_ids)

        from leech._rust_accel import HAS_RUST, _rs_read_pod5_batch

        _use_rust = HAS_RUST and _rs_read_pod5_batch is not None and self.backend != "python"

        for reader, run_infos, path_str in self._readers:
            if not remaining:
                break
            if _use_rust:
                batch = _rs_read_pod5_batch(path_str, list(remaining))
                for rid, (signal, cal_offset, cal_scale) in batch.items():
                    self._cache[rid] = (
                        signal,
                        {
                            "calibration_offset": cal_offset,
                            "calibration_scale": cal_scale,
                        },
                    )
                    remaining.discard(rid)
            else:
                reads = reader.get_reads(list(remaining))
                if not reads:
                    continue
                signals_list = reader.get_signals(reads)
                sig_by_id = dict(signals_list)
                for read_data in reads:
                    rid = read_data.read_id
                    signal = sig_by_id.get(rid)
                    if signal is not None:
                        self._cache[rid] = (
                            signal,
                            _extract_pod5_metadata(read_data, run_infos),
                        )
                        remaining.discard(rid)

        loaded = len(self._cache)
        missing = len(read_ids) - loaded
        if missing > 0:
            logger.debug(f"Preloaded {loaded}/{len(read_ids)} reads ({missing} not found)")

    def get_signal(self, read_id: str) -> tuple[np.ndarray, dict]:
        """
        Get signal for a single read. Uses cache if available.

        For directory sources, iterates underlying readers until the read
        is found (O(files) in the worst case; typically O(1) since each
        read lives in exactly one file).

        Args:
            read_id: Read identifier

        Returns:
            Tuple of (signal, metadata)

        Raises:
            ValueError: If read not found in any underlying file.
            RuntimeError: If reader not opened (use as context manager)
        """
        if not self._readers:
            raise RuntimeError("POD5Reader must be used as a context manager")

        # Check cache first (from preload)
        cached = self._cache.get(read_id)
        if cached is not None:
            return cached

        for reader, run_infos, _ in self._readers:
            try:
                read_data = reader.get_read(read_id)
            except Exception:
                # Reader raises when the id isn't in this file.
                continue
            signal = reader.get_signal(read_data)
            return signal, _extract_pod5_metadata(read_data, run_infos)
        raise ValueError(f"read_id {read_id!r} not found under {self.pod5_path}")
