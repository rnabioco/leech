"""
POD5 file reading utilities with batched access support.

Provides efficient reading of nanopore signal data from POD5 files,
with support for batched reading to improve I/O performance.
Uses escapepod-rs Python bindings for fast Rust-based POD5 access.

POD5 sources may be either a single ``.pod5`` file or a directory
containing one or more ``.pod5`` files (the "sequencing run" layout
emitted by MinKNOW). Both are handled natively by
:class:`escapepod.DatasetReader`, which scans the directory, opens each
underlying file once, and routes every read to its owning file via a
lazily-built id index — so leech no longer maintains its own per-file
dispatch or handle cache.
"""

import logging
from pathlib import Path

import numpy as np
from escapepod import DatasetReader

logger = logging.getLogger("leech.io.pod5_reader")


# Process-local cache of opened datasets, keyed by the source path string.
# A ``DatasetReader`` opens every ``.pod5`` under the source once and holds
# the handles (mmap-backed) until process exit; caching by path lets repeated
# batch calls over the same source skip the re-open + directory scan. The
# ``run_infos`` list is cached alongside the reader because the property
# rebuilds a fresh list on each access.
_DATASET_CACHE: dict[str, tuple[DatasetReader, list]] = {}


def _get_cached_entry(pod5_path: Path | str) -> tuple[DatasetReader, list]:
    """Open (or reuse) a ``(DatasetReader, run_infos)`` pair for a source."""
    key = str(pod5_path)
    entry = _DATASET_CACHE.get(key)
    if entry is None:
        ds = DatasetReader(key)
        entry = (ds, ds.run_infos)
        _DATASET_CACHE[key] = entry
    return entry


def _get_cached_dataset(pod5_path: Path | str) -> DatasetReader:
    """Open (or reuse) a :class:`DatasetReader` for a POD5 source."""
    return _get_cached_entry(pod5_path)[0]


def _sample_rate_for(read, run_infos: list) -> int:
    """Sample rate for a read, guarding a run_info_index that falls outside
    the dataset's deduplicated run-info list (sample rate is constant within
    a run in practice, so the first entry is a safe fallback)."""
    idx = read.run_info_index
    if 0 <= idx < len(run_infos):
        return run_infos[idx].sample_rate
    return run_infos[0].sample_rate if run_infos else 0


def _extract_pod5_metadata(read, run_infos: list) -> dict:
    """
    Extract standard metadata dict from an escapepod ReadData object.

    Args:
        read: An escapepod ReadData object
        run_infos: List of RunInfo objects from the dataset/reader

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
        "sample_rate": _sample_rate_for(read, run_infos),
    }


def _batch_from_dataset(
    ds: DatasetReader, read_ids: list[str], warn_missing: bool = False
) -> dict[str, tuple[np.ndarray, dict]]:
    """Fetch raw signals + metadata for ``read_ids`` from an open dataset.

    ``DatasetReader.get_signals`` decodes each read from its owning file
    (parallel VBZ decompression, GIL released), so this is a single bulk
    call regardless of how many files back the source.
    """
    run_infos = ds.run_infos
    reads = ds.reads(list(read_ids), missing_ok=True)
    sig_by_id = dict(ds.get_signals(reads))
    results: dict[str, tuple[np.ndarray, dict]] = {}
    for read_data in reads:
        signal = sig_by_id.get(read_data.read_id)
        if signal is not None:
            results[read_data.read_id] = (signal, _extract_pod5_metadata(read_data, run_infos))

    if warn_missing:
        missing = len(set(read_ids)) - len(results)
        if missing > 0:
            found = set(results)
            sample = [rid for rid in read_ids if rid not in found][:5]
            logger.warning(f"Could not find {missing} reads in POD5 source: {sample}...")
    return results


def get_cached_reader(pod5_path: Path | str) -> tuple[DatasetReader, list]:
    """Return ``(dataset, run_infos)`` for a POD5 source, cached by path.

    Handles single files and directories alike. Repeated calls for the same
    path return the same :class:`DatasetReader` instance.
    """
    return _get_cached_entry(pod5_path)


def read_pod5_signals_batch_cached(
    pod5_path: Path | str, read_ids: list[str]
) -> dict[str, tuple[np.ndarray, dict]]:
    """Batch POD5 read using a cached :class:`DatasetReader`.

    Accepts either a single ``.pod5`` file or a directory containing many.
    Use this when the caller issues many batches against the same source
    (workers, multi-shard inference, etc.) so the directory scan and file
    opens happen only once.
    """
    return _batch_from_dataset(_get_cached_dataset(pod5_path), read_ids)


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
    ds, run_infos = _get_cached_entry(pod5_path)
    reads = ds.reads([read_id], missing_ok=True)
    if not reads:
        raise ValueError(f"read_id {read_id!r} not found under {pod5_path}")
    read_data = reads[0]
    signal = ds.get_signal(read_data)
    return signal, _extract_pod5_metadata(read_data, run_infos)


def read_pod5_signals_batch(
    pod5_path: Path, read_ids: list[str]
) -> dict[str, tuple[np.ndarray, dict]]:
    """
    Read multiple signals from a POD5 source in a single batch.

    Accepts a single ``.pod5`` file or a directory of ``.pod5`` files.
    More efficient than reading one-by-one for large batches — signals are
    decoded per owning file with parallel VBZ decompression via rayon.

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
    return _batch_from_dataset(_get_cached_dataset(pod5_path), read_ids, warn_missing=True)


class POD5Reader:
    """
    Context manager for efficient POD5 reading.

    Provides a high-level interface for reading signals from POD5 sources,
    with support for batched access and caching. Accepts either a single
    ``.pod5`` file or a directory containing many; both are handled by a
    single :class:`escapepod.DatasetReader`, which opens each underlying
    file once and dispatches per read.

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
            backend: Retained for backward compatibility; POD5 access now always
                goes through :class:`escapepod.DatasetReader`.
        """
        self.pod5_path = pod5_path
        self.batch_size = batch_size
        self.backend = backend
        self._ds: DatasetReader | None = None
        self._run_infos: list = []
        self._cache: dict[str, tuple[np.ndarray, dict]] = {}

    def __enter__(self):
        """Open the POD5 source (single file or directory of files)."""
        self._ds = DatasetReader(str(self.pod5_path))
        self._run_infos = self._ds.run_infos
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Release the dataset handle and cache."""
        self._ds = None
        self._run_infos = []
        self._cache.clear()

    def preload(self, read_ids: list[str]) -> None:
        """
        Pre-load signals for a batch of reads into the internal cache.

        Uses ``DatasetReader.get_signals()`` for parallel VBZ decompression
        across the owning files.

        Args:
            read_ids: List of read identifiers to preload
        """
        if self._ds is None:
            raise RuntimeError("POD5Reader must be used as a context manager")

        self._cache.clear()
        reads = self._ds.reads(list(read_ids), missing_ok=True)
        sig_by_id = dict(self._ds.get_signals(reads))
        for read_data in reads:
            signal = sig_by_id.get(read_data.read_id)
            if signal is not None:
                self._cache[read_data.read_id] = (
                    signal,
                    _extract_pod5_metadata(read_data, self._run_infos),
                )

        loaded = len(self._cache)
        missing = len(read_ids) - loaded
        if missing > 0:
            logger.debug(f"Preloaded {loaded}/{len(read_ids)} reads ({missing} not found)")

    def get_signal(self, read_id: str) -> tuple[np.ndarray, dict]:
        """
        Get signal for a single read. Uses cache if available.

        Args:
            read_id: Read identifier

        Returns:
            Tuple of (signal, metadata)

        Raises:
            ValueError: If read not found in any underlying file.
            RuntimeError: If reader not opened (use as context manager)
        """
        if self._ds is None:
            raise RuntimeError("POD5Reader must be used as a context manager")

        cached = self._cache.get(read_id)
        if cached is not None:
            return cached

        reads = self._ds.reads([read_id], missing_ok=True)
        if not reads:
            raise ValueError(f"read_id {read_id!r} not found under {self.pod5_path}")
        read_data = reads[0]
        signal = self._ds.get_signal(read_data)
        return signal, _extract_pod5_metadata(read_data, self._run_infos)
