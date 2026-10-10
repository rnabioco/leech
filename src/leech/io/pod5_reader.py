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
        # Entering the dataset is what warms escapepod's read-id index; without
        # it every ``reads(selection=...)`` re-scans the whole reads table, so a
        # per-read lookup costs O(reads-in-file) instead of O(1). The dataset is
        # cached for the life of the process, so there is no matching __exit__ —
        # the handles are released when the process ends. ``__enter__`` is the
        # only entry point for this: DatasetReader has no public build_index().
        ds.__enter__()
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


# Per-source, per-file ``Reader`` handles, built only for a source spanning
# more than one run (see :func:`_run_ids_for_reads`).
_FILE_READERS: dict[str, list] = {}


def _run_identity(run_info) -> tuple[str, str]:
    """``(run_id, flow_cell_id)`` of a RunInfo: acquisition id, flow cell id."""
    return str(run_info.acquisition_id or ""), str(run_info.flow_cell_id or "")


def _run_ids_for_reads(
    ds: DatasetReader, run_infos: list, reads: list
) -> dict[str, tuple[str, str]]:
    """``read_id -> (run_id, flow_cell_id)`` for ``reads`` of an open dataset.

    ``ReadData.run_info_index`` indexes the run-info table of the *file* that
    holds the read, while ``ds.run_infos`` is deduplicated across files, so the
    two only line up when the source has a single run. In that (usual) case
    the answer is constant. A source spanning several runs is resolved through
    each file's own ``Reader`` instead -- indexing the deduplicated list would
    silently label reads with the wrong run, which is the very leak run-held-out
    splitting exists to prevent. A read whose run cannot be resolved gets
    ``("", "")`` rather than a guess.
    """
    if not run_infos:
        return {}
    if len(run_infos) == 1:
        ident = _run_identity(run_infos[0])
        return {r.read_id: ident for r in reads}

    from escapepod import Reader

    key = "\0".join(ds.paths)
    readers = _FILE_READERS.get(key)
    if readers is None:
        readers = [Reader(path).__enter__() for path in ds.paths]  # warms each id index
        _FILE_READERS[key] = readers
    ids = [r.read_id for r in reads]
    out: dict[str, tuple[str, str]] = {}
    for reader in readers:
        infos = reader.run_infos
        for rd in reader.reads(ids, missing_ok=True):
            idx = rd.run_info_index
            if 0 <= idx < len(infos):
                out[rd.read_id] = _run_identity(infos[idx])
    return out


def _extract_pod5_metadata(
    read, run_infos: list, run_identity: tuple[str, str] | None = None
) -> dict:
    """
    Extract standard metadata dict from an escapepod ReadData object.

    Args:
        read: An escapepod ReadData object
        run_infos: List of RunInfo objects from the dataset/reader
        run_identity: ``(run_id, flow_cell_id)`` already resolved for this read
            by its owning file. When omitted it is taken from ``run_infos``
            only if that list holds a single run; otherwise both are ``""``.

    Returns:
        Dictionary with read_id, channel, well, pore_type,
        calibration_offset, calibration_scale, sample_rate, run_id and
        flow_cell_id (empty strings when the POD5 carries no run info).
    """
    if run_identity is None:
        run_identity = _run_identity(run_infos[0]) if len(run_infos) == 1 else ("", "")
    return {
        "read_id": read.read_id,
        "channel": read.channel,
        "well": read.well,
        "pore_type": read.pore_type,
        "calibration_offset": read.calibration_offset,
        "calibration_scale": read.calibration_scale,
        "sample_rate": _sample_rate_for(read, run_infos),
        "run_id": run_identity[0],
        "flow_cell_id": run_identity[1],
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
    runs = _run_ids_for_reads(ds, run_infos, reads)
    results: dict[str, tuple[np.ndarray, dict]] = {}
    for read_data in reads:
        signal = sig_by_id.get(read_data.read_id)
        if signal is not None:
            results[read_data.read_id] = (
                signal,
                _extract_pod5_metadata(read_data, run_infos, runs.get(read_data.read_id, ("", ""))),
            )

    if warn_missing:
        missing = len(set(read_ids)) - len(results)
        if missing > 0:
            found = set(results)
            sample = [rid for rid in read_ids if rid not in found][:5]
            logger.warning(f"Could not find {missing} reads in POD5 source: {sample}...")
    return results


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
    runs = _run_ids_for_reads(ds, run_infos, reads)
    return signal, _extract_pod5_metadata(
        read_data, run_infos, runs.get(read_data.read_id, ("", ""))
    )


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

    def __init__(self, pod5_path: Path, batch_size: int = 100):
        """
        Initialize POD5 reader.

        Args:
            pod5_path: Path to a ``.pod5`` file or a directory of ``.pod5`` files.
            batch_size: Number of reads to fetch in each batch (for batch mode)
        """
        self.pod5_path = pod5_path
        self.batch_size = batch_size
        self._ds: DatasetReader | None = None
        self._run_infos: list = []
        self._cache: dict[str, tuple[np.ndarray, dict]] = {}

    def __enter__(self):
        """Open the POD5 source (single file or directory of files)."""
        ds = DatasetReader(str(self.pod5_path))
        # Entering the dataset warms escapepod's read-id index, which is what
        # makes the uncached ``get_signal`` path an indexed lookup rather than a
        # full scan of the reads table.
        ds.__enter__()
        self._ds = ds
        self._run_infos = ds.run_infos
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the dataset handle and cache."""
        ds, self._ds = self._ds, None
        self._run_infos = []
        self._cache.clear()
        if ds is not None:
            ds.__exit__(exc_type, exc_val, exc_tb)

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
        runs = _run_ids_for_reads(self._ds, self._run_infos, reads)
        for read_data in reads:
            signal = sig_by_id.get(read_data.read_id)
            if signal is not None:
                self._cache[read_data.read_id] = (
                    signal,
                    _extract_pod5_metadata(
                        read_data, self._run_infos, runs.get(read_data.read_id, ("", ""))
                    ),
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
        runs = _run_ids_for_reads(self._ds, self._run_infos, reads)
        return signal, _extract_pod5_metadata(
            read_data, self._run_infos, runs.get(read_data.read_id, ("", ""))
        )
