"""
POD5 file reading utilities with batched access support.

Provides efficient reading of nanopore signal data from POD5 files,
with support for batched reading to improve I/O performance.
Uses escapepod-rs Python bindings for fast Rust-based POD5 access.
"""

import logging
from pathlib import Path

import numpy as np
from escapepod import Reader

logger = logging.getLogger("leech.io.pod5_reader")


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
    Read raw signal from POD5 file for a specific read.

    Args:
        pod5_path: Path to POD5 file
        read_id: Read identifier

    Returns:
        Tuple of (signal_array, metadata_dict)

    Raises:
        ValueError: If read_id not found in POD5 file

    Examples:
        >>> signal, meta = read_pod5_signal(Path("reads.pod5"), "read_001")
        >>> print(f"Signal length: {len(signal)}")
        >>> print(f"Sample rate: {meta['sample_rate']}")
    """
    reader = Reader(str(pod5_path))
    run_infos = reader.run_infos()
    read_data = reader.get_read(read_id)
    signal = reader.get_signal(read_data)
    metadata = _extract_pod5_metadata(read_data, run_infos)
    return signal, metadata


def read_pod5_signals_batch(
    pod5_path: Path, read_ids: list[str]
) -> dict[str, tuple[np.ndarray, dict]]:
    """
    Read multiple signals from POD5 file in a single batch.

    This is more efficient than reading one-by-one for large batches,
    as it uses parallel VBZ decompression via rayon.

    Args:
        pod5_path: Path to POD5 file
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
    reader = Reader(str(pod5_path))
    run_infos = reader.run_infos()
    reads = reader.get_reads(read_ids)
    signals_list = reader.get_signals(reads)
    sig_by_id = dict(signals_list)

    results = {}
    for read_data in reads:
        rid = read_data.read_id
        signal = sig_by_id.get(rid)
        if signal is not None:
            results[rid] = (signal, _extract_pod5_metadata(read_data, run_infos))

    # Log if any reads were not found
    missing = set(read_ids) - set(results.keys())
    if missing:
        logger.warning(f"Could not find {len(missing)} reads in POD5: {list(missing)[:5]}...")

    return results


class POD5Reader:
    """
    Context manager for efficient POD5 reading.

    Provides a high-level interface for reading signals from POD5 files,
    with support for batched access and caching.

    Examples:
        >>> with POD5Reader(Path("reads.pod5")) as reader:
        ...     for read_id in ["read_001", "read_002"]:
        ...         signal, meta = reader.get_signal(read_id)
        ...         print(f"{read_id}: {len(signal)} samples")
    """

    def __init__(self, pod5_path: Path, batch_size: int = 100, backend: str = "auto"):
        """
        Initialize POD5 reader.

        Args:
            pod5_path: Path to POD5 file
            batch_size: Number of reads to fetch in each batch (for batch mode)
            backend: "auto" (Rust if available), "rust" (force), or "python" (force)
        """
        self.pod5_path = pod5_path
        self.batch_size = batch_size
        self.backend = backend
        self._reader = None
        self._run_infos = None
        self._cache: dict[str, tuple[np.ndarray, dict]] = {}

    def __enter__(self):
        """Open POD5 file."""
        self._reader = Reader(str(self.pod5_path))
        self._run_infos = self._reader.run_infos()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Close POD5 file."""
        self._reader = None
        self._run_infos = None
        self._cache.clear()

    def preload(self, read_ids: list[str]) -> None:
        """
        Pre-load signals for a batch of reads into the internal cache.

        When Rust acceleration (leech_core) is available, uses its
        read_pod5_batch for slightly lower overhead. Otherwise uses
        escapepod.Reader.get_signals() for parallel VBZ decompression.

        Args:
            read_ids: List of read identifiers to preload
        """
        if self._reader is None:
            raise RuntimeError("POD5Reader must be used as a context manager")

        self._cache.clear()

        from leech._rust_accel import HAS_RUST, _rs_read_pod5_batch

        _use_rust = HAS_RUST and _rs_read_pod5_batch is not None and self.backend != "python"
        if _use_rust:
            batch = _rs_read_pod5_batch(str(self.pod5_path), read_ids)
            for rid, (signal, cal_offset, cal_scale) in batch.items():
                self._cache[rid] = (
                    signal,
                    {
                        "calibration_offset": cal_offset,
                        "calibration_scale": cal_scale,
                    },
                )
        else:
            reads = self._reader.get_reads(read_ids)
            signals_list = self._reader.get_signals(reads)
            sig_by_id = dict(signals_list)
            for read_data in reads:
                rid = read_data.read_id
                signal = sig_by_id.get(rid)
                if signal is not None:
                    self._cache[rid] = (
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
            ValueError: If read not found
            RuntimeError: If reader not opened (use as context manager)
        """
        if self._reader is None:
            raise RuntimeError("POD5Reader must be used as a context manager")

        # Check cache first (from preload)
        cached = self._cache.get(read_id)
        if cached is not None:
            return cached

        read_data = self._reader.get_read(read_id)
        signal = self._reader.get_signal(read_data)
        metadata = _extract_pod5_metadata(read_data, self._run_infos)
        return signal, metadata
