"""
POD5 file reading utilities with batched access support.

Provides efficient reading of nanopore signal data from POD5 files,
with support for batched reading to improve I/O performance.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from pod5 import DatasetReader

logger = logging.getLogger("leech.io.pod5_reader")


def _extract_pod5_metadata(read) -> dict:
    """
    Extract standard metadata dict from a pod5 read object.

    Args:
        read: A pod5 read object (from DatasetReader.reads())

    Returns:
        Dictionary with read_id, channel, well, pore_type,
        calibration_offset, calibration_scale, and sample_rate.
    """
    return {
        "read_id": str(read.read_id),
        "channel": read.pore.channel,
        "well": read.pore.well,
        "pore_type": read.pore.pore_type,
        "calibration_offset": read.calibration.offset,
        "calibration_scale": read.calibration.scale,
        "sample_rate": read.run_info.sample_rate,
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
    with DatasetReader(pod5_path) as reader:
        for read in reader.reads([read_id]):
            signal = read.signal
            metadata = _extract_pod5_metadata(read)
            return signal, metadata

    raise ValueError(f"Read {read_id} not found in {pod5_path}")


def read_pod5_signals_batch(
    pod5_path: Path, read_ids: list[str]
) -> dict[str, tuple[np.ndarray, dict]]:
    """
    Read multiple signals from POD5 file in a single batch.

    This is more efficient than reading one-by-one for large batches,
    as it opens the POD5 file once and reads all requested signals.

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
    results = {}

    with DatasetReader(pod5_path) as reader:
        for read in reader.reads(read_ids):
            read_id = str(read.read_id)
            signal = read.signal
            metadata = _extract_pod5_metadata(read)
            results[read_id] = (signal, metadata)

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

    def __init__(self, pod5_path: Path, batch_size: int = 100):
        """
        Initialize POD5 reader.

        Args:
            pod5_path: Path to POD5 file
            batch_size: Number of reads to fetch in each batch (for batch mode)
        """
        self.pod5_path = pod5_path
        self.batch_size = batch_size
        self._reader = None
        self._cache: dict[str, tuple[np.ndarray, dict]] = {}

    def __enter__(self):
        """Open POD5 file."""
        self._reader = DatasetReader(self.pod5_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close POD5 file."""
        # DatasetReader handles cleanup via its own context manager
        self._reader = None
        self._cache.clear()

    def preload(self, read_ids: list[str]) -> None:
        """
        Pre-load signals for a batch of reads into the internal cache.

        When Rust acceleration is available, uses escapepod-rs for ~25-43x
        faster signal decompression. Falls back to Python pod5 library.

        Args:
            read_ids: List of read identifiers to preload
        """
        if self._reader is None:
            raise RuntimeError("POD5Reader must be used as a context manager")

        self._cache.clear()

        from leech._rust_accel import HAS_RUST, _rs_read_pod5_batch

        if HAS_RUST and _rs_read_pod5_batch is not None:
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
            for read in self._reader.reads(read_ids):
                rid = str(read.read_id)
                self._cache[rid] = (read.signal, _extract_pod5_metadata(read))

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

        for read in self._reader.reads([read_id]):
            signal = read.signal
            metadata = _extract_pod5_metadata(read)
            return signal, metadata

        raise ValueError(f"Read {read_id} not found in {self.pod5_path}")

    def get_signals_batch(self, read_ids: list[str]) -> dict[str, tuple[np.ndarray, dict]]:
        """
        Get signals for multiple reads in a batch.

        Args:
            read_ids: List of read identifiers

        Returns:
            Dictionary mapping read_id to (signal, metadata) tuples

        Raises:
            RuntimeError: If reader not opened (use as context manager)
        """
        if self._reader is None:
            raise RuntimeError("POD5Reader must be used as a context manager")

        results = {}

        for read in self._reader.reads(read_ids):
            read_id = str(read.read_id)
            signal = read.signal
            metadata = _extract_pod5_metadata(read)
            results[read_id] = (signal, metadata)

        return results

    def iter_all_reads(self) -> Iterator[tuple[str, np.ndarray, dict]]:
        """
        Iterate over all reads in POD5 file.

        Yields:
            Tuples of (read_id, signal, metadata)

        Raises:
            RuntimeError: If reader not opened (use as context manager)

        Examples:
            >>> with POD5Reader(Path("reads.pod5")) as reader:
            ...     for read_id, signal, meta in reader.iter_all_reads():
            ...         print(f"{read_id}: {len(signal)} samples")
        """
        if self._reader is None:
            raise RuntimeError("POD5Reader must be used as a context manager")

        for read in self._reader.reads():
            read_id = str(read.read_id)
            signal = read.signal
            metadata = _extract_pod5_metadata(read)
            yield read_id, signal, metadata
