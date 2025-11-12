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
            metadata = {
                "read_id": str(read.read_id),
                "channel": read.pore.channel,
                "well": read.pore.well,
                "pore_type": read.pore.pore_type,
                "calibration_offset": read.calibration.offset,
                "calibration_scale": read.calibration.scale,
                "sample_rate": read.run_info.sample_rate,
            }
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
            metadata = {
                "read_id": read_id,
                "channel": read.pore.channel,
                "well": read.pore.well,
                "pore_type": read.pore.pore_type,
                "calibration_offset": read.calibration.offset,
                "calibration_scale": read.calibration.scale,
                "sample_rate": read.run_info.sample_rate,
            }
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

    def __enter__(self):
        """Open POD5 file."""
        self._reader = DatasetReader(self.pod5_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close POD5 file."""
        if self._reader is not None:
            self._reader.close()

    def get_signal(self, read_id: str) -> tuple[np.ndarray, dict]:
        """
        Get signal for a single read.

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

        for read in self._reader.reads([read_id]):
            signal = read.signal
            metadata = {
                "read_id": str(read.read_id),
                "channel": read.pore.channel,
                "well": read.pore.well,
                "pore_type": read.pore.pore_type,
                "calibration_offset": read.calibration.offset,
                "calibration_scale": read.calibration.scale,
                "sample_rate": read.run_info.sample_rate,
            }
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
            metadata = {
                "read_id": read_id,
                "channel": read.pore.channel,
                "well": read.pore.well,
                "pore_type": read.pore.pore_type,
                "calibration_offset": read.calibration.offset,
                "calibration_scale": read.calibration.scale,
                "sample_rate": read.run_info.sample_rate,
            }
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
            metadata = {
                "read_id": read_id,
                "channel": read.pore.channel,
                "well": read.pore.well,
                "pore_type": read.pore.pore_type,
                "calibration_offset": read.calibration.offset,
                "calibration_scale": read.calibration.scale,
                "sample_rate": read.run_info.sample_rate,
            }
            yield read_id, signal, metadata
