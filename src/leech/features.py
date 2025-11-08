"""
Feature extraction from nanopore signal data.

This module provides core functionality for extracting dwell time and signal level
features from ONT nanopore data. Features are computed from:
- BAM move tables (mv tag): Per-base dwell times
- Raw signal (POD5): Signal statistics per base

The key innovation is explicit dwell time modeling, which improves classification
of charged vs. uncharged tRNAs in aa-tRNA-seq experiments.

Core Classes:
    MoveTable: Parsed basecaller move table with sequence-to-signal mapping

Core Functions:
    extract_move_table(): Parse move table from BAM alignment
    compute_dwell_times(): Calculate per-base dwell times from move table
    compute_signal_levels(): Calculate signal statistics (mean, median, std, range)
    normalize_signal(): Normalize raw signal using median-MAD, z-score, or quantile

Feature Engineering:
    Dwell features (5 total):
    - dwell: Raw dwell time (number of signal samples per base)
    - dwell_log: Log-transformed dwell time
    - dwell_mean: Mean dwell in sliding window
    - dwell_std: Std dev of dwell in sliding window
    - dwell_ratio: Ratio to mean dwell time

    Signal features (4 total):
    - level_mean: Mean signal level per base
    - level_median: Median signal level per base
    - level_std: Std dev of signal per base
    - level_range: Range (max - min) of signal per base

Example:
    >>> from leech.features import extract_move_table, compute_dwell_times
    >>> import pysam
    >>>
    >>> # Extract move table from BAM
    >>> bam = pysam.AlignmentFile("alignments.bam")
    >>> aln = next(bam)
    >>> move_table = extract_move_table(aln)
    >>>
    >>> # Compute dwell times
    >>> dwells = compute_dwell_times(move_table)
    >>> print(f"Mean dwell: {dwells.mean():.2f} samples/base")

Based on concepts from Remora (https://github.com/nanoporetech/remora) but
rewritten with explicit dwell time integration and modern dependencies.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pysam


@dataclass
class MoveTable:
    """
    Parsed move table from basecaller output.

    Attributes:
        stride: Neural network downsampling factor
        moves: Binary array where 1 indicates a new base
        read_id: Read identifier
        num_samples: Total number of raw signal samples (from ns tag)
        trim_offset: Signal trim offset (from ts tag)
    """

    stride: int
    moves: np.ndarray
    read_id: str
    num_samples: int
    trim_offset: int = 0

    @property
    def num_bases(self) -> int:
        """Number of basecalled bases (count of 1s in move array)."""
        return int(np.sum(self.moves))

    def to_seq_to_sig_map(self) -> np.ndarray:
        """
        Convert move table to sequence-to-signal mapping.

        Returns:
            Array of shape (num_bases + 1,) giving signal position for each base.
            The last element is the total signal length.
        """
        # Cumulative sum of moves gives us indices
        move_positions = np.where(self.moves == 1)[0]

        # Convert move indices to signal indices
        # Each move index i corresponds to signal position (i + 1) * stride
        seq_to_sig = (move_positions + 1) * self.stride + self.trim_offset

        # Prepend 0 for the start of the first base
        seq_to_sig = np.concatenate([[self.trim_offset], seq_to_sig])

        return seq_to_sig


def extract_move_table(alignment: pysam.AlignedSegment) -> MoveTable:
    """
    Extract move table from BAM alignment record.

    Args:
        alignment: pysam AlignedSegment with mv, ns, and ts tags

    Returns:
        MoveTable object

    Raises:
        ValueError: If required tags are missing
    """
    if not alignment.has_tag("mv"):
        raise ValueError(f"Read {alignment.query_name} missing 'mv' tag")
    if not alignment.has_tag("ns"):
        raise ValueError(f"Read {alignment.query_name} missing 'ns' tag")

    # Parse move table: first element is stride, rest is the move array
    mv_tag: Any = alignment.get_tag("mv")
    stride = int(mv_tag[0])
    moves = np.array(mv_tag[1:], dtype=np.int8)

    # Get signal metadata
    num_samples = int(alignment.get_tag("ns"))
    trim_offset = int(alignment.get_tag("ts")) if alignment.has_tag("ts") else 0

    read_id = alignment.query_name
    if read_id is None:
        raise ValueError("Alignment has no query_name")

    return MoveTable(
        stride=stride,
        moves=moves,
        read_id=read_id,
        num_samples=num_samples,
        trim_offset=trim_offset,
    )


def compute_dwell_times(move_table: MoveTable) -> np.ndarray:
    """
    Compute per-base dwell times from move table.

    Dwell time = number of signal samples assigned to each base.

    Args:
        move_table: MoveTable object from extract_move_table()

    Returns:
        Array of shape (num_bases,) with dwell time for each base in signal samples

    Example:
        >>> # Move array: [1,1,0,1,0,0,0,1,...] with stride=5
        >>> # Base 0: 1 move = 1 * 5 = 5 samples
        >>> # Base 1: 2 moves (1,1) = 2 * 5 = 10 samples
        >>> # Base 2: 4 moves (0,1,0,0,0) = 4 * 5 = 20 samples
    """
    seq_to_sig = move_table.to_seq_to_sig_map()
    dwells = np.diff(seq_to_sig)
    return dwells


def compute_signal_levels(
    signal: np.ndarray, seq_to_sig_map: np.ndarray, stat: str = "mean"
) -> np.ndarray:
    """
    Compute per-base signal level statistics.

    Args:
        signal: Normalized signal array
        seq_to_sig_map: Mapping from bases to signal indices (from MoveTable.to_seq_to_sig_map())
        stat: Statistic to compute ('mean', 'median', 'std', 'min', 'max')

    Returns:
        Array of shape (num_bases,) with signal level statistic for each base
    """
    num_bases = len(seq_to_sig_map) - 1
    levels = np.zeros(num_bases, dtype=np.float32)

    stat_funcs: dict[str, Callable[[np.ndarray], Any]] = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std,
        "min": np.min,
        "max": np.max,
    }
    stat_func = stat_funcs[stat]

    for i in range(num_bases):
        start_idx = seq_to_sig_map[i]
        end_idx = seq_to_sig_map[i + 1]
        base_signal = signal[start_idx:end_idx]
        levels[i] = float(stat_func(base_signal)) if len(base_signal) > 0 else 0.0

    return levels


def normalize_signal(
    raw_signal: np.ndarray, method: str = "median_mad"
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Normalize raw signal data.

    Args:
        raw_signal: Raw DAC signal values
        method: Normalization method ('median_mad', 'zscore', 'quantile')

    Returns:
        Tuple of (normalized_signal, normalization_params)
    """
    if method == "median_mad":
        # Median Absolute Deviation normalization (robust to outliers)
        median = np.median(raw_signal)
        mad = np.median(np.abs(raw_signal - median))
        # Scale factor for consistency with standard deviation
        scale_factor = 1.4826
        normalized = (raw_signal - median) / (mad * scale_factor)
        params = {"median": float(median), "mad": float(mad), "scale_factor": scale_factor}

    elif method == "zscore":
        # Standard z-score normalization
        mean = np.mean(raw_signal)
        std = np.std(raw_signal)
        normalized = (raw_signal - mean) / std
        params = {"mean": float(mean), "std": float(std)}

    elif method == "quantile":
        # Quantile normalization (winsorize extreme values)
        q01 = np.quantile(raw_signal, 0.01)
        q99 = np.quantile(raw_signal, 0.99)
        clipped = np.clip(raw_signal, q01, q99)
        median = np.median(clipped)
        mad = np.median(np.abs(clipped - median))
        normalized = (raw_signal - median) / (mad * 1.4826)
        params = {"median": float(median), "mad": float(mad), "q01": float(q01), "q99": float(q99)}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized.astype(np.float32), params


def compute_dwell_features(dwells: np.ndarray, window: int = 5) -> dict[str, np.ndarray]:
    """
    Compute windowed dwell time features.

    Args:
        dwells: Per-base dwell times
        window: Size of sliding window for context features

    Returns:
        Dictionary with feature arrays:
            - 'dwell': raw dwell times
            - 'dwell_log': log-transformed dwell times
            - 'dwell_mean': local mean in window
            - 'dwell_std': local std in window
            - 'dwell_ratio': ratio to local mean
    """
    # Avoid log(0) by adding small epsilon
    eps = 1e-6
    dwell_log = np.log(dwells + eps)

    # Compute local statistics with padding
    pad_width = window // 2
    padded = np.pad(dwells, pad_width, mode="edge")

    dwell_mean = np.array(
        [np.mean(padded[i : i + window]) for i in range(len(dwells))], dtype=np.float32
    )

    dwell_std = np.array(
        [np.std(padded[i : i + window]) for i in range(len(dwells))], dtype=np.float32
    )

    # Ratio of dwell to local mean (normalized dwell)
    dwell_ratio = dwells / (dwell_mean + eps)

    return {
        "dwell": dwells.astype(np.float32),
        "dwell_log": dwell_log.astype(np.float32),
        "dwell_mean": dwell_mean,
        "dwell_std": dwell_std,
        "dwell_ratio": dwell_ratio,
    }


def compute_signal_features(
    signal: np.ndarray, seq_to_sig_map: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Compute comprehensive per-base signal features.

    Args:
        signal: Normalized signal array
        seq_to_sig_map: Base to signal mapping

    Returns:
        Dictionary with per-base features:
            - 'level_mean': mean signal level
            - 'level_median': median signal level
            - 'level_std': signal standard deviation
            - 'level_range': max - min signal
    """
    num_bases = len(seq_to_sig_map) - 1

    features = {
        "level_mean": np.zeros(num_bases, dtype=np.float32),
        "level_median": np.zeros(num_bases, dtype=np.float32),
        "level_std": np.zeros(num_bases, dtype=np.float32),
        "level_range": np.zeros(num_bases, dtype=np.float32),
    }

    for i in range(num_bases):
        start = seq_to_sig_map[i]
        end = seq_to_sig_map[i + 1]
        base_sig = signal[start:end]

        if len(base_sig) > 0:
            features["level_mean"][i] = np.mean(base_sig)
            features["level_median"][i] = np.median(base_sig)
            features["level_std"][i] = np.std(base_sig)
            features["level_range"][i] = np.max(base_sig) - np.min(base_sig)

    return features
