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
from numpy.lib.stride_tricks import sliding_window_view

# Try to import Rust-accelerated implementations
try:
    from leech._rust_accel import HAS_RUST, _rs_encode_signal_kmer
except ImportError:
    HAS_RUST = False


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
            The last element is num_samples (end of basecalled signal region).

        Each move=1 at position i means a new base starts at signal index
        i * stride (+ trim_offset for untrimmed signal). This matches the
        Remora convention: ``query_to_signal = np.nonzero(mv)[0] * stride``.
        """
        move_positions = np.where(self.moves == 1)[0]

        # Each move=1 at position i marks the START of a new base
        # at signal index i * stride (offset by trim for untrimmed signal)
        seq_to_sig = move_positions * self.stride + self.trim_offset

        # Append end boundary (total signal samples = ns tag)
        seq_to_sig = np.concatenate([seq_to_sig, [self.num_samples]])

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

    Examples:
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

    boundaries = seq_to_sig_map[:-1]
    lengths = np.diff(seq_to_sig_map)
    sig_len = len(signal)
    valid = (lengths > 0) & (boundaries < sig_len)

    # Vectorized mean using reduceat (fast path)
    if stat == "mean" and np.any(valid):
        sums = np.add.reduceat(signal, boundaries[valid])
        levels[valid] = (sums / lengths[valid]).astype(np.float32)
        return levels

    # Loop for median/std/min/max (no vectorized numpy equivalent)
    stat_funcs: dict[str, Callable[[np.ndarray], Any]] = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std,
        "min": np.min,
        "max": np.max,
    }
    stat_func = stat_funcs[stat]

    for i in range(num_bases):
        if lengths[i] > 0:
            start_idx = seq_to_sig_map[i]
            end_idx = seq_to_sig_map[i + 1]
            base_signal = signal[start_idx:end_idx]
            if len(base_signal) > 0:
                levels[i] = float(stat_func(base_signal))

    return levels


def normalize_signal(
    raw_signal: np.ndarray,
    method: str = "median_mad",
    pa_mean: float | None = None,
    pa_stdev: float | None = None,
    cal_offset: float | None = None,
    cal_scale: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Normalize raw signal data.

    Args:
        raw_signal: Raw DAC signal values
        method: Normalization method ('median_mad', 'zscore', 'quantile', 'pa_scaling')
        pa_mean: Global shift for pa_scaling normalization (from basecaller model)
        pa_stdev: Global scale for pa_scaling normalization (from basecaller model)
        cal_offset: POD5 calibration offset (DAC -> pA conversion)
        cal_scale: POD5 calibration scale (DAC -> pA conversion)

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

    elif method == "pa_scaling":
        # Remora-style global normalization using basecaller model parameters.
        # DAC -> pA (via POD5 calibration) -> global normalization.
        if pa_mean is None or pa_stdev is None:
            raise ValueError("pa_scaling requires pa_mean and pa_stdev parameters")
        if cal_offset is None or cal_scale is None:
            raise ValueError("pa_scaling requires cal_offset and cal_scale from POD5 calibration")

        # Convert DAC to pA: pA = (raw + offset) * scale (pod5 convention)
        pa_signal = (raw_signal + cal_offset) * cal_scale
        # Apply global normalization
        normalized = (pa_signal - pa_mean) / pa_stdev
        params = {
            "pa_mean": pa_mean,
            "pa_stdev": pa_stdev,
            "cal_offset": cal_offset,
            "cal_scale": cal_scale,
        }

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized.astype(np.float32), params


def make_ref_to_query_mapping(cigar_tuples: list[tuple[int, int]]) -> np.ndarray:
    """
    Build reference-to-query coordinate mapping from CIGAR tuples.

    Uses remora's knot algorithm: places knots at both the start and end-1
    of each match block, ensuring 1:1 mapping within match operations and
    correct interpolation across indels.

    Args:
        cigar_tuples: List of (op, length) from pysam's cigartuples.
            Standard CIGAR ops: M=0, I=1, D=2, N=3, S=4, H=5, P=6, =7, X=8

    Returns:
        Float array of length ref_len+1 mapping reference coordinates to
        query coordinates. ref_to_query[i] gives the (float) query position
        corresponding to ref position i.
    """
    # CIGAR op classification (matching remora's constants)
    # Match ops: M(0), =(7), X(8)
    MATCH_OPS = np.array([True, False, False, False, False, False, False, True, True])
    # Ops consuming reference: M(0), D(2), N(3), =(7), X(8)
    REF_OPS = np.array([True, False, True, True, False, False, False, True, True])
    # Ops consuming query: M(0), I(1), S(4), =(7), X(8)
    QUERY_OPS = np.array([True, True, False, False, True, False, False, True, True])

    # Strip trailing non-match ops (remora convention)
    cigar = list(cigar_tuples)
    match_set = set(np.where(MATCH_OPS)[0])
    while len(cigar) > 0 and cigar[-1][0] not in match_set:
        cigar = cigar[:-1]
    if len(cigar) == 0:
        return np.array([0.0], dtype=np.float64)

    ops, lens = map(np.array, zip(*cigar, strict=True))
    is_match = MATCH_OPS[ops]
    match_counts = lens[is_match]
    # offsets shape (2, n_match): row 0 = match_count, row 1 = 1
    offsets = np.array([match_counts, np.ones_like(match_counts)])

    # Build ref knots: cumulative ref-consuming lengths, then place
    # start (end - count) and end-1 (end - 1) for each match block
    ref_cumsum = np.cumsum(np.where(REF_OPS[ops], lens, 0))
    ref_knots = np.concatenate(
        [[0], (ref_cumsum[is_match] - offsets).T.flatten(), [ref_cumsum[-1]]]
    )

    # Build query knots similarly
    query_cumsum = np.cumsum(np.where(QUERY_OPS[ops], lens, 0))
    query_knots = np.concatenate(
        [[0], (query_cumsum[is_match] - offsets).T.flatten(), [query_cumsum[-1]]]
    )

    # Interpolate to get mapping for every ref position
    ref_to_query = np.interp(np.arange(ref_knots[-1] + 1), ref_knots, query_knots)

    return ref_to_query


def compute_ref_to_signal(
    query_to_signal: np.ndarray,
    cigar_tuples: list[tuple[int, int]],
) -> np.ndarray:
    """
    Compute reference-to-signal mapping via ref->query->signal chain.

    This matches remora's compute_ref_to_signal + map_ref_to_signal:
    1. Build ref->query mapping from CIGAR (float, preserving 1:1 in matches)
    2. Interpolate float query positions through query->signal mapping

    Args:
        query_to_signal: Base-to-signal mapping for basecalled sequence,
            length = query_len + 1
        cigar_tuples: CIGAR tuples from pysam alignment

    Returns:
        Array mapping each reference position to signal position,
        length = ref_len + 1
    """
    ref_to_query = make_ref_to_query_mapping(cigar_tuples)

    # Map float query positions through query_to_signal (remora's map_ref_to_signal)
    ref_to_signal = np.floor(
        np.interp(
            ref_to_query,
            np.arange(query_to_signal.size),
            query_to_signal,
        )
    ).astype(np.int64)

    return ref_to_signal


_BASE_LOOKUP = np.full(256, -1, dtype=np.int8)
_BASE_LOOKUP[ord("A")] = 0
_BASE_LOOKUP[ord("C")] = 1
_BASE_LOOKUP[ord("G")] = 2
_BASE_LOOKUP[ord("T")] = 3
_BASE_LOOKUP[ord("U")] = 3
_BASE_LOOKUP[ord("a")] = 0
_BASE_LOOKUP[ord("c")] = 1
_BASE_LOOKUP[ord("g")] = 2
_BASE_LOOKUP[ord("t")] = 3
_BASE_LOOKUP[ord("u")] = 3


def sequence_to_int(sequence: str) -> np.ndarray:
    """
    Convert DNA/RNA sequence to integer encoding for signal_kmer encoding.

    A=0, C=1, G=2, T=3 (U=3), other=-1.

    Args:
        sequence: DNA/RNA sequence string

    Returns:
        Integer array with values 0-3 for valid bases, -1 for unknown
    """
    return _BASE_LOOKUP[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]


def encode_signal_kmer(
    sequence_ints: np.ndarray,
    seq_to_sig_map: np.ndarray,
    signal_len: int,
    kmer_context: tuple[int, int] = (4, 4),
) -> np.ndarray:
    """
    Signal-level kmer encoding: (4 * kmer_len, signal_len) float32.

    For each signal position, encodes the kmer context centered on the base
    occupying that position as a one-hot vector (4 channels per kmer position).

    This reimplements Remora's encoded_kmers.pyx in pure numpy.

    Args:
        sequence_ints: Int-encoded bases with context,
            len = seq_len + kmer_before + kmer_after.
            Values 0-3 for ACGT, <0 for unknown (skipped).
        seq_to_sig_map: Base-to-signal mapping for the core seq_len bases,
            len = seq_len + 1. Values are signal indices.
        signal_len: Length of the signal array.
        kmer_context: (kmer_before, kmer_after) context bases for kmer encoding.

    Returns:
        Encoding array of shape (4 * kmer_len, signal_len), dtype float32.
        kmer_len = kmer_before + 1 + kmer_after.
    """
    kmer_before, kmer_after = kmer_context

    if HAS_RUST:
        return np.asarray(
            _rs_encode_signal_kmer(
                sequence_ints.astype(np.int8),
                seq_to_sig_map.astype(np.int64),
                signal_len,
                kmer_before,
                kmer_after,
            )
        )

    kmer_len = kmer_before + 1 + kmer_after
    enc = np.zeros((4 * kmer_len, signal_len), dtype=np.float32)
    seq_len = len(seq_to_sig_map) - 1

    for kmer_pos in range(kmer_len):
        offset = 4 * kmer_pos
        for seq_pos in range(seq_len):
            base = sequence_ints[seq_pos + kmer_pos]
            if base < 0:
                continue
            sig_start = seq_to_sig_map[seq_pos]
            sig_end = seq_to_sig_map[seq_pos + 1]
            if sig_start < signal_len and sig_end > 0:
                enc[offset + base, max(0, sig_start) : min(signal_len, sig_end)] = 1.0

    return enc


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

    # Compute local statistics with padding using vectorized sliding window
    pad_width = window // 2
    padded = np.pad(dwells, pad_width, mode="edge")

    windows = sliding_window_view(padded, window)
    dwell_mean = windows.mean(axis=1).astype(np.float32)
    dwell_std = windows.std(axis=1).astype(np.float32)

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

    boundaries = seq_to_sig_map[:-1]
    lengths = np.diff(seq_to_sig_map)
    sig_len = len(signal)
    valid = (lengths > 0) & (boundaries < sig_len)

    # Vectorized mean using reduceat
    if np.any(valid):
        sums = np.add.reduceat(signal, boundaries[valid])
        features["level_mean"][valid] = (sums / lengths[valid]).astype(np.float32)

    # Loop only for median/std/range (no vectorized numpy equivalent)
    for i in range(num_bases):
        if lengths[i] > 0:
            start = seq_to_sig_map[i]
            end = seq_to_sig_map[i + 1]
            base_sig = signal[start:end]
            if len(base_sig) > 0:
                features["level_median"][i] = np.median(base_sig)
                features["level_std"][i] = np.std(base_sig)
                features["level_range"][i] = np.max(base_sig) - np.min(base_sig)

    return features
