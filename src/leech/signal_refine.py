"""
Signal map refinement using expected kmer signal levels.

Pure numpy port of Remora's signal map refinement (refine_signal_map.py
and refine_signal_map_core.pyx). Uses banded Viterbi dynamic programming
to refine base boundaries from the initial move-table estimate.

The refinement uses expected signal levels for each kmer context (from
kmer level tables) to find the most likely base boundaries given the
observed signal.

Key functions:
    load_kmer_table(): Load kmer -> expected level mapping
    extract_levels(): Get expected levels for a sequence
    refine_signal_mapping(): Main refinement function
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("leech.signal_refine")

# Try to import Rust-accelerated implementations
try:
    from leech._rust_accel import (
        HAS_RUST,
        _rs_extract_levels,
        _rs_rough_rescale,
        _rs_seq_banded_dp,
    )
except ImportError:
    HAS_RUST = False

# CIGAR operations
_CIGAR_OPS_CONSUME_REF = {0, 2, 3, 7, 8}  # M, D, N, =, X
_CIGAR_OPS_CONSUME_QUERY = {0, 1, 4, 7, 8}  # M, I, S, =, X


def load_kmer_table(table_path: Path) -> tuple[dict[str, float], int]:
    """
    Load kmer level table from TSV file.

    Expected format: tab-separated with columns 'kmer' and 'level_mean'
    (or first two columns if no header).

    Args:
        table_path: Path to kmer level table (e.g., rna004_9mer_levels.txt)

    Returns:
        Tuple of (kmer_to_level dict, kmer_length)
    """
    kmer_to_level: dict[str, float] = {}
    kmer_len = 0

    with open(table_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue

            kmer = parts[0].upper()
            # Skip header rows
            try:
                level = float(parts[1])
            except ValueError:
                continue

            kmer_to_level[kmer] = level
            kmer_len = len(kmer)

    logger.info(f"Loaded {len(kmer_to_level)} kmer levels (k={kmer_len}) from {table_path}")
    return kmer_to_level, kmer_len


def extract_levels(
    sequence: str,
    kmer_to_level: dict[str, float],
    kmer_len: int,
) -> np.ndarray:
    """
    Extract expected signal levels for each position in a sequence.

    For each position i, looks up the kmer centered at i and returns the
    expected signal level. Positions near the edges use truncated kmers
    if available, otherwise use the nearest valid kmer's level.

    Args:
        sequence: DNA/RNA sequence
        kmer_to_level: Mapping from kmer string to expected level
        kmer_len: Length of kmers in the table

    Returns:
        Array of expected levels, length = len(sequence)
    """
    if HAS_RUST:
        return np.asarray(_rs_extract_levels(sequence, kmer_to_level, kmer_len))

    seq_len = len(sequence)
    levels = np.zeros(seq_len, dtype=np.float64)
    half_k = kmer_len // 2

    for i in range(seq_len):
        start = i - half_k
        end = i + half_k + 1

        if start < 0 or end > seq_len:
            # Edge: use nearest valid kmer
            clamped_start = max(0, min(start, seq_len - kmer_len))
            clamped_end = clamped_start + kmer_len
            kmer = sequence[clamped_start:clamped_end].upper()
        else:
            kmer = sequence[start:end].upper()

        # RNA uses U instead of T
        kmer = kmer.replace("U", "T")

        level = kmer_to_level.get(kmer)
        if level is not None:
            levels[i] = level
        elif i > 0:
            levels[i] = levels[i - 1]  # Fallback to previous

    return levels


def rough_rescale(
    signal: np.ndarray,
    expected_levels: np.ndarray,
    seq_to_sig_map: np.ndarray,
) -> np.ndarray:
    """
    Rough rescaling of signal to match expected kmer levels.

    Computes per-base mean signal and fits a linear transform to match
    expected levels. This makes the Viterbi scoring more accurate.

    Args:
        signal: Normalized signal array
        expected_levels: Expected levels per base from kmer table
        seq_to_sig_map: Base-to-signal mapping

    Returns:
        Rescaled signal
    """
    if HAS_RUST:
        result = np.asarray(
            _rs_rough_rescale(
                signal.astype(np.float64),
                expected_levels.astype(np.float64),
                seq_to_sig_map.astype(np.int64),
            )
        )
        return result.astype(signal.dtype)

    num_bases = len(seq_to_sig_map) - 1
    observed = np.zeros(num_bases, dtype=np.float64)

    for i in range(num_bases):
        start = seq_to_sig_map[i]
        end = seq_to_sig_map[i + 1]
        if end > start:
            observed[i] = np.mean(signal[start:end])
        elif i > 0:
            observed[i] = observed[i - 1]

    # Fit linear transform: observed = a * expected + b
    if len(observed) > 1:
        A = np.vstack([expected_levels, np.ones(len(expected_levels))]).T
        result = np.linalg.lstsq(A, observed, rcond=None)
        a, b = result[0]
        # Apply inverse: rescaled = (signal - b) / a
        if abs(a) > 1e-10:
            return ((signal - b) / a).astype(signal.dtype)

    return signal


def compute_sig_band(
    seq_to_sig_map: np.ndarray,
    half_bandwidth: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute signal band around the initial mapping for banded DP.

    For each base position, defines a range of valid signal positions
    centered on the initial mapping estimate.

    Args:
        seq_to_sig_map: Initial base-to-signal mapping
        half_bandwidth: Half-width of the band in signal samples

    Returns:
        Tuple of (band_starts, band_ends) arrays, each of length num_bases+1
    """
    band_starts = np.maximum(0, seq_to_sig_map - half_bandwidth)
    band_ends = seq_to_sig_map + half_bandwidth
    return band_starts, band_ends


def seq_banded_dp(
    signal: np.ndarray,
    expected_levels: np.ndarray,
    band_starts: np.ndarray,
    band_ends: np.ndarray,
    sig_len: int,
) -> np.ndarray:
    """
    Banded Viterbi dynamic programming for signal map refinement.

    Finds the optimal base boundaries within the banded region that
    maximizes the likelihood of the observed signal given expected levels.

    The scoring function is -0.5 * (signal[t] - expected_level[b])^2
    for each signal position t assigned to base b.

    Args:
        signal: Signal array (rescaled to match expected levels)
        expected_levels: Expected signal level per base
        band_starts: Start of valid signal range per base
        band_ends: End of valid signal range per base
        sig_len: Total signal length

    Returns:
        Refined seq_to_sig_map of length num_bases+1
    """
    if HAS_RUST:
        return np.asarray(
            _rs_seq_banded_dp(
                signal.astype(np.float64),
                expected_levels.astype(np.float64),
                band_starts.astype(np.int64),
                band_ends.astype(np.int64),
                sig_len,
            )
        )

    num_bases = len(expected_levels)
    num_positions = num_bases + 1

    # Allocate DP tables
    # For each base boundary position (0..num_bases), store best score
    # at each signal position within the band
    max_band = int(np.max(band_ends - band_starts)) + 1

    scores = np.full((num_positions, max_band), -np.inf, dtype=np.float64)
    traceback = np.zeros((num_positions, max_band), dtype=np.int32)

    # Initialize: first boundary
    start = band_starts[0]
    end = min(band_ends[0], sig_len)
    for t in range(start, end):
        scores[0, t - start] = 0.0  # No cost for starting position

    # Forward pass
    for b in range(1, num_positions):
        b_start = band_starts[b]
        b_end = min(band_ends[b], sig_len)
        prev_start = band_starts[b - 1]
        prev_end = min(band_ends[b - 1], sig_len)
        level = expected_levels[b - 1]  # Level for base b-1

        for t in range(b_start, b_end):
            best_score = -np.inf
            best_prev = -1

            # Previous boundary must be before current
            for prev_t in range(prev_start, min(t, prev_end)):
                prev_idx = prev_t - prev_start
                if prev_idx < 0 or prev_idx >= max_band:
                    continue
                if scores[b - 1, prev_idx] == -np.inf:
                    continue

                # Score for assigning signal[prev_t:t] to base b-1
                seg = signal[prev_t:t]
                if len(seg) == 0:
                    continue
                seg_score = -0.5 * np.sum((seg - level) ** 2)
                total = scores[b - 1, prev_idx] + seg_score

                if total > best_score:
                    best_score = total
                    best_prev = prev_t

            idx = t - b_start
            if idx >= 0 and idx < max_band:
                scores[b, idx] = best_score
                traceback[b, idx] = best_prev

    # Traceback: find best final position
    last_start = band_starts[num_positions - 1]
    last_end = min(band_ends[num_positions - 1], sig_len)
    best_final_score = -np.inf
    best_final_t = sig_len  # Default to end

    for t in range(last_start, last_end):
        idx = t - last_start
        if idx >= 0 and idx < max_band and scores[num_positions - 1, idx] > best_final_score:
            best_final_score = scores[num_positions - 1, idx]
            best_final_t = t

    # Reconstruct path
    refined_map = np.zeros(num_positions, dtype=np.int64)
    refined_map[num_positions - 1] = best_final_t

    for b in range(num_positions - 1, 0, -1):
        t = refined_map[b]
        idx = t - band_starts[b]
        if idx >= 0 and idx < max_band:
            refined_map[b - 1] = traceback[b, idx]
        else:
            refined_map[b - 1] = band_starts[b - 1]

    return refined_map


@dataclass
class SigMapRefiner:
    """
    Signal map refiner using expected kmer signal levels.

    Attributes:
        kmer_to_level: Mapping from kmer string to expected signal level
        kmer_len: Length of kmers in the table
        half_bandwidth: Half-width of the DP band in signal samples
        do_rough_rescale: Whether to rough-rescale signal before refinement
    """

    kmer_to_level: dict[str, float]
    kmer_len: int
    half_bandwidth: int = 300
    do_rough_rescale: bool = True

    @classmethod
    def from_table(
        cls,
        table_path: Path,
        half_bandwidth: int = 300,
        do_rough_rescale: bool = True,
    ) -> "SigMapRefiner":
        """Create refiner from a kmer level table file."""
        kmer_to_level, kmer_len = load_kmer_table(table_path)
        return cls(
            kmer_to_level=kmer_to_level,
            kmer_len=kmer_len,
            half_bandwidth=half_bandwidth,
            do_rough_rescale=do_rough_rescale,
        )

    def refine(
        self,
        signal: np.ndarray,
        sequence: str,
        seq_to_sig_map: np.ndarray,
    ) -> np.ndarray:
        """
        Refine signal-to-sequence mapping using expected kmer levels.

        Args:
            signal: Normalized signal array
            sequence: Base sequence
            seq_to_sig_map: Initial base-to-signal mapping (from move table)

        Returns:
            Refined seq_to_sig_map
        """
        sig_len = len(signal)
        expected = extract_levels(sequence, self.kmer_to_level, self.kmer_len)

        work_signal = signal.astype(np.float64)
        if self.do_rough_rescale:
            work_signal = rough_rescale(work_signal, expected, seq_to_sig_map).astype(np.float64)

        band_starts, band_ends = compute_sig_band(seq_to_sig_map, self.half_bandwidth)

        refined = seq_banded_dp(work_signal, expected, band_starts, band_ends, sig_len)

        # Validate: boundaries must be monotonically increasing
        for i in range(1, len(refined)):
            if refined[i] <= refined[i - 1]:
                logger.debug(
                    f"Non-monotonic refinement at position {i}, falling back to original mapping"
                )
                return seq_to_sig_map

        return refined
