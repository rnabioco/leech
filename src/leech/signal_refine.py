"""
Signal map refinement using expected kmer signal levels.

Port of Remora's signal map refinement (refine_signal_map.py and
refine_signal_map_core.pyx). Uses banded Viterbi dynamic programming
to refine base boundaries from the initial move-table estimate.

The refinement uses expected signal levels for each kmer context (from
kmer level tables) to find the most likely base boundaries given the
observed signal.

Key functions:
    load_kmer_table(): Load kmer -> expected level mapping
    extract_levels(): Get expected levels for a sequence
    refine_signal_mapping(): Refine base boundaries via banded DP
    SigMapRefiner: High-level class for signal refinement
"""

import gzip
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("leech.signal_refine")

# Try to import Rust-accelerated implementations
try:
    from leech._rust_accel import (
        HAS_RUST,
        _rs_rough_rescale_quantile,
        _rs_seq_banded_dp,
    )
except ImportError:
    HAS_RUST = False
    _rs_rough_rescale_quantile = None
    _rs_seq_banded_dp = None


# ============================================================================
# Constants (matching Remora defaults)
# ============================================================================

LARGE_SCORE = 100.0  # Large penalty for invalid positions

ALGO_VITERBI = "Viterbi"
ALGO_DWELL_PENALTY = "dwell_penalty"
DEFAULT_ALGO = ALGO_DWELL_PENALTY

DEFAULT_HALF_BANDWIDTH = 5
DEFAULT_SHORT_DWELL_PARAMS = (4, 3, 0.5)  # (target, limit, weight)
MAX_POINTS_FOR_THEIL_SEN = 1000

# Fixed seed for escapepod's Theil-Sen subsample so refinement is reproducible
# on long reads. Must match leech_core's REFINE_SUBSAMPLE_SEED (Rust path).
REFINE_SUBSAMPLE_SEED = 42


# ============================================================================
# Kmer level table loading
# ============================================================================


def load_kmer_table(table_path: Path) -> tuple[dict[str, float], int]:
    """
    Load kmer level table from TSV file, with pickle caching for fast reload.

    On first load, parses the TSV/gzip source and writes a `.pkl` cache file
    alongside it. Subsequent loads use the pickle (~10x faster than gzip TSV).

    Expected format: tab-separated with columns 'kmer' and 'level_mean'
    (or first two columns if no header).

    Args:
        table_path: Path to kmer level table (e.g., rna004_9mer_levels.txt.gz)

    Returns:
        Tuple of (kmer_to_level dict, kmer_length)
    """
    import pickle

    # Check for pickle cache (same path with .pkl extension appended)
    cache_path = Path(str(table_path) + ".pkl")
    if cache_path.exists() and cache_path.stat().st_mtime >= table_path.stat().st_mtime:
        with open(cache_path, "rb") as f:
            kmer_to_level, kmer_len = pickle.load(f)
        logger.info(
            f"Loaded {len(kmer_to_level)} kmer levels (k={kmer_len}) from cache {cache_path}"
        )
        return kmer_to_level, kmer_len

    kmer_to_level: dict[str, float] = {}
    kmer_len = 0

    opener = gzip.open if str(table_path).endswith(".gz") else open
    with opener(table_path, "rt") as f:
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
                logger.debug("Skipping unparseable kmer level line: %s", line)
                continue

            kmer_to_level[kmer] = level
            kmer_len = len(kmer)

    logger.info(f"Loaded {len(kmer_to_level)} kmer levels (k={kmer_len}) from {table_path}")

    # Write pickle cache for next time
    try:
        with open(cache_path, "wb") as f:
            pickle.dump((kmer_to_level, kmer_len), f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError:
        pass  # read-only filesystem, skip caching

    return kmer_to_level, kmer_len


# ============================================================================
# Level extraction
# ============================================================================


def extract_levels(
    sequence: str,
    kmer_to_level: dict[str, float],
    kmer_len: int,
    center_idx: int | None = None,
) -> np.ndarray:
    """
    Extract expected signal levels for each position in a sequence.

    Matches remora's extract_levels: for each valid kmer window, looks up the
    expected level and assigns it to the center position. Edge positions where
    a full kmer cannot be formed are left at 0.

    Args:
        sequence: DNA/RNA sequence
        kmer_to_level: Mapping from kmer string to expected level
        kmer_len: Length of kmers in the table
        center_idx: Position within the kmer that corresponds to the
            "center" base (from model metadata). Defaults to kmer_len // 2.

    Returns:
        Array of expected levels, length = len(sequence).
        Positions where a valid kmer cannot be formed are 0.
    """
    if center_idx is None:
        center_idx = kmer_len // 2

    seq_len = len(sequence)
    levels = np.zeros(seq_len, dtype=np.float32)

    # Pre-process sequence once: uppercase and U->T replacement
    seq_upper = sequence.upper().replace("U", "T")

    for pos in range(seq_len - kmer_len + 1):
        kmer = seq_upper[pos : pos + kmer_len]
        level = kmer_to_level.get(kmer)
        if level is not None:
            levels[pos + center_idx] = level

    return levels


# ============================================================================
# Rough rescaling (initial, quantile-based)
# ============================================================================


def rough_rescale_quantile(
    signal: np.ndarray,
    expected_levels: np.ndarray,
    seq_to_sig_map: np.ndarray,
    clip_bases: int = 10,
) -> np.ndarray:
    """
    Rough rescaling using quantile-based fitting (remora convention).

    Uses center-of-base signal values and quantile fitting to match
    expected kmer levels. Replicates remora's rough_rescale_lstsq.

    Args:
        signal: Normalized signal array
        expected_levels: Expected levels per base from kmer table
        seq_to_sig_map: Base-to-signal mapping
        clip_bases: Number of bases to clip from each end

    Returns:
        Rescaled signal

    For float32 inputs this delegates to escapepod-signal's canonical
    implementation via leech_core (rnabioco/escapepod-rs#204), which is
    pinned bit-for-bit against this NumPy version by escapepod's golden
    tests. The NumPy body below remains the fallback for rust-less
    installs and non-float32 dtypes.
    """
    if (
        HAS_RUST
        and _rs_rough_rescale_quantile is not None
        and signal.dtype == np.float32
        and expected_levels.dtype == np.float32
    ):
        return np.asarray(
            _rs_rough_rescale_quantile(
                np.ascontiguousarray(signal),
                np.ascontiguousarray(expected_levels),
                np.ascontiguousarray(seq_to_sig_map, dtype=np.int64),
                clip_bases,
            )
        )

    centers = (seq_to_sig_map[:-1] + seq_to_sig_map[1:]) // 2
    center_signal = signal[centers].astype(np.float64)
    levels = expected_levels.copy()

    if clip_bases > 0 and len(center_signal) > clip_bases * 2:
        center_signal = center_signal[clip_bases:-clip_bases]
        levels = levels[clip_bases:-clip_bases]

    quants = np.arange(0.05, 1, 0.05)
    sig_qs = np.quantile(center_signal, quants)
    level_qs = np.quantile(levels, quants)

    coeffs = np.linalg.lstsq(
        np.column_stack([np.ones_like(sig_qs), sig_qs]),
        level_qs,
        rcond=None,
    )[0]
    shift_est, scale_est = coeffs

    if abs(scale_est) < 1e-10:
        return signal

    return (scale_est * signal + shift_est).astype(signal.dtype)


# ============================================================================
# Inter-iteration rescaling (Theil-Sen)
# ============================================================================


def _compute_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute all pairwise slopes for Theil-Sen regression."""
    delta_x = x[:, np.newaxis] - x
    delta_y = y[:, np.newaxis] - y
    mask = delta_x > 0
    return delta_y[mask] / delta_x[mask]


def _theil_sen_rescale(
    signal: np.ndarray,
    levels: np.ndarray,
    seq_to_sig_map: np.ndarray,
    dwell_filter_pctls: tuple[float, float] = (10, 90),
    min_abs_level: float = 0.2,
    edge_filter_bases: int = 10,
    min_levels: int = 10,
) -> np.ndarray:
    """
    Rescale signal using Theil-Sen regression on per-base means vs expected levels.

    Matches remora's SigMapRefiner.rescale() approach: filters extreme dwells
    and edge bases, then fits a robust linear transform.

    Args:
        signal: Current signal array
        levels: Expected levels per base
        seq_to_sig_map: Current base-to-signal mapping
        dwell_filter_pctls: Percentile bounds for dwell filtering
        min_abs_level: Minimum absolute level to include
        edge_filter_bases: Number of edge bases to exclude
        min_levels: Minimum valid bases required

    Returns:
        Rescaled signal, or original if rescaling fails
    """
    # Compute per-base mean signal
    sig_cumsum = np.empty(signal.size + 1, dtype=np.float64)
    sig_cumsum[0] = 0
    sig_cumsum[1:] = np.cumsum(signal.astype(np.float64))
    dwells = np.diff(seq_to_sig_map)
    with np.errstate(invalid="ignore"):
        sig_means = np.diff(sig_cumsum[seq_to_sig_map]) / dwells

    # Filter bases
    dwell_min, dwell_max = np.percentile(dwells, dwell_filter_pctls)
    edge_filter = np.full(dwells.size, True, dtype=np.bool_)
    if edge_filter_bases > 0:
        edge_filter[:edge_filter_bases] = False
        edge_filter[-edge_filter_bases:] = False
    valid = np.logical_and.reduce(
        (
            dwells > dwell_min,
            dwells < dwell_max,
            np.abs(levels - np.mean(levels)) > min_abs_level,
            np.logical_not(np.isnan(sig_means)),
            edge_filter,
        )
    )

    filt_means = sig_means[valid].astype(np.float64)
    filt_levels = levels[valid].astype(np.float64)
    if filt_levels.size < min_levels:
        logger.debug(f"Too few valid bases for rescaling ({filt_levels.size} < {min_levels})")
        return signal

    # Subsample for Theil-Sen if too many points
    if filt_levels.size > MAX_POINTS_FOR_THEIL_SEN:
        idx = np.random.choice(filt_levels.size, MAX_POINTS_FOR_THEIL_SEN, replace=False)
        filt_means = filt_means[idx]
        filt_levels = filt_levels[idx]

    # Theil-Sen: fit level = slope * signal_mean + intercept
    slopes = _compute_slopes(filt_means, filt_levels)
    if slopes.size == 0:
        return signal
    slope = np.median(slopes)
    intercept = np.median(filt_levels - slope * filt_means)

    if abs(slope) < 1e-10:
        logger.debug("Theil-Sen slope near zero, skipping rescale")
        return signal

    # Apply: new_signal = slope * signal + intercept
    return (slope * signal + intercept).astype(signal.dtype)


# ============================================================================
# Dwell penalty computation
# ============================================================================


def compute_dwell_pen_array(target: int = 4, limit: int = 3, weight: float = 0.5) -> np.ndarray:
    """
    Compute short dwell penalty array.

    penalty[d] = weight * (d - target)^2 for d in range(limit).
    Penalizes bases with fewer than `limit` signal samples.

    Args:
        target: Target dwell (signal samples per base)
        limit: Maximum dwell that receives a penalty
        weight: Penalty weight

    Returns:
        Float32 array of length limit
    """
    if limit > target:
        logger.warning(f"Short dwell limit ({limit}) > target ({target}). Setting limit to target.")
        limit = target
    return weight * np.square(np.arange(limit, dtype=np.float32) - target)


DEFAULT_SHORT_DWELL_PEN = compute_dwell_pen_array(*DEFAULT_SHORT_DWELL_PARAMS)


# ============================================================================
# Band computation (matching Remora)
# ============================================================================


def compute_sig_band(
    bps: np.ndarray,
    levels: np.ndarray,
    bhw: int = DEFAULT_HALF_BANDWIDTH,
) -> np.ndarray:
    """
    Compute band in sequence coordinates for each signal position.

    For each signal position, defines a range of valid sequence/base
    indices centered on the initial mapping estimate.

    Args:
        bps: Breakpoints (seq_to_sig_map), int array of length seq_len+1
        levels: Expected levels per base, float array of length seq_len.
            May contain NaN values; band will route through NaN regions.
        bhw: Band half width

    Returns:
        int32 array of shape (2, sig_len). Row 0 = lower bounds,
        row 1 = upper bounds, both in sequence coordinates.
    """
    seq_len = levels.size
    assert bps.size - 1 == seq_len, f"bps ({bps.size}) must be one longer than levels ({seq_len})"
    sig_len = int(bps[-1] - bps[0])
    seq_indices = np.repeat(np.arange(seq_len), np.diff(bps))

    band = np.empty((2, sig_len), dtype=np.int32)
    band[0, :] = np.maximum(seq_indices - bhw, 0)
    band[1, :] = np.minimum(seq_indices + bhw + 1, seq_len)

    # Handle NaN levels
    nan_mask = np.isin(seq_indices, np.nonzero(np.isnan(levels))[0])
    nan_sig_indices = np.where(nan_mask)[0]
    nan_seq_indices = seq_indices[nan_mask]
    band[0, nan_sig_indices] = nan_seq_indices
    band[1, nan_sig_indices] = nan_seq_indices + 1
    band[0, :] = np.maximum.accumulate(band[0, :])
    band[1, :] = np.minimum.accumulate(band[1, ::-1])[::-1]

    return band


def convert_to_seq_band(sig_band: np.ndarray) -> np.ndarray:
    """
    Convert sig_band (sequence coords at each signal pos) to seq_band
    (signal coords at each sequence pos).

    Args:
        sig_band: int32 array of shape (2, sig_len)

    Returns:
        int32 array of shape (2, seq_len) where seq_len = sig_band[1, -1]
    """
    sig_len = sig_band.shape[1]
    seq_len = int(sig_band[1, -1])
    seq_band = np.zeros((2, seq_len), dtype=np.int32)
    seq_band[1, :] = sig_len

    # Upper signal coords define lower sequence boundaries
    lower_sig_pos = np.nonzero(np.ediff1d(sig_band[1, :], to_begin=0))[0]
    lower_base_pos = sig_band[1, lower_sig_pos - 1]
    seq_band[0, lower_base_pos] = lower_sig_pos
    seq_band[0, :] = np.maximum.accumulate(seq_band[0, :])

    upper_sig_pos = np.nonzero(np.ediff1d(sig_band[0, :], to_begin=0))[0]
    upper_base_pos = sig_band[0, upper_sig_pos]
    seq_band[1, upper_base_pos - 1] = upper_sig_pos
    seq_band[1, :] = np.minimum.accumulate(seq_band[1, ::-1])[::-1]

    return seq_band


def adjust_seq_band(seq_band: np.ndarray, min_step: int = 2) -> None:
    """
    Adjust seq_band in-place to ensure each band boundary advances by at
    least min_step. Disallows invalid paths through the band.

    Args:
        seq_band: int32 array of shape (2, seq_len), modified in place
        min_step: Minimum step between consecutive band boundaries
    """
    n = seq_band.shape[1]

    # Fix starts: sweep right-to-left ensuring each start is >= next - min_step
    band_min = int(seq_band[0, 0])
    for i in range(n - 2, -1, -1):
        if seq_band[0, i] > seq_band[0, i + 1] - min_step:
            seq_band[0, i] = seq_band[0, i + 1] - min_step

    # Restore original start and fix forward
    seq_band[0, 0] = band_min
    i = 1
    while i < n and seq_band[0, i] <= seq_band[0, i - 1]:
        seq_band[0, i] = seq_band[0, i - 1] + 1
        i += 1

    # Fix ends: sweep left-to-right ensuring each end is >= prev + min_step
    band_max = int(seq_band[1, n - 1])
    for i in range(1, n):
        if seq_band[1, i] < seq_band[1, i - 1] + min_step:
            seq_band[1, i] = seq_band[1, i - 1] + min_step

    # Restore original end and fix backward
    seq_band[1, n - 1] = band_max
    i = n - 2
    while i >= 0 and seq_band[1, i] >= seq_band[1, i + 1]:
        seq_band[1, i] = seq_band[1, i + 1] - 1
        i -= 1


def validate_band(
    band: np.ndarray,
    sig_len: int | None = None,
    seq_len: int | None = None,
) -> None:
    """
    Validate a seq_band array.

    Args:
        band: int32 array of shape (2, seq_len)
        sig_len: Expected signal length (band[1, -1] should equal this)
        seq_len: Expected sequence length (band.shape[1] should equal this)

    Raises:
        ValueError if band is invalid
    """
    if band[0, 0] != 0:
        raise ValueError(f"Band does not start with 0 (starts at {band[0, 0]})")
    if np.diff(band, axis=0)[0].min() <= 0:
        raise ValueError("Band contains zero-length region")
    if np.diff(band[0]).min() < 0:
        raise ValueError("Band starts not monotonically increasing")
    if np.diff(band[1]).min() < 0:
        raise ValueError("Band ends not monotonically increasing")
    if sig_len is not None and band[1, -1] != sig_len:
        raise ValueError(f"Band end ({band[1, -1]}) != sig_len ({sig_len})")
    if seq_len is not None and band.shape[1] != seq_len:
        raise ValueError(f"Band width ({band.shape[1]}) != seq_len ({seq_len})")


# ============================================================================
# Core Viterbi functions (pure Python fallback)
# ============================================================================


def _score(s: float, level: float) -> float:
    """Squared error between signal sample and expected level."""
    tmp = s - level
    return tmp * tmp


def _banded_forward_vit_step(
    curr_scores: np.ndarray,
    curr_tb: np.ndarray,
    prev_scores: np.ndarray,
    curr_level: float,
    curr_signal: np.ndarray,
    band_start_diff: int,
) -> None:
    """
    Standard Viterbi forward step for one base (minimizes squared error).

    For each signal position in the band, decides whether to "move" (start
    a new base assignment from the previous base) or "stay" (continue the
    current base assignment). Traceback stores the number of stays since
    the last move.

    Args:
        curr_scores: Output scores array (populated in place)
        curr_tb: Output traceback array (populated in place)
        prev_scores: Scores from previous base
        curr_level: Expected signal level for this base
        curr_signal: Signal values within this base's band
        band_start_diff: Offset between current and previous band starts
    """
    n_curr = len(curr_scores)
    n_prev = len(prev_scores)

    # Handle band start
    if band_start_diff == 0:
        # Same start position — move here would be 0-length assignment
        curr_scores[0] = LARGE_SCORE + prev_scores[n_prev - 1]
        curr_tb[0] = -1
    else:
        base_score = _score(curr_level, curr_signal[0])
        curr_scores[0] = prev_scores[band_start_diff - 1] + base_score
        curr_tb[0] = 0
        # Clip prev_scores to align with curr
        prev_scores = prev_scores[band_start_diff:]
        n_prev = len(prev_scores)

    # If bands are the same size, trim prev by one to prevent overlap
    if n_prev == n_curr:
        prev_scores = prev_scores[: n_prev - 1]
        n_prev -= 1

    # Overlap region: both move and stay are possible
    for bp in range(1, n_prev + 1):
        base_score = _score(curr_level, curr_signal[bp])
        move_score = prev_scores[bp - 1] + base_score
        stay_score = curr_scores[bp - 1] + base_score
        if move_score < stay_score:
            curr_scores[bp] = move_score
            curr_tb[bp] = 0
        else:
            curr_scores[bp] = stay_score
            curr_tb[bp] = curr_tb[bp - 1] + 1

    # Past overlap: forced stays
    for bp in range(n_prev + 1, n_curr):
        base_score = _score(curr_level, curr_signal[bp])
        curr_scores[bp] = curr_scores[bp - 1] + base_score
        curr_tb[bp] = curr_tb[bp - 1] + 1


def _banded_forward_dwell_penalty_step(
    curr_scores: np.ndarray,
    curr_tb: np.ndarray,
    prev_scores: np.ndarray,
    curr_level: float,
    curr_signal: np.ndarray,
    band_start_diff: int,
    dwell_penalty: np.ndarray,
) -> None:
    """
    Viterbi forward step with short-dwell penalty for one base.

    Like standard Viterbi but penalizes bases with few signal samples.
    For dwells shorter than len(dwell_penalty), adds a quadratic penalty.
    For longer dwells, uses unpenalized standard Viterbi scores.

    Args:
        curr_scores: Output scores array (populated in place)
        curr_tb: Output traceback array (populated in place)
        prev_scores: Scores from previous base
        curr_level: Expected signal level for this base
        curr_signal: Signal values within this base's band
        band_start_diff: Offset between current and previous band starts
        dwell_penalty: Penalty array for short dwells
    """
    n_curr = len(curr_scores)
    n_prev = len(prev_scores)
    n_pen = len(dwell_penalty)

    # Compute unpenalized (standard Viterbi) scores for dwells >= penalty length
    unpen_scores = np.empty(n_curr, dtype=np.float32)
    unpen_tb = np.empty(n_curr, dtype=np.int32)
    _banded_forward_vit_step(
        unpen_scores,
        unpen_tb,
        prev_scores,
        curr_level,
        curr_signal,
        band_start_diff,
    )

    for bp in range(n_curr):
        # Past end of prev band by more than penalty range: forced stay
        if bp + band_start_diff - n_prev >= n_pen:
            curr_scores[bp] = curr_scores[bp - 1] + _score(curr_level, curr_signal[bp])
            curr_tb[bp] = curr_tb[bp - 1] + 1
            continue

        # Default: invalid
        curr_scores[bp] = LARGE_SCORE + prev_scores[n_prev - 1]
        curr_tb[bp] = -1

        if bp == 0 and band_start_diff == 0:
            continue

        running_pos_score = 0.0
        for dwell_idx in range(n_pen):
            # Beginning of curr or prev band reached
            if dwell_idx > bp or (band_start_diff == 0 and bp == dwell_idx):
                break

            running_pos_score += _score(curr_level, curr_signal[bp - dwell_idx])

            # Check prev position is in range
            prev_idx = bp - dwell_idx - 1 + band_start_diff
            if prev_idx >= n_prev:
                continue

            # Penalized score
            pos_score = prev_scores[prev_idx] + running_pos_score + dwell_penalty[dwell_idx]
            if pos_score < curr_scores[bp]:
                curr_scores[bp] = pos_score
                curr_tb[bp] = dwell_idx

        # Check unpenalized score for dwell >= penalty length
        if bp >= n_pen:
            pos_score = unpen_scores[bp - n_pen] + running_pos_score
            if pos_score < curr_scores[bp]:
                curr_scores[bp] = pos_score
                curr_tb[bp] = unpen_tb[bp - n_pen] + n_pen


def _banded_forward_dp(
    all_scores: np.ndarray,
    traceback: np.ndarray,
    signal: np.ndarray,
    levels: np.ndarray,
    seq_band: np.ndarray,
    base_offsets: np.ndarray,
    short_dwell_penalty: np.ndarray,
    algo: str,
) -> None:
    """
    Perform banded forward dynamic programming over all bases.

    Args:
        all_scores: Ragged output scores array (populated in place)
        traceback: Ragged output traceback array (populated in place)
        signal: Normalized signal values
        levels: Expected levels per base
        seq_band: Band boundaries (2, seq_len)
        base_offsets: Start offset in ragged array for each base
        short_dwell_penalty: Penalty array for short dwells
        algo: "Viterbi" or "dwell_penalty"
    """
    if algo == ALGO_VITERBI:
        core_func = _banded_forward_vit_step
    elif algo == ALGO_DWELL_PENALTY:
        core_func = _banded_forward_dwell_penalty_step
    else:
        raise ValueError(f"Invalid refinement algorithm: {algo}")
    use_dwell_pen = algo == ALGO_DWELL_PENALTY

    # First base: spoof prev_scores to force stays (score[0]=0, rest=inf)
    curr_bw = int(seq_band[1, 0])
    prev_scores = np.full(curr_bw, np.finfo(np.float32).max, dtype=np.float32)
    prev_scores[0] = 0.0

    if use_dwell_pen:
        core_func(
            all_scores[:curr_bw],
            traceback[:curr_bw],
            prev_scores,
            levels[0],
            signal[:curr_bw],
            1,  # band_start_diff=1 to force a "move" at position 0
            short_dwell_penalty,
        )
    else:
        core_func(
            all_scores[:curr_bw],
            traceback[:curr_bw],
            prev_scores,
            levels[0],
            signal[:curr_bw],
            1,
        )

    prev_bw = curr_bw
    prev_band_st = 0
    prev_offset = 0

    # Process remaining bases
    for base_idx in range(1, levels.shape[0]):
        curr_band_st = int(seq_band[0, base_idx])
        curr_band_en = int(seq_band[1, base_idx])
        curr_bw = curr_band_en - curr_band_st
        curr_offset = int(base_offsets[base_idx])

        if use_dwell_pen:
            core_func(
                all_scores[curr_offset : curr_offset + curr_bw],
                traceback[curr_offset : curr_offset + curr_bw],
                all_scores[prev_offset : prev_offset + prev_bw],
                levels[base_idx],
                signal[curr_band_st:curr_band_en],
                curr_band_st - prev_band_st,
                short_dwell_penalty,
            )
        else:
            core_func(
                all_scores[curr_offset : curr_offset + curr_bw],
                traceback[curr_offset : curr_offset + curr_bw],
                all_scores[prev_offset : prev_offset + prev_bw],
                levels[base_idx],
                signal[curr_band_st:curr_band_en],
                curr_band_st - prev_band_st,
            )

        prev_band_st = curr_band_st
        prev_bw = curr_bw
        prev_offset = curr_offset


def _banded_traceback(
    path: np.ndarray,
    seq_band: np.ndarray,
    base_offsets: np.ndarray,
    traceback: np.ndarray,
) -> None:
    """
    Reconstruct path from forward pass traceback.

    Args:
        path: Output array of length seq_len+1 (populated in place).
            path[i] = signal position where base i starts.
        seq_band: Band boundaries (2, seq_len)
        base_offsets: Start offset in ragged array for each base
        traceback: Ragged traceback from forward pass. Each entry stores
            the number of signal points backwards to the start of that base.
    """
    n_bases = path.shape[0] - 1
    path[0] = 0
    path[n_bases] = seq_band[1, n_bases - 1]  # sig_len

    for base_idx in range(n_bases - 1, 0, -1):
        # Signal position just before base_idx+1 starts
        sig_lookup_pos = path[base_idx + 1] - 1
        # Look up traceback for this base at this signal position
        band_idx = sig_lookup_pos - seq_band[0, base_idx]
        offset = int(base_offsets[base_idx]) + band_idx
        next_sig_offset = traceback[offset]
        path[base_idx] = sig_lookup_pos - next_sig_offset


# ============================================================================
# Banded DP entry point
# ============================================================================


def seq_banded_dp(
    signal: np.ndarray,
    levels: np.ndarray,
    seq_band: np.ndarray,
    short_dwell_penalty: np.ndarray,
    algo: str = DEFAULT_ALGO,
) -> np.ndarray:
    """
    Decode the optimal path between signal and levels using banded DP.

    Implements Remora's seq_banded_dp: banded forward Viterbi pass followed
    by traceback. Minimizes squared error between signal and expected levels,
    optionally with short-dwell penalties.

    Args:
        signal: Float32 normalized signal values
        levels: Float32 expected levels per base
        seq_band: int32 array of shape (2, seq_len). Row 0 = lower band
            boundaries in signal coords, row 1 = upper boundaries.
            seq_band[0, 0] should be 0, seq_band[1, -1] should be sig_len.
        short_dwell_penalty: Float32 penalty array for short dwells
        algo: "Viterbi" or "dwell_penalty"

    Returns:
        Int32 array of length seq_len+1 containing the signal position
        where each base starts. First element is 0, last is sig_len.
    """
    if HAS_RUST:
        assert _rs_seq_banded_dp is not None
        return np.asarray(
            _rs_seq_banded_dp(
                signal.astype(np.float32),
                levels.astype(np.float32),
                seq_band.astype(np.int32),
                short_dwell_penalty.astype(np.float32),
                algo,
            )
        )

    # Compute base offsets for ragged array indexing
    band_widths = np.diff(seq_band, axis=0)[0]  # seq_band[1] - seq_band[0]
    base_offsets_raw = np.cumsum(band_widths)
    band_len = int(base_offsets_raw[-1])

    base_offsets = np.empty(seq_band.shape[1] + 1, dtype=np.uint32)
    base_offsets[0] = 0
    base_offsets[1:] = base_offsets_raw

    # Allocate ragged arrays
    all_scores = np.empty(band_len, dtype=np.float32)
    tb = np.empty(band_len, dtype=np.int32)

    # Forward pass
    _banded_forward_dp(
        all_scores,
        tb,
        signal.astype(np.float32),
        levels.astype(np.float32),
        seq_band,
        base_offsets,
        short_dwell_penalty.astype(np.float32),
        algo,
    )

    # Traceback
    seq_len = levels.shape[0]
    path = np.empty(seq_len + 1, dtype=np.int32)
    _banded_traceback(path, seq_band, base_offsets, tb)

    return path


# ============================================================================
# Full signal mapping refinement pipeline
# ============================================================================


def refine_signal_mapping(
    signal: np.ndarray,
    seq_to_sig_map: np.ndarray,
    levels: np.ndarray,
    band_half_width: int = DEFAULT_HALF_BANDWIDTH,
    algo: str = DEFAULT_ALGO,
    short_dwell_pen: np.ndarray | None = None,
    adjust_band_min_step: int = 2,
) -> np.ndarray:
    """
    Refine signal mapping to minimize difference between signal and levels.

    Computes band from initial mapping, runs banded DP, returns refined path.
    Matches Remora's refine_signal_mapping().

    Args:
        signal: Float32 normalized signal values
        seq_to_sig_map: Initial base-to-signal mapping (seq_len+1)
        levels: Expected levels per base (seq_len)
        band_half_width: Half bandwidth for banding
        algo: "Viterbi" or "dwell_penalty"
        short_dwell_pen: Penalty array (uses default if None)
        adjust_band_min_step: Minimum step for band adjustment

    Returns:
        Refined seq_to_sig_map (int32 array of length seq_len+1)
    """
    if short_dwell_pen is None:
        short_dwell_pen = DEFAULT_SHORT_DWELL_PEN

    # Trim signal to mapped region
    sig_start = int(seq_to_sig_map[0])
    signal = signal[sig_start : int(seq_to_sig_map[-1])]
    if sig_start != 0:
        seq_to_sig_map = seq_to_sig_map.copy() - sig_start

    # Compute band: sig_band -> seq_band -> adjust
    sig_band = compute_sig_band(seq_to_sig_map, levels, bhw=band_half_width)
    seq_band = convert_to_seq_band(sig_band)
    adjust_seq_band(seq_band, min_step=adjust_band_min_step)
    validate_band(seq_band, sig_len=signal.shape[0], seq_len=levels.shape[0])

    # Replace NaN levels with 0
    temp_levels = levels.copy()
    temp_levels[np.isnan(levels)] = 0

    path = seq_banded_dp(
        signal.astype(np.float32),
        temp_levels.astype(np.float32),
        seq_band,
        short_dwell_pen,
        algo,
    )

    return path + sig_start


# ============================================================================
# SigMapRefiner class
# ============================================================================


@dataclass
class SigMapRefiner:
    """
    Signal map refiner using expected kmer signal levels.

    Matches remora's SigMapRefiner architecture:
    - rough_rescale updates normalization to match kmer levels
    - refine_sig_map (banded DP) refines base boundaries
    - multi-iteration refinement with Theil-Sen rescaling

    Attributes:
        kmer_to_level: Mapping from kmer string to expected signal level
        kmer_len: Length of kmers in the table
        half_bandwidth: Half-width of the DP band (in signal samples).
            Remora default is 5; larger values explore more.
        do_rough_rescale: Whether to rough-rescale signal before refinement
        scale_iters: Number of rescaling iterations during refinement.
            -1 = no banded DP (rough rescale only).
            0 = one round of banded DP without rescaling.
            >0 = N rounds of banded DP with rescaling between rounds.
        algo: Refinement algorithm ("Viterbi" or "dwell_penalty")
        sd_params: Short dwell penalty parameters (target, limit, weight)
        center_idx: Position within kmer that is the "center" base.
    """

    kmer_to_level: dict[str, float]
    kmer_len: int
    half_bandwidth: int = DEFAULT_HALF_BANDWIDTH
    do_rough_rescale: bool = True
    scale_iters: int = 2
    algo: str = DEFAULT_ALGO
    sd_params: tuple[int, int, float] | None = None
    center_idx: int = -1
    sd_arr: np.ndarray = field(default_factory=lambda: DEFAULT_SHORT_DWELL_PEN)

    def __post_init__(self):
        if self.center_idx < 0:
            self.center_idx = self.kmer_len // 2
        if self.sd_params is not None:
            self.sd_arr = compute_dwell_pen_array(*self.sd_params)

    @classmethod
    def from_table(
        cls,
        table_path: Path,
        half_bandwidth: int = DEFAULT_HALF_BANDWIDTH,
        do_rough_rescale: bool = True,
        scale_iters: int = 2,
        algo: str = DEFAULT_ALGO,
        sd_params: tuple[int, int, float] | None = None,
        center_idx: int = -1,
    ) -> "SigMapRefiner":
        """Create refiner from a kmer level table file."""
        kmer_to_level, kmer_len = load_kmer_table(table_path)
        return cls(
            kmer_to_level=kmer_to_level,
            kmer_len=kmer_len,
            half_bandwidth=half_bandwidth,
            do_rough_rescale=do_rough_rescale,
            scale_iters=scale_iters,
            algo=algo,
            sd_params=sd_params,
            center_idx=center_idx if center_idx >= 0 else kmer_len // 2,
        )

    def refine(
        self,
        signal: np.ndarray,
        sequence: str,
        seq_to_sig_map: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Refine signal normalization and mapping using expected kmer levels.

        Matches remora's two-phase refinement:
        1. Rough rescale: quantile-based normalization correction
        2. Iterative DP: banded Viterbi with inter-iteration Theil-Sen rescaling

        Args:
            signal: Normalized signal array
            sequence: Base sequence
            seq_to_sig_map: Initial base-to-signal mapping (from move table)

        Returns:
            Tuple of (rescaled_signal, refined_seq_to_sig_map).

        Delegates to escapepod-signal's ``refine_signal_map`` (the same canonical
        implementation and settings leech_core's Rust path uses), so the Python
        fallback and the Rust path agree. escapepod runs rough rescale + iterative
        banded DP with Theil-Sen rescaling internally; the returned
        (scale, shift, drift) reconstruct the level-matched signal.
        """
        from escapepod import refine_signal_map as _epod_refine_signal_map

        expected = extract_levels(sequence, self.kmer_to_level, self.kmer_len, self.center_idx)
        if expected.size == 0 or len(seq_to_sig_map) != expected.size + 1:
            return signal, seq_to_sig_map

        # scale_iters < 0: rough-rescale-only mode (no banded DP), mapping left
        # unchanged. escapepod's refine always runs >=1 DP pass, so this niche
        # mode keeps leech's quantile rough rescale.
        if self.scale_iters < 0:
            if self.do_rough_rescale:
                signal = rough_rescale_quantile(signal, expected, seq_to_sig_map)
            return signal, seq_to_sig_map

        sig_f32 = signal.astype(np.float32, copy=False)
        map_list = [int(x) for x in seq_to_sig_map]
        if map_list[0] < 0 or map_list[0] >= map_list[-1] or map_list[-1] > sig_f32.size:
            return signal, seq_to_sig_map

        # escapepod builds RefineSettings internally (fixed banding, LSQ rough
        # rescale, Theil-Sen rescale, asymmetric dwell penalty). scale_iters maps
        # to n_refinement_iters the same way leech_core does: max(0, scale_iters).
        levels_f32 = np.nan_to_num(expected.astype(np.float32, copy=False), nan=0.0)
        try:
            refined_map, scale, shift, drift = _epod_refine_signal_map(
                sig_f32,
                map_list,
                levels_f32,
                half_bandwidth=self.half_bandwidth,
                scale_iters=max(0, self.scale_iters),
                # Fixed seed so the Theil-Sen subsample (on long reads) is
                # reproducible; must match leech_core's REFINE_SUBSAMPLE_SEED.
                seed=REFINE_SUBSAMPLE_SEED,
            )
        except Exception as e:  # noqa: BLE001 - fall back to the input on any failure
            logger.debug(f"escapepod refine_signal_map failed: {e}")
            return signal, seq_to_sig_map

        if len(refined_map) != len(seq_to_sig_map):
            return signal, seq_to_sig_map

        # Reconstruct the level-matched signal from the returned rescale params
        # (matches leech_core: (signal[i] - shift - drift*i) / scale).
        if abs(scale) > 1e-10:
            idx = np.arange(sig_f32.size, dtype=np.float32)
            rescaled = (sig_f32 - shift - drift * idx) / scale
        else:
            rescaled = sig_f32
        return rescaled, np.asarray(refined_map)
