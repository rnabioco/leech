"""Frozen reference implementation of Remora's banded-DP signal refinement.

This is the pure NumPy port of Remora's ``refine_signal_map.py`` /
``refine_signal_map_core.pyx`` (banded Viterbi DP over expected kmer signal
levels) that ``leech.signal_refine.SigMapRefiner.refine`` used to run itself,
before it was rewritten to delegate entirely to escapepod-signal's
``refine_signal_map`` (see ``leech.signal_refine.REFINE_PRESET_DOC``). Nothing
in ``src/leech`` calls this module any more -- issue #276 found that
``compute_dwell_pen_array`` through ``refine_signal_mapping`` (plus
``rough_rescale_quantile``) were reachable only from tests, and moved them
here rather than deleting them outright, because they remain useful oracles:
``test_signal_refine.py`` exercises them as an algorithmic sanity check
(trivial cases, monotonicity, band invariants) independent of escapepod.

escapepod itself carries golden tests pinning its Rust implementation of this
same NumPy body bit-for-bit (rnabioco/escapepod-rs#204), so there is no
correctness gap from leech no longer running this path in production -- this
module exists purely as a readable, dependency-free reference.

Do not modernise or re-optimise this file; its whole value is that it does
not change. There is deliberately no Rust-accelerated dispatch here any more
-- ``leech_core`` no longer exports ``seq_banded_dp`` / ``rough_rescale_quantile``
(they only ever served this reference path; see #276), so this is the one and
only implementation now.
"""

import logging

import numpy as np

from leech.signal_refine import DEFAULT_HALF_BANDWIDTH

logger = logging.getLogger("leech.tests.reference_signal_refine")

# ============================================================================
# Constants (matching Remora defaults)
# ============================================================================

LARGE_SCORE = 100.0  # Large penalty for invalid positions

ALGO_VITERBI = "Viterbi"
ALGO_DWELL_PENALTY = "dwell_penalty"
DEFAULT_ALGO = ALGO_DWELL_PENALTY

DEFAULT_SHORT_DWELL_PARAMS = (4, 3, 0.5)  # (target, limit, weight)


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
    """
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
# Core Viterbi functions (pure Python)
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
