"""
Signal map refinement using expected kmer signal levels.

``SigMapRefiner.refine`` delegates entirely to escapepod-signal's
``refine_signal_map`` (see :data:`REFINE_PRESET_DOC`) -- the banded-Viterbi DP
that used to live in this module was removed in #276 (it was reachable only
from tests) and now lives, as a frozen reference oracle, in
``tests/reference_signal_refine.py``.

Key functions:
    load_kmer_table(): Load kmer -> expected level mapping
    extract_levels(): Get expected levels for a sequence
    SigMapRefiner: High-level class for signal refinement
"""

import gzip
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("leech.signal_refine")


# ============================================================================
# Constants
# ============================================================================

DEFAULT_HALF_BANDWIDTH = 5

# Fixed seed for escapepod's Theil-Sen subsample so refinement is reproducible
# on long reads. Must match leech_core's REFINE_SUBSAMPLE_SEED (Rust path).
REFINE_SUBSAMPLE_SEED = 42

#: escapepod's named preset is the single definition of leech's refinement
#: settings, on both backends.
#:
#: There is deliberately no ``dwell_target`` constant here any more. This module
#: used to pin it to 0.0 (resolve the target from the read's own move-table
#: median) to override escapepod's old fixed 4.0 default -- but leech_core takes
#: the whole preset via ``RefineSettings::move_table_refinement``, so pinning
#: one field here would leave the two halves free to drift apart again the
#: moment the preset changed. That asymmetry is exactly what caused #193.
#:
#: escapepod >= 0.15.0 defaults ``dwell_target``/``dwell_weight`` to ``None``,
#: meaning "take the preset" (escapepod-rs#257), which is why the floor in
#: pyproject.toml is 0.15.0 and not 0.14.0: on 0.14.0 omitting the argument
#: silently reinstates the 4.0 that this module existed to override. The
#: backend parity suite fails loudly if that ever happens.
#:
#: This is also, since #276, the *only* statement of leech's refinement
#: settings: there is no local ``algo``/short-dwell-penalty/rough-rescale
#: configuration left to drift against it.
REFINE_PRESET_DOC = "escapepod RefineSettings::move_table_refinement"


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
# SigMapRefiner class
# ============================================================================


@dataclass
class SigMapRefiner:
    """
    Signal map refiner using expected kmer signal levels.

    A thin, picklable config holder: ``refine()`` delegates the actual
    refinement (rough rescale, banded DP, Theil-Sen rescale) entirely to
    escapepod-signal's ``refine_signal_map`` (:data:`REFINE_PRESET_DOC`).
    Every field below is one escapepod-signal ``refine_signal_map`` takes
    directly; there is nothing here escapepod does not also see.

    Attributes:
        kmer_to_level: Mapping from kmer string to expected signal level
        kmer_len: Length of kmers in the table
        half_bandwidth: Half-width of the DP band (in signal samples).
            Remora default is 5; larger values explore more.
        scale_iters: Number of rescaling iterations during refinement.
            -1 = no refinement (map returned unchanged).
            0 = one round of banded DP without rescaling.
            >0 = N rounds of banded DP with rescaling between rounds.
        center_idx: Position within kmer that is the "center" base.
    """

    kmer_to_level: dict[str, float]
    kmer_len: int
    half_bandwidth: int = DEFAULT_HALF_BANDWIDTH
    scale_iters: int = 2
    center_idx: int = -1

    def __post_init__(self):
        if self.center_idx < 0:
            self.center_idx = self.kmer_len // 2

    @classmethod
    def from_table(
        cls,
        table_path: Path,
        half_bandwidth: int = DEFAULT_HALF_BANDWIDTH,
        scale_iters: int = 2,
        center_idx: int = -1,
    ) -> "SigMapRefiner":
        """Create refiner from a kmer level table file."""
        kmer_to_level, kmer_len = load_kmer_table(table_path)
        return cls(
            kmer_to_level=kmer_to_level,
            kmer_len=kmer_len,
            half_bandwidth=half_bandwidth,
            scale_iters=scale_iters,
            center_idx=center_idx if center_idx >= 0 else kmer_len // 2,
        )

    def refine(
        self,
        signal: np.ndarray,
        sequence: str,
        seq_to_sig_map: np.ndarray,
        *,
        expected_levels: np.ndarray | None = None,
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
            expected_levels: Precomputed ``extract_levels(sequence,
                self.kmer_to_level, self.kmer_len, self.center_idx)``, for a
                caller that already has it (``build_leech_read`` computes it
                once per read and reuses it here and in
                ``compute_kmer_residual_features``/``compute_signal_residual``
                rather than three identical per-read calls -- issue #275).
                ``None`` (the default) computes it here, unchanged from before.

        Returns:
            Tuple of ``(signal, refined_seq_to_sig_map)``. The signal is
            returned **unchanged** — only the boundaries are taken.

        Delegates to escapepod-signal's ``refine_signal_map`` (the same canonical
        implementation leech_core's Rust path uses). Two settings have to be
        stated explicitly for the two paths to agree, because escapepod's Python
        binding and ``rust/src/inference_pipeline/refinement.rs`` each carry
        their own copy of "leech's refinement configuration" and the copies
        differ:

        - the settings: both halves take escapepod's
          ``RefineSettings::move_table_refinement`` preset rather than pinning
          fields themselves. See :data:`REFINE_PRESET_DOC`.
        - the fitted ``(scale, shift, drift)``: leech_core deliberately discards
          them. See the note at the end of this method.
        """
        from escapepod import refine_signal_map as _epod_refine_signal_map

        if expected_levels is None:
            expected = extract_levels(sequence, self.kmer_to_level, self.kmer_len, self.center_idx)
        else:
            expected = expected_levels
        if expected.size == 0 or len(seq_to_sig_map) != expected.size + 1:
            return signal, seq_to_sig_map

        # scale_iters < 0: no refinement at all.
        #
        # This used to mean "rough rescale the signal, leave the map alone".
        # Since the fitted rescale is no longer applied (see below), that is now
        # a no-op on both outputs, so it is spelled as one. leech_core does the
        # same — `process_read_signal` skips `refine_signal_map_pipeline` for a
        # negative `refine_scale_iters` — rather than clamping to 0, which is
        # escapepod's "one DP pass, no rescale" and would have refined the map
        # on the Rust path while Python left it untouched.
        if self.scale_iters < 0:
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
            refined_map, _scale, _shift, _drift = _epod_refine_signal_map(
                sig_f32,
                map_list,
                levels_f32,
                half_bandwidth=self.half_bandwidth,
                scale_iters=max(0, self.scale_iters),
                # dwell_target/dwell_weight deliberately unset: that takes
                # escapepod's preset, which is what leech_core takes too.
                # See REFINE_PRESET_DOC.
                # Fixed seed so the Theil-Sen subsample (on long reads) is
                # reproducible; must match leech_core's REFINE_SUBSAMPLE_SEED.
                seed=REFINE_SUBSAMPLE_SEED,
            )
        except Exception as e:  # noqa: BLE001 - fall back to the input on any failure
            logger.debug(f"escapepod refine_signal_map failed: {e}")
            return signal, seq_to_sig_map

        if len(refined_map) != len(seq_to_sig_map):
            return signal, seq_to_sig_map

        # Take the refined boundaries, keep our own normalization.
        #
        # Deliberately do NOT rescale the signal by the fitted (scale, shift,
        # drift). The caller hands in a median-MAD normalized signal — one
        # transform shared by every read — and that is what per-base stats,
        # k-mer residuals and the trained models are calibrated against.
        # Applying the fit replaces it with a *per-read* transform estimated on
        # a chunk that sits largely in a constant 3' adapter, where expected
        # levels barely vary and the fit is weakly identified: observed scales
        # ran from 15 to 1084 and were frequently negative, i.e. sign-flipping
        # the read. escapepod rejects the worst of those now, but rejection is
        # itself per-read, so the reads that still fit end up on a different
        # scale from the reads that do not — and cross-read comparability is
        # exactly what k-mer residuals depend on.
        #
        # This is the same call leech_core's `refine_signal_map_pipeline`
        # makes, and for the same measured reason (#168): on tRNA-Met chunks,
        # per-base level vs expected k-mer level was r = +0.72 keeping our
        # normalization against +0.03 applying the fitted one.
        return signal, np.asarray(refined_map)
