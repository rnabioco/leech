"""
Chunk extraction from LeechReads.

Provides functionality for extracting training chunks from processed reads,
with support for motif-based filtering.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from leech.constants import (
    DEFAULT_KMER_CONTEXT,
    DEFAULT_MAX_SAMPLES_PER_BASE,
    DEFAULT_SIGNAL_CONTEXT,
    DEFAULT_SIGNAL_KMER_CONTEXT,
)
from leech.io.motif_search import MotifMatch, MotifSearcher

if TYPE_CHECKING:
    import pysam

    from leech.configs import ChunkConfig, LabelConfig, MotifConfig

logger = logging.getLogger("leech.chunking.extractor")


def resolve_feature_window(
    feature_start: int | None,
    feature_end: int | None,
    kmer_context: int = DEFAULT_KMER_CONTEXT,
) -> tuple[int, int, int]:
    """Resolve a requested feature window to ``(start, end, width)``.

    ``start``/``end`` are signed offsets from the focus base, ``end``
    inclusive, so ``width`` is ``end - start + 1``. ``None`` means "the k-mer
    window", i.e. ``-kmer_context``/``+kmer_context``.

    The `is None` tests are the point of this function. ``feature_start=0``
    (features begin *at* the focus base, the right-only windows used for tRNA
    3' ends) is a legitimate value that a truthiness test turns back into the
    default, silently widening the window by ``kmer_context`` bases — that was
    issue #189, and it reached the stored chunk metadata that `dataset.py`
    slices features with. Every caller resolving a feature window must go
    through here rather than re-deriving the rule.
    """
    start = feature_start if feature_start is not None else -kmer_context
    end = feature_end if feature_end is not None else kmer_context
    return start, end, end - start + 1


def resolve_signal_context_bases(
    seq_to_sig_map: np.ndarray,
    base_idx: int,
    left_bases: int,
    right_bases: int,
    num_mapped_bases: int,
) -> tuple[int, int]:
    """Resolve a base-defined signal window to a sample interval.

    ``--signal-context-bases L,R`` (issue #278) cuts the signal window at the
    base-to-signal map positions of offsets ``-L`` and ``+R`` (inclusive)
    around ``base_idx``, so two reads at different translocation speeds read
    the same *bases* of context instead of the same number of *samples* -- a
    sample-defined window can't reach a base at +24 on a slow read
    (~36 samples/base) without also reaching a base at +25 on a fast one
    (~24 samples/base).

    Returns ``(sample_start, sample_end)``, a half-open sample interval:
    ``sample_start`` is the first sample of base ``base_idx - L`` and
    ``sample_end`` is the first sample *past* base ``base_idx + R``. Both
    bounds are clamped to the read's own mapped span (base 0 through
    ``num_mapped_bases - 1``) rather than allowed to run off it -- a focus
    base near either edge of the read gets a *narrower* window, never a
    dropped chunk, matching the one allowed drop rule (CLAUDE.md: only
    ``base_idx`` itself having no signal boundaries drops a chunk).

    This is the single definition of the rule; the Rust chunk loop
    (``rust/src/inference_pipeline/training.rs``) mirrors it exactly (it
    cannot call into Python) and the two are held equal by
    ``tests/test_backend_parity.py``.

    ``left_bases``/``right_bases`` are meant to be ``>= 0`` -- the CLI and
    ``handle_prepare`` both refuse a negative value before any read is
    touched -- but ``lo_base``/``hi_base`` are independently clamped to
    ``[0, num_mapped_bases - 1]`` regardless, rather than relying on the
    caller: an unclamped negative ``right_bases`` could otherwise turn
    ``seq_to_sig_map[hi_base + 1]`` into a silently-wrong wraparound index
    (Python's negative indexing) instead of an error, and an unclamped
    large-magnitude negative ``left_bases`` indexes past the array entirely.
    """
    last_base = num_mapped_bases - 1
    lo_base = max(0, min(last_base, base_idx - left_bases))
    hi_base = max(0, min(last_base, base_idx + right_bases))
    return int(seq_to_sig_map[lo_base]), int(seq_to_sig_map[hi_base + 1])


def default_signal_len_for_bases_context(left_bases: int, right_bases: int) -> int:
    """Default ``signal_len`` for ``--signal-context-bases L,R``.

    Used when the caller does not pass ``--signal-len`` explicitly. The
    window covers ``L + R + 1`` bases (the focus base itself, plus ``L``
    before and ``R`` after); sizing at :data:`~leech.constants.
    DEFAULT_MAX_SAMPLES_PER_BASE` samples/base -- the slow-read rate -- keeps
    a typical read's resolved window narrower than ``signal_len``, so it gets
    zero-padded (the conservative default this issue's acceptance criteria
    calls for) rather than centre-cropped.
    """
    return (left_bases + right_bases + 1) * DEFAULT_MAX_SAMPLES_PER_BASE


def _assert_constant_column(column: np.ndarray, name: str) -> None:
    """Refuse a corpus whose stored feature window is not the same on every row.

    Reading column 0 (or any single row) and trusting it for the whole corpus
    is issue #230's failure mode generalized to this field: a corpus written
    by two prepare runs with different ``--feature-start``/``--feature-end``
    would otherwise train silently on whichever value the first chunk happened
    to carry.
    """
    if column.size and not np.all(column == column[0]):
        raise ValueError(
            f"Corpus has inconsistent {name!r} across chunks: {np.unique(column).tolist()}. "
            f"A training run needs one feature window throughout; this corpus mixes prepare "
            f"runs with different --feature-start/--feature-end and must be re-prepared "
            f"consistently before it can be trained on."
        )


def _feature_window_from_table(table, kmer_context: int) -> tuple[int | None, int | None]:
    """Column form of :func:`feature_window_from_metadata`.

    ``feature_left``/``feature_right`` are never written as columns (only as
    legacy per-chunk-dict fields on the row path), so the fallback chain here
    is shorter than the mapping form's.
    """
    start_col = table.values("feature_start")
    if start_col is not None:
        _assert_constant_column(start_col, "feature_start")
        start = int(start_col[0])
    else:
        margin_col = table.values("dwell_margin_left")
        if margin_col is not None:
            _assert_constant_column(margin_col, "dwell_margin_left")
            start = -(kmer_context + int(margin_col[0]))
        else:
            start = None

    end_col = table.values("feature_end")
    if end_col is not None:
        _assert_constant_column(end_col, "feature_end")
        end = int(end_col[0])
    else:
        end = None

    return start, end


def _feature_window_from_mapping(source, kmer_context: int) -> tuple[int | None, int | None]:
    """Mapping form of :func:`feature_window_from_metadata`.

    ``source`` is a model config dict or a single chunk (a plain dict or a
    :class:`~leech.chunking.table.ChunkRow`). Mirrors the fallback chain that
    used to be pasted into ``single.py``, ``bundle.py``, ``training.py`` and
    ``dataset.py`` separately: ``feature_start``/``feature_end`` (current),
    then ``feature_left``/``feature_right`` (legacy), then
    ``dwell_margin_left``/``dwell_margin_right`` (older legacy, ``right``
    resolved from the stored feature array's width when one is present).
    """
    if source.get("feature_start") is not None:
        start = int(source["feature_start"])
    elif source.get("feature_left") is not None:
        start = -int(source["feature_left"])
    elif source.get("dwell_margin_left") is not None:
        start = -(kmer_context + int(source["dwell_margin_left"]))
    else:
        start = None

    if source.get("feature_end") is not None:
        end = int(source["feature_end"])
    elif source.get("feature_right") is not None:
        end = int(source["feature_right"])
    elif source.get("dwell_margin_right") is not None:
        raw_features = source.get("features")
        if raw_features is not None and getattr(raw_features, "ndim", 1) > 1 and start is not None:
            # A chunk dict: derive from the stored feature array's actual width.
            end = raw_features.shape[1] - 1 + start
        else:
            # A config dict never carries "features" -- this is the only branch
            # that fires for it, and single.py/bundle.py's original inline code
            # computed exactly this (kmer_context + dwell_margin_right) with no
            # array to check. Do not collapse this to None: that silently
            # narrows the window to +-kmer_context instead of using the margin
            # the config actually recorded.
            end = kmer_context + int(source["dwell_margin_right"])
    else:
        end = None

    return start, end


def feature_window_from_metadata(source, kmer_context: int) -> tuple[int | None, int | None]:
    """Resolve the stored ``(feature_start, feature_end)`` from a config or corpus.

    ``None`` for either element means nothing was stored for it -- callers
    fall back to the k-mer window via :func:`resolve_feature_window`, exactly
    as a fresh corpus with no feature-window fields at all always has.

    ``source`` is either a :class:`~leech.chunking.table.ChunkTable` (the
    window is read as a column and asserted constant across every chunk -- see
    :func:`_assert_constant_column`) or a mapping -- a model's ``config.json``
    dict or a single chunk (a plain dict or a
    :class:`~leech.chunking.table.ChunkRow`). This is the one place the
    fallback chain is written; ``training.py``, ``dataset.py`` and
    ``InferenceSpec`` all resolve through it rather than carrying their own
    copies (issue #269).
    """
    from leech.chunking.table import ChunkTable

    if isinstance(source, ChunkTable):
        return _feature_window_from_table(source, kmer_context)
    return _feature_window_from_mapping(source, kmer_context)


def merge_feature_channels(
    dwell_features: dict[str, np.ndarray],
    signal_features: dict[str, np.ndarray],
) -> list[tuple[str, np.ndarray]]:
    """Order the per-base feature arrays into the chunk's feature rows.

    Row order *is* the model's input channel order, so it is part of every
    trained checkpoint: reordering it silently feeds `level_mean` into the
    filter that learned `dwell_log`. It is the dict-merge order it has always
    been -- dwell rows in `compute_dwell_features` order, then the level rows
    in `compute_signal_features` order, then the three k-mer residual rows that
    `preparation.reader` folds into `signal_features` -- and it must stay equal
    to the order `rust/src/inference_pipeline/processing.rs` pushes rows in.
    `tests/test_data_prep.py::TestFeatureChannelOrder` pins it.

    Resolved once per read rather than once per focus base: all-bases mode
    calls `get_chunk` thousands of times per read and each call was rebuilding
    this dict.
    """
    return list({**dwell_features, **signal_features}.items())


def _mask_focus_side(
    kmer_seq: str,
    sequence_with_kmer_context: str,
    *,
    kmer_context: int,
    core_focus_idx: int,
    kmer_before: int,
    side: str,
) -> tuple[str, str]:
    """Blank ('N') sequence characters strictly to one side of the focus base.

    Applies to both sequence fields `get_chunk` builds:

    - `kmer_seq` (the `sequence` field, `base_onehot`'s k-mer window):
      always exactly `2*kmer_context+1` bases wide and centered on the focus
      base by construction (`kmer_start = base_idx - kmer_context`), so the
      focus is always at index `kmer_context`.
    - `sequence_with_kmer_context` (the `signal_kmer` window): its extent is
      `[seq_start - kmer_before, seq_end + kmer_after)`, where `seq_start`/
      `seq_end` are the SIGNAL window's covered bases and therefore move with
      `base_justify` and the signal context -- the focus is not at a fixed
      offset, so the caller passes its position as `core_focus_idx +
      kmer_before` (`core_focus_idx = base_idx - seq_start`, computed from the
      same local variables `get_chunk` used to build the window in the first
      place, not re-derived from the chunk dict afterwards).

    The focus base's own character is never masked -- masking stops just
    short of it on the requested side and leaves the other side untouched.
    'N' maps to the same "not a base" sentinel (-1) that `sequence_to_int` /
    `encode_signal_kmer` already skip (`if base < 0: continue`), so a masked
    position contributes nothing to either encoding: the one-hot channels at
    that k-mer position are all zero, for every signal sample the masked base
    covers. This is the same mechanism the manual per-corpus workaround used
    (leech#256), reused here instead of reinvented.

    A focus index outside the string (only possible when the focus base's own
    signal span reaches all the way to the edge of a very narrow window)
    clamps into range rather than raising, so masking degrades to "mask
    everything but the boundary base" instead of crashing a rare chunk.
    """

    def _mask(seq: str, focus_idx: int) -> str:
        n = len(seq)
        if n == 0:
            return seq
        focus_idx = max(0, min(focus_idx, n - 1))
        if side == "left":
            return "N" * focus_idx + seq[focus_idx:]
        return seq[: focus_idx + 1] + "N" * (n - focus_idx - 1)

    return (
        _mask(kmer_seq, kmer_context),
        _mask(sequence_with_kmer_context, core_focus_idx + kmer_before),
    )


class LeechRead:
    """
    Container for a single read's data with all features.

    Attributes:
        read_id: Unique read identifier
        sequence: Basecalled sequence
        signal: Normalized signal array (cropped to the aligned region in
            ref-anchored mode; full trimmed/reversed signal otherwise)
        seq_to_sig_map: Mapping from base indices to signal indices
        dwells: Per-base dwell times
        dwell_features: Dict of dwell-derived features
        signal_features: Dict of signal-level features
        feature_channels: The two dicts merged into the ordered ``(name,
            array)`` feature rows every chunk is cut from. Resolved once here;
            mutating either dict afterwards will not be picked up.
        feature_matrix: ``feature_channels``' arrays stacked into one ``(F,
            num_bases)`` matrix, also resolved once here -- what
            ``get_chunk`` actually slices/pads (issue #275).
        labels: Optional labels for training (e.g., 0=uncharged, 1=charged)
        metadata: Additional metadata (alignment info, etc.)
        full_signal: When ref-anchored mode crops ``signal`` to the aligned
            region, the full pre-crop normalized signal is stashed here so
            ``get_chunk`` can optionally read into soft-clipped/unaligned
            regions at chunk-window edges. ``None`` when no crop happened.
        signal_offset: Index of ``signal[0]`` within ``full_signal`` — the
            translation between cropped (``self.signal``) coordinates and
            absolute coordinates for ``full_signal``. Zero when not cropped.
    """

    def __init__(
        self,
        read_id: str,
        sequence: str,
        signal: np.ndarray,
        seq_to_sig_map: np.ndarray,
        dwells: np.ndarray,
        dwell_features: dict[str, np.ndarray],
        signal_features: dict[str, np.ndarray],
        labels: np.ndarray | None = None,
        metadata: dict | None = None,
        signal_residual: np.ndarray | None = None,
        full_signal: np.ndarray | None = None,
        signal_offset: int = 0,
    ):
        """Initialize LeechRead."""
        self.read_id = read_id
        self.sequence = sequence
        self.signal = signal
        self.seq_to_sig_map = seq_to_sig_map
        self.dwells = dwells
        self.dwell_features = dwell_features
        self.signal_features = signal_features
        self.feature_channels = merge_feature_channels(dwell_features, signal_features)
        # One (F, num_bases) matrix, stacked once per read, so `get_chunk`
        # slices/pads all F rows in a single 2D op instead of looping over
        # ~12 channels and stacking at the end of every chunk (issue #275).
        # Every channel is float32 (see compute_dwell_features /
        # compute_signal_features / compute_kmer_residual_features), so this
        # stack changes no dtype and, since every row uses the same slice
        # indices, no value either.
        self.feature_matrix: np.ndarray = (
            np.stack([arr for _, arr in self.feature_channels], axis=0)
            if self.feature_channels
            else np.zeros((0, 0), dtype=np.float32)
        )
        self.labels = labels
        self.metadata = metadata if metadata is not None else {}
        self.signal_residual = signal_residual
        self.full_signal = full_signal
        self.signal_offset = signal_offset

    @property
    def num_bases(self) -> int:
        """Number of bases in the read."""
        return len(self.sequence)

    @property
    def num_mapped_bases(self) -> int:
        """Number of bases that have signal boundaries in ``seq_to_sig_map``.

        Usually equal to :attr:`num_bases`, but not always: under
        ``anchor="reference"`` the sequence is the aligned reference slice
        ``[reference_start:reference_end]`` while the map comes from
        ``compute_ref_to_signal``, which strips trailing non-match CIGAR ops
        first. An alignment ending in a deletion therefore yields a map
        shorter than the sequence, and a focus base in the gap has no
        ``seq_to_sig_map[base_idx + 1]``.

        This is the bound a focus base must satisfy; :attr:`num_bases` is the
        bound for reading *sequence* (which pads with ``N`` past the end).
        The Rust pipeline draws the same distinction — see
        ``extract_training_chunks_from_read``.
        """
        return max(0, len(self.seq_to_sig_map) - 1)

    @property
    def num_samples(self) -> int:
        """Number of signal samples."""
        return len(self.signal)

    def get_chunk(
        self,
        base_idx: int,
        config: ChunkConfig | None = None,
        signal_context: tuple[int, int] = DEFAULT_SIGNAL_CONTEXT,
        kmer_context: int = DEFAULT_KMER_CONTEXT,
        base_justify: str = "center",
        feature_start: int | None = None,
        feature_end: int | None = None,
        recover_softclip_signal: bool = False,
        signal_context_bases: tuple[int, int] | None = None,
        signal_len: int | None = None,
        mask_seq_side: str | None = None,
    ) -> dict[str, np.ndarray | str | int | None] | None:
        """
        Extract a training chunk centered on a specific base.

        Args:
            base_idx: Index of the focus base
            config: Optional ChunkConfig that overrides individual params.
            signal_context: (left, right) signal padding around focus base.
                Ignored when ``signal_context_bases`` is set.
            kmer_context: Number of bases on each side for k-mer encoding
            base_justify: "center", "start", or "end"
            feature_start: Signed offset from focus for feature window start.
            feature_end: Signed offset from focus for feature window end (inclusive).
            recover_softclip_signal: When True and ``full_signal`` is set
                (ref-anchored mode), fill chunk-window samples that fall
                outside the aligned region with real soft-clipped signal
                instead of zeros. Off by default to preserve Remora-compatible
                behavior — see R4 in the coordinate audit. Not honored in
                base-defined mode (``signal_context_bases`` set) -- the
                windows there never underflow the read's own signal (see
                below), so there is nothing to recover.
            signal_context_bases: ``(L, R)`` base offsets (issue #278) --
                mutually exclusive with ``signal_context``. When set, the
                signal window is cut at the base-to-signal map positions of
                ``base_idx - L`` and ``base_idx + R`` (inclusive) via
                :func:`resolve_signal_context_bases`, then pads (narrower) or
                centre-crops (wider) to a fixed ``signal_len`` -- the same
                pad/crop split ``escapepod_signal::chunk::place_window``
                applies on the Rust side, since the number of samples
                spanning ``L + R`` bases varies read to read.
            signal_len: Fixed emitted signal length when
                ``signal_context_bases`` is set. Required in that case;
                ignored otherwise (sample mode's ``chunk_len`` is
                ``signal_context[0] + signal_context[1]``).
            mask_seq_side: "left", "right", or None (default). Blanks ('N')
                sequence-branch characters strictly to that side of the focus
                base in both ``sequence`` and ``sequence_with_kmer_context`` —
                see :func:`_mask_focus_side` for the exact geometry and
                leech#256 for why.

        Returns:
            Dictionary with 'signal', 'kmer', 'dwell', 'features' arrays,
            or None if chunk cannot be extracted
        """
        # Override individual params from config if provided
        if config is not None:
            signal_context = config.signal_context
            kmer_context = config.kmer_context
            base_justify = config.base_justify
            feature_start = config.feature_start
            feature_end = config.feature_end
            recover_softclip_signal = config.recover_softclip_signal
            signal_context_bases = config.signal_context_bases
            signal_len = config.signal_len
            mask_seq_side = config.mask_seq_side

        # Check boundaries: base_idx must be valid for seq_to_sig_map access.
        # Bound on the map, not the sequence — they can differ (see
        # `num_mapped_bases`), and guarding on the sequence lets
        # `seq_to_sig_map[base_idx + 1]` below raise IndexError, which the
        # prepare workers turn into dropping the whole read rather than this
        # one chunk.
        if base_idx < 0 or base_idx >= self.num_mapped_bases:
            return None

        # Extract signal chunk (remora-compatible: pad with zeros at boundaries)
        if base_justify == "start":
            focus_sig_pos = int(self.seq_to_sig_map[base_idx])
        elif base_justify == "end":
            focus_sig_pos = int(self.seq_to_sig_map[base_idx + 1])
        else:
            focus_sig_pos = int(
                (self.seq_to_sig_map[base_idx] + self.seq_to_sig_map[base_idx + 1]) // 2
            )

        if signal_context_bases is not None:
            # Base-defined window (issue #278). `sample_start`/`sample_end`
            # are always within [0, num_samples] -- resolve_signal_context_bases
            # clamps at the base level (base 0 / num_mapped_bases - 1), and
            # seq_to_sig_map never exceeds num_samples -- so there is no
            # read-underflow/overflow case to handle here, only the
            # narrower-than/wider-than-signal_len split below.
            left_bases, right_bases = signal_context_bases
            assert signal_len is not None, "signal_len is required when signal_context_bases is set"
            chunk_len = signal_len
            sample_start, sample_end = resolve_signal_context_bases(
                self.seq_to_sig_map, base_idx, left_bases, right_bases, self.num_mapped_bases
            )
            # The REQUESTED (pre-crop) window -- used below for seq_to_sig_map
            # / sequence_with_kmer_context, matching
            # escapepod_signal::chunk::cut_chunk's own SignalKmer branch,
            # which hands signal_kmer_inputs this same pre-crop pair rather
            # than the post-crop one place_window actually placed.
            sig_start = sample_start
            sig_end = sample_end
            seq_to_sig_offset = 0
            requested = max(0, sig_end - sig_start)

            # The anchor's offset within the emitted signal_len-wide array,
            # and the true copy bound -- mirrors
            # escapepod_signal::chunk::place_window exactly:
            #   - narrower (or equal): copy is bounded by the REQUESTED
            #     `sample_end`, not `win_start + chunk_len` -- the shortfall
            #     is genuine padding (bases -L..+R and then zeros), not "keep
            #     copying real signal past +R until chunk_len samples are
            #     full", which is what `win_start + chunk_len` would give
            #     whenever the read has more signal beyond `sample_end`.
            #   - wider: centre-crop, where the cropped width equals
            #     chunk_len exactly, so `win_start + chunk_len` is correct.
            # Named `win_*` (not `eff_*`) to avoid colliding with
            # `resolve_feature_window`'s unrelated `eff_start`/`eff_end` below.
            if requested <= chunk_len:
                win_start = sample_start
                win_end = sample_end
            else:
                crop = (requested - chunk_len) // 2
                win_start = sample_start + crop
                win_end = win_start + chunk_len
            focus_signal_pos_value = focus_sig_pos - win_start

            signal_chunk = np.zeros(chunk_len, dtype=np.float32)
            signal_residual_chunk = (
                np.zeros(chunk_len, dtype=np.float32) if self.signal_residual is not None else None
            )
            lo = max(win_start, 0)
            hi = min(win_end, self.num_samples)
            if hi > lo:
                off = lo - win_start
                n = min(hi - lo, chunk_len - off)
                if n > 0:
                    signal_chunk[off : off + n] = self.signal[lo : lo + n]
                    if self.signal_residual is not None and signal_residual_chunk is not None:
                        signal_residual_chunk[off : off + n] = self.signal_residual[lo : lo + n]
            chunk_sig_len = chunk_len
        else:
            chunk_len = signal_context[0] + signal_context[1]
            sig_start = focus_sig_pos - signal_context[0]
            sig_end = focus_sig_pos + signal_context[1]

            seq_to_sig_offset = 0
            if sig_start >= 0 and sig_end <= self.num_samples:
                signal_chunk = self.signal[sig_start:sig_end].copy()
                signal_residual_chunk = (
                    self.signal_residual[sig_start:sig_end].copy()
                    if self.signal_residual is not None
                    else None
                )
            else:
                signal_chunk = np.zeros(chunk_len, dtype=np.float32)
                signal_residual_chunk = (
                    np.zeros(chunk_len, dtype=np.float32)
                    if self.signal_residual is not None
                    else None
                )
                fill_st = 0
                fill_en = chunk_len
                if sig_start < 0:
                    fill_st = -sig_start
                    seq_to_sig_offset = -sig_start
                    sig_start = 0
                if sig_end > self.num_samples:
                    fill_en = self.num_samples - sig_start + seq_to_sig_offset
                    sig_end = self.num_samples
                if fill_en > fill_st:
                    signal_chunk[fill_st:fill_en] = self.signal[sig_start:sig_end]
                    if self.signal_residual is not None and signal_residual_chunk is not None:
                        signal_residual_chunk[fill_st:fill_en] = self.signal_residual[
                            sig_start:sig_end
                        ]

                # R4: in ref-anchored mode, the cropped self.signal drops
                # soft-clipped samples that may still exist in self.full_signal.
                # When the chunk window underflows past the aligned region, copy
                # those samples in instead of leaving zeros. Restricted to the
                # primary signal channel — self.signal_residual is only defined
                # for the refined aligned region, so it stays zero-padded.
                if recover_softclip_signal and self.full_signal is not None:
                    # signal_chunk[i] corresponds to absolute full_signal index
                    # (i + chunk_sig_start_in_cropped) + self.signal_offset, where
                    # chunk_sig_start_in_cropped is the original sig_start before
                    # the underflow clamping above. Reconstruct it from fill_st.
                    chunk_sig_start_in_cropped = sig_start - fill_st
                    abs_start = chunk_sig_start_in_cropped + self.signal_offset
                    full_len = len(self.full_signal)
                    # Left edge: fill [0, fill_st) from full_signal before the aligned region.
                    if fill_st > 0:
                        src_st = max(0, abs_start)
                        src_en = min(full_len, abs_start + fill_st)
                        if src_en > src_st:
                            dst_st = src_st - abs_start
                            dst_en = dst_st + (src_en - src_st)
                            signal_chunk[dst_st:dst_en] = self.full_signal[src_st:src_en].astype(
                                np.float32
                            )
                    # Right edge: fill [fill_en, chunk_len) from full_signal past the aligned region.
                    if fill_en < chunk_len:
                        src_st = max(0, abs_start + fill_en)
                        src_en = min(full_len, abs_start + chunk_len)
                        if src_en > src_st:
                            dst_st = src_st - abs_start
                            dst_en = dst_st + (src_en - src_st)
                            signal_chunk[dst_st:dst_en] = self.full_signal[src_st:src_en].astype(
                                np.float32
                            )
            chunk_sig_len = chunk_len
            focus_signal_pos_value = signal_context[0]

        # Extract k-mer sequence context with safe boundary handling
        kmer_start = base_idx - kmer_context
        kmer_end = base_idx + kmer_context + 1
        if kmer_start >= 0 and kmer_end <= self.num_bases:
            kmer_seq = self.sequence[kmer_start:kmer_end]
        else:
            parts = []
            for i in range(kmer_start, kmer_end):
                if 0 <= i < self.num_bases:
                    parts.append(self.sequence[i])
                else:
                    parts.append("N")
            kmer_seq = "".join(parts)

        # Extract dwell features with safe boundary handling
        eff_start, eff_end, dwell_width = resolve_feature_window(
            feature_start, feature_end, kmer_context
        )
        dwell_start = base_idx + eff_start
        dwell_end = base_idx + eff_end + 1
        safe_start = max(0, dwell_start)
        safe_end = min(len(self.dwells), dwell_end)
        if safe_start < safe_end:
            raw_dwell = self.dwells[safe_start:safe_end]
        else:
            raw_dwell = np.array([], dtype=self.dwells.dtype)
        if len(raw_dwell) < dwell_width:
            dwell_chunk = np.zeros(dwell_width, dtype=self.dwells.dtype)
            offset = safe_start - dwell_start
            dwell_chunk[offset : offset + len(raw_dwell)] = raw_dwell
        else:
            dwell_chunk = raw_dwell

        # Compile additional features (also with wider window, safe boundary).
        # Channel order was fixed once in __init__ -- see merge_feature_channels.
        # Sliced from the (F, num_bases) matrix __init__ stacked once, as one
        # 2D op across all F rows -- every row uses the same safe_start/
        # safe_end/dwell_width, so this was always doing identical work per
        # channel; looping and stacking after the fact just paid for it once
        # per channel per chunk instead of once per chunk (issue #275).
        n_features = self.feature_matrix.shape[0]
        if n_features == 0:
            feature_chunk = np.array([])
        else:
            if safe_start < safe_end:
                raw_feat = self.feature_matrix[:, safe_start:safe_end]
            else:
                raw_feat = np.zeros((n_features, 0), dtype=self.feature_matrix.dtype)
            if raw_feat.shape[1] < dwell_width:
                feature_chunk = np.zeros((n_features, dwell_width), dtype=self.feature_matrix.dtype)
                feat_offset = safe_start - dwell_start
                feature_chunk[:, feat_offset : feat_offset + raw_feat.shape[1]] = raw_feat
            else:
                feature_chunk = raw_feat

        # Build chunk-relative seq_to_sig_map for signal_kmer encoding.
        seq_start = int(np.searchsorted(self.seq_to_sig_map, sig_start, side="right") - 1)
        seq_end = int(np.searchsorted(self.seq_to_sig_map, sig_end, side="left"))
        seq_start = max(0, seq_start)
        seq_end = min(self.num_bases, seq_end)

        chunk_seq_to_sig = self.seq_to_sig_map[seq_start : seq_end + 1].copy()
        chunk_seq_to_sig -= sig_start - seq_to_sig_offset
        chunk_seq_to_sig[0] = 0
        chunk_seq_to_sig[-1] = chunk_sig_len
        chunk_seq_to_sig = chunk_seq_to_sig.astype(np.int64)

        # Extended sequence for signal_kmer encoding: core bases + kmer context
        kmer_before, kmer_after = DEFAULT_SIGNAL_KMER_CONTEXT
        ext_start = seq_start - kmer_before
        ext_end = seq_end + kmer_after
        if ext_start >= 0 and ext_end <= self.num_bases:
            sequence_with_kmer_context = self.sequence[ext_start:ext_end]
        else:
            parts = []
            for i in range(ext_start, ext_end):
                if 0 <= i < self.num_bases:
                    parts.append(self.sequence[i])
                else:
                    parts.append("N")
            sequence_with_kmer_context = "".join(parts)

        if mask_seq_side is not None:
            kmer_seq, sequence_with_kmer_context = _mask_focus_side(
                kmer_seq,
                sequence_with_kmer_context,
                kmer_context=kmer_context,
                core_focus_idx=base_idx - seq_start,
                kmer_before=kmer_before,
                side=mask_seq_side,
            )

        chunk_dict: dict[str, np.ndarray | str | int | None] = {
            "signal": signal_chunk,
            "sequence": kmer_seq,
            "dwell": dwell_chunk,
            "features": feature_chunk,
            "feature_start": eff_start,
            "feature_end": eff_end,
            "base_idx": base_idx,
            "label": self.labels[base_idx] if self.labels is not None else None,
            "seq_to_sig_map": chunk_seq_to_sig,
            "sequence_with_kmer_context": sequence_with_kmer_context,
        }
        if signal_residual_chunk is not None:
            chunk_dict["signal_residual"] = signal_residual_chunk
        # Store the focus base position within the signal chunk so that
        # downstream consumers (dataset.py) can crop asymmetrically without
        # assuming the focus is at center. In sample mode the focus is always
        # at signal_context[0] samples from the left edge, regardless of
        # boundary zero-padding; in base-defined mode it is resolved per
        # chunk above, since the window's width relative to signal_len (pad
        # vs. centre-crop) varies read to read.
        chunk_dict["focus_signal_pos"] = focus_signal_pos_value
        return chunk_dict


def extraction_sequence(
    *,
    anchor: str,
    basecall: str,
    reference_sequence: str | None,
    cigar_tuples: list[tuple[int, int]] | None,
) -> str:
    """The sequence chunks are cut from, and that focus bases index into.

    Must track ``build_leech_read``'s choice exactly: under
    ``anchor="reference"``, with both a reference sequence and a CIGAR to map
    through, chunks come from the aligned reference slice; otherwise from the
    basecall. Motif positions are indices into this string, so handing the
    searcher the other one returns coordinates in the wrong frame.

    Only observable with a ``BasecalledMotifSearcher`` --
    ``ReferenceMotifSearcher`` ignores the sequence argument and reads the
    alignment instead -- which is why two of the three inference paths could
    pass the basecall under ``anchor="reference"`` without anyone noticing.
    That combination is reachable: `predict` selects the searcher with
    ``mode="fasta" if reference_sequences else "bam"``, so a run without a
    reference FASTA gets the basecalled searcher while chunks are still cut in
    reference coordinates.
    """
    if anchor == "reference" and reference_sequence is not None and cigar_tuples is not None:
        return reference_sequence
    return basecall


def find_focus_bases(
    read_id: str,
    sequence: str,
    alignment: pysam.AlignedSegment | None,
    motif_config: MotifConfig,
    motif_searcher: MotifSearcher | None,
) -> list[MotifMatch]:
    """Which bases of a read contribute chunks.

    The single definition of that rule. Both prepare backends call it: the
    Python one from :func:`extract_training_chunks` with a built
    :class:`LeechRead`, the Rust one from
    ``leech.preparation.parallel._find_motif_positions`` with the ``ReadInfo``
    it has not yet turned into a read. They used to carry a copy each, which
    drifted — the Rust copy fell back to all-bases over the *query* sequence
    while this one used the reference (issue #185).

    Args:
        read_id: Read identifier, for the searcher's diagnostics.
        sequence: The sequence chunks are cut from — the aligned reference
            slice under ``anchor="reference"``, the basecall otherwise. Must
            be the same string both backends extract against, since the
            returned indices are positions in it.
        alignment: BAM alignment (or mock), required for reference search.
        motif_config: Motif, offset, and search mode.
        motif_searcher: Searcher strategy; required when a motif is set.

    Returns:
        One :class:`~leech.io.motif_search.MotifMatch` per focus base, with
        ``position`` indexing into ``sequence`` (motif offset already
        applied) and ``junction_indel``/``junction_mapped`` carrying the
        CIGAR-measured junction disruption at that position (issue #282).
        Out-of-range positions are possible and are the caller's to reject.
        No-motif (all-bases) mode has no junction to measure, so every match
        carries the ``MotifMatch`` defaults (``junction_indel=0``,
        ``junction_mapped=False``).
    """
    if motif_config.motif is None:
        # No motif: every base, minus the edges that cannot hold a k-mer.
        return [MotifMatch(pos) for pos in range(5, max(5, len(sequence) - 5))]

    if motif_searcher is None:
        raise ValueError("motif_searcher required when motif is provided")

    matches = motif_searcher.find_motif_positions(
        read_id=read_id,
        sequence=sequence,
        alignment=alignment,
        motif=motif_config.motif,
    )
    return [
        MotifMatch(m.position + motif_config.motif_offset, m.junction_indel, m.junction_mapped)
        for m in matches
    ]


def extract_training_chunks(
    leech_read: LeechRead,
    motif_config: MotifConfig,
    chunk_config: ChunkConfig,
    labeling: LabelConfig,
    motif_searcher: MotifSearcher | None = None,
) -> list[dict[str, np.ndarray | str | int | None]]:
    """
    Extract all training chunks from a read, optionally filtered by motif.

    Args:
        leech_read: LeechRead object
        motif_config: Motif configuration (motif, motif_offset)
        chunk_config: Chunk configuration (base_justify, feature_start/end, etc.)
        labeling: Label configuration (label, label_int)
        motif_searcher: MotifSearcher instance (required if motif is provided)

    Returns:
        List of chunk dictionaries
    """
    chunks: list[dict] = []

    # Per-read labeling + externally-anchored chunk: short-circuits both
    # the motif search and the default single-file label. Used by pipelines
    # that have already computed a region of interest (e.g. an adapter
    # region) per read — they pass a {read_id: (label_int, anchor_sample)}
    # map via LabelConfig.focus_map and get exactly one chunk per kept read
    # at that sample offset. Reads not in the map are skipped.
    if labeling.focus_map is not None:
        entry = labeling.focus_map.get(leech_read.read_id)
        if entry is None:
            return chunks
        focus_label_int, anchor_sample = entry
        leech_read.labels = np.full(leech_read.num_bases, focus_label_int, dtype=np.int64)
        # Convert signal-sample anchor to base index via the read's
        # move-table-derived map. `searchsorted(..., side="right") - 1`
        # gives the base whose signal window contains the anchor sample.
        base_idx = int(np.searchsorted(leech_read.seq_to_sig_map, anchor_sample, side="right") - 1)
        # Respect the same edge guard the all-bases fallback uses below so
        # chunks always have enough kmer context on both sides.
        base_idx = int(np.clip(base_idx, 5, leech_read.num_bases - 6))
        focus_bases = [MotifMatch(base_idx)]
    else:
        # Set numeric labels for all bases if provided (file-level mode).
        if labeling.label_int is not None:
            leech_read.labels = np.full(leech_read.num_bases, labeling.label_int, dtype=np.int64)

        # Find focus bases (either all or motif matches). Shared with the
        # Rust backend — see find_focus_bases.
        focus_bases = find_focus_bases(
            read_id=leech_read.read_id,
            # The extraction sequence: reference slice under anchor="reference".
            sequence=leech_read.sequence,
            # Alignment from metadata (may be None for basecalled search).
            alignment=leech_read.metadata.get("alignment"),
            motif_config=motif_config,
            motif_searcher=motif_searcher,
        )

    # Extract chunks
    cl_value = leech_read.metadata.get("cl_value")
    reference_name = leech_read.metadata.get("reference_name", "")
    for match in focus_bases:
        chunk = leech_read.get_chunk(match.position, config=chunk_config)
        if chunk is not None:
            chunk["read_id"] = leech_read.read_id
            # Rename numeric "label" from get_chunk() to "label_int"
            chunk["label_int"] = chunk.pop("label", None)
            # Add string label
            chunk["label"] = labeling.label
            # Add charging level (may be None)
            chunk["cl_value"] = cl_value
            # Add alignment reference name (e.g., tRNA isodecoder identity)
            chunk["reference_name"] = reference_name
            # Junction disruption at this focus base's motif span (issue
            # #282): mapped_len - len(motif) through the CIGAR, and whether
            # that measurement was possible at all. A sampling/abstention
            # field, never a model input -- see MotifMatch.
            chunk["junction_indel"] = match.junction_indel
            chunk["junction_mapped"] = match.junction_mapped
            chunks.append(chunk)

    return chunks
