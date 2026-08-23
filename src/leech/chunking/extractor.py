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
    DEFAULT_SIGNAL_CONTEXT,
    DEFAULT_SIGNAL_KMER_CONTEXT,
)
from leech.io.motif_search import MotifSearcher

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
    ) -> dict[str, np.ndarray | str | int | None] | None:
        """
        Extract a training chunk centered on a specific base.

        Args:
            base_idx: Index of the focus base
            config: Optional ChunkConfig that overrides individual params.
            signal_context: (left, right) signal padding around focus base
            kmer_context: Number of bases on each side for k-mer encoding
            base_justify: "center", "start", or "end"
            feature_start: Signed offset from focus for feature window start.
            feature_end: Signed offset from focus for feature window end (inclusive).
            recover_softclip_signal: When True and ``full_signal`` is set
                (ref-anchored mode), fill chunk-window samples that fall
                outside the aligned region with real soft-clipped signal
                instead of zeros. Off by default to preserve Remora-compatible
                behavior — see R4 in the coordinate audit.

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
                np.zeros(chunk_len, dtype=np.float32) if self.signal_residual is not None else None
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
                    signal_residual_chunk[fill_st:fill_en] = self.signal_residual[sig_start:sig_end]

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

        # Compile additional features (also with wider window, safe boundary)
        features = []
        for _feat_name, feat_array in {**self.dwell_features, **self.signal_features}.items():
            if safe_start < safe_end:
                raw_feat = feat_array[safe_start:safe_end]
            else:
                raw_feat = np.array([], dtype=feat_array.dtype)
            if len(raw_feat) < dwell_width:
                padded = np.zeros(dwell_width, dtype=feat_array.dtype)
                feat_offset = safe_start - dwell_start
                padded[feat_offset : feat_offset + len(raw_feat)] = raw_feat
                features.append(padded)
            else:
                features.append(raw_feat)

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

        chunk_dict: dict[str, np.ndarray | str | int | None] = {
            "signal": signal_chunk,
            "sequence": kmer_seq,
            "dwell": dwell_chunk,
            "features": np.stack(features, axis=0) if features else np.array([]),
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
        # assuming the focus is at center.  The focus is always at
        # signal_context[0] samples from the left edge, regardless of
        # boundary zero-padding.
        chunk_dict["focus_signal_pos"] = signal_context[0]
        return chunk_dict


def find_focus_bases(
    read_id: str,
    sequence: str,
    alignment: pysam.AlignedSegment | None,
    motif_config: MotifConfig,
    motif_searcher: MotifSearcher | None,
) -> list[int]:
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
        Focus base indices into ``sequence``, motif offset already applied.
        Out-of-range indices are possible and are the caller's to reject.
    """
    if motif_config.motif is None:
        # No motif: every base, minus the edges that cannot hold a k-mer.
        return list(range(5, max(5, len(sequence) - 5)))

    if motif_searcher is None:
        raise ValueError("motif_searcher required when motif is provided")

    positions = motif_searcher.find_motif_positions(
        read_id=read_id,
        sequence=sequence,
        alignment=alignment,
        motif=motif_config.motif,
    )
    return [pos + motif_config.motif_offset for pos in positions]


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
        focus_bases = [base_idx]
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
    for base_idx in focus_bases:
        chunk = leech_read.get_chunk(base_idx, config=chunk_config)
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
            chunks.append(chunk)

    return chunks
