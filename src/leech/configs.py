"""
Composable configuration dataclasses for prep/inference pipelines.

Replaces individual parameter threading with structured config objects.
Both the preparation and inference paths share leaf configs (SignalConfig,
MotifConfig, ChunkConfig), eliminating divergence bugs.

All dataclasses are standard ``@dataclass`` — picklable for multiprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from leech.constants import DEFAULT_KMER_CONTEXT, DEFAULT_SIGNAL_CONTEXT

if TYPE_CHECKING:
    from leech.signal_refine import SigMapRefiner


@dataclass
class SignalConfig:
    """How raw signal is processed into a LeechRead."""

    reverse_signal: bool = True
    anchor: str = "reference"
    norm_method: str = "median_mad"
    pa_mean: float | None = None
    pa_stdev: float | None = None
    refine_signal_map: bool = True
    refine_scale_iters: int = 2
    refine_half_bandwidth: int = 5
    refine_do_rough_rescale: bool = True
    refine_kmer_center_idx: int = -1
    signal_refiner: SigMapRefiner | None = None
    # Path of the kmer level table the refiner was loaded from. Captured
    # so PrepareConfig.to_dict() can hash it as a provenance fingerprint;
    # the refiner object itself doesn't track its source path.
    kmer_table_path: Path | None = None
    compute_features: bool = True


@dataclass
class MotifConfig:
    """How motif positions are found in reads."""

    motif: str | None = None
    motif_offset: int = 0
    motif_reference: str = "fasta"
    reference_sequences: dict[str, str] | None = None
    skip_motif_indels: bool = False


@dataclass
class ChunkConfig:
    """How training chunks are extracted from a LeechRead."""

    base_justify: str = "center"
    feature_start: int | None = None
    feature_end: int | None = None
    signal_context: tuple[int, int] = DEFAULT_SIGNAL_CONTEXT
    kmer_context: int = DEFAULT_KMER_CONTEXT
    # When True and a LeechRead has a stashed full pre-crop signal
    # (ref-anchored mode), fill chunk samples that extend past the aligned
    # region with real soft-clipped signal instead of zeros. Default False
    # preserves the Remora-compatible zero-pad behavior; see R4 in the
    # coordinate-positioning audit for why it's opt-in.
    recover_softclip_signal: bool = False


@dataclass
class LabelConfig:
    """Labels assigned to extracted chunks.

    Two labeling modes:

    - **File-level (default):** every read in the input POD5/BAM gets the
      same ``label_int`` / ``label``. This matches leech's historical
      one-class-per-input-file workflow.
    - **Per-read (focus_map):** a mapping ``{read_id: (label_int,
      anchor_sample)}`` selects a subset of reads, assigns each its own
      label, and anchors extraction at a caller-provided signal-sample
      offset (e.g. an adapter-region midpoint). Reads not in the map are
      skipped. Takes precedence over motif search when set — one chunk
      per read at the anchor position. Intended for downstream pipelines
      (like the 005 adapter-barcode classifier) where per-read labels
      come from an external source and chunks are centered on an
      externally-detected region, not a sequence motif.
    """

    label: str | None = None
    label_int: int | None = None
    focus_map: dict[str, tuple[int, int]] | None = None


@dataclass
class PrepareConfig:
    """Full config for data preparation (parallel or sequential)."""

    pod5_path: Path
    signal: SignalConfig = field(default_factory=SignalConfig)
    motif: MotifConfig = field(default_factory=MotifConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    labeling: LabelConfig = field(default_factory=LabelConfig)
    reference_fasta: Path | None = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict (excludes non-serializable fields)."""
        kmer_table_sha256: str | None = None
        if self.signal.refine_signal_map and self.signal.kmer_table_path is not None:
            from leech.data import compute_kmer_table_sha256

            kmer_table_sha256 = compute_kmer_table_sha256(self.signal.kmer_table_path)

        return {
            "anchor": self.signal.anchor,
            "reverse_signal": self.signal.reverse_signal,
            "signal_norm": self.signal.norm_method,
            "refine_signal_map": self.signal.refine_signal_map,
            "refine_scale_iters": self.signal.refine_scale_iters,
            "refine_half_bandwidth": self.signal.refine_half_bandwidth,
            "refine_do_rough_rescale": self.signal.refine_do_rough_rescale,
            "refine_kmer_center_idx": self.signal.refine_kmer_center_idx,
            "kmer_table_sha256": kmer_table_sha256,
            "pa_mean": self.signal.pa_mean,
            "pa_stdev": self.signal.pa_stdev,
            "motif": self.motif.motif,
            "motif_offset": self.motif.motif_offset,
            "motif_reference": self.motif.motif_reference,
            "skip_motif_indels": self.motif.skip_motif_indels,
            "base_justify": self.chunk.base_justify,
            "feature_start": self.chunk.feature_start,
            "feature_end": self.chunk.feature_end,
            "signal_context": list(self.chunk.signal_context),
            "kmer_context": self.chunk.kmer_context,
            "recover_softclip_signal": self.chunk.recover_softclip_signal,
            "label": self.labeling.label,
            "reference_fasta": str(self.reference_fasta) if self.reference_fasta else None,
        }


@dataclass
class InferenceConfig:
    """Full config for inference workers. Shares signal/motif/chunk with prep."""

    pod5_path: Path
    signal: SignalConfig = field(default_factory=SignalConfig)
    motif: MotifConfig = field(default_factory=MotifConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    # Inference-specific
    seq_encoding: str = "signal_kmer"
    signal_kmer_context: tuple[int, int] = (4, 4)
    signal_len: int = 400
    kmer_len: int = 11
    dwell_offset: int = 0
    wide_features: bool = False
    requires_features: bool = True
    signal_in_channels: int = 1
