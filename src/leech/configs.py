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
    signal_refiner: SigMapRefiner | None = None
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


@dataclass
class LabelConfig:
    """Labels assigned to extracted chunks."""

    label: str | None = None
    label_int: int | None = None


@dataclass
class PrepareConfig:
    """Full config for data preparation (parallel or sequential)."""

    pod5_path: Path
    signal: SignalConfig = field(default_factory=SignalConfig)
    motif: MotifConfig = field(default_factory=MotifConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    labeling: LabelConfig = field(default_factory=LabelConfig)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict (excludes non-serializable fields)."""
        return {
            "anchor": self.signal.anchor,
            "reverse_signal": self.signal.reverse_signal,
            "signal_norm": self.signal.norm_method,
            "refine_signal_map": self.signal.refine_signal_map,
            "motif": self.motif.motif,
            "motif_offset": self.motif.motif_offset,
            "motif_reference": self.motif.motif_reference,
            "skip_motif_indels": self.motif.skip_motif_indels,
            "base_justify": self.chunk.base_justify,
            "feature_start": self.chunk.feature_start,
            "feature_end": self.chunk.feature_end,
            "signal_context": list(self.chunk.signal_context),
            "kmer_context": self.chunk.kmer_context,
            "label": self.labeling.label,
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
