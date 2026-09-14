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

from leech.constants import (
    DEFAULT_KMER_CONTEXT,
    DEFAULT_REFINE_HALF_BANDWIDTH,
    DEFAULT_SIGNAL_CONTEXT,
)

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
    refine_half_bandwidth: int = DEFAULT_REFINE_HALF_BANDWIDTH
    refine_kmer_center_idx: int = -1
    signal_refiner: SigMapRefiner | None = None
    # Path of the kmer level table the refiner was loaded from. Captured
    # so PrepareConfig.to_dict() can hash it as a provenance fingerprint;
    # the refiner object itself doesn't track its source path.
    kmer_table_path: Path | None = None
    compute_features: bool = True

    def __post_init__(self) -> None:
        # `refine_signal_map=False` with a `signal_refiner` attached is
        # refused rather than unified across backends (issue #265). Python's
        # `build_leech_read` computed the 3 k-mer residual feature rows off
        # `signal_refiner` alone, ignoring `refine_signal_map`; the Rust
        # dispatch only forwarded the kmer table -- and Rust's own
        # `process_read_signal` only computes the expected levels the
        # residuals need -- when `refine_signal_map` is also True. Unifying
        # on the looser (Python) rule would need Rust's internal gate lifted
        # too, not just the Python dispatch call site; refusing the
        # combination is the contained fix and costs nothing, because no
        # caller relies on it: every `SignalConfig` construction in this
        # codebase (CLI `data prepare`, single-model and bundle inference)
        # already sets `refine_signal_map=True` whenever it builds a
        # `signal_refiner`. Only direct `PrepareConfig`/`SignalConfig`
        # construction through the Python API could reach it.
        if not self.refine_signal_map and self.signal_refiner is not None:
            raise ValueError(
                "SignalConfig: refine_signal_map=False with a signal_refiner "
                "attached is not supported. The two prepare backends disagree "
                "on what this means (Python computes k-mer residual features "
                "off the refiner regardless of refine_signal_map; Rust "
                "requires refine_signal_map=True to compute them at all), so "
                "the combination silently produced different chunk widths "
                "per backend (#265). Either drop the refiner "
                "(signal_refiner=None) or set refine_signal_map=True."
            )


@dataclass
class MotifConfig:
    """How motif positions are found in reads."""

    motif: str | None = None
    motif_offset: int = 0
    motif_reference: str = "fasta"
    reference_sequences: dict[str, str] | None = None
    skip_motif_indels: bool = False
    #: Accept a reference motif only if it also maps cleanly to query
    #: coordinates. Under anchor="reference" that mapping is a quality gate
    #: whose result is discarded, so setting this False keeps reads whose motif
    #: basecalled badly without moving any chunk. See ReferenceMotifSearcher.
    require_query_mapping: bool = True


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
    # Base-defined signal window (issue #278): `(L, R)` base offsets around
    # the focus base, mutually exclusive with `signal_context`. When set,
    # `signal_len` (the fixed emitted width) must also be set -- see
    # `LeechRead.get_chunk` and `chunking.resolve_signal_context_bases`.
    signal_context_bases: tuple[int, int] | None = None
    signal_len: int | None = None

    def resolved_feature_window(self) -> tuple[int, int, int]:
        """``(start, end, width)`` of the feature window this config asks for.

        ``feature_start``/``feature_end`` are optional and mean "the k-mer
        window" when unset; resolving them anywhere other than
        ``resolve_feature_window`` is how issue #189 happened.
        """
        from leech.chunking import resolve_feature_window

        return resolve_feature_window(self.feature_start, self.feature_end, self.kmer_context)


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

        _feat_start, _feat_end, _feat_width = self.chunk.resolved_feature_window()

        return {
            "anchor": self.signal.anchor,
            "reverse_signal": self.signal.reverse_signal,
            "signal_norm": self.signal.norm_method,
            "refine_signal_map": self.signal.refine_signal_map,
            "refine_scale_iters": self.signal.refine_scale_iters,
            "refine_half_bandwidth": self.signal.refine_half_bandwidth,
            "refine_kmer_center_idx": self.signal.refine_kmer_center_idx,
            "kmer_table_sha256": kmer_table_sha256,
            "pa_mean": self.signal.pa_mean,
            "pa_stdev": self.signal.pa_stdev,
            "motif": self.motif.motif,
            "motif_offset": self.motif.motif_offset,
            "motif_reference": self.motif.motif_reference,
            "skip_motif_indels": self.motif.skip_motif_indels,
            "require_query_mapping": self.motif.require_query_mapping,
            "base_justify": self.chunk.base_justify,
            "feature_start": self.chunk.feature_start,
            "feature_end": self.chunk.feature_end,
            # The requested window above may be null (meaning "the k-mer
            # window"); these are what extraction actually used, so a corpus
            # states its own feature geometry instead of leaving the reader to
            # re-derive it (issue #189).
            "feature_start_resolved": _feat_start,
            "feature_end_resolved": _feat_end,
            "feature_width": _feat_width,
            "signal_context": list(self.chunk.signal_context),
            "signal_context_bases": (
                list(self.chunk.signal_context_bases)
                if self.chunk.signal_context_bases is not None
                else None
            ),
            "signal_len": self.chunk.signal_len,
            "kmer_context": self.chunk.kmer_context,
            "recover_softclip_signal": self.chunk.recover_softclip_signal,
            "label": self.labeling.label,
            "reference_fasta": str(self.reference_fasta) if self.reference_fasta else None,
        }


@dataclass
class OptimConfig:
    """Optimizer and gradient-handling knobs."""

    learning_rate: float = 0.001
    weight_decay: float = 0.0
    max_grad_norm: float = 0.0
    quantile_grad_clip: bool = False
    grad_accum_split: int = 1
    save_optim_every: int = 1


@dataclass
class SchedulerConfig:
    """LR schedule knobs."""

    scheduler_type: str = "none"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    warmup_epochs: int = 0


@dataclass
class AugmentConfig:
    """Signal/feature augmentation knobs applied to the training dataset."""

    jitter: float = 0.0
    scale_min: float = 1.0
    scale_max: float = 1.0
    time_mask_bases: int = 0
    time_mask_count: int = 1
    shift_max_bases: float = 0.0
    feature_noise_scale: float = 0.0
    time_stretch: tuple[float, float] = (1.0, 1.0)


@dataclass
class AuxHeadConfig:
    """Auxiliary head knobs: adversarial (confound) and CL regression."""

    adversarial_lambda: float = 0.0
    adversarial_anneal_epochs: int = 0
    cl_regression: bool = False
    cl_lambda: float = 1.0


@dataclass
class TrainConfig:
    """The training recipe: everything that describes *what* a run trains,

    as opposed to *where* (paths, device, worker count) or *how many ranks*
    (``--gpus``). One object threaded through ``train_model``/``Trainer``
    instead of the ~50-parameter lists that used to be hand-copied across
    ``train_model``, ``Trainer.__init__``, ``handle_train``,
    ``GridSearchConfig``, ``run_grid_point`` and ``_grid_point_worker`` (#270).

    ``train_model`` and ``Trainer`` both accept ``cfg=None`` and build one
    from their individual keyword arguments in that case, so every existing
    caller (tests included) that passes loose kwargs keeps working
    unchanged; ``cfg`` is the path new callers (``handle_train``, the grid
    search worker) use directly.
    """

    epochs: int = 50
    batch_size: int = 128
    early_stopping_patience: int = 10
    use_class_weights: bool = True
    pos_weight: float | None = None
    loss_type: str = "bce"
    focal_gamma: float = 2.0
    #: ``{source_group: flip_rate}`` for ``loss_type="noise_corrected_bce"``,
    #: parsed from ``--label-noise-rate`` by ``leech.losses.parse_label_noise_rate``.
    #: Unmapped groups get rate 0. Recorded verbatim in config.json.
    label_noise_rates: dict[str, float] | None = None
    # Asymmetric focal loss (--focal-neg-gamma, issue #280): None keeps the
    # symmetric loss bit-for-bit; see FocalBCEWithLogitsLoss for why.
    focal_neg_gamma: float | None = None
    label_smoothing: float = 0.0
    mixed_precision: bool = False
    checkpoint_metric: str = "auto"
    num_out: int = 1
    signal_mode: str = "both"
    # Per-channel feature standardization: compute corpus-wide mean/std once
    # and freeze them into the feature branch's first layer (issue #283).
    standardize_features: bool = False

    # Provenance / extraction geometry recorded in config.json, not used to
    # extract anything here -- see chunking.resolve_feature_window and
    # extraction_sequence for where these are authoritative.
    motif: str | None = None
    motif_offset: int = 0
    base_justify: str = "center"
    seq_encoding: str = "signal_kmer"
    signal_kmer_context: tuple[int, int] = (4, 4)
    allow_encoding_fallback: bool = True
    left_context: int | None = None
    right_context: int | None = None
    strict_window: bool = False

    # Sampling strategy (mutually exclusive; enforced in train_model)
    balance_groups: bool = False
    oversample_minority: bool = False
    #: Chunk metadata field to inverse-frequency weight sampling by, e.g.
    #: "junction_indel" to over-sample the disrupted-junction population
    #: (issue #282) or "source_group" (equivalent to balance_groups). Any
    #: field ChunkTable/a chunk dict exposes is valid.
    sample_weight_field: str | None = None
    label_map: dict[str, int] | None = None

    # Confound / adversarial provenance token (parsed in train_model)
    confound: str | None = None

    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    aux_head: AuxHeadConfig = field(default_factory=AuxHeadConfig)


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
