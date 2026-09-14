"""Helper functions shared across inference submodules."""

import array
import json
import logging
import threading
from collections.abc import Callable, Hashable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pysam
import torch

from leech._rust_accel import (
    RUST_NORM_METHOD,
    make_kmer_levels,
    rust_supports_norm_method,
    rust_supports_softclip_recovery,
)
from leech.chunking import extraction_sequence, feature_window_from_metadata
from leech.constants import (
    BELOW_THRESHOLD_LABEL,
    DEFAULT_REFINE_HALF_BANDWIDTH,
    DEFAULT_SEQ_ENCODING_FALLBACK,
)
from leech.features import encode_signal_kmer, encode_signal_kmer_batch, sequence_to_int
from leech.model_loading import load_model_from_checkpoint
from leech.models import requires_features as _requires_features
from leech.models import wide_features as _wide_features
from leech.models.inference_wrapper import ModelInferenceWrapper, TracedModelWrapper
from leech.models.remora_compat import RemoraModelWrapper
from leech.preparation import encode_kmer

logger = logging.getLogger("leech.inference")


def is_multiclass(config: Mapping) -> bool:
    """Whether a model's config describes a categorical (softmax) output.

    One definition, for every place that decides how predictions are
    represented, tagged and calibrated: ``num_out > 1``. Before this,
    ``evaluation.py`` and ``commands/calibrate.py`` used ``> 2`` here while
    predict used ``> 1``, so a 2-output cross-entropy model got multiclass BAM
    tags (``aa``/``pn``/``pp``) from predict but binary Platt calibration that
    predict never applied to it (issue #269).

    This is a different question from ``Trainer``'s internal ``_num_out > 2``
    branches in the validation loop, which pick a *metric* (macro-F1 plus
    accuracy for 3+ classes vs. an AUROC-style treatment for a 1- or 2-output
    model) given an already-known loss type, and are left alone here.
    """
    return config.get("num_out", 1) > 1


def _write_prediction_tags(
    aln: pysam.AlignedSegment,
    predicted_aa: str,
    conf: float,
    class_names_str: str,
    probs: list[float],
    raw: bool,
    min_confidence: int,
    min_margin: int = 0,
    predicted_cl: float | None = None,
    junction_indel: int | None = None,
    junction_mapped: bool = False,
    abstain_on_junction_indel: bool = False,
) -> None:
    """Write prediction tags to a BAM alignment.

    Args:
        aln: pysam alignment to tag
        predicted_aa: predicted class label
        conf: max class probability (0.0-1.0); written to ``ac`` unchanged,
            including for reads that fail the thresholds below
        class_names_str: comma-separated class names for pn tag
        probs: full probability distribution
        raw: if True, write float tags; otherwise compact uint8
        min_confidence: threshold in 0-255 uint8 space
        min_margin: margin threshold in 0-255 uint8 space
        predicted_cl: predicted charging level in [0, 1] (None = no CL head)
        junction_indel: CIGAR-measured indel at this call's motif junction
            (issue #282), or ``None`` when no motif measurement was made.
            Written to the ``ji`` tag when available.
        junction_mapped: Whether ``junction_indel`` was actually measured
            (False means "unknown", not "exact") -- see ``MotifMatch``.
        abstain_on_junction_indel: When True, the margin threshold below is
            enforced only on chunks whose junction is disrupted (an unmapped
            junction or a nonzero indel) rather than on every chunk -- the
            compound "badly called junction AND marginal score" abstention
            rule this field exists for (charging-waveform-plan.md §13.12).
            Reads with an intact junction bypass the margin check entirely
            under this flag; ``min_confidence`` is unaffected either way.
    """
    sorted_probs = sorted(probs, reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
    margin_uint8 = int(min(255, max(0, round(margin * 255))))
    ac_uint8 = int(min(255, max(0, round(conf * 255))))

    if abstain_on_junction_indel:
        disrupted = not junction_mapped or (junction_indel is not None and junction_indel != 0)
        passed_margin = (margin_uint8 >= min_margin) if disrupted else True
    else:
        passed_margin = margin_uint8 >= min_margin
    passed_threshold = ac_uint8 >= min_confidence and passed_margin

    # `ac` always carries the winning class probability, whether or not the call
    # passed. Reporting 1-conf for filtered reads only makes sense when there are
    # exactly two classes; for N-way models it is not the probability of anything,
    # and it makes `ac` unfilterable because its meaning depends on the class tag.
    aln.set_tag("aa", predicted_aa if passed_threshold else BELOW_THRESHOLD_LABEL)

    if raw:
        aln.set_tag("ac", conf)
        aln.set_tag("am", margin)
    else:
        aln.set_tag("ac", ac_uint8, value_type="C")
        aln.set_tag("am", margin_uint8, value_type="C")

    # Junction disruption, so the abstention rule can be re-applied or audited
    # offline without re-running inference (issue #282). Omitted when no motif
    # measurement was ever attempted (no motif, or a Remora model).
    if junction_indel is not None:
        aln.set_tag("ji", int(junction_indel), value_type="i")
        aln.set_tag("jm", int(junction_mapped), value_type="C")

    aln.set_tag("pn", class_names_str)
    if raw:
        aln.set_tag("pp", probs)
    else:
        aln.set_tag(
            "pp",
            array.array("B", [int(min(255, max(0, round(p * 255)))) for p in probs]),
        )

    # Predicted charging level (CL regression head)
    if predicted_cl is not None:
        if raw:
            aln.set_tag("pc", predicted_cl)
        else:
            aln.set_tag("pc", int(min(255, max(0, round(predicted_cl * 255)))), value_type="C")


class InferenceConfigError(RuntimeError):
    """Raised when inference input shapes don't match the model's config."""


def _warn_if_kmer_table_drifted(config_sha256: str | None, live_table_path: Path) -> None:
    """Warn when the kmer level table on disk no longer matches the one the
    model was trained against.

    Models persist the table's SHA256 in their config (added by R3); inference
    always loads the table via ``leech.data.get_kmer_table()``, which can
    silently return a different file after a leech upgrade. We warn rather
    than hard-error because (a) a refresh of the bundled table may be
    intentional and (b) old models predate the field — config_sha256 is None.
    """
    if config_sha256 is None:
        return
    from leech.data import compute_kmer_table_sha256

    live_sha256 = compute_kmer_table_sha256(live_table_path)
    if live_sha256 != config_sha256:
        logger.warning(
            "Kmer level table at %s (sha256=%s) does not match the table "
            "the model was trained against (sha256=%s). Signal-map refinement "
            "may produce different base boundaries than at training time, "
            "which can degrade prediction quality. Pin the table version or "
            "retrain if this is unintentional.",
            live_table_path,
            live_sha256[:12],
            config_sha256[:12],
        )


def _check_config_consistency[T](
    param_name: str,
    cli_value: T,
    config_value: T | None,
    cli_default: T,
    *,
    explicit: bool | None = None,
) -> T:
    """Resolve inference param from config, erroring on CLI conflict.

    Logic:
    - config has value + CLI is default -> use config (normal auto-read)
    - config has value + CLI differs   -> raise InferenceConfigError
    - config is None/missing           -> use CLI value (old models without field)

    ``explicit``, when given, comes from click's own
    ``get_parameter_source(param_name) is COMMANDLINE`` rather than being
    inferred from ``cli_value != cli_default`` -- the inferred form cannot
    tell an explicit ``--motif-offset 0`` from "not passed" when the default
    is also 0, so that call silently lost to the config instead of raising
    on a real conflict. Callers with no click context (tests, programmatic
    use) omit it and keep the inferred heuristic.
    """
    if config_value is not None:
        is_explicit = explicit if explicit is not None else cli_value != cli_default
        if is_explicit and cli_value != config_value:
            raise InferenceConfigError(
                f"CLI --{param_name}={cli_value!r} conflicts with training config "
                f"{param_name}={config_value!r}. Inference must use the same "
                f"parameters the model was trained with."
            )
        return config_value
    return cli_value


@dataclass(frozen=True)
class InferenceSpec:
    """Everything predict needs to know about a model, resolved once.

    Replaces the config-resolution block that used to be pasted into
    ``run_inference`` and ``run_bundle_inference`` separately (motif/anchor/
    base_justify via ``_check_config_consistency``, refinement setup, the
    feature-window fallback chain, ``wide_features`` detection, the
    ``is_multiclass``/``seq_encoding`` default disagreements) -- issue #269.
    Both call :meth:`from_config` once and read every other field off the
    result rather than re-deriving it.
    """

    # Model geometry
    signal_len: int
    kmer_len: int
    signal_context: tuple[int, int]
    signal_context_bases: tuple[int, int] | None
    kmer_context: int
    dwell_offset: int
    seq_encoding: str
    signal_kmer_context: tuple[int, int]
    wide_features: bool
    requires_features: bool
    signal_in_channels: int
    feature_start: int | None
    feature_end: int | None

    # Motif / anchoring
    motif: str
    motif_offset: int
    anchor: str
    base_justify: str
    reference_fasta: Path | None
    skip_motif_indels: bool
    require_query_mapping: bool
    recover_softclip_signal: bool

    # Signal map refinement
    refine_signal_map: bool
    refine_half_bandwidth: int
    refine_scale_iters: int
    refine_kmer_center_idx: int

    # Output representation
    is_multiclass: bool
    num_out: int
    label_map: dict[str, int] | None
    calibration: dict | None
    cl_regression: bool
    dwell_template_table: str | None

    @classmethod
    def from_config(
        cls,
        config: Mapping,
        *,
        model_type: str = "",
        motif: str | None = None,
        motif_offset: int = 0,
        base_justify: str = "center",
        anchor: str = "reference",
        reference_fasta: Path | None = None,
        default_refine_scale_iters: int = 2,
        parameter_sources: Mapping[str, bool] | None = None,
        strict_model_type: bool = False,
        model_dwell_margin: Callable[[], int] | None = None,
    ) -> "InferenceSpec":
        """Resolve one :class:`InferenceSpec` from a model/bundle config dict.

        Args:
            config: A leech checkpoint's ``config.json``, a bundle's
                ``config``, or the dict :func:`load_model_auto` synthesizes
                for a Remora model. All three share this shape.
            model_type: The registry name (``ModelInferenceWrapper.model_type``
                or a bundle's ``model_name``), used only to resolve
                ``wide_features``/``requires_features`` structurally via
                :func:`leech.models.wide_features` /
                :func:`leech.models.requires_features`. An unresolvable name
                (e.g. a Remora model, or the empty default) falls back to
                ``False`` for both rather than raising -- unlike bundling,
                grid search and export, which read ``model_type`` from a real
                config and treat an unresolvable name as a bug worth raising.
            motif, motif_offset, base_justify, anchor, reference_fasta:
                CLI-provided values, checked against the config for conflicts
                via :func:`_check_config_consistency`.
            default_refine_scale_iters: The literal fallback for
                ``refine_scale_iters`` when the config lacks the key --
                Remora configs default to ``-1`` (no refinement DP pass),
                leech configs to ``2``. Callers must pass the right one; this
                function does not infer it from the config shape.
            parameter_sources: Optional ``{param_name: was_explicit}`` map
                from click's ``get_parameter_source``, passed through to
                :func:`_check_config_consistency` so an explicit CLI value
                that happens to equal the default is still checked against
                the config rather than assumed to be "not passed".
            strict_model_type: When ``True`` and ``model_type`` is truthy but
                not a recognized registry name, raise ``KeyError`` instead of
                falling back to ``False`` for ``wide_features``/
                ``requires_features``. Bundle callers pass this: a bundle's
                ``model_type`` is read back from its own saved metadata, so
                an unresolvable name means the bundle is stale, renamed out
                from under it, or corrupted -- a real bug worth raising
                loudly rather than silently guessing its feature-window
                convention. Single-model callers leave this ``False``: a
                Remora model has no registry name at all (``model_type`` is
                the empty-string default), which is expected, not an error.
            model_dwell_margin: Optional zero-arg callable returning a wide-
                feature model's real ``dwell_margin`` attribute (e.g.
                ``lambda: getattr(model_wrapper.model, "dwell_margin", 0)``),
                for the rare fallback below. ``dwell_margin`` is a model
                constructor default, never written into ``config.json``, so
                reading it needs the model itself -- lazy because building
                one (bundle.py's caller has to ``_instantiate_model(config)``)
                is wasted work on every call that never reaches this branch.
        """
        sources = parameter_sources or {}

        motif = _check_config_consistency(
            "motif", motif, config.get("motif"), None, explicit=sources.get("motif")
        )
        motif_offset = _check_config_consistency(
            "motif-offset",
            motif_offset,
            config.get("motif_offset"),
            0,
            explicit=sources.get("motif_offset"),
        )
        if motif is None:
            raise InferenceConfigError(
                "motif is None after auto-read from config. Either pass --motif on "
                "the CLI or ensure the config contains a non-null 'motif' field. "
                "Without a motif, inference predicts at every position, producing noise."
            )

        base_justify = _check_config_consistency(
            "base-justify",
            base_justify,
            config.get("base_justify"),
            "center",
            explicit=sources.get("base_justify"),
        )
        anchor = _check_config_consistency(
            "anchor", anchor, config.get("anchor"), "reference", explicit=sources.get("anchor")
        )

        resolved_reference_fasta = reference_fasta
        if resolved_reference_fasta is None:
            cfg_ref = config.get("reference_fasta")
            if cfg_ref is not None:
                cfg_path = Path(cfg_ref)
                if cfg_path.exists():
                    resolved_reference_fasta = cfg_path
                else:
                    logger.warning(
                        f"reference_fasta from config ({cfg_ref}) not found; "
                        f"pass --reference-fasta explicitly"
                    )

        signal_len = int(config.get("signal_len", 100))
        kmer_len = int(config.get("kmer_len", 9))
        kmer_context = kmer_len // 2
        left_ctx = config.get("left_context")
        right_ctx = config.get("right_context")
        signal_context = (
            (int(left_ctx), int(right_ctx))
            if left_ctx is not None and right_ctx is not None
            else (signal_len // 2, signal_len // 2)
        )
        # Base-defined signal window (issue #278): recorded by `model train`
        # from the `data prepare` sidecar, `None` for a sample-context model.
        # `signal_len` above already holds the model's actual trained width
        # either way, so predict re-derives the identical window via
        # `LeechRead.get_chunk`'s `signal_context_bases`/`signal_len` pair.
        _scb = config.get("signal_context_bases")
        signal_context_bases = (int(_scb[0]), int(_scb[1])) if _scb is not None else None

        seq_encoding = config.get("seq_encoding", DEFAULT_SEQ_ENCODING_FALLBACK)
        signal_kmer_context = tuple(config.get("signal_kmer_context", (4, 4)))
        dwell_offset = int(config.get("dwell_offset", 0))

        try:
            wide_features = _wide_features(model_type)
        except KeyError as e:
            if strict_model_type and model_type:
                raise KeyError(
                    f"Model architecture '{model_type}' is not a recognized "
                    f"model (renamed, removed, or a corrupted checkpoint/bundle). "
                    f"Cannot determine its feature-window convention."
                ) from e
            wide_features = False
        feature_start, feature_end = feature_window_from_metadata(config, kmer_context)
        if wide_features and feature_start is None and feature_end is None:
            model_margin = model_dwell_margin() if model_dwell_margin is not None else 0
            if model_margin:
                feature_start = -(kmer_context + model_margin)
                feature_end = kmer_context + model_margin
                logger.warning(
                    f"Config missing feature_start/end, "
                    f"falling back to model default margin: {model_margin}"
                )

        num_out = int(config.get("num_out", 1))
        label_map = config.get("label_map")

        signal_in_channels = int(config.get("signal_in_channels", 1))
        try:
            requires_features = _requires_features(model_type)
        except KeyError:
            requires_features = False

        refine_signal_map = bool(config.get("refine_signal_map", True) or signal_in_channels > 1)
        refine_scale_iters = int(config.get("refine_scale_iters", default_refine_scale_iters))

        return cls(
            signal_len=signal_len,
            kmer_len=kmer_len,
            signal_context=signal_context,
            signal_context_bases=signal_context_bases,
            kmer_context=kmer_context,
            dwell_offset=dwell_offset,
            seq_encoding=seq_encoding,
            signal_kmer_context=signal_kmer_context,
            wide_features=wide_features,
            requires_features=requires_features,
            signal_in_channels=signal_in_channels,
            feature_start=feature_start,
            feature_end=feature_end,
            motif=motif,
            motif_offset=motif_offset,
            anchor=anchor,
            base_justify=base_justify,
            reference_fasta=resolved_reference_fasta,
            skip_motif_indels=bool(config.get("skip_motif_indels", False)),
            require_query_mapping=bool(config.get("require_query_mapping", True)),
            recover_softclip_signal=bool(config.get("recover_softclip_signal", False)),
            refine_signal_map=refine_signal_map,
            refine_half_bandwidth=int(
                config.get("refine_half_bandwidth", DEFAULT_REFINE_HALF_BANDWIDTH)
            ),
            refine_scale_iters=refine_scale_iters,
            refine_kmer_center_idx=int(config.get("refine_kmer_center_idx", -1)),
            is_multiclass=num_out > 1,
            num_out=num_out,
            label_map=label_map,
            calibration=config.get("calibration") if num_out > 1 else None,
            cl_regression=bool(config.get("cl_regression", False)),
            dwell_template_table=config.get("dwell_template_table") or None,
        )


def validate_inference_shapes(
    signal: np.ndarray,
    features: np.ndarray | None,
    config: dict,
) -> None:
    """Validate that inference input shapes match the model's expected config.

    Checks signal channels, feature count, and signal length. Call once on the
    first chunk to catch mismatches early instead of silently producing garbage.

    Args:
        signal: Signal array -- 1D (single channel) or 2D (channels, signal_len).
        features: Feature array (num_features, kmer_len), or None if model has no feature branch.
        config: Model config dict with signal_in_channels, num_features, signal_len.

    Raises:
        InferenceConfigError: On any shape mismatch.
    """
    expected_channels = config.get("signal_in_channels", 1)
    if signal.ndim == 1:
        actual_channels = 1
    elif signal.ndim == 2:
        actual_channels = signal.shape[0]
    else:
        raise InferenceConfigError(f"Signal has unexpected ndim={signal.ndim}; expected 1D or 2D")

    if actual_channels != expected_channels:
        raise InferenceConfigError(
            f"Signal has {actual_channels} channel(s), but model expects "
            f"signal_in_channels={expected_channels}. "
            f"This usually means signal_residual is missing (model trained with 2-channel input)."
        )

    expected_signal_len = config.get("signal_len")
    if expected_signal_len is not None:
        actual_signal_len = signal.shape[-1]
        if actual_signal_len != expected_signal_len:
            raise InferenceConfigError(
                f"Signal length {actual_signal_len} != expected {expected_signal_len}"
            )

    if features is not None:
        expected_features = config.get("num_features")
        if expected_features is not None:
            actual_features = features.shape[0]
            if actual_features != expected_features:
                raise InferenceConfigError(
                    f"Feature array has {actual_features} features, "
                    f"but model expects num_features={expected_features}"
                )


def prepare_inference_features(
    features: np.ndarray | None,
    *,
    kmer_len: int,
    feature_start: int | None,
    dwell_offset: int = 0,
    wide_features: bool = False,
    dwell_templates: np.ndarray | None = None,
    template_min_pos: int = 0,
) -> np.ndarray | None:
    """Put a chunk's feature array into the shape the model was trained on.

    The single definition of that transform for inference. It mirrors
    ``ChunkDataset._prepare_features`` in ``dataset.py``, which is what ran at
    training time, and the two must not drift: the model sees whatever comes
    out of here.

    Two things happen, in this order, because that is the order training used:

    1. Dwell template channels are appended, keyed to the *stored* feature
       window (``feat_start`` is the coordinate of column 0). Appending after a
       narrowing would key them to the wrong columns.
    2. The array is narrowed to the model's k-mer window and shifted by
       ``dwell_offset``.

    Before this existed there were four copies of step 2 and none of them
    agreed with training on all of it. The Rust extraction paths in ``single``
    and ``bundle`` skipped it entirely, so a model trained on a wide feature
    window was handed the full window at predict time and ``dwell_offset`` did
    nothing; the bundle's Python path did step 2 before step 1.

    Args:
        features: ``(num_features, feat_width)`` array, or None.
        kmer_len: The model's k-mer width; the returned window is this wide.
        feature_start: Signed offset of column 0 from the focus base. ``None``
            means the k-mer window, i.e. ``-(kmer_len // 2)``.
        dwell_offset: Shift the window toward the 3' end, in bases.
        wide_features: Model consumes the full window; skip step 2.
        dwell_templates: ``(n_aa, n_positions)`` template table, or None.
        template_min_pos: First position covered by ``dwell_templates``.

    Returns:
        The prepared array, or ``features`` unchanged when there is nothing to
        do (None, empty, or already at the target width).

    Raises:
        InferenceConfigError: The requested window falls outside the stored
            one. Training raises for the same condition rather than sliding the
            window to fit, because a window that does not fit means the model
            config and the corpus disagree about the feature geometry.
    """
    if features is None or features.size == 0:
        return features

    kmer_context = kmer_len // 2
    start_offset = feature_start if feature_start is not None else -kmer_context

    if dwell_templates is not None:
        from leech.dataset import append_dwell_template_channels

        features = append_dwell_template_channels(
            features,
            feat_start=start_offset,
            dwell_templates=dwell_templates,
            template_min_pos=template_min_pos,
        )

    if wide_features or features.shape[1] <= kmer_len:
        return features

    start = (-kmer_context - start_offset) + dwell_offset
    if start < 0 or start + kmer_len > features.shape[1]:
        raise InferenceConfigError(
            f"feature window [{start}, {start + kmer_len}) does not fit the "
            f"stored width {features.shape[1]} "
            f"(kmer_len={kmer_len}, feature_start={start_offset}, "
            f"dwell_offset={dwell_offset})"
        )
    return features[:, start : start + kmer_len]


def _extract_remora_metadata(model_path: Path) -> dict:
    """Extract metadata from a Remora TorchScript model's embedded meta.txt."""
    import json as _json

    extra_files = {"meta.txt": ""}
    torch.jit.load(str(model_path), map_location="cpu", _extra_files=extra_files)
    meta_str = extra_files.get("meta.txt", "")
    if not meta_str:
        return {}
    raw = _json.loads(meta_str)

    # Derive chunk_context / chunk_len
    if "chunk_context" not in raw:
        raw["chunk_context"] = (
            int(raw.get("chunk_context_0", 50)),
            int(raw.get("chunk_context_1", 50)),
        )
    raw["chunk_len"] = sum(raw["chunk_context"])

    # Derive kmer_context_bases / kmer_len
    if "kmer_context_bases" not in raw:
        raw["kmer_context_bases"] = (
            int(raw.get("kmer_context_bases_0", 4)),
            int(raw.get("kmer_context_bases_1", 4)),
        )
    raw["kmer_len"] = sum(raw["kmer_context_bases"]) + 1

    # Derive motif
    if "num_motifs" in raw:
        num = int(raw["num_motifs"])
        motifs = []
        for i in range(num):
            motifs.append((raw[f"motif_{i}"], int(raw[f"motif_offset_{i}"])))
        raw["motifs"] = motifs
        raw["motif"] = motifs[0]
    elif "motif" in raw and "motif_offset" in raw:
        raw["motif"] = (raw["motif"], int(raw["motif_offset"]))

    # Derive signal refinement parameters
    if "refine_half_bandwidth" in raw:
        raw["refine_signal_map"] = True
        raw["refine_half_bandwidth"] = int(raw["refine_half_bandwidth"])
        raw["refine_scale_iters"] = int(raw.get("refine_scale_iters", 2))
        raw["refine_kmer_center_idx"] = int(raw.get("refine_kmer_center_idx", -1))

    return raw


def load_model_auto(
    model_path: Path, device: str = "cpu"
) -> tuple[ModelInferenceWrapper | TracedModelWrapper | RemoraModelWrapper, dict]:
    """
    Load leech model (directory), leech TorchScript, or Remora TorchScript (.pt file).

    Auto-detects format:
    - Directory with config.json -> leech checkpoint
    - .pt file with leech_meta.txt -> leech torch.export or TorchScript export
    - .pt file with meta.txt -> Remora TorchScript model

    Args:
        model_path: Path to model directory or .pt file
        device: Device to load model on

    Returns:
        Tuple of (wrapper, config_dict)
    """
    path = Path(model_path)
    if path.is_dir():
        model, config = load_model_from_checkpoint(path, device=device)
        model_type = config["model_name"]
        return ModelInferenceWrapper(model, model_type), config
    elif path.suffix == ".pt" and not (path.parent / "config.json").exists():
        # Try torch.export format first (PyTorch 2+)
        extra = {"leech_meta.txt": ""}
        try:
            ep = torch.export.load(str(path), extra_files=extra)
            if extra.get("leech_meta.txt", ""):
                config = json.loads(extra["leech_meta.txt"])
                model_name = config.get("model_name", "")
                requires_features = bool(model_name) and _requires_features(model_name)
                loaded_model = ep.module().to(device)
                wrapper = TracedModelWrapper(loaded_model, requires_features=requires_features)
                logger.info(
                    f"Leech exported model: {model_name}, "
                    f"signal_len={config.get('signal_len')}, kmer_len={config.get('kmer_len')}"
                )
                return wrapper, config
        except Exception as e:
            logger.debug("torch.export load failed, trying TorchScript: %s", e)

        # Try legacy TorchScript format
        extra = {"leech_meta.txt": ""}
        try:
            traced = torch.jit.load(str(path), map_location=device, _extra_files=extra)
        except Exception as e:
            logger.debug("TorchScript load failed: %s", e)
            traced = None

        if traced is not None and extra.get("leech_meta.txt", ""):
            config = json.loads(extra["leech_meta.txt"])
            model_name = config.get("model_name", "")
            # model_name is read back from the TorchScript file's own sidecar,
            # so it should always be a real registry name; unlike the
            # torch.export attempt above (inside a broad except that falls
            # through to try this format instead), there is no further format
            # to fall back to here, so a bad name must fail loudly rather than
            # silently guess the model's feature-window shape.
            try:
                requires_features = bool(model_name) and _requires_features(model_name)
            except KeyError as e:
                raise KeyError(
                    f"TorchScript model '{model_name}' ({path}) is not a "
                    f"recognized model architecture (renamed, removed, or a "
                    f"corrupted sidecar). Cannot determine whether it takes a "
                    f"features input."
                ) from e
            wrapper = TracedModelWrapper(traced, requires_features=requires_features)
            logger.info(
                f"Leech TorchScript model: {model_name}, "
                f"signal_len={config.get('signal_len')}, kmer_len={config.get('kmer_len')}"
            )
            return wrapper, config

        # Fall back to Remora TorchScript
        wrapper = RemoraModelWrapper(path, device=device)

        # Extract metadata from the model
        remora_meta = _extract_remora_metadata(path)

        kmer_context = remora_meta.get("kmer_context_bases", (4, 4))
        if isinstance(kmer_context, list):
            kmer_context = tuple(kmer_context)
        chunk_context = remora_meta.get("chunk_context", (50, 50))

        config = {
            "seq_encoding": "signal_kmer",
            "signal_kmer_context": list(kmer_context),
            "is_remora": True,
            "signal_len": sum(chunk_context),
            "kmer_len": sum(kmer_context) + 1,
            "chunk_context": list(chunk_context),
        }

        # Pass through motif if available
        motif_info = remora_meta.get("motif")
        if isinstance(motif_info, (list, tuple)) and len(motif_info) == 2:
            config["motif"] = motif_info[0]
            config["motif_offset"] = int(motif_info[1])

        # Pass through signal refinement parameters
        if remora_meta.get("refine_signal_map", False):
            config["refine_signal_map"] = True
            config["refine_half_bandwidth"] = remora_meta.get(
                "refine_half_bandwidth", DEFAULT_REFINE_HALF_BANDWIDTH
            )
            config["refine_scale_iters"] = remora_meta.get("refine_scale_iters", -1)
            config["refine_kmer_center_idx"] = remora_meta.get("refine_kmer_center_idx", -1)

        logger.info(
            f"Remora model: signal_len={config['signal_len']}, kmer_len={config['kmer_len']}"
        )

        return wrapper, config
    else:
        raise ValueError(f"Cannot auto-detect model format for {model_path}")


def _encode_sequence_for_inference(
    chunk: dict,
    seq_encoding: str,
    signal_len: int,
    signal_kmer_context: tuple[int, int] = (4, 4),
) -> torch.Tensor | None:
    """Encode sequence from a chunk for inference.

    Args:
        chunk: Chunk dict from LeechRead.get_chunk()
        seq_encoding: "base_onehot" or "signal_kmer"
        signal_len: Target signal length
        signal_kmer_context: Kmer context for signal_kmer encoding

    Returns:
        Encoded sequence tensor, or None if signal_kmer encoding is required
        but the chunk lacks the necessary fields.
    """
    if seq_encoding == "signal_kmer":
        seq_ctx = chunk.get("sequence_with_kmer_context")
        seq_to_sig = chunk.get("seq_to_sig_map")
        if seq_ctx is not None and seq_to_sig is not None:
            seq_ints = sequence_to_int(seq_ctx)
            enc = encode_signal_kmer(seq_ints, seq_to_sig, signal_len, signal_kmer_context)
            return torch.from_numpy(enc)
        else:
            # Cannot fall back to base_onehot for signal_kmer models
            logger.debug("Chunk lacks signal_kmer fields, skipping")
            return None
    else:
        return encode_kmer(chunk["sequence"])


def chunks_for_read(
    leech_read,
    positions: list[int],
    *,
    chunk_config,
    signal_len: int,
    seq_encoding: str,
    signal_kmer_context: tuple[int, int],
    requires_features: bool,
    kmer_len: int,
    dwell_offset: int,
    wide_features: bool,
    dwell_templates: np.ndarray | None = None,
    template_min_pos: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None, int]]:
    """Extract ready-to-batch ``(signal, sequence, features, base_idx)`` tuples
    from an already-built :class:`~leech.chunking.LeechRead` at each of
    ``positions``.

    The one place the per-chunk array-building triplet runs --
    :func:`prepare_signal_channels`, :func:`_encode_sequence_for_inference`,
    :func:`prepare_inference_features` -- for the Python serial extraction
    path. ``run_inference``'s sequential path and ``run_bundle_inference``'s
    serial path each carried their own copy of this loop, and only the bundle
    copy applied dwell templates; the bundle copy also built its signal array
    by hand rather than through ``prepare_signal_channels``, so it never
    padded/cropped ``signal`` to ``signal_len`` the way every other extraction
    path does (issue #268 -- found while unifying, not by design).

    Building the ``LeechRead`` itself, and finding ``positions`` on it, stays
    with the caller: a ``pysam.AlignedSegment`` and a ``ReadInfo`` differ too
    much above this point (motif search takes a real alignment object one
    caller has and the other reconstructs a mock of) for a shared protocol to
    buy anything; this is the part that was actually duplicated array-copy
    logic rather than read-shape plumbing.

    ``signal_kmer`` is batched once per read rather than once per chunk
    (issue #260): every chunk's raw ``sequence_with_kmer_context``/
    ``seq_to_sig_map`` is collected first and run through
    :func:`~leech.features.encode_signal_kmer_batch` in one call, instead of
    one pyo3 call per chunk via :func:`_encode_sequence_for_inference`. Both
    ``run_inference`` and ``run_bundle_inference`` get this from being routed
    through this one function -- before #268 unified them, only the
    ``run_inference`` copy of this loop had been given the batched path.
    """
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, int]] = []

    if seq_encoding != "signal_kmer":
        for base_idx in positions:
            chunk = leech_read.get_chunk(base_idx, config=chunk_config)
            if chunk is None:
                continue

            sig = prepare_signal_channels(chunk, signal_len)

            seq_enc = _encode_sequence_for_inference(
                chunk, seq_encoding, signal_len, signal_kmer_context
            )
            if seq_enc is None:
                continue
            seq_arr = seq_enc.numpy() if isinstance(seq_enc, torch.Tensor) else seq_enc

            feat = None
            if requires_features:
                features_array = chunk["features"]
                assert isinstance(features_array, np.ndarray)
                feat = prepare_inference_features(
                    features_array.astype(np.float32),
                    kmer_len=kmer_len,
                    feature_start=chunk.get("feature_start"),
                    dwell_offset=dwell_offset,
                    wide_features=wide_features,
                    dwell_templates=dwell_templates,
                    template_min_pos=template_min_pos,
                )

            results.append((sig, seq_arr, feat, base_idx))
        return results

    # signal_kmer: collect every chunk's raw inputs first, batch-encode once.
    pending: list[tuple[np.ndarray, np.ndarray | None, int, np.ndarray, np.ndarray]] = []
    for base_idx in positions:
        chunk = leech_read.get_chunk(base_idx, config=chunk_config)
        if chunk is None:
            continue

        seq_ctx = chunk.get("sequence_with_kmer_context")
        seq_to_sig = chunk.get("seq_to_sig_map")
        if seq_ctx is None or seq_to_sig is None:
            continue

        sig = prepare_signal_channels(chunk, signal_len)

        feat = None
        if requires_features:
            features_array = chunk["features"]
            assert isinstance(features_array, np.ndarray)
            feat = prepare_inference_features(
                features_array.astype(np.float32),
                kmer_len=kmer_len,
                feature_start=chunk.get("feature_start"),
                dwell_offset=dwell_offset,
                wide_features=wide_features,
                dwell_templates=dwell_templates,
                template_min_pos=template_min_pos,
            )

        pending.append((sig, feat, base_idx, sequence_to_int(seq_ctx), seq_to_sig))

    if pending:
        n = len(pending)
        max_si = max(p[3].shape[0] for p in pending)
        max_s2 = max(p[4].shape[0] for p in pending)
        padded_seq_ints = np.full((n, max_si), -1, dtype=np.int8)
        padded_s2s = np.full((n, max_s2), signal_len, dtype=np.int32)
        for i, (_, _, _, seq_ints, s2s) in enumerate(pending):
            padded_seq_ints[i, : seq_ints.shape[0]] = seq_ints
            padded_s2s[i, : s2s.shape[0]] = s2s
        batch_enc = encode_signal_kmer_batch(
            padded_seq_ints, padded_s2s, signal_len, signal_kmer_context
        )
        for i, (sig, feat, base_idx, _, _) in enumerate(pending):
            results.append((sig, batch_enc[i], feat, base_idx))

    return results


def _unpack_multiclass_pred(
    pred: tuple,
) -> tuple[int, int, float, list[float], float | None, int | None, bool]:
    """Unpack one ``pending[read_id]`` entry from ``_run_batch_multiclass``.

    Three historical widths, oldest first: 4-tuple (no CL head), 5-tuple
    (+ CL prediction), 7-tuple (current, + junction_indel/junction_mapped --
    issue #282). Centralized so the BAM-tag and TSV writers -- the two
    consumers of this shape -- decode it identically.
    """
    if len(pred) == 7:
        return pred
    if len(pred) == 5:
        base_idx, cls_idx, conf, all_probs, cl_pred = pred
        return base_idx, cls_idx, conf, all_probs, cl_pred, None, False
    base_idx, cls_idx, conf, all_probs = pred
    return base_idx, cls_idx, conf, all_probs, None, None, False


def _write_mega_batch_predictions(
    aln_batch: list[pysam.AlignedSegment],
    pending: dict[str, list],
    bam_out: pysam.AlignmentFile,
    is_multiclass: bool,
    int_to_label: dict[int, str] | None,
    class_names_str: str | None,
    raw: bool,
    min_confidence: int,
    min_margin: int,
    abstain_on_junction_indel: bool = False,
) -> int:
    """Write predictions for a mega-batch of alignments. Returns prediction count."""
    n_preds = 0
    if is_multiclass:
        for aln in aln_batch:
            preds = pending.get(aln.query_name)
            if preds:
                _, cls_idx, conf, all_probs, cl_pred, junction_indel, junction_mapped = (
                    _unpack_multiclass_pred(preds[0])
                )
                if int_to_label:
                    predicted_aa = int_to_label.get(cls_idx, str(cls_idx))
                else:
                    predicted_aa = str(cls_idx)
                _write_prediction_tags(
                    aln,
                    predicted_aa,
                    conf,
                    class_names_str,
                    all_probs,
                    raw,
                    min_confidence,
                    min_margin,
                    predicted_cl=cl_pred,
                    junction_indel=junction_indel,
                    junction_mapped=junction_mapped,
                    abstain_on_junction_indel=abstain_on_junction_indel,
                )
                n_preds += 1
            bam_out.write(aln)
    else:
        for aln in aln_batch:
            preds = pending.get(aln.query_name)
            if preds:
                preds.sort(key=lambda x: x[0])
                positions_list = [int(p[0]) for p in preds]
                ml_scores = [int(min(255, max(0, p[1] * 255))) for p in preds]
                aln.set_tag("MP", array.array("i", positions_list))
                aln.set_tag("ML", array.array("B", ml_scores))
                n_preds += 1
            bam_out.write(aln)
    return n_preds


class GpuBatchRunner:
    """Double-buffered, single-worker async submission of one scoring function.

    The pattern ``run_inference``'s and ``run_bundle_inference``'s serial
    paths each hand-rolled: a dedicated single-thread executor runs
    ``score_fn`` on one batch while the caller's extraction loop keeps filling
    the next one. Submitting a new batch waits for the previous one to finish
    first, so at most one is ever in flight — the ``max_workers=1`` GPU
    executor invariant PR #253 established, preserved here rather than
    reintroduced independently on the bundle path (issue #268).

    ``score_fn(signals, sequences, features, meta)`` is the same shape
    :class:`BatchAccumulator`'s flush callback expects, so a bound
    :meth:`submit` is a drop-in ``flush_fn``::

        runner = GpuBatchRunner(score_fn)
        accumulator = BatchAccumulator(batch_size, runner.submit)
        ...
        accumulator.flush()
        runner.drain()
        runner.shutdown()
    """

    __slots__ = ("_executor", "_future", "score_fn")

    def __init__(self, score_fn: Callable[[list, list, list, list], None]):
        self.score_fn = score_fn
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None

    def submit(self, signals: list, sequences: list, features: list, meta: list) -> None:
        """Wait for any in-flight batch, then submit this one (non-blocking)."""
        if self._future is not None:
            self._future.result()
        self._future = self._executor.submit(self.score_fn, signals, sequences, features, meta)

    def drain(self) -> None:
        """Block until the most recently submitted batch has finished."""
        if self._future is not None:
            self._future.result()
            self._future = None

    def shutdown(self) -> None:
        """Wait for the executor's thread to exit.

        Always call after :meth:`drain` — every batch is done by then, so
        this costs nothing, but a caller that goes on to fork (an ``mp.Pool``
        on the parallel path) must not leave the thread alive across it: a
        fork inherits a running thread's memory but not the thread itself, so
        any lock it held never releases (see ``run_inference``'s own note on
        this at its ``_extract_pool.shutdown(wait=True)`` call).
        """
        self._executor.shutdown(wait=True)


class BatchAccumulator:
    """Four parallel chunk buffers, a size check, and a flush — in one place.

    Every extraction path in :mod:`leech.inference` produces the same four
    parallel streams (signal, sequence, feature, per-chunk metadata) and turns
    them into fixed-size model batches. That state machine used to be written
    out three times — twice in ``single.py`` (multiprocessing workers, threaded
    Rust) and once in ``bundle.py`` — so the paths differed in more than the
    only thing that actually differs between them: what produces the chunks.

    ``flush_fn`` is handed the four buffers *after* they have been detached
    from the accumulator, so it is free to keep them. The sequential path in
    ``single.py`` relies on that: it submits them to a single-worker GPU thread
    and keeps filling the next batch while that runs.

    Element types are whatever the caller appends — ``single.py`` accumulates
    numpy arrays for :func:`_run_batch`, ``bundle.py`` accumulates torch
    tensors for its own multi-model flush.
    """

    __slots__ = ("batch_size", "features", "flush_fn", "meta", "sequences", "signals")

    def __init__(self, batch_size: int, flush_fn: Callable[[list, list, list, list], None]):
        self.batch_size = batch_size
        self.flush_fn = flush_fn
        self.signals: list = []
        self.sequences: list = []
        self.features: list = []
        self.meta: list = []

    def __len__(self) -> int:
        return len(self.signals)

    def take(self) -> tuple[list, list, list, list]:
        """Detach and return the current buffers, leaving the accumulator empty."""
        buffers = (self.signals, self.sequences, self.features, self.meta)
        self.signals = []
        self.sequences = []
        self.features = []
        self.meta = []
        return buffers

    def flush(self) -> None:
        """Run ``flush_fn`` on the buffered chunks. No-op when empty."""
        if not self.signals:
            return
        self.flush_fn(*self.take())

    def add(self, signal, sequence, feature, meta) -> None:
        """Append one chunk, flushing once the batch is full."""
        self.signals.append(signal)
        self.sequences.append(sequence)
        self.features.append(feature)
        self.meta.append(meta)
        if len(self.signals) >= self.batch_size:
            self.flush()


def prepare_signal_channels(chunk: dict, signal_len: int) -> np.ndarray:
    """Pad/crop a chunk's signal to ``signal_len`` and stack the residual channel.

    Returns ``(signal_len,)`` for a single-channel model, or
    ``(2, signal_len)`` when the chunk carries a k-mer residual channel. Shared
    by both Python extraction paths in ``single.py``; they had byte-identical
    copies of this.
    """
    signal_array = chunk["signal"]
    sig = signal_array.astype(np.float32)
    sig_residual = chunk.get("signal_residual")
    if len(sig) < signal_len:
        sig = np.pad(sig, (0, signal_len - len(sig)), mode="constant")
        if sig_residual is not None:
            sig_residual = np.pad(
                sig_residual.astype(np.float32),
                (0, signal_len - len(sig_residual)),
                mode="constant",
            )
    elif len(sig) > signal_len:
        start = (len(sig) - signal_len) // 2
        sig = sig[start : start + signal_len]
        if sig_residual is not None:
            sig_residual = sig_residual.astype(np.float32)[start : start + signal_len]
    if sig_residual is not None:
        sig_residual = sig_residual.astype(np.float32)
        sig = np.stack([sig, sig_residual], axis=0)
    return sig


# Per-thread pinned staging buffers for host->device batch copies.
#
# `torch.from_numpy(np.stack(x)).to(device)` copies out of pageable memory: the
# driver has to stage it through a pinned bounce buffer of its own, and the copy
# is synchronous, so the calling thread blocks for its full duration. On the
# sequential inference path that thread is the dedicated GPU worker, and the
# extraction threads feeding it sit idle meanwhile. Staging into a pinned buffer
# we own lets `np.stack` write straight into page-locked memory and the copy be
# issued with `non_blocking=True`.
#
# Reuse is only safe while no copy out of the buffer is still in flight. Every
# caller here happens to end its batch with a `.cpu()` on the same stream, which
# synchronizes -- but that is an invariant a future caller could quietly break,
# so each buffer carries a CUDA event recorded after its copy and waits on it
# before being overwritten. In the normal case the event is long since complete
# and the wait returns immediately. Buffers are thread-local on top of that, so
# two threads never share one.
_pinned_staging = threading.local()


def _stack_to_device(arrays: list[np.ndarray], device: str, slot: Hashable) -> torch.Tensor:
    """Stack ``arrays`` into one tensor on ``device``.

    On CUDA the stack lands in a per-thread pinned staging buffer (keyed by
    ``slot`` plus row shape and dtype) and the copy is issued asynchronously.
    Everywhere else this is exactly ``torch.from_numpy(np.stack(arrays)).to(device)``.

    ``slot`` names the input the batch belongs to ("signal", "sequence",
    "features"); two inputs must not share a buffer. Every row is expected to
    have the row shape and dtype of ``arrays[0]`` — which is what every caller
    produces (float32 throughout) and what plain ``np.stack`` requires for the
    shape anyway.
    """
    if not device.startswith("cuda"):
        return torch.from_numpy(np.stack(arrays)).to(device)

    first = arrays[0]
    n = len(arrays)
    key = (slot, first.shape, first.dtype.str)
    cache = getattr(_pinned_staging, "buffers", None)
    if cache is None:
        cache = {}
        _pinned_staging.buffers = cache
    entry = cache.get(key)
    if entry is None or entry[0].shape[0] < n:
        entry = (
            torch.empty(
                (n,) + first.shape,
                dtype=torch.from_numpy(first).dtype,
                pin_memory=True,
            ),
            torch.cuda.Event(),
        )
        cache[key] = entry
    else:
        # Do not overwrite a buffer whose last copy is still in flight.
        entry[1].synchronize()
    buf, copied = entry
    view = buf[:n]
    np.stack(arrays, out=view.numpy())
    out = view.to(device, non_blocking=True)
    copied.record()
    return out


def _run_batch_multiclass(
    signals: list[np.ndarray],
    sequences: list[np.ndarray],
    features: list[np.ndarray | None],
    meta: list[tuple[str, int, int, bool]],
    model_wrapper: ModelInferenceWrapper | TracedModelWrapper | RemoraModelWrapper,
    requires_features: bool,
    device: str,
    pending: dict[str, list[tuple[int, int, float, list[float], float | None, int, bool]]],
    calibration: dict | None = None,
    cl_regression_head: "torch.nn.Module | None" = None,
) -> None:
    """Run a multi-class batch: store (base_idx, class_idx, confidence, all_probs,
    cl_pred, junction_indel, junction_mapped) per read. The last two pass
    through unchanged from ``meta`` -- see issue #282."""
    signal_t = _stack_to_device(signals, device, "signal")
    seq_t = _stack_to_device(sequences, device, "sequence")
    batch = {"signal": signal_t, "sequence": seq_t}

    if requires_features:
        valid_feats = [f for f in features if f is not None]
        if valid_feats:
            batch["features"] = _stack_to_device(valid_feats, device, "features")

    with torch.inference_mode():
        logits = model_wrapper.forward_batch(batch, device)
        if calibration is not None:
            from leech.calibration import apply_calibration

            logits = apply_calibration(logits, calibration)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        class_indices = np.argmax(probs, axis=-1)
        confidences = probs.max(axis=-1)

        # CL regression prediction from captured representation
        cl_preds: np.ndarray | None = None
        if (
            cl_regression_head is not None
            and isinstance(model_wrapper, ModelInferenceWrapper)
            and model_wrapper.captured_repr is not None
        ):
            cl_preds = cl_regression_head(model_wrapper.captured_repr).cpu().numpy()

    # One `tolist()` for the whole batch, not `float(p)` per class per chunk.
    # numpy promotes float32 -> Python float identically either way.
    prob_lists = probs.tolist()
    for i, ((read_id, base_idx, junction_indel, junction_mapped), cls_idx, conf) in enumerate(
        zip(meta, class_indices.flatten(), confidences.flatten(), strict=True)
    ):
        cl_val = float(cl_preds[i]) if cl_preds is not None else None
        if read_id not in pending:
            pending[read_id] = []
        pending[read_id].append(
            (
                base_idx,
                int(cls_idx),
                float(conf),
                prob_lists[i],
                cl_val,
                junction_indel,
                junction_mapped,
            )
        )


def _run_batch(
    signals: list[np.ndarray],
    sequences: list[np.ndarray],
    features: list[np.ndarray | None],
    meta: list[tuple[str, int, int, bool]],
    model_wrapper: ModelInferenceWrapper | RemoraModelWrapper,
    requires_features: bool,
    device: str,
    pending: dict[str, list[tuple[int, float]]],
) -> None:
    """Run a batch through the model and accumulate results into pending.

    ``meta``'s junction fields are unused here -- the binary (pairwise) BAM
    tag path (``MP``/``ML``) has no BELOW_THRESHOLD_LABEL-style abstention
    for issue #282's rule to plug into. See ``_run_batch_multiclass``.
    """
    signal_t = _stack_to_device(signals, device, "signal")
    seq_t = _stack_to_device(sequences, device, "sequence")
    batch = {"signal": signal_t, "sequence": seq_t}

    if requires_features:
        valid_feats = [f for f in features if f is not None]
        if valid_feats:
            batch["features"] = _stack_to_device(valid_feats, device, "features")

    with torch.inference_mode():
        logits = model_wrapper.forward_batch(batch, device)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    for meta_entry, prob in zip(meta, probs.tolist(), strict=True):
        read_id, base_idx = meta_entry[0], meta_entry[1]
        if read_id not in pending:
            pending[read_id] = []
        pending[read_id].append((base_idx, prob))


# =============================================================================
# Rust monolithic extraction path — shared between run_inference (single.py)
# and run_bundle_inference (bundle.py).
#
# The functions below let both call sites use the same logic for (a) deciding
# whether the rust hot path is available, (b) capping rayon threads to the
# SLURM allocation, (c) building the kwargs dict for leech_core, and (d)
# collecting pysam alignment metadata into the parallel-list format rust
# expects. Before this refactor the logic was duplicated across both files.
# =============================================================================


def check_rust_extraction_available(
    backend: str,
    norm_method: str = RUST_NORM_METHOD,
    recover_softclip_signal: bool = False,
    signal_context_bases: tuple[int, int] | None = None,
) -> tuple[bool, object, object, object]:
    """Decide whether the rust monolithic extraction hot path is usable.

    Args:
        backend: CLI-provided backend selector ("auto", "rust", or "python").
        norm_method: Signal normalization method for this run. The Rust
            pipeline only implements ``median_mad`` (see
            ``rust_supports_norm_method``), so anything else falls back to
            Python rather than being silently normalized the wrong way.
        recover_softclip_signal: Whether ref-anchored chunks should recover
            real soft-clipped samples at alignment edges. The Rust pipeline
            discards the pre-crop signal that recovery reads from (see
            ``rust_supports_softclip_recovery``), so this also forces Python.
        signal_context_bases: The model's base-defined signal window
            (``(L, R)``), or ``None`` for a sample-context model. The Rust
            *inference* pipeline (``inference.rs``) has no base-defined
            window support -- that landed only in the training/prepare path
            (``training.rs``, issue #278) -- so a model trained with
            ``--signal-context-bases`` always forces Python at predict time.

    Returns:
        (use_rust, extract_inference_chunks, preload_pod5_signals,
        extract_chunks_from_preloaded) — the three function handles are None
        when rust is unavailable.

    Raises:
        RuntimeError: if ``backend == "rust"`` but ``leech_core`` is not
        importable, or if it cannot honor the requested options.
    """
    from leech._rust_accel import (
        HAS_RUST,
        _rs_extract_chunks_from_preloaded,
        _rs_extract_inference_chunks,
        _rs_preload_pod5_signals,
        rust_supports_signal_context_bases,
    )

    rust_available = HAS_RUST and _rs_extract_inference_chunks is not None
    if backend == "rust" and not rust_available:
        raise RuntimeError(
            "--backend rust requested but leech_core is not installed. "
            "Build with: cd rust && uv run maturin develop --release"
        )
    norm_ok = rust_supports_norm_method(norm_method)
    softclip_ok = rust_supports_softclip_recovery(recover_softclip_signal)
    bases_ok = rust_supports_signal_context_bases(signal_context_bases)
    if backend == "rust" and not norm_ok:
        raise RuntimeError(
            f"--backend rust requested but the Rust pipeline only implements "
            f"'{RUST_NORM_METHOD}' normalization, not '{norm_method}'. "
            f"Use --backend python or --signal-norm {RUST_NORM_METHOD}."
        )
    if backend == "rust" and not softclip_ok:
        raise RuntimeError(
            "--backend rust requested but the Rust pipeline does not implement "
            "recover_softclip_signal, which this model's config enables. "
            "Use --backend python."
        )
    if backend == "rust" and not bases_ok:
        raise RuntimeError(
            "--backend rust requested but this model was trained with "
            "--signal-context-bases and the Rust inference pipeline does not "
            "implement a base-defined signal window. Use --backend python."
        )
    use_rust = rust_available and backend != "python" and norm_ok and softclip_ok and bases_ok
    if rust_available and backend != "python" and not norm_ok:
        logger.warning(
            f"Signal normalization '{norm_method}' is not implemented in the Rust "
            f"extraction path (which always applies '{RUST_NORM_METHOD}'); falling "
            f"back to the Python path so chunks match how the model was trained."
        )
    if rust_available and backend != "python" and norm_ok and not softclip_ok:
        logger.warning(
            "recover_softclip_signal is not implemented in the Rust extraction "
            "path (it discards the pre-crop signal the recovery reads from); "
            "falling back to the Python path so chunks match how the model was "
            "trained."
        )
    if rust_available and backend != "python" and norm_ok and softclip_ok and not bases_ok:
        logger.warning(
            "This model was trained with --signal-context-bases, which the Rust "
            "inference pipeline does not implement; falling back to the Python "
            "path so chunks match how the model was trained."
        )
    return (
        use_rust,
        _rs_extract_inference_chunks,
        _rs_preload_pod5_signals,
        _rs_extract_chunks_from_preloaded,
    )


def cap_rayon_threads_for_slurm(max_cap: int | None = None) -> int:
    """Clamp ``RAYON_NUM_THREADS`` to the SLURM allocation.

    Without this, rayon defaults to all visible system CPUs (e.g. 63 on a
    64-core shared node) and oversubscribes whenever multiple jobs share a
    node. Idempotent — respects any existing ``RAYON_NUM_THREADS`` value.

    Args:
        max_cap: Optional upper bound, applied ONLY when ``SLURM_CPUS_PER_TASK``
            is not set. On a SLURM allocation we always honor the allocation.
            On a dev machine ``os.cpu_count()`` can be 64+ and we don't want
            rayon to claim them all — the cap kicks in there.

    Returns:
        The number of CPUs the caller should reason about (after any cap).
    """
    import os

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    if slurm_cpus > 0:
        avail = slurm_cpus
    else:
        avail = os.cpu_count() or 4
        if max_cap is not None:
            avail = min(avail, max_cap)
    if "RAYON_NUM_THREADS" not in os.environ:
        rayon_threads = max(1, avail - 6)  # reserve headroom for main + GPU I/O
        os.environ["RAYON_NUM_THREADS"] = str(rayon_threads)
        logger.info(f"Set RAYON_NUM_THREADS={rayon_threads} (from {avail} available CPUs)")
    return avail


def build_rust_extraction_kwargs(
    *,
    signal_context: tuple[int, int],
    kmer_context: int,
    signal_len: int,
    compute_features: bool,
    reverse_signal: bool,
    feature_start: int | None,
    feature_end: int | None,
    anchor: str,
    seq_encoding: str,
    signal_kmer_context: tuple[int, int],
    refine_signal_map: bool,
    signal_refiner,
    refine_half_bandwidth: int,
    refine_scale_iters: int,
    signal_in_channels: int,
    base_justify: str,
) -> dict:
    """Build the kwargs dict passed to the rust extraction functions.

    Extracts kmer table + kmer length + kmer center from ``signal_refiner``
    when refinement is enabled; otherwise fills in inert defaults.

    Every key the Rust entry points accept and the Python extraction path
    honors belongs here. ``base_justify`` did not, so the Rust signature's
    ``"center"`` default silently overrode a model trained with
    ``--base-justify end`` -- which moves the focus sample within the base and
    so shifts every signal window. ``data prepare`` passed it; only predict
    dropped it.

    This function runs once per predict run (the returned dict is reused
    across every mega-batch via ``**_rs_kwargs``), which is exactly where the
    k-mer table's ``dict -> KmerLevels`` conversion belongs: building it here
    means every batch call after this one borrows the same handle instead of
    re-marshalling the 262,144-entry table itself (issue #259).
    """
    kmer_table_handle = None
    kmer_table_len = 9
    kmer_table_center = -1
    if refine_signal_map and signal_refiner is not None:
        kmer_to_level = getattr(signal_refiner, "kmer_to_level", None)
        kmer_table_handle = make_kmer_levels(kmer_to_level)
        kmer_table_len = getattr(signal_refiner, "kmer_len", 9)
        kmer_table_center = getattr(signal_refiner, "center_idx", -1)

    return {
        "signal_context_left": signal_context[0],
        "signal_context_right": signal_context[1],
        "kmer_context": kmer_context,
        "signal_len": signal_len,
        "compute_features": compute_features,
        "reverse_signal": reverse_signal,
        "feature_start": feature_start,
        "feature_end": feature_end,
        "anchor": anchor,
        "seq_encoding": seq_encoding,
        "signal_kmer_context": (signal_kmer_context if seq_encoding == "signal_kmer" else None),
        "refine_signal_map": refine_signal_map,
        "kmer_table": kmer_table_handle,
        "kmer_len": kmer_table_len,
        "kmer_center_idx": kmer_table_center,
        "refine_half_bandwidth": refine_half_bandwidth,
        "refine_scale_iters": refine_scale_iters,
        "signal_in_channels": signal_in_channels,
        "base_justify": base_justify,
    }


def collect_bam_metadata_for_rust(
    aln_batch: list,
    *,
    motif: str,
    motif_offset: int,
    motif_searcher,
    anchor: str,
    reference_sequences: dict[str, str] | None,
    max_positions_per_read: int | None = None,
) -> tuple[
    list[str],
    list[str],
    list[int],
    list[np.ndarray],
    list[int],
    list[int],
    list[list[int]],
    list[list[tuple[int, int]]],
    list[str | None],
    list[dict[int, tuple[int, bool]]],
]:
    """Collect pysam alignment metadata into the parallel-list format expected
    by ``_rs_extract_inference_chunks`` / ``_rs_preload_pod5_signals``.

    Must be called on the thread that owns ``aln_batch`` — pysam is not
    thread-safe across iteration.

    Args:
        aln_batch: Batch of pysam ``AlignedSegment`` objects.
        motif: Motif to search for in each read's sequence.
        motif_offset: Offset added to each motif match position.
        motif_searcher: Configured motif searcher.
        anchor: "reference" or "basecall".
        reference_sequences: Per-reference sequence dict; required for
            ``anchor == "reference"``.
        max_positions_per_read: Cap on motif positions emitted per read.
            Bundle inference passes 1 to force one-chunk-per-read.

    Returns:
        Ten parallel lists: (rids, seqs, strides, moves, num_samples,
        trim_offsets, motif_positions, cigar_tuples, reference_sequences,
        junctions). ``junctions[i]`` maps each of ``motif_positions[i]`` to
        its ``(junction_indel, junction_mapped)`` (issue #282) -- the Rust
        extraction call itself is unaware of it; a caller correlates it back
        onto Rust's returned ``(sig, seq_arr, feat, read_id, base_idx)``
        chunks by ``base_idx``, which Rust always sets to the position that
        produced the chunk. See :func:`junction_lookup_from_rs_meta`.
    """
    from leech.features import extract_move_table

    rs_rids: list[str] = []
    rs_seqs: list[str] = []
    rs_strides: list[int] = []
    rs_mvs: list[np.ndarray] = []
    rs_ns: list[int] = []
    rs_trims: list[int] = []
    rs_motifs: list[list[int]] = []
    rs_cigars: list[list[tuple[int, int]]] = []
    rs_refs: list[str | None] = []
    rs_junctions: list[dict[int, tuple[int, bool]]] = []

    for aln in aln_batch:
        if aln.query_name is None or aln.query_sequence is None:
            continue
        try:
            mt = extract_move_table(aln)

            # Resolve the reference slice BEFORE searching. Motif positions are
            # indices into whatever sequence chunks are cut from, which under
            # anchor="reference" is that slice and not the basecall. This
            # searched `aln.query_sequence` unconditionally, so a
            # BasecalledMotifSearcher -- which `predict` selects whenever there
            # is no reference FASTA -- returned query coordinates for windows
            # cut in reference coordinates.
            ref_seq = None
            cigar_list: list[tuple[int, int]] = []
            if anchor == "reference":
                cigar_list = list(aln.cigartuples) if aln.cigartuples else []
                if reference_sequences and aln.reference_name in reference_sequences:
                    full_ref = reference_sequences[aln.reference_name]
                    ref_seq = full_ref[aln.reference_start : aln.reference_end]
                else:
                    try:
                        ref_seq = aln.get_reference_sequence()
                    except Exception:
                        ref_seq = None

            matches = motif_searcher.find_motif_positions(
                aln.query_name,
                extraction_sequence(
                    anchor=anchor,
                    basecall=aln.query_sequence,
                    reference_sequence=ref_seq,
                    cigar_tuples=cigar_list or None,
                ),
                aln,
                motif,
            )
            positions = [m.position + motif_offset for m in matches]
            if not positions:
                continue
            junctions = {
                m.position + motif_offset: (m.junction_indel, m.junction_mapped) for m in matches
            }
            if max_positions_per_read is not None:
                positions = positions[:max_positions_per_read]

            rs_rids.append(aln.query_name)
            rs_seqs.append(aln.query_sequence)
            rs_strides.append(mt.stride)
            # Zero-copy uint8 view (moves are 0/1, same bit pattern as int8) --
            # the Rust entry point borrows this array's buffer directly
            # instead of re-marshalling `.tolist()` per read (issue #259).
            rs_mvs.append(mt.moves.view(np.uint8))
            rs_ns.append(mt.num_samples)
            rs_trims.append(mt.trim_offset)
            rs_motifs.append(positions)
            rs_cigars.append(cigar_list)
            rs_refs.append(ref_seq)
            rs_junctions.append(junctions)
        except Exception as e:
            logger.warning(f"Skipping read {aln.query_name}: {e}")
            continue

    return (
        rs_rids,
        rs_seqs,
        rs_strides,
        rs_mvs,
        rs_ns,
        rs_trims,
        rs_motifs,
        rs_cigars,
        rs_refs,
        rs_junctions,
    )


def junction_lookup_from_rs_meta(
    rs_meta: tuple,
) -> dict[tuple[str, int], tuple[int, bool]]:
    """Flatten ``collect_bam_metadata_for_rust``'s per-read junction dicts.

    ``rs_meta[0]`` (read ids) and ``rs_meta[9]`` (per-read
    ``{base_idx: (junction_indel, junction_mapped)}``) are parallel lists;
    this builds the single ``(read_id, base_idx) -> (junction_indel,
    junction_mapped)`` lookup a Rust chunk consumer keys into (issue #282).
    """
    lookup: dict[tuple[str, int], tuple[int, bool]] = {}
    for read_id, per_base in zip(rs_meta[0], rs_meta[9], strict=True):
        for base_idx, info in per_base.items():
            lookup[(read_id, base_idx)] = info
    return lookup
