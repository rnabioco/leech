"""Helper functions shared across inference submodules."""

import array
import json
import logging
from pathlib import Path

import numpy as np
import pysam
import torch

from leech.constants import BELOW_THRESHOLD_LABEL
from leech.features import encode_signal_kmer, sequence_to_int
from leech.model_loading import load_model_from_checkpoint
from leech.models.inference_wrapper import ModelInferenceWrapper, TracedModelWrapper
from leech.models.remora_compat import RemoraModelWrapper
from leech.preparation import encode_kmer

logger = logging.getLogger("leech.inference")


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
    """
    sorted_probs = sorted(probs, reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
    margin_uint8 = int(min(255, max(0, round(margin * 255))))
    ac_uint8 = int(min(255, max(0, round(conf * 255))))
    passed_threshold = ac_uint8 >= min_confidence and margin_uint8 >= min_margin

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
) -> T:
    """Resolve inference param from config, erroring on CLI conflict.

    Logic:
    - config has value + CLI is default -> use config (normal auto-read)
    - config has value + CLI differs   -> raise InferenceConfigError
    - config is None/missing           -> use CLI value (old models without field)
    """
    if config_value is not None:
        if cli_value != cli_default and cli_value != config_value:
            raise InferenceConfigError(
                f"CLI --{param_name}={cli_value!r} conflicts with training config "
                f"{param_name}={config_value!r}. Inference must use the same "
                f"parameters the model was trained with."
            )
        return config_value
    return cli_value


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
        raw["refine_do_rough_rescale"] = raw.get("refine_do_rough_rescale", True)
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
                requires_features = model_name in ModelInferenceWrapper.FEATURE_MODELS
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
            requires_features = model_name in ModelInferenceWrapper.FEATURE_MODELS
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
            config["refine_half_bandwidth"] = remora_meta.get("refine_half_bandwidth", 5)
            config["refine_do_rough_rescale"] = remora_meta.get("refine_do_rough_rescale", True)
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
) -> int:
    """Write predictions for a mega-batch of alignments. Returns prediction count."""
    n_preds = 0
    if is_multiclass:
        for aln in aln_batch:
            preds = pending.get(aln.query_name)
            if preds:
                pred = preds[0]
                # Unpack: 4-tuple (old) or 5-tuple (with CL prediction)
                if len(pred) == 5:
                    _, cls_idx, conf, all_probs, cl_pred = pred
                else:
                    _, cls_idx, conf, all_probs = pred
                    cl_pred = None
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


def _run_batch_multiclass(
    signals: list[np.ndarray],
    sequences: list[np.ndarray],
    features: list[np.ndarray | None],
    meta: list[tuple[str, int]],
    model_wrapper: ModelInferenceWrapper | TracedModelWrapper | RemoraModelWrapper,
    requires_features: bool,
    device: str,
    pending: dict[str, list[tuple[int, int, float, list[float], float | None]]],
    calibration: dict | None = None,
    cl_regression_head: "torch.nn.Module | None" = None,
) -> None:
    """Run a multi-class batch: store (base_idx, class_idx, confidence, all_probs, cl_pred) per read."""
    signal_t = torch.from_numpy(np.stack(signals)).to(device)
    seq_t = torch.from_numpy(np.stack(sequences)).to(device)
    batch = {"signal": signal_t, "sequence": seq_t}

    if requires_features:
        valid_feats = [f for f in features if f is not None]
        if valid_feats:
            batch["features"] = torch.from_numpy(np.stack(valid_feats)).to(device)

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

    for i, ((read_id, base_idx), cls_idx, conf, prob_vec) in enumerate(
        zip(meta, class_indices.flatten(), confidences.flatten(), probs, strict=True)
    ):
        cl_val = float(cl_preds[i]) if cl_preds is not None else None
        if read_id not in pending:
            pending[read_id] = []
        pending[read_id].append(
            (base_idx, int(cls_idx), float(conf), [float(p) for p in prob_vec], cl_val)
        )


def _run_batch(
    signals: list[np.ndarray],
    sequences: list[np.ndarray],
    features: list[np.ndarray | None],
    meta: list[tuple[str, int]],
    model_wrapper: ModelInferenceWrapper | RemoraModelWrapper,
    requires_features: bool,
    device: str,
    pending: dict[str, list[tuple[int, float]]],
) -> None:
    """Run a batch through the model and accumulate results into pending."""
    signal_t = torch.from_numpy(np.stack(signals)).to(device)
    seq_t = torch.from_numpy(np.stack(sequences)).to(device)
    batch = {"signal": signal_t, "sequence": seq_t}

    if requires_features:
        valid_feats = [f for f in features if f is not None]
        if valid_feats:
            batch["features"] = torch.from_numpy(np.stack(valid_feats)).to(device)

    with torch.inference_mode():
        logits = model_wrapper.forward_batch(batch, device)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    for (read_id, base_idx), prob in zip(meta, probs, strict=True):
        if read_id not in pending:
            pending[read_id] = []
        pending[read_id].append((base_idx, float(prob)))


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
) -> tuple[bool, object, object, object]:
    """Decide whether the rust monolithic extraction hot path is usable.

    Args:
        backend: CLI-provided backend selector ("auto", "rust", or "python").

    Returns:
        (use_rust, extract_inference_chunks, preload_pod5_signals,
        extract_chunks_from_preloaded) — the three function handles are None
        when rust is unavailable.

    Raises:
        RuntimeError: if ``backend == "rust"`` but ``leech_core`` is not
        importable.
    """
    from leech._rust_accel import (
        HAS_RUST,
        _rs_extract_chunks_from_preloaded,
        _rs_extract_inference_chunks,
        _rs_preload_pod5_signals,
    )

    rust_available = HAS_RUST and _rs_extract_inference_chunks is not None
    if backend == "rust" and not rust_available:
        raise RuntimeError(
            "--backend rust requested but leech_core is not installed. "
            "Build with: cd rust && uv run maturin develop --release"
        )
    use_rust = rust_available and backend != "python"
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
) -> dict:
    """Build the kwargs dict passed to the rust extraction functions.

    Extracts kmer table + kmer length + kmer center from ``signal_refiner``
    when refinement is enabled; otherwise fills in inert defaults.
    """
    kmer_table_dict = None
    kmer_table_len = 9
    kmer_table_center = -1
    if refine_signal_map and signal_refiner is not None:
        kmer_table_dict = getattr(signal_refiner, "kmer_to_level", None)
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
        "kmer_table": kmer_table_dict,
        "kmer_len": kmer_table_len,
        "kmer_center_idx": kmer_table_center,
        "refine_half_bandwidth": refine_half_bandwidth,
        "refine_scale_iters": refine_scale_iters,
        "signal_in_channels": signal_in_channels,
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
    list[list[int]],
    list[int],
    list[int],
    list[list[int]],
    list[list[tuple[int, int]]],
    list[str | None],
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
        Nine parallel lists: (rids, seqs, strides, moves, num_samples,
        trim_offsets, motif_positions, cigar_tuples, reference_sequences).
    """
    from leech.features import extract_move_table

    rs_rids: list[str] = []
    rs_seqs: list[str] = []
    rs_strides: list[int] = []
    rs_mvs: list[list[int]] = []
    rs_ns: list[int] = []
    rs_trims: list[int] = []
    rs_motifs: list[list[int]] = []
    rs_cigars: list[list[tuple[int, int]]] = []
    rs_refs: list[str | None] = []

    for aln in aln_batch:
        if aln.query_name is None or aln.query_sequence is None:
            continue
        try:
            mt = extract_move_table(aln)
            positions = [
                pos + motif_offset
                for pos in motif_searcher.find_motif_positions(
                    aln.query_name, aln.query_sequence, aln, motif
                )
            ]
            if not positions:
                continue
            if max_positions_per_read is not None:
                positions = positions[:max_positions_per_read]

            rs_rids.append(aln.query_name)
            rs_seqs.append(aln.query_sequence)
            rs_strides.append(mt.stride)
            rs_mvs.append(mt.moves.tolist())
            rs_ns.append(mt.num_samples)
            rs_trims.append(mt.trim_offset)
            rs_motifs.append(positions)

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
            rs_cigars.append(cigar_list)
            rs_refs.append(ref_seq)
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
    )
