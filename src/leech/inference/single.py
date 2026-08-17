"""Single-model inference: run_inference and _inference_worker."""

import functools
import logging
import math
import multiprocessing as mp
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pysam
import torch
from rich.progress import Progress

from leech.configs import ChunkConfig, InferenceConfig, MotifConfig, SignalConfig
from leech.features import encode_signal_kmer, extract_move_table, sequence_to_int
from leech.inference.helpers import (
    _check_config_consistency,
    _encode_sequence_for_inference,
    _run_batch,
    _run_batch_multiclass,
    _write_mega_batch_predictions,
    build_rust_extraction_kwargs,
    cap_rayon_threads_for_slurm,
    check_rust_extraction_available,
    collect_bam_metadata_for_rust,
    load_model_auto,
    validate_inference_shapes,
)
from leech.io.bam_reader import count_bam_reads, iter_bam_batches
from leech.io.motif_search import get_motif_searcher
from leech.io.pod5_reader import POD5Reader
from leech.models.inference_wrapper import ModelInferenceWrapper
from leech.models.remora_compat import RemoraModelWrapper
from leech.preparation.reader import build_leech_read

logger = logging.getLogger("leech.inference")


def _inference_worker(
    args: tuple[list, InferenceConfig],
) -> list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray | None]]:
    """
    Worker for parallel chunk extraction during inference.

    Extracts chunks from reads and optionally pre-computes signal_kmer encoding.

    Returns:
        List of (read_id, base_idx, signal, encoded_sequence, features_or_none) tuples
    """
    from leech.io.pod5_reader import read_pod5_signals_batch_cached
    from leech.preparation.reader import build_leech_read

    read_infos, config = args

    results: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray | None]] = []
    _shape_validated = False

    # Batch-read all POD5 signals via the process-local reader cache.
    read_info_by_id = {ri.read_id: ri for ri in read_infos}
    pod5_cache = read_pod5_signals_batch_cached(config.pod5_path, list(read_info_by_id.keys()))

    for read_info in read_infos:
        try:
            cached = pod5_cache.get(read_info.read_id)
            if cached is None:
                continue
            raw_signal, pod5_metadata = cached

            # Build SignalConfig with compute_features override
            sig_cfg = SignalConfig(
                reverse_signal=config.signal.reverse_signal,
                anchor=config.signal.anchor,
                norm_method=config.signal.norm_method,
                pa_mean=config.signal.pa_mean,
                pa_stdev=config.signal.pa_stdev,
                refine_signal_map=config.signal.refine_signal_map,
                signal_refiner=config.signal.signal_refiner,
                compute_features=config.requires_features or config.signal_in_channels > 1,
            )

            leech_read = build_leech_read(
                read_id=read_info.read_id,
                sequence=read_info.sequence,
                raw_signal=raw_signal,
                move_table=read_info.to_move_table(),
                signal_config=sig_cfg,
                reference_sequence=read_info.reference_sequence,
                cigar_tuples=read_info.cigar_tuples,
                cal_offset=pod5_metadata.get("calibration_offset"),
                cal_scale=pod5_metadata.get("calibration_scale"),
            )

            # Find motif positions
            if config.motif.motif is not None:
                searcher = get_motif_searcher(
                    mode="fasta" if config.motif.reference_sequences else "bam",
                    reference_sequences=config.motif.reference_sequences,
                    skip_indels=config.motif.skip_motif_indels,
                    anchor=config.signal.anchor,
                )
                aln = read_info.to_mock_alignment()
                positions = [
                    pos + config.motif.motif_offset
                    for pos in searcher.find_motif_positions(
                        read_info.read_id, read_info.sequence, aln, config.motif.motif
                    )
                ]
            else:
                kmer_context = config.chunk.kmer_context
                positions = list(range(kmer_context, leech_read.num_bases - kmer_context))

            for base_idx in positions:
                chunk = leech_read.get_chunk(base_idx, config=config.chunk)
                if chunk is None:
                    continue

                # Signal (with optional kmer residual channel)
                sig = chunk["signal"].astype(np.float32)
                sig_residual = chunk.get("signal_residual")
                if len(sig) < config.signal_len:
                    sig = np.pad(sig, (0, config.signal_len - len(sig)), mode="constant")
                    if sig_residual is not None:
                        sig_residual = np.pad(
                            sig_residual.astype(np.float32),
                            (0, config.signal_len - len(sig_residual)),
                            mode="constant",
                        )
                elif len(sig) > config.signal_len:
                    start = (len(sig) - config.signal_len) // 2
                    sig = sig[start : start + config.signal_len]
                    if sig_residual is not None:
                        sig_residual = sig_residual.astype(np.float32)[
                            start : start + config.signal_len
                        ]
                if sig_residual is not None:
                    sig_residual = sig_residual.astype(np.float32)
                    sig = np.stack([sig, sig_residual], axis=0)  # (2, signal_len)

                # Sequence encoding
                if config.seq_encoding == "signal_kmer":
                    seq_ctx = chunk.get("sequence_with_kmer_context")
                    seq_to_sig = chunk.get("seq_to_sig_map")
                    if seq_ctx is not None and seq_to_sig is not None:
                        seq_ints = sequence_to_int(seq_ctx)
                        enc_seq = encode_signal_kmer(
                            seq_ints,
                            seq_to_sig,
                            config.signal_len,
                            tuple(config.signal_kmer_context),
                        )
                    else:
                        from leech.preparation.encoding import encode_kmer as _enc

                        enc_seq = _enc(chunk["sequence"]).numpy()
                else:
                    from leech.preparation.encoding import encode_kmer as _enc

                    enc_seq = _enc(chunk["sequence"]).numpy()

                # Features
                feat = None
                if config.requires_features:
                    feat_arr = chunk["features"]
                    if feat_arr.size > 0:
                        feat_arr = feat_arr.astype(np.float32)
                        if config.wide_features:
                            pass
                        elif feat_arr.shape[1] > config.kmer_len:
                            kmer_ctx = config.kmer_len // 2
                            fs = int(chunk.get("feature_start", -kmer_ctx))
                            s = (-kmer_ctx - fs) + config.dwell_offset
                            feat_arr = feat_arr[:, s : s + config.kmer_len]
                        feat = feat_arr

                if not _shape_validated:
                    # config is InferenceConfig dataclass; build dict for validator
                    _cfg_dict = {
                        "signal_in_channels": config.signal_in_channels,
                        "signal_len": config.signal_len,
                    }
                    validate_inference_shapes(sig, feat, _cfg_dict)
                    _shape_validated = True

                results.append((read_info.read_id, base_idx, sig, enc_seq, feat))

        except Exception as e:
            logger.warning(f"Worker: skipping read {read_info.read_id}: {e}")
            continue

    return results


def run_inference(
    model_and_config: tuple[torch.nn.Module | ModelInferenceWrapper | RemoraModelWrapper, dict]
    | None = None,
    model_path: Path | None = None,
    pod5_path: Path | None = None,
    bam_path: Path | None = None,
    output_path: Path | None = None,
    device: str = "cuda",
    min_mapq: int = 0,
    motif: str | None = None,
    motif_offset: int = 0,
    batch_size: int = 256,
    base_justify: str = "center",
    reverse_signal: bool = True,
    num_workers: int = 0,
    chunk_size: int = 100,
    anchor: str = "reference",
    reference_fasta: Path | None = None,
    raw: bool = False,
    min_confidence: int = 0,
    min_margin: int = 0,
    read_batch_size: int = 10_000,
    backend: str = "auto",
    no_compile: bool = False,
    output_format: str = "bam",
    copy_tags: list[str] | None = None,
) -> None:
    """
    Run inference on POD5 and BAM files.

    Supports both leech native models and Remora TorchScript models (auto-detected).
    Supports parallel chunk extraction via num_workers > 0.

    Args:
        model_and_config: Pre-loaded (wrapper_or_model, config) tuple.
        model_path: Path to model checkpoint directory or Remora .pt file.
        pod5_path: Path to POD5 file with raw signal
        bam_path: Path to input BAM file with alignments
        output_path: Path to output BAM file with predictions
        raw: Write full float probabilities (default: compact uint8)
        min_confidence: Confidence threshold in 0-255 uint8 space
        min_margin: Margin threshold in 0-255 uint8 space
        device: Device for inference
        min_mapq: Minimum mapping quality
        motif: Optional motif to filter predictions (auto-read from config if None)
        motif_offset: Offset within motif for prediction (auto-read from config if 0)
        batch_size: Chunks per forward pass
        base_justify: Signal justification within focus base
        reverse_signal: Whether to reverse signal for RNA
        num_workers: Parallel chunk extraction workers (0=sequential).
            Only beneficial with GPU inference, where CPU chunk extraction
            overlaps with GPU forward passes. For CPU-only inference, the
            sequential path (0) is faster due to batched POD5 access and
            no multiprocessing overhead.
        backend: Extraction backend. "auto" uses Rust if available, "rust"
            forces Rust (error if unavailable), "python" forces Python.
        chunk_size: Reads per worker batch
        anchor: "basecall" or "reference" for reference-anchored mode
        reference_fasta: Path to reference FASTA (for reference-anchored mode)
        read_batch_size: Reads per mega-batch for memory-bounded streaming (default 50K).
            Each mega-batch loads BAM alignments + POD5 signals, runs inference,
            writes predictions, then frees memory. Set to 0 to disable (load all).
        output_format: "bam" for BAM output with tags, "tsv" for gzipped TSV.
            TSV mode requires a multiclass model.
    """
    # Apply backend override to signal_refine module
    logger.info(f"Extraction backend: {backend}")
    if backend == "python":
        import leech.signal_refine as _sr

        _sr.HAS_RUST = False
    elif backend == "rust":
        import leech.signal_refine as _sr

        if not _sr.HAS_RUST:
            logger.warning("Backend rust requested but signal_refine Rust not available")

    # Load model
    if model_and_config is not None:
        wrapper_or_model, config = model_and_config
    elif model_path is not None:
        logger.info(f"Loading model from {model_path}")
        wrapper_or_model, config = load_model_auto(model_path, device=device)
    else:
        raise ValueError("Either model_and_config or model_path must be provided")

    # Determine if this is a Remora model or a leech model
    is_remora = config.get("is_remora", False)

    # Signal map refinement setup
    refine_signal_map = False
    signal_refiner = None

    if is_remora:
        model_wrapper = wrapper_or_model
        signal_len = config.get("signal_len", 100)
        kmer_len = config.get("kmer_len", 9)
        seq_encoding = "signal_kmer"
        signal_kmer_context = tuple(config.get("signal_kmer_context", (4, 4)))
        dwell_offset = 0

        # Resolve motif/offset from config, erroring on CLI conflict
        motif = _check_config_consistency("motif", motif, config.get("motif"), None)
        motif_offset = _check_config_consistency(
            "motif-offset", motif_offset, config.get("motif_offset"), 0
        )
        if motif is not None:
            logger.info(f"Motif from remora config: {motif} (offset={motif_offset})")

        if motif is None:
            raise ValueError("--motif is required for Remora models (no config.json)")

        # Set up signal map refinement if model specifies it
        if config.get("refine_signal_map", True):
            from leech.data import get_kmer_table
            from leech.inference.helpers import _warn_if_kmer_table_drifted
            from leech.signal_refine import SigMapRefiner

            kmer_table_path = get_kmer_table()
            _warn_if_kmer_table_drifted(config.get("kmer_table_sha256"), kmer_table_path)
            half_bw = config.get("refine_half_bandwidth", 5)
            do_rescale = config.get("refine_do_rough_rescale", True)
            scale_iters = config.get("refine_scale_iters", -1)
            center_idx = config.get("refine_kmer_center_idx", -1)
            signal_refiner = SigMapRefiner.from_table(
                kmer_table_path,
                half_bandwidth=half_bw,
                do_rough_rescale=do_rescale,
                scale_iters=scale_iters,
                center_idx=center_idx,
            )
            refine_signal_map = True
            logger.info(
                f"Signal map refinement: half_bw={half_bw}, "
                f"scale_iters={scale_iters}, center_idx={center_idx}"
            )
    else:
        # Leech model
        if isinstance(wrapper_or_model, ModelInferenceWrapper):
            model_wrapper = wrapper_or_model
        else:
            model_type = config["model_name"]
            model_wrapper = ModelInferenceWrapper(wrapper_or_model, model_type)

        signal_len = config["signal_len"]
        kmer_len = config["kmer_len"]
        dwell_offset = config.get("dwell_offset", 0)
        seq_encoding = config.get("seq_encoding", "signal_kmer")
        signal_kmer_context = tuple(config.get("signal_kmer_context", (4, 4)))

        # Resolve motif/offset from config, erroring on CLI conflict
        motif = _check_config_consistency("motif", motif, config.get("motif"), None)
        motif_offset = _check_config_consistency(
            "motif-offset", motif_offset, config.get("motif_offset"), 0
        )
        if motif is not None:
            logger.info(f"Motif from config: {motif} (offset={motif_offset})")

        if motif is None:
            raise ValueError(
                "motif is None after auto-read from config. "
                "Either pass --motif on the CLI or ensure config.json contains a non-null 'motif' field. "
                "Without a motif, inference predicts at every position, producing noise."
            )

        # Signal map refinement for leech models (needed for kmer residual signal channel)
        if config.get("refine_signal_map", True) or config.get("signal_in_channels", 1) > 1:
            from leech.data import get_kmer_table
            from leech.inference.helpers import _warn_if_kmer_table_drifted
            from leech.signal_refine import SigMapRefiner

            kmer_table_path = get_kmer_table()
            _warn_if_kmer_table_drifted(config.get("kmer_table_sha256"), kmer_table_path)
            half_bw = config.get("refine_half_bandwidth", 5)
            do_rescale = config.get("refine_do_rough_rescale", True)
            scale_iters = config.get("refine_scale_iters", 2)
            center_idx = config.get("refine_kmer_center_idx", -1)
            signal_refiner = SigMapRefiner.from_table(
                kmer_table_path,
                half_bandwidth=half_bw,
                do_rough_rescale=do_rescale,
                scale_iters=scale_iters,
                center_idx=center_idx,
            )
            refine_signal_map = True
            logger.info(
                f"Signal map refinement enabled for leech model "
                f"(signal_in_channels={config.get('signal_in_channels', 1)})"
            )

    # Use asymmetric context if available, otherwise fall back to symmetric
    left_ctx = config.get("left_context")
    right_ctx = config.get("right_context")
    if left_ctx is not None and right_ctx is not None:
        signal_context = (left_ctx, right_ctx)
    else:
        signal_context = (signal_len // 2, signal_len // 2)
    kmer_context = kmer_len // 2
    requires_features = getattr(model_wrapper, "requires_features", False)

    # Determine feature_start/feature_end from config (must match training data)
    _model_type = getattr(model_wrapper, "model_type", "")
    wide_features = _model_type in ModelInferenceWrapper.WIDE_FEATURE_MODELS
    _kmer_context = kmer_len // 2

    # Read new params, falling back to old dwell_margin_* for backward compat
    _feature_start = config.get("feature_start")
    _feature_end = config.get("feature_end")
    if _feature_start is None and "feature_left" in config:
        _feature_start = -config["feature_left"]
    if _feature_end is None and "feature_right" in config:
        _feature_end = config["feature_right"]
    if _feature_start is None and "dwell_margin_left" in config:
        _feature_start = -(_kmer_context + config["dwell_margin_left"])
    if _feature_end is None and "dwell_margin_right" in config:
        _feature_end = _kmer_context + config["dwell_margin_right"]
    if wide_features and _feature_start is None and _feature_end is None:
        _model_margin = (
            getattr(model_wrapper.model, "dwell_margin", 0)
            if hasattr(model_wrapper, "model")
            else 0
        )
        if _model_margin:
            _feature_start = -(_kmer_context + _model_margin)
            _feature_end = _kmer_context + _model_margin
            logger.warning(
                f"Config missing feature_start/end, "
                f"falling back to model default margin: {_model_margin}"
            )

    # Detect multi-class model
    num_out = config.get("num_out", 1)
    label_map = config.get("label_map")  # {name: int} or None
    if label_map:
        # Invert to {int: name}
        int_to_label = {v: k for k, v in label_map.items()}
    else:
        int_to_label = None
    is_multiclass = num_out > 1

    # Resolve base_justify from config, erroring on CLI conflict
    base_justify = _check_config_consistency(
        "base-justify", base_justify, config.get("base_justify"), "center"
    )
    logger.info(f"base_justify: {base_justify}")

    anchor = _check_config_consistency("anchor", anchor, config.get("anchor"), "reference")
    logger.info(f"anchor: {anchor}")

    if reference_fasta is None:
        cfg_ref = config.get("reference_fasta")
        if cfg_ref is not None:
            cfg_path = Path(cfg_ref)
            if cfg_path.exists():
                reference_fasta = cfg_path
                logger.info(f"reference_fasta from config: {reference_fasta}")
            else:
                logger.warning(
                    f"reference_fasta from config ({cfg_ref}) not found; "
                    f"pass --reference-fasta explicitly"
                )

    logger.info(f"Signal length: {signal_len}, K-mer length: {kmer_len}")
    if is_multiclass:
        logger.info(f"Multi-class model: num_out={num_out}")
    logger.info(f"Signal context: {signal_context}")
    logger.info(f"Sequence encoding: {seq_encoding}, base_justify: {base_justify}")
    if _feature_start is not None or _feature_end is not None:
        _fs = _feature_start if _feature_start is not None else -_kmer_context
        _fe = _feature_end if _feature_end is not None else _kmer_context
        logger.info(f"Feature window: [{_fs}, {_fe}] relative to focus (width={_fe - _fs + 1})")
    if motif:
        logger.info(f"Motif: {motif} (offset={motif_offset})")

    # Open BAM for header and normalization detection
    bam_in = pysam.AlignmentFile(str(bam_path), "rb")

    # Detect normalization method: read from config, with sm/sd tag override for Remora
    norm_method = config.get("signal_norm", "median_mad")
    pa_mean = config.get("pa_mean")
    pa_stdev = config.get("pa_stdev")
    if is_remora:
        # Peek at first alignment for pa_scaling tags
        for first_aln in bam_in.fetch(until_eof=True):
            if first_aln.query_name is not None:
                if first_aln.has_tag("sm") and first_aln.has_tag("sd"):
                    pa_mean = float(first_aln.get_tag("sm"))
                    pa_stdev = float(first_aln.get_tag("sd"))
                    norm_method = "pa_scaling"
                    logger.info(f"Using pa_scaling normalization (sm={pa_mean}, sd={pa_stdev})")
                break

    # Create output writer (BAM or TSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tsv_writer = None
    bam_out = None
    if output_format == "tsv":
        if not is_multiclass:
            raise RuntimeError(
                "TSV output is only supported for multiclass models. "
                "Use .bam extension for binary models."
            )
        from leech.io.tsv_writer import TsvPredictionWriter

        _has_cl = (
            config.get("cl_regression", False)
            and config.get("cl_regression_head_state_dict") is not None
        )
        if int_to_label:
            _tsv_class_names = [int_to_label[i] for i in range(num_out)]
        else:
            _tsv_class_names = [str(i) for i in range(num_out)]
        tsv_writer = TsvPredictionWriter(
            output_path, _tsv_class_names, _has_cl, copy_tags=copy_tags
        )
        logger.info(f"TSV output: {output_path} ({len(_tsv_class_names)} classes)")
    else:
        bam_out = pysam.AlignmentFile(str(output_path), "wb", template=bam_in)
    bam_in.close()

    total_reads = 0
    total_predictions = 0

    if hasattr(model_wrapper, "model"):
        model_wrapper.model.eval()
    if hasattr(model_wrapper, "eval"):
        model_wrapper.eval()

    # Set up CL regression head if present in config (multiclass bundles)
    _cl_head: torch.nn.Module | None = None
    if config.get("cl_regression") and isinstance(model_wrapper, ModelInferenceWrapper):
        from leech.losses import RegressionHead

        cl_state = config.get("cl_regression_head_state_dict")
        if cl_state is not None:
            repr_dim = model_wrapper.enable_repr_capture()
            _cl_head = RegressionHead(input_dim=repr_dim)
            _cl_head.load_state_dict(cl_state)
            _cl_head.to(device)
            _cl_head.eval()
            logger.info(f"CL regression head loaded (repr_dim={repr_dim})")

    # Enable TF32 matmul for better performance on Ampere+ GPUs.
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")

    # torch.compile decision deferred until after BAM read count is known (see below)

    # Skip feature computation when model doesn't need them (big speedup)
    # But always compute when signal_in_channels > 1 (needed for kmer residual)
    signal_in_channels = config.get("signal_in_channels", 1)
    compute_features = requires_features or signal_in_channels > 1

    # Load reference sequences for reference-anchored mode and/or reference-based motif search
    reference_sequences = None
    if anchor == "reference" or motif is not None:
        from leech.io import get_reference_sequences

        reference_sequences = get_reference_sequences(bam_path, reference_fasta)
        logger.info(f"Loaded {len(reference_sequences)} reference sequences")

    # Create motif searcher (reference-based when reference_sequences available)
    motif_searcher = get_motif_searcher(
        mode="fasta" if reference_sequences else "bam",
        reference_sequences=reference_sequences,
        skip_indels=config.get("skip_motif_indels", False),
        anchor=anchor,
    )

    # Prepare class_names_str for multiclass (shared across mega-batches)
    class_names_str = None
    if is_multiclass:
        if int_to_label:
            class_names = [int_to_label[i] for i in range(num_out)]
            class_names_str = ",".join(class_names)
        else:
            class_names_str = ",".join(str(i) for i in range(num_out))

    logger.info(f"Streaming inference with read_batch_size={read_batch_size}")

    n_total_reads = count_bam_reads(bam_path)
    n_total_mega_batches = math.ceil(n_total_reads / read_batch_size) if n_total_reads > 0 else 0
    logger.info(
        f"BAM contains ~{n_total_reads} mapped reads -> ~{n_total_mega_batches} mega-batches "
        f"of {read_batch_size}"
    )
    mega_batch_idx = 0

    # torch.compile the model for faster inference (CUDA graph + kernel fusion).
    # Auto-skip for small runs (<5000 reads) where compilation overhead (~15-30s)
    # outweighs the speedup. Also skip when repr capture hooks are active or
    # when --no-compile is set.
    _COMPILE_THRESHOLD = 5000
    _has_repr_hook = (
        isinstance(model_wrapper, ModelInferenceWrapper) and model_wrapper._repr_hook is not None
    )
    if no_compile:
        logger.info("torch.compile disabled (--no-compile)")
    elif n_total_reads < _COMPILE_THRESHOLD:
        logger.info(
            f"torch.compile auto-skipped ({n_total_reads} reads < {_COMPILE_THRESHOLD} threshold)"
        )
    elif (
        isinstance(model_wrapper, ModelInferenceWrapper)
        and device.startswith("cuda")
        and hasattr(torch, "compile")
        and not _has_repr_hook
    ):
        try:
            model_wrapper.model = torch.compile(model_wrapper.model, mode="reduce-overhead")  # ty: ignore[invalid-assignment]
            logger.info("torch.compile enabled (mode=reduce-overhead)")
        except Exception as e:
            logger.warning(f"torch.compile failed, using eager mode: {e}")
    elif _has_repr_hook:
        logger.info("torch.compile skipped (repr capture hooks incompatible with CUDA graphs)")

    if num_workers > 0:
        # ---- Parallel path (mega-batched) ----
        from leech.io.bam_reader import ReadInfo

        logger.info(f"Parallel inference with {num_workers} workers")

        inf_config = InferenceConfig(
            pod5_path=pod5_path,
            signal=SignalConfig(
                reverse_signal=reverse_signal,
                anchor=anchor,
                norm_method=norm_method,
                pa_mean=pa_mean,
                pa_stdev=pa_stdev,
                refine_signal_map=refine_signal_map,
                signal_refiner=signal_refiner,
            ),
            motif=MotifConfig(
                motif=motif,
                motif_offset=motif_offset,
                reference_sequences=reference_sequences,
            ),
            chunk=ChunkConfig(
                base_justify=base_justify,
                feature_start=_feature_start,
                feature_end=_feature_end,
                signal_context=signal_context,
                kmer_context=kmer_context,
                recover_softclip_signal=config.get("recover_softclip_signal", False),
            ),
            seq_encoding=seq_encoding,
            signal_kmer_context=signal_kmer_context,
            signal_len=signal_len,
            kmer_len=kmer_len,
            dwell_offset=dwell_offset,
            wide_features=wide_features,
            requires_features=requires_features,
            signal_in_channels=signal_in_channels,
        )

        calibration = config.get("calibration") if is_multiclass else None
        _batch_fn_p = (
            functools.partial(
                _run_batch_multiclass,
                calibration=calibration,
                cl_regression_head=_cl_head,
            )
            if is_multiclass
            else _run_batch
        )

        with Progress() as progress:
            task = progress.add_task("[cyan]Running inference...", total=None)

            with mp.Pool(processes=num_workers) as pool:
                for aln_batch in iter_bam_batches(
                    bam_path, batch_size=read_batch_size, min_mapq=min_mapq
                ):
                    logger.info(f"Mega-batch: {len(aln_batch)} alignments read from BAM")

                    # Build ReadInfo objects from this mega-batch
                    read_infos = []
                    for aln in aln_batch:
                        try:
                            read_infos.append(ReadInfo(aln))
                        except Exception as e:
                            logger.warning(f"Skipping read {aln.query_name}: {e}")

                    logger.info(
                        f"Built {len(read_infos)} ReadInfo objects, "
                        f"dispatching to {num_workers} workers"
                    )

                    if not read_infos:
                        if bam_out is not None:
                            for aln in aln_batch:
                                bam_out.write(aln)
                        total_reads += len(aln_batch)
                        continue

                    # Split into worker sub-batches and dispatch
                    worker_batches = [
                        read_infos[i : i + chunk_size]
                        for i in range(0, len(read_infos), chunk_size)
                    ]
                    worker_args = [(wb, inf_config) for wb in worker_batches]

                    pending: dict[str, list] = {}
                    for worker_results in pool.imap_unordered(_inference_worker, worker_args):
                        signals_buf = []
                        seqs_buf = []
                        feats_buf = []
                        meta_buf = []

                        for read_id, base_idx, sig, enc_seq, feat in worker_results:
                            signals_buf.append(sig)
                            seqs_buf.append(enc_seq)
                            feats_buf.append(feat)
                            meta_buf.append((read_id, base_idx))

                            if len(signals_buf) >= batch_size:
                                _batch_fn_p(
                                    signals_buf,
                                    seqs_buf,
                                    feats_buf,
                                    meta_buf,
                                    model_wrapper,
                                    requires_features,
                                    device,
                                    pending,
                                )
                                signals_buf, seqs_buf, feats_buf, meta_buf = [], [], [], []

                        if signals_buf:
                            _batch_fn_p(
                                signals_buf,
                                seqs_buf,
                                feats_buf,
                                meta_buf,
                                model_wrapper,
                                requires_features,
                                device,
                                pending,
                            )

                    # Write this mega-batch's predictions
                    if tsv_writer is not None:
                        batch_preds = tsv_writer.write_predictions(aln_batch, pending, int_to_label)
                    else:
                        batch_preds = _write_mega_batch_predictions(
                            aln_batch,
                            pending,
                            bam_out,
                            is_multiclass,
                            int_to_label,
                            class_names_str,
                            raw,
                            min_confidence,
                            min_margin,
                        )
                    total_reads += len(aln_batch)
                    total_predictions += batch_preds
                    mega_batch_idx += 1
                    logger.info(
                        f"Mega-batch {mega_batch_idx}/{n_total_mega_batches} complete: "
                        f"wrote {batch_preds} predictions for {len(aln_batch)} reads"
                    )

                    progress.update(
                        task,
                        advance=0,
                        description=(
                            f"[cyan]Processed {total_reads} reads "
                            f"({total_predictions} predictions)..."
                        ),
                    )

    else:
        # ---- Sequential path (mega-batched, double-buffered GPU) ----
        from concurrent.futures import Future, ThreadPoolExecutor

        batch_signals: list[np.ndarray] = []
        batch_seqs: list[np.ndarray] = []
        batch_feats: list[np.ndarray | None] = []
        batch_meta: list[tuple[str, int]] = []
        pending: dict[str, list] = {}
        _shape_validated = False

        calibration = config.get("calibration") if is_multiclass else None
        _batch_fn = (
            functools.partial(
                _run_batch_multiclass,
                calibration=calibration,
                cl_regression_head=_cl_head,
            )
            if is_multiclass
            else _run_batch
        )

        _gpu_executor = ThreadPoolExecutor(max_workers=1)
        _gpu_future: Future | None = None
        _bam_write_executor = ThreadPoolExecutor(max_workers=1)
        _bam_write_future: Future | None = None

        def _flush_batch() -> None:
            """Submit accumulated chunks to GPU thread (double-buffered)."""
            nonlocal batch_signals, batch_seqs, batch_feats, batch_meta, _gpu_future
            if not batch_signals:
                return
            # Wait for previous GPU batch before submitting next
            if _gpu_future is not None:
                _gpu_future.result()
            # Capture current batch and reset buffers
            sigs, seqs, feats, meta = (
                batch_signals,
                batch_seqs,
                batch_feats,
                batch_meta,
            )
            batch_signals, batch_seqs, batch_feats, batch_meta = [], [], [], []
            # Submit GPU work -- runs while main thread continues extraction
            _gpu_future = _gpu_executor.submit(
                _batch_fn,
                sigs,
                seqs,
                feats,
                meta,
                model_wrapper,
                requires_features,
                device,
                pending,
            )

        def _drain_gpu() -> None:
            """Wait for any in-flight GPU batch to complete."""
            nonlocal _gpu_future
            if _gpu_future is not None:
                _gpu_future.result()
                _gpu_future = None

        seq_signal_config = SignalConfig(
            reverse_signal=reverse_signal,
            anchor=anchor,
            norm_method=norm_method,
            pa_mean=pa_mean,
            pa_stdev=pa_stdev,
            refine_signal_map=refine_signal_map,
            signal_refiner=signal_refiner,
            compute_features=compute_features,
        )
        seq_chunk_config = ChunkConfig(
            base_justify=base_justify,
            feature_start=_feature_start,
            feature_end=_feature_end,
            signal_context=signal_context,
            kmer_context=kmer_context,
            recover_softclip_signal=config.get("recover_softclip_signal", False),
        )

        # Extraction thread count + rust setup (all three shared with
        # run_bundle_inference via helpers.py).
        _MAX_THREADS = 8  # sensible cap to avoid oversubscription on shared nodes
        _avail_cpus = cap_rayon_threads_for_slurm(max_cap=_MAX_THREADS)
        n_extract = max(1, _avail_cpus - 6)  # reserve headroom for main + GPU I/O

        logger.info(f"Sequential path with {n_extract} extraction threads, double-buffered GPU")

        (
            _use_rust_extraction,
            _rs_extract_inference_chunks,
            _rs_preload_pod5_signals,
            _rs_extract_chunks_from_preloaded,
        ) = check_rust_extraction_available(
            backend, norm_method, seq_chunk_config.recover_softclip_signal
        )
        if _use_rust_extraction:
            logger.info("Using Rust monolithic extraction (escapepod-rs + leech_core)")
        else:
            logger.info(
                "Using Python extraction path"
                + (" (forced via --backend python)" if backend == "python" else "")
            )

        # Check for prefetch support (split POD5 preload + chunk extraction)
        _has_prefetch = (
            _use_rust_extraction
            and _rs_preload_pod5_signals is not None
            and _rs_extract_chunks_from_preloaded is not None
        )
        if _has_prefetch:
            logger.info("POD5 prefetch pipeline enabled (overlapped I/O)")

        def _extract_one_read(
            aln: pysam.AlignedSegment,
        ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple[str, int]]]:
            """Extract ready-to-batch chunks from one alignment. Thread-safe."""
            read_id = aln.query_name
            read_seq = aln.query_sequence
            if read_id is None or read_seq is None:
                return []

            try:
                move_table = extract_move_table(aln)
                raw_signal, pod5_metadata = pod5_reader.get_signal(read_id)

                ref_seq = None
                cigar_tuples = None
                if seq_signal_config.anchor == "reference":
                    if reference_sequences and aln.reference_name in reference_sequences:
                        full_ref = reference_sequences[aln.reference_name]
                        ref_seq = full_ref[aln.reference_start : aln.reference_end]
                    else:
                        try:
                            ref_seq = aln.get_reference_sequence()
                        except Exception:
                            ref_seq = None
                    cigar_tuples = aln.cigartuples

                leech_read = build_leech_read(
                    read_id=read_id,
                    sequence=read_seq,
                    raw_signal=raw_signal,
                    move_table=move_table,
                    signal_config=seq_signal_config,
                    metadata={},
                    reference_sequence=ref_seq,
                    cigar_tuples=cigar_tuples,
                    cal_offset=pod5_metadata.get("calibration_offset"),
                    cal_scale=pod5_metadata.get("calibration_scale"),
                )
            except Exception as e:
                logger.warning(f"Skipping read {read_id}: {e}")
                return []

            # Find positions to predict
            assert motif is not None
            positions = [
                pos + motif_offset
                for pos in motif_searcher.find_motif_positions(
                    leech_read.read_id, leech_read.sequence, aln, motif
                )
            ]

            results: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple[str, int]]] = []
            for base_idx in positions:
                chunk = leech_read.get_chunk(base_idx, config=seq_chunk_config)
                if chunk is None:
                    continue

                # Signal (with optional kmer residual channel)
                signal_array = chunk["signal"]
                assert isinstance(signal_array, np.ndarray)
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

                # Sequence
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
                    feat = features_array.astype(np.float32)
                    if wide_features:
                        pass
                    elif feat.size > 0 and feat.shape[1] > kmer_len:
                        fs = chunk.get("feature_start", -_kmer_context)
                        s = (-_kmer_context - fs) + dwell_offset
                        feat = feat[:, s : s + kmer_len]

                results.append((sig, seq_arr, feat, (read_id, base_idx)))
            return results

        _extract_pool = ThreadPoolExecutor(max_workers=n_extract)

        # Shared Rust kwargs + metadata collection (shared with
        # run_bundle_inference via helpers.py).
        assert motif is not None  # type narrowing for helpers below
        _rs_kwargs = build_rust_extraction_kwargs(
            signal_context=signal_context,
            kmer_context=kmer_context,
            signal_len=signal_len,
            compute_features=compute_features,
            reverse_signal=reverse_signal,
            feature_start=_feature_start,
            feature_end=_feature_end,
            anchor=anchor,
            seq_encoding=seq_encoding,
            signal_kmer_context=signal_kmer_context,
            refine_signal_map=refine_signal_map,
            signal_refiner=signal_refiner,
            refine_half_bandwidth=config.get("refine_half_bandwidth", 5),
            refine_scale_iters=config.get("refine_scale_iters", 2),
            signal_in_channels=signal_in_channels,
        )

        def _collect_bam_metadata(aln_batch: list) -> tuple:
            return collect_bam_metadata_for_rust(
                aln_batch,
                motif=motif,
                motif_offset=motif_offset,
                motif_searcher=motif_searcher,
                anchor=anchor,
                reference_sequences=reference_sequences,
            )

        _SUB_BATCH_SIZE = 50_000  # Sub-batch Rust extraction for continuous GPU feeding

        def _extract_chunks_from_preloaded(preloaded, rs_meta):
            """Yield chunks in sub-batches for continuous GPU feeding.

            Instead of extracting all reads at once (blocking GPU for ~2 min),
            process ~25K reads at a time (~15-20s each) so chunks flow to GPU
            after each sub-batch completes.
            """
            (
                rs_rids,
                rs_seqs,
                rs_strides,
                rs_mvs,
                rs_ns,
                rs_trims,
                rs_motifs,
                rs_cigars,
                rs_refs,
            ) = rs_meta
            assert _rs_extract_chunks_from_preloaded is not None
            n = len(rs_rids)
            for start in range(0, n, _SUB_BATCH_SIZE):
                end = min(start + _SUB_BATCH_SIZE, n)
                if n > _SUB_BATCH_SIZE:
                    logger.info(
                        f"  Sub-batch {start // _SUB_BATCH_SIZE + 1}/"
                        f"{(n + _SUB_BATCH_SIZE - 1) // _SUB_BATCH_SIZE}: "
                        f"reads {start}-{end} of {n}"
                    )
                sub_chunks = _rs_extract_chunks_from_preloaded(
                    preloaded,
                    read_ids=rs_rids[start:end],
                    sequences=rs_seqs[start:end],
                    mv_strides=rs_strides[start:end],
                    mv_arrays=rs_mvs[start:end],
                    num_samples_list=rs_ns[start:end],
                    trim_offsets=rs_trims[start:end],
                    motif_positions=rs_motifs[start:end],
                    cigar_tuples=rs_cigars[start:end] if anchor == "reference" else None,
                    reference_sequences=rs_refs[start:end] if anchor == "reference" else None,
                    **_rs_kwargs,
                )
                yield from sub_chunks

        def _consume_rust_chunks(chunks):
            """Iterate Rust chunks into batch buffers, flushing to GPU as needed."""
            nonlocal _shape_validated
            for sig, seq_arr, feat, read_id, base_idx in chunks:
                if signal_in_channels > 1 and sig.ndim == 1:
                    sig = sig.reshape(signal_in_channels, -1)
                if not _shape_validated:
                    validate_inference_shapes(sig, feat, config)
                    _shape_validated = True
                batch_signals.append(sig)
                batch_seqs.append(seq_arr)
                batch_feats.append(feat)
                batch_meta.append((read_id, base_idx))
                if len(batch_signals) >= batch_size:
                    _flush_batch()

        def _wait_for_bam_write():
            """Wait for any in-flight async BAM write to complete."""
            nonlocal _bam_write_future
            if _bam_write_future is not None:
                _bam_write_future.result()
                _bam_write_future = None

        def _finalize_mega_batch(aln_batch_to_write):
            """Flush GPU, submit async BAM write, update counters.

            BAM writes are overlapped with the next mega-batch's extraction.
            We serialize writes (wait for previous) since pysam is not thread-safe.
            """
            nonlocal total_reads, total_predictions, mega_batch_idx
            nonlocal pending, _bam_write_future
            _flush_batch()
            _drain_gpu()
            # Wait for any previous BAM write (serializes bam_out access)
            _wait_for_bam_write()
            # Swap pending -> snapshot; next mega-batch gets a fresh dict
            write_pending = pending
            pending = {}
            batch_preds = len(write_pending)
            # Submit write to background thread
            if tsv_writer is not None:
                _bam_write_future = _bam_write_executor.submit(
                    tsv_writer.write_predictions,
                    aln_batch_to_write,
                    write_pending,
                    int_to_label,
                )
            else:
                _bam_write_future = _bam_write_executor.submit(
                    _write_mega_batch_predictions,
                    aln_batch_to_write,
                    write_pending,
                    bam_out,
                    is_multiclass,
                    int_to_label,
                    class_names_str,
                    raw,
                    min_confidence,
                    min_margin,
                )
            total_reads += len(aln_batch_to_write)
            total_predictions += batch_preds
            mega_batch_idx += 1
            logger.info(
                f"Mega-batch {mega_batch_idx}/{n_total_mega_batches} complete: "
                f"wrote {batch_preds} predictions for {len(aln_batch_to_write)} reads"
            )

        import queue as _queue
        import threading as _threading
        import time as _time

        _t_total_start = _time.perf_counter()

        with Progress() as progress:
            task = progress.add_task("[cyan]Running inference...", total=None)

            # Skip opening Python DatasetReader when Rust handles all POD5 I/O.
            # Opening a 40+ GB POD5 file in Python just to index it is expensive
            # and completely unused on the Rust extraction paths.
            _pod5_ctx = nullcontext() if _use_rust_extraction else POD5Reader(pod5_path)
            with _pod5_ctx as pod5_reader:
                if _has_prefetch:
                    # ---- Queue-based extraction pipeline ----
                    # A producer thread handles BAM reading, metadata collection,
                    # POD5 prefetch, and Rust extraction. It pushes (aln_batch, chunks)
                    # to a bounded queue. The main thread consumes from the queue,
                    # runs GPU inference, and writes results.
                    #
                    # Benefits over the previous prefetch pipeline:
                    # - Consumer (GPU + finalize) runs concurrently with producer
                    # - Metadata for batch N+1 overlaps with extraction of batch N
                    # - No synchronization gap between mega-batches
                    assert _rs_preload_pod5_signals is not None
                    assert _rs_extract_chunks_from_preloaded is not None

                    _SENTINEL = object()
                    _extraction_queue: _queue.Queue = _queue.Queue(maxsize=2)
                    _producer_error: BaseException | None = None

                    def _extraction_producer():
                        """Background thread: reads BAM -> metadata -> prefetch -> extract -> queue."""
                        nonlocal _producer_error
                        try:
                            _meta_exec = ThreadPoolExecutor(max_workers=1)
                            _prefetch_exec = ThreadPoolExecutor(max_workers=1)

                            # Pipeline state for overlapping metadata and prefetch
                            _prev = None  # (prefetch_future, rs_meta, aln_batch) or None
                            _meta_future = None

                            for aln_batch in iter_bam_batches(
                                bam_path, batch_size=read_batch_size, min_mapq=min_mapq
                            ):
                                if _prev is not None:
                                    p_future, p_meta, p_aln = _prev
                                    preloaded = p_future.result()
                                    p_rids = p_meta[0]

                                    # Overlap: start metadata for CURRENT batch
                                    # while extracting PREVIOUS (Rust releases GIL)
                                    _meta_future = _meta_exec.submit(
                                        _collect_bam_metadata, aln_batch
                                    )

                                    chunk_list: list = []
                                    if p_rids:
                                        for chunk in _extract_chunks_from_preloaded(
                                            preloaded, p_meta
                                        ):
                                            chunk_list.append(chunk)

                                    # Push to queue (blocks if queue full -- backpressure)
                                    _extraction_queue.put((p_aln, chunk_list, len(p_rids)))

                                    # Get metadata result (should be done by now)
                                    rs_meta = _meta_future.result()
                                else:
                                    # First batch -- no previous, collect metadata sync
                                    rs_meta = _collect_bam_metadata(aln_batch)

                                # Submit prefetch for current batch (overlaps with
                                # consumer processing + next BAM read)
                                cur_future = _prefetch_exec.submit(
                                    _rs_preload_pod5_signals,
                                    str(pod5_path),
                                    rs_meta[0],
                                )
                                _prev = (cur_future, rs_meta, aln_batch)

                            # Process final batch
                            if _prev is not None:
                                p_future, p_meta, p_aln = _prev
                                preloaded = p_future.result()
                                p_rids = p_meta[0]
                                chunk_list = []
                                if p_rids:
                                    for chunk in _extract_chunks_from_preloaded(preloaded, p_meta):
                                        chunk_list.append(chunk)
                                _extraction_queue.put((p_aln, chunk_list, len(p_rids)))

                            _meta_exec.shutdown(wait=True)
                            _prefetch_exec.shutdown(wait=True)
                        except BaseException as exc:
                            _producer_error = exc
                        finally:
                            _extraction_queue.put(_SENTINEL)

                    _producer_thread = _threading.Thread(target=_extraction_producer, daemon=True)
                    _producer_thread.start()
                    logger.info("Queue-based extraction pipeline started (producer thread)")

                    # Consumer loop: pull from queue -> GPU -> finalize
                    while True:
                        item = _extraction_queue.get()
                        if item is _SENTINEL:
                            break
                        aln_batch, chunk_list, n_rids = item
                        _t_mb_start = _time.perf_counter()

                        logger.info(
                            f"Mega-batch: {len(aln_batch)} alignments, {n_rids} for Rust extraction"
                        )

                        if chunk_list:
                            _consume_rust_chunks(iter(chunk_list))
                        _t_consume = _time.perf_counter()

                        _finalize_mega_batch(aln_batch)
                        _t_finalize = _time.perf_counter()

                        logger.debug(
                            f"  Timing: consume+gpu={_t_consume - _t_mb_start:.2f}s "
                            f"finalize={_t_finalize - _t_consume:.2f}s"
                        )
                        progress.update(
                            task,
                            advance=0,
                            description=(
                                f"[cyan]Processed {total_reads} reads "
                                f"({total_predictions} predictions)..."
                            ),
                        )

                    _producer_thread.join()
                    if _producer_error is not None:
                        raise RuntimeError("Extraction producer thread failed") from _producer_error

                else:
                    # ---- Non-prefetch fallback paths ----
                    for aln_batch in iter_bam_batches(
                        bam_path, batch_size=read_batch_size, min_mapq=min_mapq
                    ):
                        if _use_rust_extraction:
                            # ---- Rust monolithic hot path (no prefetch) ----
                            _t_mb_start = _time.perf_counter()
                            rs_meta = _collect_bam_metadata(aln_batch)
                            _t_meta = _time.perf_counter()
                            rs_read_ids = rs_meta[0]

                            logger.info(
                                f"Mega-batch: {len(aln_batch)} alignments, "
                                f"{len(rs_read_ids)} for Rust extraction"
                            )

                            if rs_read_ids:
                                assert _rs_extract_inference_chunks is not None
                                chunks = _rs_extract_inference_chunks(
                                    str(pod5_path),
                                    read_ids=rs_read_ids,
                                    sequences=rs_meta[1],
                                    mv_strides=rs_meta[2],
                                    mv_arrays=rs_meta[3],
                                    num_samples_list=rs_meta[4],
                                    trim_offsets=rs_meta[5],
                                    motif_positions=rs_meta[6],
                                    cigar_tuples=(rs_meta[7] if anchor == "reference" else None),
                                    reference_sequences=(
                                        rs_meta[8] if anchor == "reference" else None
                                    ),
                                    **_rs_kwargs,
                                )
                                _consume_rust_chunks(chunks)
                            _t_extract = _time.perf_counter()
                            logger.debug(
                                f"  Timing: metadata={_t_meta - _t_mb_start:.2f}s "
                                f"extract+gpu={_t_extract - _t_meta:.2f}s"
                            )
                        else:
                            # ---- Python extraction path ----
                            batch_read_ids = [
                                aln.query_name
                                for aln in aln_batch
                                if aln.query_name is not None and aln.query_sequence is not None
                            ]
                            if batch_read_ids:
                                pod5_reader.preload(batch_read_ids)

                            logger.info(
                                f"Mega-batch: {len(aln_batch)} alignments, "
                                f"{len(batch_read_ids)} preloaded from POD5"
                            )

                            # Parallel extraction -> GPU batching
                            for chunks in _extract_pool.map(_extract_one_read, aln_batch):
                                for sig, seq_arr, feat, meta in chunks:
                                    if not _shape_validated:
                                        validate_inference_shapes(sig, feat, config)
                                        _shape_validated = True

                                    batch_signals.append(sig)
                                    batch_seqs.append(seq_arr)
                                    batch_feats.append(feat)
                                    batch_meta.append(meta)

                                    if len(batch_signals) >= batch_size:
                                        _flush_batch()

                        _finalize_mega_batch(aln_batch)

                        progress.update(
                            task,
                            advance=0,
                            description=(
                                f"[cyan]Processed {total_reads} reads "
                                f"({total_predictions} predictions)..."
                            ),
                        )

        _t_total = _time.perf_counter() - _t_total_start
        logger.info(
            f"Inference wall time: {_t_total:.1f}s "
            f"({total_reads} reads, {total_predictions} predictions, "
            f"{total_reads / _t_total:.0f} reads/s)"
        )

        _extract_pool.shutdown(wait=False)
        _gpu_executor.shutdown(wait=False)
        _wait_for_bam_write()  # Ensure final BAM write completes before close
        _bam_write_executor.shutdown(wait=False)

    if tsv_writer is not None:
        tsv_writer.close()
    if bam_out is not None:
        bam_out.close()

    logger.info("Inference complete!")
    logger.info(f"Reads processed: {total_reads}")
    logger.info(f"Total predictions: {total_predictions}")
    logger.info(f"Output written to: {output_path}")
