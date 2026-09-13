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
    BatchAccumulator,
    _check_config_consistency,
    _encode_sequence_for_inference,
    _htslib_write_threads,
    _run_batch,
    _run_batch_multiclass,
    _write_mega_batch_predictions,
    build_rust_extraction_kwargs,
    cap_rayon_threads_for_slurm,
    check_rust_extraction_available,
    collect_bam_metadata_for_rust,
    load_model_auto,
    prepare_inference_features,
    prepare_signal_channels,
    resolve_pipeline_depths,
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
                    require_query_mapping=config.motif.require_query_mapping,
                )
                aln = read_info.to_mock_alignment()
                positions = [
                    pos + config.motif.motif_offset
                    for pos in searcher.find_motif_positions(
                        read_info.read_id,
                        # The sequence chunks are cut from -- under
                        # anchor="reference" that is the aligned reference
                        # slice, which `leech_read.sequence` already is.
                        # `read_info.sequence` is the basecall, a different
                        # coordinate frame.
                        leech_read.sequence,
                        aln,
                        config.motif.motif,
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
                sig = prepare_signal_channels(chunk, config.signal_len)

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
                        feat = prepare_inference_features(
                            feat_arr.astype(np.float32),
                            kmer_len=config.kmer_len,
                            feature_start=chunk.get("feature_start"),
                            dwell_offset=config.dwell_offset,
                            wide_features=config.wide_features,
                        )

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
        # BGZF compression in htslib's own thread pool rather than on the
        # writer thread. The writer is the one stage here that is pure Python
        # plus a blocking C call: every record pays `set_tag` under the GIL and
        # then a synchronous deflate. Handing the deflate to htslib threads
        # takes it off the GIL entirely, which matters because the GPU worker
        # needs the GIL for every eager kernel launch and the consumer holds it
        # in a per-chunk loop.
        bam_out = pysam.AlignmentFile(
            str(output_path), "wb", template=bam_in, threads=_htslib_write_threads()
        )
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
        # Recorded by `data prepare` and carried through `model train`. Without
        # it, a corpus prepared with --no-require-query-mapping was scored at
        # predict time with the gate back on, i.e. on a different read
        # population than the model was trained on.
        require_query_mapping=config.get("require_query_mapping", True),
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

    # torch.compile the model for faster inference (kernel fusion, and CUDA
    # graphs when nothing blocks them). Auto-skip for runs too short to amortize
    # compilation; also skip when --no-compile is set.
    #
    # Compiling this model costs ~60s wall -- measured on the 20-class
    # production bundle on an A30, from the compile call to the first
    # mega-batch landing -- and buys ~10% end to end at the size this pipeline
    # is deployed at (16 logical CPUs, i.e. 8 physical cores, four jobs sharing
    # a 64-core/4-GPU node), because extraction rather than the GPU is that
    # configuration's bottleneck. Break-even is therefore around ten minutes of
    # inference, ~2M reads at the ~3,800 reads/s it sustains there. The old
    # 5,000-read threshold turned it on for essentially every run: a 176k-read
    # sample measured 45.8s uncompiled against 101.5s compiled, 60s of
    # compilation against a ~4s gain.
    _COMPILE_THRESHOLD = 2_000_000
    _compiled = False
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
    ):
        # Assign `forward_module`, not `model`. `forward_batch` calls
        # `self.forward_module` (see ModelInferenceWrapper.__init__); replacing
        # `self.model` leaves `forward_module` bound to the original eager
        # module, so compiling was a no-op on every inference path that went
        # through this branch. Measured on the production 20-class bundle,
        # batch 1024: 16,556 chunks/s assigning `.model` against 16,571
        # chunks/s with no compile at all -- 1.00x, i.e. nothing.
        #
        # Only CUDA graphs are incompatible with a repr-capture hook, because
        # the hook's Python side effect cannot be replayed by a graph. Plain
        # inductor just breaks the graph there and still fuses everything
        # around it, so a bundle with a CL-regression head gets essentially the
        # whole win rather than none of it: same benchmark, 30,019 chunks/s
        # compiled-with-hook against 30,227 chunks/s reduce-overhead-no-hook
        # and 16,571 eager -- 1.81x vs 1.83x. The old branch skipped compile
        # outright whenever the hook was present, which is every production
        # multiclass bundle.
        _mode = "default" if _has_repr_hook else "reduce-overhead"
        try:
            model_wrapper.forward_module = torch.compile(model_wrapper.model, mode=_mode)  # ty: ignore[invalid-assignment]
            _compiled = True
            logger.info(
                f"torch.compile enabled (mode={_mode}"
                + (", repr capture hook present)" if _has_repr_hook else ")")
            )
        except Exception as e:
            logger.warning(f"torch.compile failed, using eager mode: {e}")

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
                skip_motif_indels=config.get("skip_motif_indels", False),
                require_query_mapping=config.get("require_query_mapping", True),
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

        def _run_worker_batch(sigs, seqs, feats, meta) -> None:
            """Flush callback: score one batch into the current mega-batch's ``pending``."""
            _batch_fn_p(
                sigs,
                seqs,
                feats,
                meta,
                model_wrapper,
                requires_features,
                device,
                pending,
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
                    accumulator = BatchAccumulator(batch_size, _run_worker_batch)
                    for worker_results in pool.imap_unordered(_inference_worker, worker_args):
                        for read_id, base_idx, sig, enc_seq, feat in worker_results:
                            accumulator.add(sig, enc_seq, feat, (read_id, base_idx))
                        # One worker's results never share a batch with the
                        # next worker's -- imap_unordered hands them back
                        # whole, and this is the batching the path has always
                        # had.
                        accumulator.flush()

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
        # ---- Sequential path (mega-batched, pipelined GPU) ----
        from collections import deque
        from concurrent.futures import Future, ThreadPoolExecutor

        (
            _extract_chunk_reads,
            _extract_queue_depth,
            _gpu_in_flight,
            _write_queue_depth,
        ) = resolve_pipeline_depths(read_batch_size=read_batch_size, batch_size=batch_size)

        pending: dict[str, list] = {}
        _shape_validated = False

        calibration = config.get("calibration") if is_multiclass else None
        # Only pad when compiled: a fixed batch shape is what keeps
        # `torch.compile` from recompiling on every mega-batch's short tail
        # (see `_stack_to_device`). Uncompiled, padding would just be wasted
        # forward work on rows nobody reads.
        _pad_to = batch_size if _compiled else None
        _batch_fn = (
            functools.partial(
                _run_batch_multiclass,
                calibration=calibration,
                cl_regression_head=_cl_head,
                pad_to=_pad_to,
            )
            if is_multiclass
            else functools.partial(_run_batch, pad_to=_pad_to)
        )

        # One GPU worker, several batches queued behind it. `max_workers=1` is
        # load-bearing, not conservatism: it keeps GPU calls sequential, keeps
        # `_stack_to_device`'s thread-local pinned staging buffer single-owner,
        # and makes the order in which batches append to `pending[read_id]`
        # exactly the order they were extracted in. The depth is what changed --
        # see `_gpu_in_flight` below.
        _gpu_executor = ThreadPoolExecutor(max_workers=1)
        _gpu_futures: deque[Future] = deque()

        def _submit_gpu_batch(sigs, seqs, feats, meta) -> None:
            """Flush callback: hand one batch to the GPU thread.

            The accumulator has already detached these buffers, so the GPU
            thread owns them and extraction can keep filling the next batch.

            This used to wait for the *immediately* preceding batch before
            submitting, which is a one-deep handshake: the thread that fills
            batches could never be more than one batch ahead of the GPU, so
            every GPU batch stalled extraction for its full duration -- and a
            "GPU batch" here is mostly host work (stack, D2H, scatter), not
            kernel time. Queuing `_gpu_in_flight` batches instead lets
            extraction run a whole prep cycle ahead; the executor still runs
            them one at a time, in order.
            """
            while len(_gpu_futures) >= _gpu_in_flight:
                _gpu_futures.popleft().result()
            _gpu_futures.append(
                _gpu_executor.submit(
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
            )

        accumulator = BatchAccumulator(batch_size, _submit_gpu_batch)

        def _drain_gpu() -> None:
            """Wait for every in-flight GPU batch to complete.

            Called at a mega-batch boundary, where `pending` is about to be
            snapshotted for writing, so every batch of this mega-batch must
            have landed in it.
            """
            while _gpu_futures:
                _gpu_futures.popleft().result()

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
                sig = prepare_signal_channels(chunk, signal_len)

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
                    feat = prepare_inference_features(
                        features_array.astype(np.float32),
                        kmer_len=kmer_len,
                        feature_start=chunk.get("feature_start"),
                        dwell_offset=dwell_offset,
                        wide_features=wide_features,
                    )

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
            base_justify=base_justify,
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

        def _extract_chunks_from_preloaded(preloaded, rs_meta):
            """Yield one *list* of chunks per extraction sub-batch.

            Yielding per sub-batch rather than per chunk is what lets the
            producer hand each piece to the consumer as it is built, instead of
            accumulating a whole mega-batch first. The sub-batch size is
            `_extract_chunk_reads` (see `resolve_pipeline_depths`); it was a
            fixed 50,000, which is five times the default `read_batch_size`, so
            this loop always ran exactly once and the "continuous GPU feeding"
            it was written for never happened.
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
            for start in range(0, n, _extract_chunk_reads):
                end = min(start + _extract_chunk_reads, n)
                yield _rs_extract_chunks_from_preloaded(
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

        def _consume_rust_chunks(chunks):
            """Iterate Rust chunks into batch buffers, flushing to GPU as needed."""
            nonlocal _shape_validated
            for sig, seq_arr, feat, read_id, base_idx in chunks:
                if signal_in_channels > 1 and sig.ndim == 1:
                    sig = sig.reshape(signal_in_channels, -1)
                # Rust returns features at the full requested window width, the
                # same as a Python chunk's `features`. Both need the same
                # narrowing to the model's k-mer window -- this loop used to
                # skip it, so a wide-window corpus fed the model the wrong
                # width and `dwell_offset` was inert on the Rust path.
                feat = prepare_inference_features(
                    feat,
                    kmer_len=kmer_len,
                    feature_start=_feature_start,
                    dwell_offset=dwell_offset,
                    wide_features=wide_features,
                )
                if not _shape_validated:
                    validate_inference_shapes(sig, feat, config)
                    _shape_validated = True
                accumulator.add(sig, seq_arr, feat, (read_id, base_idx))

        import queue as _queue
        import threading as _threading
        import time as _time

        # One writer thread behind a bounded queue, rather than one write
        # behind a single future the consumer blocks on.
        #
        # pysam is not thread-safe, so there is still exactly one thread
        # touching `bam_out` and writes still happen in mega-batch order --
        # that part is unchanged and must stay. What changed is who waits: the
        # consumer used to call `.result()` on the previous write before it
        # could submit the next one, so whenever writing a mega-batch took
        # longer than producing one (it does -- pysam tagging is ~7k reads/s),
        # the consumer sat blocked with the GPU queue empty. That stall was
        # half the wall clock on a 176k-read sample, and it is what made GPU
        # utilization arrive in bursts.
        _write_queue: _queue.Queue = _queue.Queue(maxsize=_write_queue_depth)
        _WRITE_SENTINEL = object()
        _writer_error: BaseException | None = None

        def _bam_writer_loop() -> None:
            nonlocal _writer_error
            while True:
                item = _write_queue.get()
                if item is _WRITE_SENTINEL:
                    return
                aln_batch_to_write, write_pending = item
                try:
                    if tsv_writer is not None:
                        tsv_writer.write_predictions(
                            aln_batch_to_write, write_pending, int_to_label
                        )
                    else:
                        _write_mega_batch_predictions(
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
                except BaseException as exc:  # surfaced by _wait_for_bam_write
                    _writer_error = exc
                    return

        _writer_thread = _threading.Thread(target=_bam_writer_loop, daemon=True)
        _writer_thread.start()

        def _queue_bam_write(item) -> None:
            """Hand one mega-batch to the writer, without risking a deadlock.

            A plain blocking `put` on a bounded queue hangs forever if the
            writer has already died: nothing will ever drain it again. Time the
            put out and re-check instead, so a writer failure surfaces as its
            own exception rather than as a stalled run.
            """
            while True:
                if _writer_error is not None:
                    raise RuntimeError("BAM writer thread failed") from _writer_error
                try:
                    _write_queue.put(item, timeout=5.0)
                    return
                except _queue.Full:
                    continue

        def _wait_for_bam_write():
            """Drain every queued write and stop the writer thread."""
            if _writer_thread.is_alive():
                _queue_bam_write(_WRITE_SENTINEL)
            _writer_thread.join()
            if _writer_error is not None:
                raise RuntimeError("BAM writer thread failed") from _writer_error

        def _finalize_mega_batch(aln_batch_to_write):
            """Flush GPU, queue the BAM write, update counters."""
            nonlocal total_reads, total_predictions, mega_batch_idx
            nonlocal pending
            accumulator.flush()
            _drain_gpu()
            # Swap pending -> snapshot; next mega-batch gets a fresh dict
            write_pending = pending
            pending = {}
            batch_preds = len(write_pending)
            # Blocks only once `_write_queue_depth` mega-batches are already
            # queued, which bounds memory without putting the writer on the
            # consumer's critical path.
            _queue_bam_write((aln_batch_to_write, write_pending))
            total_reads += len(aln_batch_to_write)
            total_predictions += batch_preds
            mega_batch_idx += 1
            logger.info(
                f"Mega-batch {mega_batch_idx}/{n_total_mega_batches} complete: "
                f"wrote {batch_preds} predictions for {len(aln_batch_to_write)} reads"
            )

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
                    _extraction_queue: _queue.Queue = _queue.Queue(maxsize=_extract_queue_depth)
                    _producer_error: BaseException | None = None

                    def _emit_mega_batch(preloaded, p_meta, p_aln) -> None:
                        """Push one mega-batch to the consumer, a sub-batch at a time.

                        The consumer needs `aln_batch` only to finalize, so it
                        rides on the last item; every earlier item carries just
                        chunks. Emitting per sub-batch is the point: the GPU
                        starts on the first sub-batch while rayon is still
                        extracting the rest, instead of waiting for the whole
                        mega-batch to be materialized.
                        """
                        p_rids = p_meta[0]
                        sub_batches = (
                            list(_extract_chunks_from_preloaded(preloaded, p_meta))
                            if p_rids
                            else []
                        )
                        if not sub_batches:
                            _extraction_queue.put(([], p_aln, len(p_rids), True))
                            return
                        last = len(sub_batches) - 1
                        for i, chunks in enumerate(sub_batches):
                            _extraction_queue.put(
                                (
                                    chunks,
                                    p_aln if i == last else None,
                                    len(p_rids) if i == last else 0,
                                    i == last,
                                )
                            )

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

                                    # Overlap: start metadata for CURRENT batch
                                    # while extracting PREVIOUS (Rust releases GIL)
                                    _meta_future = _meta_exec.submit(
                                        _collect_bam_metadata, aln_batch
                                    )

                                    _emit_mega_batch(preloaded, p_meta, p_aln)

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
                                _emit_mega_batch(p_future.result(), p_meta, p_aln)

                            _meta_exec.shutdown(wait=True)
                            _prefetch_exec.shutdown(wait=True)
                        except BaseException as exc:
                            _producer_error = exc
                        finally:
                            _extraction_queue.put(_SENTINEL)

                    _producer_thread = _threading.Thread(target=_extraction_producer, daemon=True)
                    _producer_thread.start()
                    logger.info("Queue-based extraction pipeline started (producer thread)")

                    # Consumer loop: pull sub-batches -> GPU; finalize on the
                    # sub-batch flagged as its mega-batch's last.
                    _t_mb_start = _time.perf_counter()
                    while True:
                        item = _extraction_queue.get()
                        if item is _SENTINEL:
                            break
                        chunk_list, aln_batch, n_rids, is_last = item

                        if chunk_list:
                            _consume_rust_chunks(iter(chunk_list))
                        if not is_last:
                            continue
                        _t_consume = _time.perf_counter()

                        assert aln_batch is not None
                        logger.info(
                            f"Mega-batch: {len(aln_batch)} alignments, {n_rids} for Rust extraction"
                        )
                        _finalize_mega_batch(aln_batch)
                        _t_finalize = _time.perf_counter()

                        logger.debug(
                            f"  Timing: consume+gpu={_t_consume - _t_mb_start:.2f}s "
                            f"finalize={_t_finalize - _t_consume:.2f}s"
                        )
                        _t_mb_start = _t_finalize
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

                                    accumulator.add(sig, seq_arr, feat, meta)

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

        # wait=True, not False, and the writer thread is joined rather than
        # left daemonized. Everything here is already drained (`_drain_gpu`
        # above), so waiting costs nothing -- but `wait=False` leaves worker
        # threads alive past the return, and the parallel path above forks an
        # `mp.Pool`. A fork inherits the memory of a process with running
        # threads, including any lock those threads hold, but not the threads
        # themselves, so nothing ever releases it: calling `run_inference` with
        # `num_workers=0` and then with `num_workers>0` in one process hung
        # forever, with no error.
        _extract_pool.shutdown(wait=True)
        _gpu_executor.shutdown(wait=True)
        _wait_for_bam_write()  # Drain queued writes before closing the file

    if tsv_writer is not None:
        tsv_writer.close()
    if bam_out is not None:
        bam_out.close()

    logger.info("Inference complete!")
    logger.info(f"Reads processed: {total_reads}")
    logger.info(f"Total predictions: {total_predictions}")
    logger.info(f"Output written to: {output_path}")
