"""Bundle inference: run_bundle_inference."""

import logging
import math
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import pysam
import torch
from rich.progress import Progress

from leech.configs import ChunkConfig, InferenceConfig, MotifConfig, SignalConfig
from leech.features import extract_move_table
from leech.inference.aggregation import (
    aggregate_one_vs_all,
    aggregate_pairwise,
    aggregate_pairwise_tournament,
    aggregate_pairwise_weighted,
)
from leech.inference.helpers import (
    BatchAccumulator,
    _check_config_consistency,
    _encode_sequence_for_inference,
    _write_prediction_tags,
    build_rust_extraction_kwargs,
    cap_rayon_threads_for_slurm,
    check_rust_extraction_available,
    collect_bam_metadata_for_rust,
    prepare_inference_features,
    validate_inference_shapes,
)
from leech.io.bam_reader import ReadInfo, count_bam_reads, iter_bam_batches
from leech.io.motif_search import get_motif_searcher
from leech.io.pod5_reader import POD5Reader
from leech.model_export import deserialize_exported_model, deserialize_traced_model
from leech.model_loading import _instantiate_model
from leech.models.inference_wrapper import ModelInferenceWrapper, TracedModelWrapper
from leech.preparation.reader import build_leech_read

logger = logging.getLogger("leech.inference")


def _write_bundle_mega_batch(
    aln_batch: list[pysam.AlignedSegment],
    read_probs: dict[str, np.ndarray],
    pairs: list,
    pair_to_idx: dict,
    aggregate_fn,
    pair_names_str: str,
    bam_out: pysam.AlignmentFile,
    raw: bool,
    min_confidence: int,
    min_margin: int,
) -> int:
    """Aggregate per-read probabilities and write one mega-batch of alignments.

    Every alignment is written, tagged or not; the return value is how many
    carried a prediction. All three extraction paths below shared a verbatim
    copy of this loop.
    """
    n_written = 0
    for aln in aln_batch:
        prob_vec = read_probs.get(aln.query_name)
        if prob_vec is None:
            bam_out.write(aln)
            continue

        probs = [float(prob_vec[pair_to_idx[pair]]) for pair in pairs]
        predicted_aa, confidence, _ = aggregate_fn(pairs, probs)

        _write_prediction_tags(
            aln,
            predicted_aa,
            confidence,
            pair_names_str,
            probs,
            raw,
            min_confidence,
            min_margin,
        )
        bam_out.write(aln)
        n_written += 1
    return n_written


def run_bundle_inference(
    bundle_path: Path,
    pod5_path: Path,
    bam_path: Path,
    output_path: Path,
    device: str = "cuda",
    min_mapq: int = 10,
    motif: str | None = None,
    motif_offset: int = 0,
    base_justify: str = "center",
    reverse_signal: bool = True,
    raw: bool = False,
    min_confidence: int = 0,
    min_margin: int = 0,
    aggregation: str = "naive",
    anchor: str = "reference",
    reference_fasta: Path | None = None,
    batch_size: int = 512,
    num_workers: int = 0,
    read_batch_size: int = 10_000,
    backend: str = "auto",
) -> None:
    """
    Run all models from a bundle on each read, aggregate to a single AA prediction.

    Writes output BAM with tags (compact by default, float with --raw):
    - aa:Z:Gly -- predicted amino acid (or "unc" if below threshold)
    - ac:C:238 (compact) or ac:f:0.93 (raw) -- confidence score
    - pn:Z:Ala_Gly,... -- pair names (omitted for uncharged unless --raw)
    - pp:B:B,... (compact) or pp:B:f,... (raw) -- probabilities

    Args:
        bundle_path: Path to bundle .pt file
        pod5_path: Path to POD5 file with raw signal
        bam_path: Path to input BAM file with alignments
        output_path: Path to output BAM file with predictions
        device: Device for inference
        min_mapq: Minimum mapping quality
        motif: Optional motif to filter predictions
        motif_offset: Offset within motif for prediction
        base_justify: Signal justification within focus base
        reverse_signal: Whether to reverse signal for RNA
        raw: If True, also write per-pair probabilities (pn/pp tags)
        anchor: "basecall" or "reference" for reference-anchored mode
        reference_fasta: Path to reference FASTA (for reference-anchored mode)
    """
    logger.info(f"Extraction backend: {backend}")
    if backend == "python":
        import leech.signal_refine as _sr

        _sr.HAS_RUST = False

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    metadata = bundle["metadata"]
    config = bundle["config"]
    pairs = metadata["pairs"]
    comparison_type = metadata["comparison_type"]
    is_torchscript = metadata.get("torchscript", False)

    signal_len = config["signal_len"]
    kmer_len = config["kmer_len"]
    model_type = config.get("model_name", metadata.get("architecture", ""))
    dwell_offset = config.get("dwell_offset", 0)
    seq_encoding = config.get("seq_encoding", "signal_kmer")
    signal_kmer_context = tuple(config.get("signal_kmer_context", (4, 4)))

    # Resolve motif/offset/justify from config, erroring on CLI conflict
    motif = _check_config_consistency("motif", motif, config.get("motif"), None)
    motif_offset = _check_config_consistency(
        "motif-offset", motif_offset, config.get("motif_offset"), 0
    )
    if motif is not None:
        logger.info(f"Motif from bundle config: {motif} (offset={motif_offset})")

    if motif is None:
        raise ValueError(
            "motif is None after auto-read from bundle config. "
            "Either pass --motif on the CLI or ensure the bundle config contains a non-null 'motif' field. "
            "Without a motif, inference predicts at every position, producing noise."
        )

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

    # Use asymmetric context if available, otherwise fall back to symmetric
    left_ctx = config.get("left_context")
    right_ctx = config.get("right_context")
    if left_ctx is not None and right_ctx is not None:
        signal_context = (left_ctx, right_ctx)
    else:
        signal_context = (signal_len // 2, signal_len // 2)
    kmer_context = kmer_len // 2

    logger.info(
        f"Bundle: {metadata['architecture']}, {len(pairs)} models, v{metadata['bundle_version']}"
        f"{' (TorchScript)' if is_torchscript else ''}"
    )

    # Determine feature_start/feature_end from config (must match training data)
    wide_features = model_type in ModelInferenceWrapper.WIDE_FEATURE_MODELS
    _kmer_context = kmer_len // 2
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
        _model_margin = getattr(_instantiate_model(config), "dwell_margin", 0)
        if _model_margin:
            _feature_start = -(_kmer_context + _model_margin)
            _feature_end = _kmer_context + _model_margin
            logger.warning(
                f"Bundle config missing feature_start/end, "
                f"falling back to model default margin: {_model_margin}"
            )

    logger.info(f"Signal context: {signal_context}, kmer_len: {kmer_len}")
    logger.info(f"seq_encoding: {seq_encoding}, base_justify: {base_justify}")
    _fs = _feature_start if _feature_start is not None else -_kmer_context
    _fe = _feature_end if _feature_end is not None else _kmer_context
    logger.info(f"dwell_offset: {dwell_offset}, feature window: [{_fs}, {_fe}]")
    if _feature_start is not None or _feature_end is not None:
        logger.info(f"Feature window: [{_fs}, {_fe}] relative to focus (width={_fe - _fs + 1})")

    # Select aggregation function
    if comparison_type == "pairwise":
        _pairwise_agg = {
            "naive": aggregate_pairwise,
            "weighted": aggregate_pairwise_weighted,
            "tournament": aggregate_pairwise_tournament,
        }
        if aggregation not in _pairwise_agg:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. "
                f"Available for pairwise: {list(_pairwise_agg.keys())}"
            )
        aggregate_fn = _pairwise_agg[aggregation]
        logger.info(f"Pairwise aggregation method: {aggregation}")
    else:
        aggregate_fn = aggregate_one_vs_all

    # Bind the bundle's recorded class names so aggregation never has to recover
    # them by splitting pair strings. Absent on bundles built before this was
    # stored; resolve_pair_labels() warns and falls back in that case.
    aggregate_fn = partial(aggregate_fn, pair_labels=metadata.get("pair_labels"))

    # Load all models (skip for vmap bundles -- stacked params loaded in vmap setup)
    wrappers: dict[str, ModelInferenceWrapper | TracedModelWrapper] = {}
    format_version = metadata.get("format_version", 1)
    platt_params: dict[str, tuple[float, float]] = {}

    if format_version == 4:
        # vmap bundle: params are pre-stacked, Platt scaling is pre-vectorized
        logger.info("vmap bundle (format_version=4): using vectorized inference")
    elif is_torchscript and format_version >= 3:
        # torch.export format (format_version 3+)
        requires_features = metadata.get("requires_features", False)
        for pair in pairs:
            model = deserialize_exported_model(
                bundle["models"][pair]["exported_bytes"], device=device
            )
            wrappers[pair] = TracedModelWrapper(model, requires_features=requires_features)
    elif is_torchscript:
        # Legacy TorchScript format (format_version 2)
        requires_features = metadata.get("requires_features", False)
        for pair in pairs:
            traced = deserialize_traced_model(bundle["models"][pair]["traced_bytes"], device=device)
            wrappers[pair] = TracedModelWrapper(traced, requires_features=requires_features)
    else:
        for pair in pairs:
            m = _instantiate_model(config)
            state_dict = bundle["models"][pair]["state_dict"]
            # Strip _orig_mod. prefix from torch.compile'd state dicts
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
            from leech.model_loading import _migrate_state_dict_keys

            state_dict = _migrate_state_dict_keys(state_dict)
            m.load_state_dict(state_dict)
            m = m.to(device)
            m.eval()
            wrappers[pair] = ModelInferenceWrapper(m, model_type)

    # Load per-model Platt scaling params (for post-hoc calibration, non-vmap bundles)
    if format_version != 4 and "models" in bundle:
        for pair in pairs:
            a = bundle["models"][pair].get("platt_a")
            b = bundle["models"][pair].get("platt_b")
            if a is not None and b is not None:
                platt_params[pair] = (a, b)
        if platt_params:
            logger.info(f"Platt scaling enabled for {len(platt_params)}/{len(pairs)} models")

    pair_names_str = ",".join(pairs)

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

    # Open BAM for header only
    bam_in = pysam.AlignmentFile(str(bam_path), "rb")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bam_out = pysam.AlignmentFile(str(output_path), "wb", template=bam_in)
    bam_in.close()

    # Determine feature requirements (needs_features for vmap detection too)
    if format_version == 4:
        needs_features = metadata.get("requires_features", False)
    else:
        first_wrapper = next(iter(wrappers.values()))
        needs_features = first_wrapper.requires_features
    bundle_signal_in_channels = config.get("signal_in_channels", 1)
    bundle_compute_features = needs_features or bundle_signal_in_channels > 1

    # Signal map refinement for bundle models (needed for kmer residual signal channel)
    bundle_refine = False
    bundle_refiner = None
    if config.get("refine_signal_map", True) or bundle_signal_in_channels > 1:
        from leech.data import get_kmer_table
        from leech.inference.helpers import _warn_if_kmer_table_drifted
        from leech.signal_refine import SigMapRefiner

        kmer_table_path = get_kmer_table()
        _warn_if_kmer_table_drifted(config.get("kmer_table_sha256"), kmer_table_path)
        bundle_refiner = SigMapRefiner.from_table(
            kmer_table_path,
            half_bandwidth=config.get("refine_half_bandwidth", 5),
            do_rough_rescale=config.get("refine_do_rough_rescale", True),
            scale_iters=config.get("refine_scale_iters", 2),
            center_idx=config.get("refine_kmer_center_idx", -1),
        )
        bundle_refine = True
        logger.info(
            f"Signal map refinement enabled for bundle "
            f"(signal_in_channels={bundle_signal_in_channels})"
        )

    # Load per-AA dwell templates if the bundle was trained with them. The
    # model expects `base_features + N_AA` channels, so inference must apply
    # the same template-append that the training dataset did. This is
    # label-independent (computes dwell/expected for all AAs), so it works
    # verbatim at inference time.
    dwell_templates_arr: np.ndarray | None = None
    dwell_template_min_pos: int = 0
    bundle_dwell_template_table = config.get("dwell_template_table") or None
    if bundle_dwell_template_table:
        from leech.dataset import load_dwell_template_table

        dwell_templates_arr, dwell_template_min_pos, _ = load_dwell_template_table(
            Path(bundle_dwell_template_table)
        )
        logger.info(
            f"Dwell template append enabled at inference "
            f"(+{dwell_templates_arr.shape[0]} channels from "
            f"{bundle_dwell_template_table})"
        )

    # -- Setup vmap forward for format_version 4, else use sequential wrappers --
    is_vmap = format_version == 4
    vmapped_forward = None
    vmap_stacked_params = None
    vmap_stacked_buffers = None
    vmap_platt_a = None
    vmap_platt_b = None
    vmap_base_model = None

    if is_vmap:
        vmap_stacked_params = {k: v.to(device) for k, v in bundle["stacked_params"].items()}
        vmap_stacked_buffers = {k: v.to(device) for k, v in bundle["stacked_buffers"].items()}
        vmap_platt_a = bundle["platt_a"].to(device)  # (n_models,)
        vmap_platt_b = bundle["platt_b"].to(device)  # (n_models,)

        vmap_base_model = _instantiate_model(config).to(device).eval()

        def _single_forward(params, buffers, signal, sequence, features):
            if needs_features:
                return torch.func.functional_call(
                    vmap_base_model, (params, buffers), (signal, sequence, features)
                )
            else:
                return torch.func.functional_call(
                    vmap_base_model, (params, buffers), (signal, sequence)
                )

        vmapped_forward = torch.vmap(_single_forward, in_dims=(0, 0, None, None, None))
        logger.info("vmap vectorized forward initialized")

    # -- Streaming mega-batch inference --
    pair_to_idx = {pair: i for i, pair in enumerate(pairs)}
    n_pairs = len(pairs)

    read_probs: dict[str, np.ndarray] = {}  # read_id -> shape (n_pairs,)
    _shape_validated = False
    n_chunks = 0
    n_batches_done = 0

    def _run_bundle_batch(sigs: list, seqs: list, feats: list, rids: list) -> None:
        """Flush callback: run every model in the bundle over one batch."""
        nonlocal n_batches_done

        sig_t = torch.stack(sigs).to(device)
        seq_t = torch.stack(seqs).to(device)
        feat_t = None
        if needs_features:
            valid_feats = [f for f in feats if f is not None]
            if valid_feats:
                feat_t = torch.stack(valid_feats).to(device)

        with torch.inference_mode():
            if is_vmap and vmapped_forward is not None:
                if needs_features:
                    all_logits = vmapped_forward(
                        vmap_stacked_params, vmap_stacked_buffers, sig_t, seq_t, feat_t
                    )
                else:
                    all_logits = vmapped_forward(
                        vmap_stacked_params, vmap_stacked_buffers, sig_t, seq_t, None
                    )
                all_logits = vmap_platt_a[:, None, None] * all_logits + vmap_platt_b[:, None, None]
                all_p = torch.sigmoid(all_logits).squeeze(-1).cpu().numpy()
            else:
                all_p = np.empty((n_pairs, len(sigs)), dtype=np.float32)
                for pair in pairs:
                    batch_dict: dict[str, torch.Tensor] = {
                        "signal": sig_t,
                        "sequence": seq_t,
                    }
                    if feat_t is not None:
                        batch_dict["features"] = feat_t
                    logits = wrappers[pair].forward_batch(batch_dict, device)
                    pp = platt_params.get(pair)
                    if pp is not None:
                        a, b = pp
                        logits = a * logits + b
                    all_p[pair_to_idx[pair]] = torch.sigmoid(logits).cpu().numpy().flatten()

        for i, rid in enumerate(rids):
            read_probs[rid] = all_p[:, i]

        n_batches_done += 1

    accumulator = BatchAccumulator(batch_size, _run_bundle_batch)

    n_reads = 0
    n_predicted = 0

    bundle_signal_config = SignalConfig(
        reverse_signal=reverse_signal,
        anchor=anchor,
        norm_method=config.get("signal_norm", "median_mad"),
        pa_mean=config.get("pa_mean"),
        pa_stdev=config.get("pa_stdev"),
        compute_features=bundle_compute_features,
        refine_signal_map=bundle_refine,
        signal_refiner=bundle_refiner,
    )
    bundle_chunk_config = ChunkConfig(
        base_justify=base_justify,
        feature_start=_feature_start,
        feature_end=_feature_end,
        signal_context=signal_context,
        kmer_context=kmer_context,
        recover_softclip_signal=config.get("recover_softclip_signal", False),
    )

    logger.info(f"Streaming bundle inference with read_batch_size={read_batch_size}")

    n_total_reads = count_bam_reads(bam_path)
    n_total_mega_batches = math.ceil(n_total_reads / read_batch_size) if n_total_reads > 0 else 0
    logger.info(
        f"BAM contains ~{n_total_reads} mapped reads -> ~{n_total_mega_batches} mega-batches "
        f"of {read_batch_size}"
    )
    mega_batch_idx = 0

    # ---- Rust monolithic extraction setup ----
    #
    # The fast path lives in leech_core (escapepod-rs + rust chunk extractor).
    # All three helpers (availability check, rayon thread cap, kwargs builder)
    # are shared with run_inference in single.py via helpers.py.
    assert motif is not None  # type narrowing for helpers below
    (
        _use_rust_extraction,
        _rs_extract_inference_chunks,
        _rs_preload_pod5_signals,
        _rs_extract_chunks_from_preloaded,
    ) = check_rust_extraction_available(
        backend,
        bundle_signal_config.norm_method,
        bundle_chunk_config.recover_softclip_signal,
    )
    if _use_rust_extraction:
        logger.info("Using Rust monolithic extraction (escapepod-rs + leech_core)")
        cap_rayon_threads_for_slurm()
    elif backend == "python":
        logger.info("Using Python extraction path (forced via --backend python)")
    else:
        logger.info("Using Python extraction path (leech_core unavailable)")

    _rs_kwargs: dict | None = None
    if _use_rust_extraction:
        _rs_kwargs = build_rust_extraction_kwargs(
            signal_context=signal_context,
            kmer_context=kmer_context,
            signal_len=signal_len,
            compute_features=bundle_compute_features,
            reverse_signal=reverse_signal,
            feature_start=_feature_start,
            feature_end=_feature_end,
            anchor=anchor,
            seq_encoding=seq_encoding,
            signal_kmer_context=signal_kmer_context,
            refine_signal_map=bundle_refine,
            signal_refiner=bundle_refiner,
            refine_half_bandwidth=config.get("refine_half_bandwidth", 5),
            refine_scale_iters=config.get("refine_scale_iters", 2),
            signal_in_channels=bundle_signal_in_channels,
            base_justify=base_justify,
        )

    # Build InferenceConfig for parallel workers (reused across mega-batches).
    # Only constructed when num_workers > 0 — the serial path uses its own
    # local SignalConfig/ChunkConfig.
    inf_config: InferenceConfig | None = None
    if num_workers > 0 and not _use_rust_extraction:
        inf_config = InferenceConfig(
            pod5_path=pod5_path,
            signal=SignalConfig(
                reverse_signal=reverse_signal,
                anchor=anchor,
                norm_method=config.get("signal_norm", "median_mad"),
                pa_mean=config.get("pa_mean"),
                pa_stdev=config.get("pa_stdev"),
                refine_signal_map=bundle_refine,
                signal_refiner=bundle_refiner,
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
            requires_features=needs_features,
            signal_in_channels=bundle_signal_in_channels,
        )

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing reads...", total=None)

        if _use_rust_extraction:
            # ---- Rust monolithic hot path ----
            #
            # Mirrors run_inference's rust sequential path (single.py:1375-1413):
            # main thread collects pysam metadata for a mega-batch, rust
            # consumes the metadata + POD5 path and returns extracted chunks,
            # main thread batches them into torch tensors and runs the existing
            # multi-model batch accumulator.
            assert _rs_extract_inference_chunks is not None
            assert _rs_kwargs is not None

            logger.info("Rust bundle inference: serial mega-batches")

            for aln_batch in iter_bam_batches(
                bam_path, batch_size=read_batch_size, min_mapq=min_mapq
            ):
                read_probs.clear()

                rs_meta = collect_bam_metadata_for_rust(
                    aln_batch,
                    motif=motif,
                    motif_offset=motif_offset,
                    motif_searcher=motif_searcher,
                    anchor=anchor,
                    reference_sequences=reference_sequences,
                    # Bundle semantics: one chunk per read (matches the serial
                    # path's positions[0] behavior)
                    max_positions_per_read=1,
                )
                rs_read_ids = rs_meta[0]

                n_reads += len(aln_batch)

                logger.info(
                    f"Mega-batch: {len(aln_batch)} alignments, "
                    f"{len(rs_read_ids)} for Rust extraction"
                )

                if rs_read_ids:
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
                        reference_sequences=(rs_meta[8] if anchor == "reference" else None),
                        **_rs_kwargs,
                    )

                    # Consume rust chunks into batch buffers. Rust emits at most
                    # one chunk per read (we truncated motif positions above),
                    # so no dedupe is needed.
                    for sig, seq_arr, feat, read_id, _base_idx in chunks:
                        if bundle_signal_in_channels > 1 and sig.ndim == 1:
                            sig = sig.reshape(bundle_signal_in_channels, -1)

                        # Rust hands back the full requested feature window,
                        # exactly as a Python chunk's `features` does, so it
                        # needs the same preparation: templates appended
                        # against the stored window, then narrowed to the
                        # model's k-mer window. This loop used to append but
                        # never narrow.
                        if needs_features and feat is not None:
                            feat = prepare_inference_features(
                                feat,
                                kmer_len=kmer_len,
                                feature_start=_feature_start,
                                dwell_offset=dwell_offset,
                                wide_features=wide_features,
                                dwell_templates=dwell_templates_arr,
                                template_min_pos=dwell_template_min_pos,
                            )

                        if not _shape_validated:
                            validate_inference_shapes(sig, feat, config)
                            _shape_validated = True

                        n_chunks += 1
                        accumulator.add(
                            torch.from_numpy(sig),
                            torch.from_numpy(seq_arr),
                            torch.from_numpy(feat) if needs_features and feat is not None else None,
                            read_id,
                        )

                # Flush remaining chunks for this mega-batch
                accumulator.flush()

                # -- Aggregate per-read and write BAM for this mega-batch --
                batch_preds = _write_bundle_mega_batch(
                    aln_batch,
                    read_probs,
                    pairs,
                    pair_to_idx,
                    aggregate_fn,
                    pair_names_str,
                    bam_out,
                    raw,
                    min_confidence,
                    min_margin,
                )
                n_predicted += batch_preds

                mega_batch_idx += 1
                logger.info(
                    f"Mega-batch {mega_batch_idx}/{n_total_mega_batches} "
                    f"complete: wrote {batch_preds} predictions for "
                    f"{len(aln_batch)} reads"
                )

                progress.update(
                    task,
                    advance=0,
                    description=(
                        f"[cyan]Processed {n_chunks} chunks from "
                        f"{n_reads} reads ({n_batches_done} batches, "
                        f"{n_predicted} predicted)..."
                    ),
                )

            logger.info(f"Extracted and inferred {n_chunks} chunks from {n_reads} reads")
            logger.info(f"Predicted {n_predicted} reads")
            bam_out.close()

            logger.info(f"Bundle inference complete: {n_reads} reads, {len(pairs)} models")
            logger.info(f"Output written to: {output_path}")
            return

        if num_workers > 0:
            # ---- Parallel path: dispatch chunk extraction to mp.Pool workers ----
            #
            # Each worker opens its own escapepod Reader, extracts chunks for a
            # sub-batch of ReadInfo objects, and returns (read_id, base_idx, sig,
            # enc_seq, feat) tuples. The main process collects results, dedupes
            # to one chunk per read (matching the serial path's positions[0]
            # behavior), batches, and runs the multi-model GPU forward via the
            # shared batch accumulator.
            from leech.inference.single import _inference_worker

            assert inf_config is not None  # constructed above
            logger.info(f"Parallel bundle inference with {num_workers} workers")

            # reads per worker sub-batch. 100 is too small — mp.Pool IPC
            # overhead dominates with many tiny sub-batches. Large sub-batches
            # amortize escapepod open + kmer table load + pickling cost across
            # many reads. Requires read_batch_size >= num_workers * this value
            # to actually saturate the worker pool within a mega-batch.
            _worker_chunk_size = 50000

            with mp.Pool(processes=num_workers) as pool:
                for aln_batch in iter_bam_batches(
                    bam_path, batch_size=read_batch_size, min_mapq=min_mapq
                ):
                    read_probs.clear()

                    # Build ReadInfo objects (lightweight, picklable) from this
                    # mega-batch. This is single-threaded but cheap.
                    read_infos: list[ReadInfo] = []
                    for aln in aln_batch:
                        try:
                            read_infos.append(ReadInfo(aln))
                        except Exception as e:
                            logger.warning(f"Skipping read {aln.query_name}: {e}")

                    logger.info(
                        f"Mega-batch: {len(aln_batch)} alignments, "
                        f"{len(read_infos)} ReadInfo objects "
                        f"-> {num_workers} workers"
                    )

                    n_reads += len(read_infos)

                    if not read_infos:
                        # Nothing to infer but still write the mega-batch through
                        for aln in aln_batch:
                            bam_out.write(aln)
                        mega_batch_idx += 1
                        continue

                    # Split into sub-batches and dispatch
                    worker_batches = [
                        read_infos[i : i + _worker_chunk_size]
                        for i in range(0, len(read_infos), _worker_chunk_size)
                    ]
                    worker_args = [(wb, inf_config) for wb in worker_batches]

                    # Dedupe: bundle uses only the first chunk per read
                    seen_in_batch: set[str] = set()

                    for worker_results in pool.imap_unordered(_inference_worker, worker_args):
                        for (
                            read_id,
                            _base_idx,
                            sig,
                            enc_seq,
                            feat,
                        ) in worker_results:
                            if read_id in seen_in_batch:
                                continue
                            seen_in_batch.add(read_id)

                            if (
                                needs_features
                                and feat is not None
                                and dwell_templates_arr is not None
                            ):
                                from leech.dataset import (
                                    append_dwell_template_channels,
                                )

                                feat = append_dwell_template_channels(
                                    feat,
                                    feat_start=_feature_start
                                    if _feature_start is not None
                                    else -_kmer_context,
                                    dwell_templates=dwell_templates_arr,
                                    template_min_pos=dwell_template_min_pos,
                                )

                            if not _shape_validated:
                                validate_inference_shapes(sig, feat, config)
                                _shape_validated = True

                            n_chunks += 1
                            accumulator.add(
                                torch.from_numpy(sig),
                                torch.from_numpy(enc_seq),
                                torch.from_numpy(feat)
                                if needs_features and feat is not None
                                else None,
                                read_id,
                            )

                    # Flush remaining chunks for this mega-batch
                    accumulator.flush()

                    # -- Aggregate per-read and write BAM for this mega-batch --
                    batch_preds = _write_bundle_mega_batch(
                        aln_batch,
                        read_probs,
                        pairs,
                        pair_to_idx,
                        aggregate_fn,
                        pair_names_str,
                        bam_out,
                        raw,
                        min_confidence,
                        min_margin,
                    )
                    n_predicted += batch_preds

                    mega_batch_idx += 1
                    logger.info(
                        f"Mega-batch {mega_batch_idx}/{n_total_mega_batches} "
                        f"complete: wrote {batch_preds} predictions for "
                        f"{len(aln_batch)} reads"
                    )

                    progress.update(
                        task,
                        advance=0,
                        description=(
                            f"[cyan]Processed {n_chunks} chunks from "
                            f"{n_reads} reads ({n_batches_done} batches, "
                            f"{n_predicted} predicted)..."
                        ),
                    )

            logger.info(f"Extracted and inferred {n_chunks} chunks from {n_reads} reads")
            logger.info(f"Predicted {n_predicted} reads")
            bam_out.close()

            logger.info(f"Bundle inference complete: {n_reads} reads, {len(pairs)} models")
            logger.info(f"Output written to: {output_path}")
            return

        with POD5Reader(pod5_path, backend=backend) as pod5_reader:
            for aln_batch in iter_bam_batches(
                bam_path, batch_size=read_batch_size, min_mapq=min_mapq
            ):
                # Preload POD5 signals for this mega-batch
                batch_rids = [
                    aln.query_name
                    for aln in aln_batch
                    if aln.query_name is not None and aln.query_sequence is not None
                ]
                if batch_rids:
                    pod5_reader.preload(batch_rids)

                logger.info(
                    f"Mega-batch: {len(aln_batch)} alignments, "
                    f"{len(batch_rids)} preloaded from POD5"
                )

                read_probs.clear()

                for aln in aln_batch:
                    read_id = aln.query_name
                    read_seq = aln.query_sequence
                    if read_id is None or read_seq is None:
                        continue

                    try:
                        move_table = extract_move_table(aln)
                        raw_signal, pod5_metadata = pod5_reader.get_signal(read_id)

                        ref_seq = None
                        cigar_tuples = None
                        if bundle_signal_config.anchor == "reference":
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
                            signal_config=bundle_signal_config,
                            metadata={"alignment": aln},
                            reference_sequence=ref_seq,
                            cigar_tuples=cigar_tuples,
                            cal_offset=pod5_metadata.get("calibration_offset"),
                            cal_scale=pod5_metadata.get("calibration_scale"),
                        )
                    except Exception as e:
                        logger.warning(f"Skipping read {read_id}: {e}")
                        continue

                    n_reads += 1

                    # Find prediction position(s)
                    assert motif is not None, "motif must not be None at inference time"
                    aln_meta = leech_read.metadata.get("alignment")
                    positions = [
                        pos + motif_offset
                        for pos in motif_searcher.find_motif_positions(
                            leech_read.read_id, leech_read.sequence, aln_meta, motif
                        )
                    ]

                    if not positions:
                        continue

                    base_idx = positions[0]
                    chunk = leech_read.get_chunk(base_idx, config=bundle_chunk_config)
                    if chunk is None:
                        continue

                    # Prepare tensors (with optional kmer residual channel)
                    signal_array = chunk["signal"]
                    assert isinstance(signal_array, np.ndarray)
                    sig = signal_array.astype(np.float32)
                    sig_residual = chunk.get("signal_residual")
                    if sig_residual is not None:
                        sig_residual = sig_residual.astype(np.float32)
                        if len(sig_residual) < len(sig):
                            sig_residual = np.pad(
                                sig_residual,
                                (0, len(sig) - len(sig_residual)),
                                mode="constant",
                            )
                        elif len(sig_residual) > len(sig):
                            sig_residual = sig_residual[: len(sig)]
                        sig = np.stack([sig, sig_residual], axis=0)
                    signal_t = torch.from_numpy(sig)

                    seq_t = _encode_sequence_for_inference(
                        chunk, seq_encoding, signal_len, signal_kmer_context
                    )
                    if seq_t is None:
                        continue

                    n_chunks += 1

                    feat_t = None
                    if needs_features:
                        features_array = chunk["features"]
                        assert isinstance(features_array, np.ndarray)
                        # Templates first, then narrowing -- the order
                        # `dataset.py` used at training time. This site had it
                        # the other way round, so the template channels were
                        # keyed to the pre-narrowing column 0 while the array
                        # had already been shifted out from under them.
                        features_array = prepare_inference_features(
                            features_array.astype(np.float32),
                            kmer_len=kmer_len,
                            feature_start=chunk.get("feature_start"),
                            dwell_offset=dwell_offset,
                            wide_features=wide_features,
                            dwell_templates=dwell_templates_arr,
                            template_min_pos=dwell_template_min_pos,
                        )
                        feat_t = torch.from_numpy(features_array)

                    if not _shape_validated:
                        _feat_for_check = (
                            features_array.astype(np.float32) if needs_features else None
                        )
                        validate_inference_shapes(sig, _feat_for_check, config)
                        _shape_validated = True

                    accumulator.add(signal_t, seq_t, feat_t, leech_read.read_id)

                # Flush remaining chunks for this mega-batch
                accumulator.flush()

                # -- Aggregate per-read and write BAM for this mega-batch --
                batch_preds = _write_bundle_mega_batch(
                    aln_batch,
                    read_probs,
                    pairs,
                    pair_to_idx,
                    aggregate_fn,
                    pair_names_str,
                    bam_out,
                    raw,
                    min_confidence,
                    min_margin,
                )
                n_predicted += batch_preds

                mega_batch_idx += 1
                logger.info(
                    f"Mega-batch {mega_batch_idx}/{n_total_mega_batches} complete: "
                    f"wrote {batch_preds} predictions for {len(aln_batch)} reads"
                )

                progress.update(
                    task,
                    advance=0,
                    description=(
                        f"[cyan]Processed {n_chunks} chunks from {n_reads} reads "
                        f"({n_batches_done} batches, {n_predicted} predicted)..."
                    ),
                )

    logger.info(f"Extracted and inferred {n_chunks} chunks from {n_reads} reads")
    logger.info(f"Predicted {n_predicted} reads")
    bam_out.close()

    logger.info(f"Bundle inference complete: {n_reads} reads, {len(pairs)} models")
    logger.info(f"Output written to: {output_path}")
