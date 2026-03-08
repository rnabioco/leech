"""
Inference engine for running predictions on new data.

Reads POD5 and BAM files, extracts features, runs model predictions,
and writes modification probabilities to output BAM files.

Supports:
- Leech native models (checkpoint directories)
- Remora TorchScript models (.pt files) via auto-detection
- Parallel chunk extraction with multiprocessing
- Signal-level kmer encoding (signal_kmer) and base-level one-hot (base_onehot)
"""

import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pysam
import torch
from rich.progress import Progress

from leech.features import encode_signal_kmer, sequence_to_int
from leech.io.motif_search import find_motif_in_sequence
from leech.models.inference_wrapper import ModelInferenceWrapper
from leech.models.remora_compat import RemoraModelWrapper
from leech.preparation import encode_kmer, iter_bam_with_pod5
from leech.util import _instantiate_model, load_model_from_checkpoint

logger = logging.getLogger("leech.inference")


def load_model_auto(
    model_path: Path, device: str = "cpu"
) -> tuple[ModelInferenceWrapper | RemoraModelWrapper, dict]:
    """
    Load leech model (directory) or Remora TorchScript (.pt file).

    Auto-detects format:
    - Directory with config.json → leech checkpoint
    - .pt file → Remora TorchScript model

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
        # Standalone .pt file without config.json → Remora TorchScript
        wrapper = RemoraModelWrapper(path, device=device)
        config = {
            "seq_encoding": "signal_kmer",
            "signal_kmer_context": [4, 4],
            "is_remora": True,
        }
        return wrapper, config
    else:
        raise ValueError(f"Cannot auto-detect model format for {model_path}")


def _encode_sequence_for_inference(
    chunk: dict,
    seq_encoding: str,
    signal_len: int,
    signal_kmer_context: tuple[int, int] = (4, 4),
) -> torch.Tensor:
    """Encode sequence from a chunk for inference.

    Args:
        chunk: Chunk dict from LeechRead.get_chunk()
        seq_encoding: "base_onehot" or "signal_kmer"
        signal_len: Target signal length
        signal_kmer_context: Kmer context for signal_kmer encoding

    Returns:
        Encoded sequence tensor
    """
    if seq_encoding == "signal_kmer":
        seq_ctx = chunk.get("sequence_with_kmer_context")
        seq_to_sig = chunk.get("seq_to_sig_map")
        if seq_ctx is not None and seq_to_sig is not None:
            seq_ints = sequence_to_int(seq_ctx)
            enc = encode_signal_kmer(seq_ints, seq_to_sig, signal_len, signal_kmer_context)
            return torch.from_numpy(enc)
        else:
            # Fall back to base_onehot if chunk lacks signal_kmer data
            logger.debug("Chunk lacks signal_kmer fields, falling back to base_onehot")
            return encode_kmer(chunk["sequence"])
    else:
        return encode_kmer(chunk["sequence"])


def aggregate_pairwise(pairs: list[str], probs: list[float]) -> tuple[str, float, dict[str, float]]:
    """
    Aggregate pairwise model probabilities into a single amino acid prediction.

    Label convention: pair "A_B" (alphabetical) -> A = label 0, B = label 1.
    Sigmoid probability p: vote strength (1-p) goes to A, p goes to B.

    Args:
        pairs: List of pair names (e.g., ["Ala_Gly", "Ala_Ser", "Gly_Ser"])
        probs: List of sigmoid probabilities, one per pair

    Returns:
        Tuple of (predicted_aa, confidence, vote_totals) where:
        - predicted_aa: amino acid with highest total vote strength
        - confidence: winner's vote fraction (winner_total / sum_all_totals)
        - vote_totals: dict mapping amino acid -> total vote strength
    """
    votes: dict[str, float] = {}
    for pair, prob in zip(pairs, probs, strict=True):
        aa_a, aa_b = pair.split("_", 1)
        votes[aa_a] = votes.get(aa_a, 0.0) + (1.0 - prob)
        votes[aa_b] = votes.get(aa_b, 0.0) + prob

    total = sum(votes.values())
    predicted_aa = max(votes, key=votes.__getitem__)
    confidence = votes[predicted_aa] / total if total > 0 else 0.0
    return predicted_aa, confidence, votes


def aggregate_one_vs_all(
    pairs: list[str], probs: list[float]
) -> tuple[str, float, dict[str, float]]:
    """
    Aggregate one-vs-all model probabilities into a single amino acid prediction.

    Label convention: pair "A_notA" -> A = label 0, notA = label 1.
    Score for AA = (1 - p), since low probability means more likely to be the target.

    Args:
        pairs: List of pair names (e.g., ["Ala_notAla", "Gly_notGly"])
        probs: List of sigmoid probabilities, one per pair

    Returns:
        Tuple of (predicted_aa, confidence, scores) where:
        - predicted_aa: amino acid with highest score
        - confidence: max score
        - scores: dict mapping amino acid -> score
    """
    scores: dict[str, float] = {}
    for pair, prob in zip(pairs, probs, strict=True):
        aa = pair.split("_", 1)[0]
        scores[aa] = 1.0 - prob

    predicted_aa = max(scores, key=scores.__getitem__)
    confidence = scores[predicted_aa]
    return predicted_aa, confidence, scores


def _inference_worker(
    args: tuple,
) -> list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray | None]]:
    """
    Worker for parallel chunk extraction during inference.

    Extracts chunks from reads and optionally pre-computes signal_kmer encoding.

    Returns:
        List of (read_id, base_idx, signal, encoded_sequence, features_or_none) tuples
    """
    from pod5 import DatasetReader

    (
        read_infos,
        pod5_path,
        motif,
        motif_offset,
        signal_context,
        kmer_context,
        base_justify,
        seq_encoding,
        signal_kmer_context,
        signal_len,
        kmer_len,
        dwell_offset,
        requires_features,
        reverse_signal,
    ) = args

    results: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray | None]] = []

    with DatasetReader(pod5_path) as pod5_reader:
        for read_info in read_infos:
            try:
                # Read signal from POD5
                signal_found = False
                for read in pod5_reader.reads([read_info.read_id]):
                    raw_signal = read.signal
                    signal_found = True
                    break

                if not signal_found:
                    continue

                # Build LeechRead via shared helper
                from leech.preparation.reader import build_leech_read

                leech_read = build_leech_read(
                    read_id=read_info.read_id,
                    sequence=read_info.sequence,
                    raw_signal=raw_signal,
                    move_table=read_info.to_move_table(),
                    reverse_signal=reverse_signal,
                )

                # Find motif positions
                if motif is not None:
                    positions = [
                        pos + motif_offset
                        for pos in find_motif_in_sequence(leech_read.sequence, motif)
                    ]
                else:
                    positions = list(range(kmer_context, leech_read.num_bases - kmer_context))

                for base_idx in positions:
                    chunk = leech_read.get_chunk(
                        base_idx,
                        signal_context=signal_context,
                        kmer_context=kmer_context,
                        base_justify=base_justify,
                    )
                    if chunk is None:
                        continue

                    # Signal
                    sig = chunk["signal"].astype(np.float32)
                    if len(sig) < signal_len:
                        sig = np.pad(sig, (0, signal_len - len(sig)), mode="constant")
                    elif len(sig) > signal_len:
                        start = (len(sig) - signal_len) // 2
                        sig = sig[start : start + signal_len]

                    # Sequence encoding
                    if seq_encoding == "signal_kmer":
                        seq_ctx = chunk.get("sequence_with_kmer_context")
                        seq_to_sig = chunk.get("seq_to_sig_map")
                        if seq_ctx is not None and seq_to_sig is not None:
                            seq_ints = sequence_to_int(seq_ctx)
                            enc_seq = encode_signal_kmer(
                                seq_ints, seq_to_sig, signal_len, tuple(signal_kmer_context)
                            )
                        else:
                            from leech.preparation.encoding import encode_kmer as _enc

                            enc_seq = _enc(chunk["sequence"]).numpy()
                    else:
                        from leech.preparation.encoding import encode_kmer as _enc

                        enc_seq = _enc(chunk["sequence"]).numpy()

                    # Features
                    feat = None
                    if requires_features:
                        feat_arr = chunk["features"]
                        if feat_arr.size > 0:
                            feat_arr = feat_arr.astype(np.float32)
                            if feat_arr.shape[1] > kmer_len:
                                margin = (feat_arr.shape[1] - kmer_len) // 2
                                s = margin + dwell_offset
                                feat_arr = feat_arr[:, s : s + kmer_len]
                            feat = feat_arr

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
    min_mapq: int = 10,
    motif: str | None = None,
    motif_offset: int = 0,
    batch_size: int = 256,
    base_justify: str = "center",
    reverse_signal: bool = True,
    num_workers: int = 0,
    chunk_size: int = 100,
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
        device: Device for inference
        min_mapq: Minimum mapping quality
        motif: Optional motif to filter predictions (auto-read from config if None)
        motif_offset: Offset within motif for prediction (auto-read from config if 0)
        batch_size: Chunks per forward pass
        base_justify: Signal justification within focus base
        reverse_signal: Whether to reverse signal for RNA
        num_workers: Parallel chunk extraction workers (0=sequential)
        chunk_size: Reads per worker batch
    """
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

    if is_remora:
        model_wrapper = wrapper_or_model
        # Remora models: motif MUST be provided
        if motif is None:
            raise ValueError("--motif is required for Remora models (no config.json)")
        # Infer signal_len from a dummy pass or default
        signal_len = 400  # Remora default
        kmer_len = 9  # Remora default (4+1+4 kmer context)
        seq_encoding = "signal_kmer"
        signal_kmer_context = tuple(config.get("signal_kmer_context", (4, 4)))
        dwell_offset = 0
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
        seq_encoding = config.get("seq_encoding", "base_onehot")
        signal_kmer_context = tuple(config.get("signal_kmer_context", (4, 4)))

        # Auto-read motif from model config if not provided
        if motif is None and config.get("motif"):
            motif = config["motif"]
            motif_offset = config.get("motif_offset", motif_offset)
            logger.info(f"Auto-read motif from config: {motif} (offset={motif_offset})")

    signal_context = (signal_len // 2, signal_len // 2)
    kmer_context = kmer_len // 2
    requires_features = getattr(model_wrapper, "requires_features", False)

    logger.info(f"Signal length: {signal_len}, K-mer length: {kmer_len}")
    logger.info(f"Sequence encoding: {seq_encoding}")
    if motif:
        logger.info(f"Motif: {motif} (offset={motif_offset})")

    # Open input BAM and index alignments by read ID
    bam_in = pysam.AlignmentFile(str(bam_path), "rb")
    alignment_by_read_id: dict[str, pysam.AlignedSegment] = {}
    for a in bam_in.fetch(until_eof=True):
        if a.query_name is not None:
            alignment_by_read_id[a.query_name] = a

    # Create output BAM
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bam_out = pysam.AlignmentFile(str(output_path), "wb", template=bam_in)

    total_reads = 0
    total_predictions = 0

    if hasattr(model_wrapper, "model"):
        model_wrapper.model.eval()
    if hasattr(model_wrapper, "eval"):
        model_wrapper.eval()

    if num_workers > 0:
        # ---- Parallel path ----
        from leech.io import collect_read_infos

        logger.info(f"Parallel inference with {num_workers} workers")

        read_infos = collect_read_infos(bam_path, min_mapq=min_mapq)
        total_reads = len(read_infos)
        logger.info(f"Found {total_reads} reads")

        if total_reads == 0:
            bam_in.close()
            bam_out.close()
            return

        read_batches = [
            read_infos[i : i + chunk_size] for i in range(0, len(read_infos), chunk_size)
        ]

        worker_args = [
            (
                batch_reads,
                pod5_path,
                motif,
                motif_offset,
                signal_context,
                kmer_context,
                base_justify,
                seq_encoding,
                signal_kmer_context,
                signal_len,
                kmer_len,
                dwell_offset,
                requires_features,
                reverse_signal,
            )
            for batch_reads in read_batches
        ]

        # Collect all results, then batch and run model
        pending: dict[str, list[tuple[int, float]]] = {}

        with Progress() as progress:
            task = progress.add_task("[cyan]Extracting chunks...", total=len(read_batches))

            with mp.Pool(processes=num_workers) as pool:
                for worker_results in pool.imap_unordered(_inference_worker, worker_args):
                    # Accumulate into batch buffer
                    signals_buf = []
                    seqs_buf = []
                    feats_buf = []
                    meta_buf = []  # (read_id, base_idx)

                    for read_id, base_idx, sig, enc_seq, feat in worker_results:
                        signals_buf.append(sig)
                        seqs_buf.append(enc_seq)
                        feats_buf.append(feat)
                        meta_buf.append((read_id, base_idx))

                        if len(signals_buf) >= batch_size:
                            _run_batch(
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

                    # Flush remaining
                    if signals_buf:
                        _run_batch(
                            signals_buf,
                            seqs_buf,
                            feats_buf,
                            meta_buf,
                            model_wrapper,
                            requires_features,
                            device,
                            pending,
                        )

                    progress.advance(task)

        # Write all results to BAM (iterate dict to preserve original BAM order)
        total_predictions = sum(len(v) for v in pending.values())
        for aln in alignment_by_read_id.values():
            preds = pending.get(aln.query_name)
            if preds:
                preds.sort(key=lambda x: x[0])
                positions_list = [p[0] for p in preds]
                ml_scores = [int(min(255, max(0, p[1] * 255))) for p in preds]
                aln.set_tag("MP", positions_list, value_type="B")
                aln.set_tag("ML", ml_scores, value_type="B")
            bam_out.write(aln)

    else:
        # ---- Sequential path ----
        with Progress() as progress:
            task = progress.add_task("[cyan]Running inference...", total=None)

            for leech_read in iter_bam_with_pod5(
                bam_path, pod5_path, min_mapq=min_mapq, reverse_signal=reverse_signal
            ):
                total_reads += 1
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Processed {total_reads} reads...",
                )

                aln = alignment_by_read_id.get(leech_read.read_id)
                if aln is None:
                    continue

                # Find positions to predict
                if motif is None:
                    positions = list(range(kmer_context, leech_read.num_bases - kmer_context))
                else:
                    positions = [
                        pos + motif_offset
                        for pos in find_motif_in_sequence(leech_read.sequence, motif)
                    ]

                predictions = []

                for base_idx in positions:
                    chunk = leech_read.get_chunk(
                        base_idx,
                        signal_context=signal_context,
                        kmer_context=kmer_context,
                        base_justify=base_justify,
                    )
                    if chunk is None:
                        continue

                    # Signal
                    signal_array = chunk["signal"]
                    assert isinstance(signal_array, np.ndarray)
                    sig = signal_array.astype(np.float32)
                    if len(sig) < signal_len:
                        sig = np.pad(sig, (0, signal_len - len(sig)), mode="constant")
                    elif len(sig) > signal_len:
                        start = (len(sig) - signal_len) // 2
                        sig = sig[start : start + signal_len]
                    signal_t = torch.from_numpy(sig).to(device).unsqueeze(0)

                    # Sequence
                    seq_t = (
                        _encode_sequence_for_inference(
                            chunk, seq_encoding, signal_len, signal_kmer_context
                        )
                        .to(device)
                        .unsqueeze(0)
                    )

                    batch = {"signal": signal_t, "sequence": seq_t}

                    # Features
                    if requires_features:
                        features_array = chunk["features"]
                        assert isinstance(features_array, np.ndarray)
                        feat = features_array.astype(np.float32)
                        if feat.size > 0 and feat.shape[1] > kmer_len:
                            margin = (feat.shape[1] - kmer_len) // 2
                            s = margin + dwell_offset
                            feat = feat[:, s : s + kmer_len]
                        batch["features"] = torch.from_numpy(feat).to(device).unsqueeze(0)

                    with torch.no_grad():
                        logits = model_wrapper.forward_batch(batch, device)
                        prob = torch.sigmoid(logits).item()

                    predictions.append((base_idx, prob))

                if predictions:
                    predictions.sort(key=lambda x: x[0])
                    positions_list = [p[0] for p in predictions]
                    probs_list = [p[1] for p in predictions]
                    ml_scores = [int(min(255, max(0, p * 255))) for p in probs_list]
                    aln.set_tag("MP", positions_list, value_type="B")
                    aln.set_tag("ML", ml_scores, value_type="B")
                    total_predictions += len(predictions)

                bam_out.write(aln)

    bam_in.close()
    bam_out.close()

    logger.info("Inference complete!")
    logger.info(f"Reads processed: {total_reads}")
    logger.info(f"Total predictions: {total_predictions}")
    logger.info(f"Output written to: {output_path}")


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

    with torch.no_grad():
        logits = model_wrapper.forward_batch(batch, device)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    for (read_id, base_idx), prob in zip(meta, probs, strict=True):
        if read_id not in pending:
            pending[read_id] = []
        pending[read_id].append((base_idx, float(prob)))


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
) -> None:
    """
    Run all models from a bundle on each read, aggregate to a single AA prediction.

    Writes output BAM with tags:
    - aa:Z:Gly — predicted amino acid
    - ac:f:0.93 — confidence score (vote fraction or max score)
    With --raw, additionally:
    - pn:Z:Ala_Gly,Ala_Ser,... — comma-separated pair names
    - pp:B:f,0.95,0.23,... — float array of probabilities in matching order

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
    """
    bundle = torch.load(bundle_path, map_location=device)
    metadata = bundle["metadata"]
    config = bundle["config"]
    pairs = metadata["pairs"]
    comparison_type = metadata["comparison_type"]

    signal_len = config["signal_len"]
    kmer_len = config["kmer_len"]
    model_type = config["model_name"]
    dwell_offset = config.get("dwell_offset", 0)
    seq_encoding = config.get("seq_encoding", "base_onehot")
    signal_kmer_context = tuple(config.get("signal_kmer_context", (4, 4)))

    signal_context = (signal_len // 2, signal_len // 2)
    kmer_context = kmer_len // 2

    logger.info(
        f"Bundle: {metadata['architecture']}, {len(pairs)} models, v{metadata['bundle_version']}"
    )

    # Select aggregation function
    if comparison_type == "pairwise":
        aggregate_fn = aggregate_pairwise
    else:
        aggregate_fn = aggregate_one_vs_all

    # Load all models
    wrappers = {}
    for pair in pairs:
        m = _instantiate_model(config)
        m.load_state_dict(bundle["models"][pair]["state_dict"])
        m = m.to(device)
        m.eval()
        wrappers[pair] = ModelInferenceWrapper(m, model_type)

    pair_names_str = ",".join(pairs)

    # Open BAM files and index alignments by read ID (avoids O(n^2) scanning)
    bam_in = pysam.AlignmentFile(str(bam_path), "rb")
    alignment_by_read_id: dict[str, pysam.AlignedSegment] = {}
    for a in bam_in.fetch(until_eof=True):
        if a.query_name is not None:
            alignment_by_read_id[a.query_name] = a
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bam_out = pysam.AlignmentFile(str(output_path), "wb", template=bam_in)

    total_reads = 0

    with Progress() as progress:
        task = progress.add_task("[cyan]Running bundle inference...", total=None)

        for leech_read in iter_bam_with_pod5(
            bam_path, pod5_path, min_mapq=min_mapq, reverse_signal=reverse_signal
        ):
            total_reads += 1
            progress.update(task, advance=1, description=f"[cyan]Processed {total_reads} reads...")

            # Find alignment
            aln = alignment_by_read_id.get(leech_read.read_id)
            if aln is None:
                continue

            # Find prediction position(s)
            if motif is None:
                positions = list(range(kmer_context, leech_read.num_bases - kmer_context))
            else:
                positions = [
                    pos + motif_offset for pos in find_motif_in_sequence(leech_read.sequence, motif)
                ]

            # Extract chunk once (use first valid position)
            if not positions:
                bam_out.write(aln)
                continue

            base_idx = positions[0]
            chunk = leech_read.get_chunk(
                base_idx,
                signal_context=signal_context,
                kmer_context=kmer_context,
                base_justify=base_justify,
            )
            if chunk is None:
                bam_out.write(aln)
                continue

            # Prepare input tensors (once for all models)
            signal_array = chunk["signal"]
            assert isinstance(signal_array, np.ndarray)
            signal = torch.from_numpy(signal_array.astype(np.float32)).to(device).unsqueeze(0)

            sequence = (
                _encode_sequence_for_inference(chunk, seq_encoding, signal_len, signal_kmer_context)
                .to(device)
                .unsqueeze(0)
            )

            batch = {"signal": signal, "sequence": sequence}

            # Add features if needed (check first wrapper)
            first_wrapper = next(iter(wrappers.values()))
            if first_wrapper.requires_features:
                features_array = chunk["features"]
                assert isinstance(features_array, np.ndarray)
                if features_array.size > 0 and features_array.shape[1] > kmer_len:
                    margin = (features_array.shape[1] - kmer_len) // 2
                    start = margin + dwell_offset
                    features_array = features_array[:, start : start + kmer_len]
                features = torch.from_numpy(features_array.astype(np.float32)).to(device)
                batch["features"] = features.unsqueeze(0)

            # Run all models
            probs = []
            with torch.no_grad():
                for pair in pairs:
                    logits = wrappers[pair].forward_batch(batch, device)
                    prob = torch.sigmoid(logits).item()
                    probs.append(prob)

            # Aggregate to single AA prediction
            predicted_aa, confidence, _ = aggregate_fn(pairs, probs)

            # Write tags
            aln.set_tag("aa", predicted_aa)  # Z type (string)
            aln.set_tag("ac", confidence)  # f type (float)
            if raw:
                aln.set_tag("pn", pair_names_str)  # Z type (string)
                aln.set_tag("pp", probs)  # B:f type (float array)
            bam_out.write(aln)

    bam_in.close()
    bam_out.close()

    logger.info(f"Bundle inference complete: {total_reads} reads, {len(pairs)} models")
    logger.info(f"Output written to: {output_path}")


def load_predictions_from_bam(bam_path: Path) -> dict:
    """
    Load predictions from a BAM file with modification tags.

    Args:
        bam_path: Path to BAM file with predictions

    Returns:
        Dictionary mapping read_id -> list of (position, probability) tuples
    """
    predictions: dict[str, list[tuple]] = {}

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for aln in bam:
            if aln.has_tag("MP") and aln.has_tag("ML"):
                positions_raw = aln.get_tag("MP")
                ml_scores_raw = aln.get_tag("ML")

                # Ensure we have array types
                if not isinstance(positions_raw, (list, np.ndarray)):
                    continue
                if not isinstance(ml_scores_raw, (list, np.ndarray)):
                    continue

                # Convert ML scores back to probabilities
                probs = [float(score) / 255.0 for score in ml_scores_raw]

                read_name = aln.query_name
                if read_name is not None:
                    predictions[read_name] = list(zip(positions_raw, probs, strict=False))

    return predictions
