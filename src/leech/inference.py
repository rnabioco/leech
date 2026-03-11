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

import array
import json
import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pysam
import torch
from rich.progress import Progress

from leech.features import encode_signal_kmer, sequence_to_int
from leech.io.motif_search import find_motif_in_sequence
from leech.models.inference_wrapper import ModelInferenceWrapper, TracedModelWrapper
from leech.models.remora_compat import RemoraModelWrapper
from leech.preparation import encode_kmer, iter_bam_with_pod5
from leech.util import _instantiate_model, deserialize_traced_model, load_model_from_checkpoint

if TYPE_CHECKING:
    from leech.signal_refine import SigMapRefiner

logger = logging.getLogger("leech.inference")


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
        raw["refine_scale_iters"] = int(raw.get("refine_scale_iters", -1))
        raw["refine_kmer_center_idx"] = int(raw.get("refine_kmer_center_idx", -1))

    return raw


def _is_leech_torchscript(path: Path) -> bool:
    """Check if a .pt file is a leech TorchScript export (has leech_meta.txt)."""
    try:
        extra = {"leech_meta.txt": ""}
        torch.jit.load(str(path), map_location="cpu", _extra_files=extra)
        return bool(extra.get("leech_meta.txt", ""))
    except Exception:
        return False


def load_model_auto(
    model_path: Path, device: str = "cpu"
) -> tuple[ModelInferenceWrapper | TracedModelWrapper | RemoraModelWrapper, dict]:
    """
    Load leech model (directory), leech TorchScript, or Remora TorchScript (.pt file).

    Auto-detects format:
    - Directory with config.json → leech checkpoint
    - .pt file with leech_meta.txt → leech TorchScript export
    - .pt file with meta.txt → Remora TorchScript model

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
        # Try leech TorchScript first
        extra = {"leech_meta.txt": ""}
        try:
            traced = torch.jit.load(str(path), map_location=device, _extra_files=extra)
        except Exception:
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


@dataclass
class InferenceWorkerConfig:
    """Configuration shared by all parallel inference workers."""

    pod5_path: Path
    motif: str | None
    motif_offset: int
    signal_context: tuple[int, int]
    kmer_context: int
    base_justify: str
    seq_encoding: str
    signal_kmer_context: tuple[int, int]
    signal_len: int
    kmer_len: int
    dwell_offset: int
    dwell_margin_left: int
    dwell_margin_right: int
    wide_features: bool
    requires_features: bool
    reverse_signal: bool
    # Reference-anchored mode + normalization + refinement
    anchor: str
    norm_method: str
    pa_mean: float | None
    pa_stdev: float | None
    refine_signal_map: bool
    signal_refiner: "SigMapRefiner | None"


def _inference_worker(
    args: tuple[list, InferenceWorkerConfig],
) -> list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray | None]]:
    """
    Worker for parallel chunk extraction during inference.

    Extracts chunks from reads and optionally pre-computes signal_kmer encoding.

    Returns:
        List of (read_id, base_idx, signal, encoded_sequence, features_or_none) tuples
    """
    from pod5 import DatasetReader

    from leech.io.pod5_reader import _extract_pod5_metadata
    from leech.preparation.reader import build_leech_read

    read_infos, config = args

    results: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray | None]] = []

    # Batch-read all POD5 signals in one traversal (avoids per-read seeks on large files)
    read_info_by_id = {ri.read_id: ri for ri in read_infos}
    pod5_cache: dict[str, tuple] = {}  # read_id -> (signal, metadata)

    with DatasetReader(config.pod5_path) as pod5_reader:
        for read in pod5_reader.reads(list(read_info_by_id.keys())):
            rid = str(read.read_id)
            pod5_cache[rid] = (read.signal, _extract_pod5_metadata(read))

        for read_info in read_infos:
            try:
                cached = pod5_cache.get(read_info.read_id)
                if cached is None:
                    continue
                raw_signal, pod5_metadata = cached

                # Build LeechRead via shared helper (with full params)
                leech_read = build_leech_read(
                    read_id=read_info.read_id,
                    sequence=read_info.sequence,
                    raw_signal=raw_signal,
                    move_table=read_info.to_move_table(),
                    reverse_signal=config.reverse_signal,
                    compute_features=config.requires_features,
                    anchor=config.anchor,
                    reference_sequence=read_info.reference_sequence,
                    cigar_tuples=read_info.cigar_tuples,
                    norm_method=config.norm_method,
                    pa_mean=config.pa_mean,
                    pa_stdev=config.pa_stdev,
                    cal_offset=pod5_metadata.get("calibration_offset"),
                    cal_scale=pod5_metadata.get("calibration_scale"),
                    refine_signal_map=config.refine_signal_map,
                    signal_refiner=config.signal_refiner,
                )

                # Find motif positions
                if config.motif is not None:
                    positions = [
                        pos + config.motif_offset
                        for pos in find_motif_in_sequence(leech_read.sequence, config.motif)
                    ]
                else:
                    positions = list(
                        range(config.kmer_context, leech_read.num_bases - config.kmer_context)
                    )

                for base_idx in positions:
                    chunk = leech_read.get_chunk(
                        base_idx,
                        signal_context=config.signal_context,
                        kmer_context=config.kmer_context,
                        base_justify=config.base_justify,
                        dwell_margin_left=config.dwell_margin_left if config.dwell_margin_left else None,
                        dwell_margin_right=config.dwell_margin_right if config.dwell_margin_right else None,
                    )
                    if chunk is None:
                        continue

                    # Signal
                    sig = chunk["signal"].astype(np.float32)
                    if len(sig) < config.signal_len:
                        sig = np.pad(sig, (0, config.signal_len - len(sig)), mode="constant")
                    elif len(sig) > config.signal_len:
                        start = (len(sig) - config.signal_len) // 2
                        sig = sig[start : start + config.signal_len]

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
                                # Wide feature models get full margin
                                pass
                            elif feat_arr.shape[1] > config.kmer_len:
                                margin = (feat_arr.shape[1] - config.kmer_len) // 2
                                s = margin + config.dwell_offset
                                feat_arr = feat_arr[:, s : s + config.kmer_len]
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
    min_mapq: int = 0,
    motif: str | None = None,
    motif_offset: int = 0,
    batch_size: int = 256,
    base_justify: str = "center",
    reverse_signal: bool = True,
    num_workers: int = 0,
    chunk_size: int = 100,
    anchor: str = "basecall",
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
        num_workers: Parallel chunk extraction workers (0=sequential).
            Only beneficial with GPU inference, where CPU chunk extraction
            overlaps with GPU forward passes. For CPU-only inference, the
            sequential path (0) is faster due to batched POD5 access and
            no multiprocessing overhead.
        chunk_size: Reads per worker batch
        anchor: "basecall" or "reference" for reference-anchored mode
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

        # Auto-read motif from model metadata
        if motif is None and config.get("motif"):
            motif = config["motif"]
            motif_offset = config.get("motif_offset", motif_offset)
            logger.info(f"Auto-read motif from remora model: {motif} (offset={motif_offset})")

        if motif is None:
            raise ValueError("--motif is required for Remora models (no config.json)")

        # Set up signal map refinement if model specifies it
        if config.get("refine_signal_map", False):
            from leech.data import get_kmer_table
            from leech.signal_refine import SigMapRefiner

            kmer_table_path = get_kmer_table()
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
        seq_encoding = config.get("seq_encoding", "base_onehot")
        signal_kmer_context = tuple(config.get("signal_kmer_context", (4, 4)))

        # Auto-read motif from model config if not provided
        if motif is None and config.get("motif"):
            motif = config["motif"]
            motif_offset = config.get("motif_offset", motif_offset)
            logger.info(f"Auto-read motif from config: {motif} (offset={motif_offset})")

    # Use asymmetric context if available, otherwise fall back to symmetric
    left_ctx = config.get("left_context")
    right_ctx = config.get("right_context")
    if left_ctx is not None and right_ctx is not None:
        signal_context = (left_ctx, right_ctx)
    else:
        signal_context = (signal_len // 2, signal_len // 2)
    kmer_context = kmer_len // 2
    requires_features = getattr(model_wrapper, "requires_features", False)

    # Determine dwell margins from config (must match training data preparation)
    _model_type = getattr(model_wrapper, "model_type", "")
    wide_features = _model_type in ModelInferenceWrapper.WIDE_FEATURE_MODELS
    dwell_margin_left = config.get("dwell_margin_left", 0)
    dwell_margin_right = config.get("dwell_margin_right", 0)
    if wide_features and dwell_margin_left == 0 and dwell_margin_right == 0:
        # Fallback for old models without explicit margins: use model default
        _model_margin = getattr(model_wrapper.model, "dwell_margin", 0) if hasattr(model_wrapper, "model") else 0
        if _model_margin:
            dwell_margin_left = _model_margin
            dwell_margin_right = _model_margin
            logger.warning(
                f"Config missing dwell_margin_left/right, "
                f"falling back to model default: {_model_margin}"
            )

    logger.info(f"Signal length: {signal_len}, K-mer length: {kmer_len}")
    logger.info(f"Signal context: {signal_context}")
    logger.info(f"Sequence encoding: {seq_encoding}")
    if dwell_margin_left or dwell_margin_right:
        logger.info(
            f"Dwell margins: left={dwell_margin_left}, right={dwell_margin_right} "
            f"(feature width={kmer_len + dwell_margin_left + dwell_margin_right})"
        )
    if motif:
        logger.info(f"Motif: {motif} (offset={motif_offset})")

    # Open input BAM and index alignments by read ID
    bam_in = pysam.AlignmentFile(str(bam_path), "rb")
    alignment_by_read_id: dict[str, pysam.AlignedSegment] = {}
    for a in bam_in.fetch(until_eof=True):
        if a.query_name is not None:
            alignment_by_read_id[a.query_name] = a

    # Detect normalization method: for remora models, use sm/sd tags if available
    norm_method = "median_mad"
    pa_mean = None
    pa_stdev = None
    if is_remora and alignment_by_read_id:
        first_aln = next(iter(alignment_by_read_id.values()))
        if first_aln.has_tag("sm") and first_aln.has_tag("sd"):
            pa_mean = float(first_aln.get_tag("sm"))
            pa_stdev = float(first_aln.get_tag("sd"))
            norm_method = "pa_scaling"
            logger.info(f"Using pa_scaling normalization (sm={pa_mean}, sd={pa_stdev})")

    # Create output BAM
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bam_out = pysam.AlignmentFile(str(output_path), "wb", template=bam_in)

    total_reads = 0
    total_predictions = 0

    if hasattr(model_wrapper, "model"):
        model_wrapper.model.eval()
    if hasattr(model_wrapper, "eval"):
        model_wrapper.eval()

    # Skip feature computation when model doesn't need them (big speedup)
    compute_features = requires_features

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

        config = InferenceWorkerConfig(
            pod5_path=pod5_path,
            motif=motif,
            motif_offset=motif_offset,
            signal_context=signal_context,
            kmer_context=kmer_context,
            base_justify=base_justify,
            seq_encoding=seq_encoding,
            signal_kmer_context=signal_kmer_context,
            signal_len=signal_len,
            kmer_len=kmer_len,
            dwell_offset=dwell_offset,
            dwell_margin_left=dwell_margin_left,
            dwell_margin_right=dwell_margin_right,
            wide_features=wide_features,
            requires_features=requires_features,
            reverse_signal=reverse_signal,
            anchor=anchor,
            norm_method=norm_method,
            pa_mean=pa_mean,
            pa_stdev=pa_stdev,
            refine_signal_map=refine_signal_map,
            signal_refiner=signal_refiner,
        )

        worker_args = [(batch_reads, config) for batch_reads in read_batches]

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
                positions_list = [int(p[0]) for p in preds]
                ml_scores = [int(min(255, max(0, p[1] * 255))) for p in preds]
                aln.set_tag("MP", array.array("i", positions_list))
                aln.set_tag("ML", array.array("B", ml_scores))
            bam_out.write(aln)

    else:
        # ---- Sequential path ----
        # Accumulate chunks across reads for batched model forward passes
        batch_signals: list[np.ndarray] = []
        batch_seqs: list[np.ndarray] = []
        batch_feats: list[np.ndarray | None] = []
        batch_meta: list[tuple[str, int]] = []  # (read_id, base_idx)
        pending: dict[str, list[tuple[int, float]]] = {}

        def _flush_batch() -> None:
            """Run accumulated chunks through model."""
            nonlocal batch_signals, batch_seqs, batch_feats, batch_meta
            if not batch_signals:
                return
            _run_batch(
                batch_signals,
                batch_seqs,
                batch_feats,
                batch_meta,
                model_wrapper,
                requires_features,
                device,
                pending,
            )
            batch_signals, batch_seqs, batch_feats, batch_meta = [], [], [], []

        with Progress() as progress:
            task = progress.add_task("[cyan]Running inference...", total=None)

            for leech_read in iter_bam_with_pod5(
                bam_path,
                pod5_path,
                min_mapq=min_mapq,
                reverse_signal=reverse_signal,
                anchor=anchor,
                norm_method=norm_method,
                pa_mean=pa_mean,
                pa_stdev=pa_stdev,
                refine_signal_map=refine_signal_map,
                signal_refiner=signal_refiner,
                compute_features=compute_features,
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

                for base_idx in positions:
                    chunk = leech_read.get_chunk(
                        base_idx,
                        signal_context=signal_context,
                        kmer_context=kmer_context,
                        base_justify=base_justify,
                        dwell_margin_left=dwell_margin_left if dwell_margin_left else None,
                        dwell_margin_right=dwell_margin_right if dwell_margin_right else None,
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

                    # Sequence
                    seq_enc = _encode_sequence_for_inference(
                        chunk, seq_encoding, signal_len, signal_kmer_context
                    )
                    if seq_enc is None:
                        continue

                    batch_signals.append(sig)
                    batch_seqs.append(
                        seq_enc.numpy() if isinstance(seq_enc, torch.Tensor) else seq_enc
                    )
                    # Features
                    feat = None
                    if requires_features:
                        features_array = chunk["features"]
                        assert isinstance(features_array, np.ndarray)
                        feat = features_array.astype(np.float32)
                        if wide_features:
                            # Wide feature models get full margin
                            pass
                        elif feat.size > 0 and feat.shape[1] > kmer_len:
                            margin = (feat.shape[1] - kmer_len) // 2
                            s = margin + dwell_offset
                            feat = feat[:, s : s + kmer_len]
                    batch_feats.append(feat)
                    batch_meta.append((leech_read.read_id, base_idx))

                    if len(batch_signals) >= batch_size:
                        _flush_batch()

            # Flush remaining chunks
            _flush_batch()

        # Write all results to BAM
        total_predictions = sum(len(v) for v in pending.values())
        for aln in alignment_by_read_id.values():
            preds = pending.get(aln.query_name)
            if preds:
                preds.sort(key=lambda x: x[0])
                positions_list = [int(p[0]) for p in preds]
                ml_scores = [int(min(255, max(0, p[1] * 255))) for p in preds]
                aln.set_tag("MP", array.array("i", positions_list))
                aln.set_tag("ML", array.array("B", ml_scores))
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
    is_torchscript = metadata.get("torchscript", False)

    signal_len = config["signal_len"]
    kmer_len = config["kmer_len"]
    model_type = config.get("model_name", metadata.get("architecture", ""))
    dwell_offset = config.get("dwell_offset", 0)
    seq_encoding = config.get("seq_encoding", "base_onehot")
    signal_kmer_context = tuple(config.get("signal_kmer_context", (4, 4)))

    # Auto-read motif/offset/justify from bundle config if not provided on CLI
    if motif is None and config.get("motif"):
        motif = config["motif"]
        motif_offset = config.get("motif_offset", motif_offset)
        logger.info(f"Auto-read motif from bundle config: {motif} (offset={motif_offset})")

    if base_justify == "center" and config.get("base_justify"):
        base_justify = config["base_justify"]
        logger.info(f"Auto-read base_justify from bundle config: {base_justify}")

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

    # Determine dwell margins from config (must match training data preparation)
    wide_features = model_type in ModelInferenceWrapper.WIDE_FEATURE_MODELS
    dwell_margin_left = config.get("dwell_margin_left", 0)
    dwell_margin_right = config.get("dwell_margin_right", 0)
    if wide_features and dwell_margin_left == 0 and dwell_margin_right == 0:
        # Fallback for old bundles without explicit margins: use model default
        _model_margin = getattr(_instantiate_model(config), "dwell_margin", 0)
        if _model_margin:
            dwell_margin_left = _model_margin
            dwell_margin_right = _model_margin
            logger.warning(
                f"Bundle config missing dwell_margin_left/right, "
                f"falling back to model default: {_model_margin}"
            )

    logger.info(f"Signal context: {signal_context}, kmer_len: {kmer_len}")
    if dwell_margin_left or dwell_margin_right:
        logger.info(
            f"Dwell margins: left={dwell_margin_left}, right={dwell_margin_right} "
            f"(feature width={kmer_len + dwell_margin_left + dwell_margin_right})"
        )

    # Select aggregation function
    if comparison_type == "pairwise":
        aggregate_fn = aggregate_pairwise
    else:
        aggregate_fn = aggregate_one_vs_all

    # Load all models
    wrappers: dict[str, ModelInferenceWrapper | TracedModelWrapper] = {}
    if is_torchscript:
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
            m.load_state_dict(state_dict)
            m = m.to(device)
            m.eval()
            wrappers[pair] = ModelInferenceWrapper(m, model_type)

    # Load per-model Platt scaling params (for post-hoc calibration)
    platt_params: dict[str, tuple[float, float]] = {}
    for pair in pairs:
        a = bundle["models"][pair].get("platt_a")
        b = bundle["models"][pair].get("platt_b")
        if a is not None and b is not None:
            platt_params[pair] = (a, b)
    if platt_params:
        logger.info(f"Platt scaling enabled for {len(platt_params)}/{len(pairs)} models")

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
                dwell_margin_left=dwell_margin_left if dwell_margin_left else None,
                dwell_margin_right=dwell_margin_right if dwell_margin_right else None,
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
                if wide_features:
                    # Wide feature models (attention variants) get full margin
                    pass
                elif features_array.size > 0 and features_array.shape[1] > kmer_len:
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
                    pp = platt_params.get(pair)
                    if pp is not None:
                        a, b = pp
                        logits = a * logits + b
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
