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
import functools
import json
import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pysam
import torch
from rich.progress import Progress

from leech.configs import ChunkConfig, InferenceConfig, MotifConfig, SignalConfig
from leech.features import encode_signal_kmer, sequence_to_int
from leech.io.motif_search import get_motif_searcher
from leech.models.inference_wrapper import ModelInferenceWrapper, TracedModelWrapper
from leech.models.remora_compat import RemoraModelWrapper
from leech.preparation import encode_kmer, iter_bam_with_pod5
from leech.util import (
    _instantiate_model,
    deserialize_exported_model,
    deserialize_traced_model,
    load_model_from_checkpoint,
)

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
) -> None:
    """Write prediction tags to a BAM alignment.

    Args:
        aln: pysam alignment to tag
        predicted_aa: predicted amino acid label
        conf: max class probability (0.0-1.0)
        class_names_str: comma-separated class names for pn tag
        probs: full probability distribution
        raw: if True, write float tags; otherwise compact uint8
        min_confidence: threshold in 0-255 uint8 space
        min_margin: margin threshold in 0-255 uint8 space
    """
    sorted_probs = sorted(probs, reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
    margin_uint8 = int(min(255, max(0, round(margin * 255))))
    ac_uint8 = int(min(255, max(0, round(conf * 255))))
    is_charged = ac_uint8 >= min_confidence and margin_uint8 >= min_margin

    if is_charged:
        aln.set_tag("aa", predicted_aa)
        ac_val = conf
    else:
        aln.set_tag("aa", "unc")
        ac_val = 1.0 - conf

    if raw:
        aln.set_tag("ac", ac_val)
        aln.set_tag("am", margin)
    else:
        aln.set_tag("ac", int(min(255, max(0, round(ac_val * 255)))), value_type="C")
        aln.set_tag("am", margin_uint8, value_type="C")

    aln.set_tag("pn", class_names_str)
    if raw:
        aln.set_tag("pp", probs)
    else:
        aln.set_tag(
            "pp",
            array.array("B", [int(min(255, max(0, round(p * 255)))) for p in probs]),
        )


class InferenceConfigError(RuntimeError):
    """Raised when inference input shapes don't match the model's config."""


def _check_config_consistency[T](
    param_name: str,
    cli_value: T,
    config_value: T | None,
    cli_default: T,
) -> T:
    """Resolve inference param from config, erroring on CLI conflict.

    Logic:
    - config has value + CLI is default → use config (normal auto-read)
    - config has value + CLI differs   → raise InferenceConfigError
    - config is None/missing           → use CLI value (old models without field)
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
        signal: Signal array — 1D (single channel) or 2D (channels, signal_len).
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


def _is_leech_export(path: Path) -> bool:
    """Check if a .pt file is a leech export (torch.export or TorchScript with leech_meta.txt)."""
    # Try torch.export format first (format_version 3+)
    try:
        extra = {"leech_meta.txt": ""}
        torch.export.load(str(path), extra_files=extra)
        if extra.get("leech_meta.txt", ""):
            return True
    except Exception as e:
        logger.debug("Not torch.export format: %s", e)
    # Fall back to legacy TorchScript format
    try:
        extra = {"leech_meta.txt": ""}
        torch.jit.load(str(path), map_location="cpu", _extra_files=extra)
        return bool(extra.get("leech_meta.txt", ""))
    except Exception as e:
        logger.debug("Not TorchScript format: %s", e)
        return False


def load_model_auto(
    model_path: Path, device: str = "cpu"
) -> tuple[ModelInferenceWrapper | TracedModelWrapper | RemoraModelWrapper, dict]:
    """
    Load leech model (directory), leech TorchScript, or Remora TorchScript (.pt file).

    Auto-detects format:
    - Directory with config.json → leech checkpoint
    - .pt file with leech_meta.txt → leech torch.export or TorchScript export
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


def aggregate_pairwise_weighted(
    pairs: list[str], probs: list[float]
) -> tuple[str, float, dict[str, float]]:
    """Pairwise aggregation with confidence weighting.

    Models seeing OOD data produce p ~ 0.5 (uncertain). Weight each
    model's vote by its confidence: w = |2p - 1|, so uncertain models
    contribute near-zero and confident models dominate.
    """
    votes: dict[str, float] = {}
    for pair, prob in zip(pairs, probs, strict=True):
        aa_a, aa_b = pair.split("_", 1)
        confidence = abs(2 * prob - 1)  # 0 at p=0.5, 1 at p=0 or p=1
        votes[aa_a] = votes.get(aa_a, 0.0) + (1.0 - prob) * confidence
        votes[aa_b] = votes.get(aa_b, 0.0) + prob * confidence
    total = sum(votes.values())
    predicted_aa = max(votes, key=votes.__getitem__)
    confidence_score = votes[predicted_aa] / total if total > 0 else 0.0
    return predicted_aa, confidence_score, votes


def aggregate_pairwise_tournament(
    pairs: list[str], probs: list[float], top_k: int = 5
) -> tuple[str, float, dict[str, float]]:
    """Tournament-style aggregation.

    Round 1: naive vote to identify top-K candidates.
    Round 2: only aggregate models involving top-K AAs.
    Eliminates 90%+ of OOD noise.
    """
    # Round 1: get initial rankings
    _, _, initial_votes = aggregate_pairwise(pairs, probs)
    top_aas = sorted(initial_votes, key=initial_votes.__getitem__, reverse=True)[:top_k]

    # Round 2: only use relevant models
    final_votes: dict[str, float] = dict.fromkeys(top_aas, 0.0)
    for pair, prob in zip(pairs, probs, strict=True):
        aa_a, aa_b = pair.split("_", 1)
        if aa_a in final_votes and aa_b in final_votes:
            final_votes[aa_a] += 1.0 - prob
            final_votes[aa_b] += prob

    total = sum(final_votes.values())
    predicted_aa = max(final_votes, key=final_votes.__getitem__)
    confidence_score = final_votes[predicted_aa] / total if total > 0 else 0.0
    return predicted_aa, confidence_score, final_votes


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
    args: tuple[list, InferenceConfig],
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
    _shape_validated = False

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
        chunk_size: Reads per worker batch
        anchor: "basecall" or "reference" for reference-anchored mode
        reference_fasta: Path to reference FASTA (for reference-anchored mode)
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
            from leech.signal_refine import SigMapRefiner

            kmer_table_path = get_kmer_table()
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

    # Open input BAM and index alignments by read ID
    bam_in = pysam.AlignmentFile(str(bam_path), "rb")
    alignment_by_read_id: dict[str, pysam.AlignedSegment] = {}
    for a in bam_in.fetch(until_eof=True):
        if a.query_name is not None:
            alignment_by_read_id[a.query_name] = a

    # Detect normalization method: read from config, with sm/sd tag override for Remora
    norm_method = config.get("signal_norm", "median_mad")
    pa_mean = config.get("pa_mean")
    pa_stdev = config.get("pa_stdev")
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

        worker_args = [(batch_reads, inf_config) for batch_reads in read_batches]

        # Collect all results, then batch and run model
        pending: dict[str, list] = {}
        calibration = config.get("calibration") if is_multiclass else None
        _batch_fn_p = (
            functools.partial(_run_batch_multiclass, calibration=calibration)
            if is_multiclass
            else _run_batch
        )

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

                    # Flush remaining
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

                    progress.advance(task)

        # Write all results to BAM (iterate dict to preserve original BAM order)
        total_predictions = sum(len(v) for v in pending.values())
        if is_multiclass:
            if int_to_label:
                class_names = [int_to_label[i] for i in range(num_out)]
                class_names_str = ",".join(class_names)
            else:
                class_names_str = ",".join(str(i) for i in range(num_out))

            for aln in alignment_by_read_id.values():
                preds = pending.get(aln.query_name)
                if preds:
                    _, cls_idx, conf, all_probs = preds[0]
                    if int_to_label:
                        predicted_aa = int_to_label.get(cls_idx, str(cls_idx))
                    else:
                        predicted_aa = str(cls_idx)
                    _write_prediction_tags(
                        aln, predicted_aa, conf, class_names_str,
                        all_probs, raw, min_confidence, min_margin,
                    )
                bam_out.write(aln)
        else:
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
        pending: dict[str, list] = {}
        _shape_validated = False

        calibration = config.get("calibration") if is_multiclass else None
        _batch_fn = (
            functools.partial(_run_batch_multiclass, calibration=calibration)
            if is_multiclass
            else _run_batch
        )

        def _flush_batch() -> None:
            """Run accumulated chunks through model."""
            nonlocal batch_signals, batch_seqs, batch_feats, batch_meta
            if not batch_signals:
                return
            _batch_fn(
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
            )

            for leech_read in iter_bam_with_pod5(
                bam_path,
                pod5_path,
                signal_config=seq_signal_config,
                min_mapq=min_mapq,
                reference_sequences=reference_sequences,
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

                # Find positions to predict (motif is validated non-None above)
                assert motif is not None, "motif must not be None at inference time"
                aln_meta = leech_read.metadata.get("alignment")
                positions = [
                    pos + motif_offset
                    for pos in motif_searcher.find_motif_positions(
                        leech_read.read_id, leech_read.sequence, aln_meta, motif
                    )
                ]

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
                            sig_residual = sig_residual.astype(np.float32)[
                                start : start + signal_len
                            ]
                    if sig_residual is not None:
                        sig_residual = sig_residual.astype(np.float32)
                        sig = np.stack([sig, sig_residual], axis=0)  # (2, signal_len)

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
                            pass
                        elif feat.size > 0 and feat.shape[1] > kmer_len:
                            fs = chunk.get("feature_start", -_kmer_context)
                            s = (-_kmer_context - fs) + dwell_offset
                            feat = feat[:, s : s + kmer_len]
                    # Validate shapes on first chunk
                    if not _shape_validated:
                        validate_inference_shapes(sig, feat, config)
                        _shape_validated = True

                    batch_feats.append(feat)
                    batch_meta.append((leech_read.read_id, base_idx))

                    if len(batch_signals) >= batch_size:
                        _flush_batch()

            # Flush remaining chunks
            _flush_batch()

        # Write all results to BAM
        total_predictions = sum(len(v) for v in pending.values())
        if is_multiclass:
            # Multi-class: write aa/ac tags + full softmax (pn/pp)
            # pn:Z — comma-separated class names (sorted by label_int)
            if int_to_label:
                class_names = [int_to_label[i] for i in range(num_out)]
                class_names_str = ",".join(class_names)
            else:
                class_names_str = ",".join(str(i) for i in range(num_out))

            for aln in alignment_by_read_id.values():
                preds = pending.get(aln.query_name)
                if preds:
                    # Take first prediction (typically one motif per read)
                    _, cls_idx, conf, all_probs = preds[0]
                    if int_to_label:
                        predicted_aa = int_to_label.get(cls_idx, str(cls_idx))
                    else:
                        predicted_aa = str(cls_idx)
                    _write_prediction_tags(
                        aln, predicted_aa, conf, class_names_str,
                        all_probs, raw, min_confidence, min_margin,
                    )
                bam_out.write(aln)
        else:
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


def _run_batch_multiclass(
    signals: list[np.ndarray],
    sequences: list[np.ndarray],
    features: list[np.ndarray | None],
    meta: list[tuple[str, int]],
    model_wrapper: ModelInferenceWrapper | TracedModelWrapper | RemoraModelWrapper,
    requires_features: bool,
    device: str,
    pending: dict[str, list[tuple[int, int, float, list[float]]]],
    calibration: dict | None = None,
) -> None:
    """Run a multi-class batch: store (base_idx, class_idx, confidence, all_probs) per read."""
    signal_t = torch.from_numpy(np.stack(signals)).to(device)
    seq_t = torch.from_numpy(np.stack(sequences)).to(device)
    batch = {"signal": signal_t, "sequence": seq_t}

    if requires_features:
        valid_feats = [f for f in features if f is not None]
        if valid_feats:
            batch["features"] = torch.from_numpy(np.stack(valid_feats)).to(device)

    with torch.no_grad():
        logits = model_wrapper.forward_batch(batch, device)
        if calibration is not None:
            from leech.calibration import apply_calibration

            logits = apply_calibration(logits, calibration)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        class_indices = np.argmax(probs, axis=-1)
        confidences = probs.max(axis=-1)

    for (read_id, base_idx), cls_idx, conf, prob_vec in zip(
        meta, class_indices.flatten(), confidences.flatten(), probs, strict=True
    ):
        if read_id not in pending:
            pending[read_id] = []
        pending[read_id].append((base_idx, int(cls_idx), float(conf), [float(p) for p in prob_vec]))


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
    min_confidence: int = 0,
    min_margin: int = 0,
    aggregation: str = "naive",
    anchor: str = "reference",
    reference_fasta: Path | None = None,
    batch_size: int = 512,
    num_workers: int = 0,
) -> None:
    """
    Run all models from a bundle on each read, aggregate to a single AA prediction.

    Writes output BAM with tags (compact by default, float with --raw):
    - aa:Z:Gly — predicted amino acid (or "unc" if below threshold)
    - ac:C:238 (compact) or ac:f:0.93 (raw) — confidence score
    - pn:Z:Ala_Gly,... — pair names (omitted for uncharged unless --raw)
    - pp:B:B,... (compact) or pp:B:f,... (raw) — probabilities

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

    # Load all models (skip for vmap bundles — stacked params loaded in vmap setup)
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
            from leech.util import _migrate_state_dict_keys

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
    )

    # Open BAM files and index alignments by read ID (avoids O(n^2) scanning)
    bam_in = pysam.AlignmentFile(str(bam_path), "rb")
    alignment_by_read_id: dict[str, pysam.AlignedSegment] = {}
    for a in bam_in.fetch(until_eof=True):
        if a.query_name is not None:
            alignment_by_read_id[a.query_name] = a
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bam_out = pysam.AlignmentFile(str(output_path), "wb", template=bam_in)

    first_wrapper = next(iter(wrappers.values()))
    needs_features = first_wrapper.requires_features
    bundle_signal_in_channels = config.get("signal_in_channels", 1)
    bundle_compute_features = needs_features or bundle_signal_in_channels > 1

    # Signal map refinement for bundle models (needed for kmer residual signal channel)
    bundle_refine = False
    bundle_refiner = None
    if config.get("refine_signal_map", True) or bundle_signal_in_channels > 1:
        from leech.data import get_kmer_table
        from leech.signal_refine import SigMapRefiner

        kmer_table_path = get_kmer_table()
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

    # ── Setup vmap forward for format_version 4, else use sequential wrappers ──
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

    # ── Streaming batch inference ──
    # Accumulate chunks into batches, flush when full. This avoids materializing
    # all chunks in memory before inference starts.
    read_probs: dict[str, np.ndarray] = {}  # read_id → shape (n_pairs,)
    pair_to_idx = {pair: i for i, pair in enumerate(pairs)}
    n_pairs = len(pairs)

    batch_signals: list[torch.Tensor] = []
    batch_sequences: list[torch.Tensor] = []
    batch_features: list[torch.Tensor] = []
    batch_read_ids: list[str] = []
    _shape_validated = False
    n_chunks = 0
    n_batches_done = 0

    def _flush_batch() -> None:
        nonlocal n_batches_done
        if not batch_signals:
            return

        sig_t = torch.stack(batch_signals).to(device)
        seq_t = torch.stack(batch_sequences).to(device)
        feat_t = torch.stack(batch_features).to(device) if needs_features else None

        with torch.inference_mode():
            if is_vmap and vmapped_forward is not None:
                # Vectorized: run all N models in one pass
                if needs_features:
                    all_logits = vmapped_forward(
                        vmap_stacked_params, vmap_stacked_buffers, sig_t, seq_t, feat_t
                    )
                else:
                    all_logits = vmapped_forward(
                        vmap_stacked_params, vmap_stacked_buffers, sig_t, seq_t, None
                    )
                # all_logits: (n_models, batch_size, 1) → apply Platt scaling
                all_logits = vmap_platt_a[:, None, None] * all_logits + vmap_platt_b[:, None, None]
                all_p = torch.sigmoid(all_logits).squeeze(-1).cpu().numpy()  # (n_models, batch)
            else:
                # Sequential: run each model on the batch
                all_p = np.empty((n_pairs, len(batch_signals)), dtype=np.float32)
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

        # Scatter into read_probs
        for i, rid in enumerate(batch_read_ids):
            read_probs[rid] = all_p[:, i]  # shape (n_pairs,)

        batch_signals.clear()
        batch_sequences.clear()
        batch_features.clear()
        batch_read_ids.clear()
        n_batches_done += 1

    n_reads = 0

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
    )

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing reads...", total=None)

        for leech_read in iter_bam_with_pod5(
            bam_path,
            pod5_path,
            signal_config=bundle_signal_config,
            min_mapq=min_mapq,
            reference_sequences=reference_sequences,
        ):
            n_reads += 1

            # Find prediction position(s) (motif is validated non-None above)
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
                        sig_residual, (0, len(sig) - len(sig_residual)), mode="constant"
                    )
                elif len(sig_residual) > len(sig):
                    sig_residual = sig_residual[: len(sig)]
                sig = np.stack([sig, sig_residual], axis=0)  # (2, signal_len)
            signal_t = torch.from_numpy(sig)

            seq_t = _encode_sequence_for_inference(
                chunk, seq_encoding, signal_len, signal_kmer_context
            )
            if seq_t is None:
                continue

            n_chunks += 1
            batch_signals.append(signal_t)
            batch_sequences.append(seq_t)
            batch_read_ids.append(leech_read.read_id)

            if needs_features:
                features_array = chunk["features"]
                assert isinstance(features_array, np.ndarray)
                if wide_features:
                    pass
                elif features_array.size > 0 and features_array.shape[1] > kmer_len:
                    fs = chunk.get("feature_start", -_kmer_context)
                    feat_start = (-_kmer_context - fs) + dwell_offset
                    features_array = features_array[:, feat_start : feat_start + kmer_len]
                batch_features.append(torch.from_numpy(features_array.astype(np.float32)))

            if not _shape_validated:
                _feat_for_check = features_array.astype(np.float32) if needs_features else None
                validate_inference_shapes(sig, _feat_for_check, config)
                _shape_validated = True

            if len(batch_signals) >= batch_size:
                _flush_batch()

            progress.update(
                task,
                advance=0,
                description=(
                    f"[cyan]Processed {n_chunks} chunks from {n_reads} reads "
                    f"({n_batches_done} batches)..."
                ),
            )

        # Flush remaining chunks
        _flush_batch()

    logger.info(f"Extracted and inferred {n_chunks} chunks from {n_reads} reads")

    if n_chunks == 0:
        # No valid chunks — write all reads without tags
        for aln in alignment_by_read_id.values():
            bam_out.write(aln)
        bam_in.close()
        bam_out.close()
        return

    # ── Aggregate per-read and write BAM ──
    n_predicted = 0
    for read_id, aln in alignment_by_read_id.items():
        prob_vec = read_probs.get(read_id)
        if prob_vec is None:
            bam_out.write(aln)
            continue

        probs = [float(prob_vec[pair_to_idx[pair]]) for pair in pairs]
        predicted_aa, confidence, _ = aggregate_fn(pairs, probs)

        _write_prediction_tags(
            aln, predicted_aa, confidence, pair_names_str,
            probs, raw, min_confidence, min_margin,
        )
        bam_out.write(aln)
        n_predicted += 1

    logger.info(f"Predicted {n_predicted}/{len(alignment_by_read_id)} reads")
    bam_in.close()
    bam_out.close()

    logger.info(f"Bundle inference complete: {n_reads} reads, {len(pairs)} models")
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
