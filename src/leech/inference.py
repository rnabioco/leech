"""
Inference engine for running predictions on new data.

Reads POD5 and BAM files, extracts features, runs model predictions,
and writes modification probabilities to output BAM files.
"""

import logging
from pathlib import Path

import numpy as np
import pysam
import torch
from rich.progress import Progress

from leech.models.inference_wrapper import ModelInferenceWrapper
from leech.preparation import encode_kmer, iter_bam_with_pod5
from leech.util import _instantiate_model, load_model_from_checkpoint

logger = logging.getLogger("leech.inference")


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


def run_inference(
    model_and_config: tuple[torch.nn.Module, dict] | None = None,
    model_path: Path | None = None,
    pod5_path: Path | None = None,
    bam_path: Path | None = None,
    output_path: Path | None = None,
    device: str = "cuda",
    min_mapq: int = 10,
    motif: str | None = None,
    motif_offset: int = 0,
    batch_size: int = 128,
    base_justify: str = "center",
    reverse_signal: bool = True,
) -> None:
    """
    Run inference on POD5 and BAM files.

    Writes predictions to output BAM with modification probability tags.

    Args:
        model_and_config: Pre-loaded (model, config) tuple. If provided, model_path is ignored.
        model_path: Path to model checkpoint directory (used if model_and_config is None)
        pod5_path: Path to POD5 file with raw signal
        bam_path: Path to input BAM file with alignments
        output_path: Path to output BAM file with predictions
        device: Device for inference
        min_mapq: Minimum mapping quality
        motif: Optional motif to filter predictions
        motif_offset: Offset within motif for prediction
        batch_size: Batch size for inference
    """
    if model_and_config is not None:
        model, config = model_and_config
    elif model_path is not None:
        logger.info(f"Loading model from {model_path}")
        model, config = load_model_from_checkpoint(model_path, device=device)
    else:
        raise ValueError("Either model_and_config or model_path must be provided")

    signal_len = config["signal_len"]
    kmer_len = config["kmer_len"]
    model_type = config["model_name"]
    dwell_offset = config.get("dwell_offset", 0)

    # Wrap model for unified forward pass
    model_wrapper = ModelInferenceWrapper(model, model_type)

    # Calculate context from signal_len (assumes symmetric)
    signal_context = (signal_len // 2, signal_len // 2)
    kmer_context = kmer_len // 2

    logger.info(f"Model: {model_type}")
    logger.info(f"Signal length: {signal_len}")
    logger.info(f"K-mer length: {kmer_len}")
    logger.info(f"Signal context: {signal_context}")

    # Open input BAM and index alignments by read ID (avoids O(n^2) scanning)
    bam_in = pysam.AlignmentFile(str(bam_path), "rb")
    alignment_by_read_id: dict[str, pysam.AlignedSegment] = {}
    for a in bam_in.fetch(until_eof=True):
        if a.query_name is not None:
            alignment_by_read_id[a.query_name] = a

    # Create output BAM
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bam_out = pysam.AlignmentFile(str(output_path), "wb", template=bam_in)

    logger.info(f"\nProcessing reads from {bam_path}")
    logger.info(f"Output: {output_path}")

    total_reads = 0
    total_predictions = 0

    model.eval()

    # Process reads with progress bar
    with Progress() as progress:
        task = progress.add_task("[cyan]Running inference...", total=None)

        for leech_read in iter_bam_with_pod5(
            bam_path, pod5_path, min_mapq=min_mapq, reverse_signal=reverse_signal
        ):
            total_reads += 1
            progress.update(task, advance=1, description=f"[cyan]Processed {total_reads} reads...")

            # Get corresponding alignment from input BAM
            aln = alignment_by_read_id.get(leech_read.read_id)
            if aln is None:
                continue

            # Find positions to predict
            if motif is None:
                # Predict all positions (avoid edges)
                positions = list(range(kmer_context, leech_read.num_bases - kmer_context))
            else:
                # Find motif positions
                positions = []
                motif_len = len(motif)
                for i in range(len(leech_read.sequence) - motif_len + 1):
                    if leech_read.sequence[i : i + motif_len] == motif:
                        positions.append(i + motif_offset)

            # Extract chunks and run predictions
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

                # Prepare input tensors
                signal_array = chunk["signal"]
                assert isinstance(signal_array, np.ndarray)
                signal = torch.from_numpy(signal_array.astype(np.float32)).to(device)

                sequence_str = chunk["sequence"]
                assert isinstance(sequence_str, str)
                sequence = encode_kmer(sequence_str).to(device)

                # Add batch dimension
                signal = signal.unsqueeze(0)
                sequence = sequence.unsqueeze(0)

                # Prepare batch dict for wrapper
                batch = {
                    "signal": signal,
                    "sequence": sequence,
                }

                # Add features if model requires them
                if model_wrapper.requires_features:
                    features_array = chunk["features"]
                    assert isinstance(features_array, np.ndarray)
                    # Apply dwell_offset slicing if wider margin exists
                    if features_array.size > 0 and features_array.shape[1] > kmer_len:
                        margin = (features_array.shape[1] - kmer_len) // 2
                        start = margin + dwell_offset
                        features_array = features_array[:, start : start + kmer_len]
                    features = torch.from_numpy(features_array.astype(np.float32)).to(device)
                    batch["features"] = features.unsqueeze(0)

                # Run inference
                with torch.no_grad():
                    logits = model_wrapper.forward_batch(batch, device)
                    prob = torch.sigmoid(logits).item()

                predictions.append((base_idx, prob))

            # Add predictions to BAM tags
            # Using MM (base modification) and ML (modification likelihood) tags
            # Format: MM:Z:C+m,0,1,2;  ML:B:C,255,128,64
            if predictions:
                # Sort by position
                predictions.sort(key=lambda x: x[0])

                # Convert probabilities to phred-like scores (0-255)
                positions_list = [p[0] for p in predictions]
                probs_list = [p[1] for p in predictions]
                ml_scores = [int(min(255, max(0, p * 255))) for p in probs_list]

                # Add tags to alignment
                aln.set_tag("MP", positions_list, value_type="B")  # Modification positions
                aln.set_tag("ML", ml_scores, value_type="B")  # Modification likelihoods (0-255)

                total_predictions += len(predictions)

            # Write to output
            bam_out.write(aln)

    bam_in.close()
    bam_out.close()

    logger.info("\nInference complete!")
    logger.info(f"Reads processed: {total_reads}")
    logger.info(f"Total predictions: {total_predictions}")
    logger.info(f"Output written to: {output_path}")


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
                positions = []
                motif_len = len(motif)
                for i in range(len(leech_read.sequence) - motif_len + 1):
                    if leech_read.sequence[i : i + motif_len] == motif:
                        positions.append(i + motif_offset)

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

            sequence_str = chunk["sequence"]
            assert isinstance(sequence_str, str)
            sequence = encode_kmer(sequence_str).to(device).unsqueeze(0)

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
