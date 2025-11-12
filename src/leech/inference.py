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

from leech.preparation import encode_kmer, iter_bam_with_pod5
from leech.models.inference_wrapper import ModelInferenceWrapper
from leech.util import load_model_from_checkpoint

logger = logging.getLogger("leech.inference")


def run_inference(
    model_path: Path,
    pod5_path: Path,
    bam_path: Path,
    output_path: Path,
    device: str = "cuda",
    min_mapq: int = 10,
    motif: str | None = None,
    motif_offset: int = 0,
    batch_size: int = 128,
) -> None:
    """
    Run inference on POD5 and BAM files.

    Writes predictions to output BAM with modification probability tags.

    Args:
        model_path: Path to model checkpoint directory
        pod5_path: Path to POD5 file with raw signal
        bam_path: Path to input BAM file with alignments
        output_path: Path to output BAM file with predictions
        device: Device for inference
        min_mapq: Minimum mapping quality
        motif: Optional motif to filter predictions
        motif_offset: Offset within motif for prediction
        batch_size: Batch size for inference
    """
    logger.info(f"Loading model from {model_path}")

    # Load model and config
    model, config = load_model_from_checkpoint(model_path, device=device)

    signal_len = config["signal_len"]
    kmer_len = config["kmer_len"]
    model_type = config["model_name"]
    config["num_features"]

    # Wrap model for unified forward pass
    model_wrapper = ModelInferenceWrapper(model, model_type)

    # Calculate context from signal_len (assumes symmetric)
    signal_context = (signal_len // 2, signal_len // 2)
    kmer_context = kmer_len // 2

    logger.info(f"Model: {model_type}")
    logger.info(f"Signal length: {signal_len}")
    logger.info(f"K-mer length: {kmer_len}")
    logger.info(f"Signal context: {signal_context}")

    # Open input BAM
    bam_in = pysam.AlignmentFile(str(bam_path), "rb")

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

        for leech_read in iter_bam_with_pod5(bam_path, pod5_path, min_mapq=min_mapq):
            total_reads += 1
            progress.update(task, advance=1, description=f"[cyan]Processed {total_reads} reads...")

            # Get corresponding alignment from input BAM
            bam_in.reset()
            aln = None
            for a in bam_in.fetch(until_eof=True):
                if a.query_name == leech_read.read_id:
                    aln = a
                    break

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
                    base_idx, signal_context=signal_context, kmer_context=kmer_context
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
