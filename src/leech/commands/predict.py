"""
Handler for the 'predict' command.

This module contains the business logic for running inference on new data.
"""

import logging
from pathlib import Path

import rich_click as click
from rich.console import Console

logger = logging.getLogger("leech.commands.predict")
console = Console()


def validate_predict_args(
    model: Path | None,
    bundle_path: Path | None,
    pair: str | None,
    run_all: bool,
) -> None:
    """Validate mutually exclusive predict command arguments."""
    if model and bundle_path:
        raise click.UsageError("--model and --bundle are mutually exclusive")
    if not model and not bundle_path:
        raise click.UsageError("Either --model or --bundle is required")
    if bundle_path and not pair and not run_all:
        raise click.UsageError("--bundle requires either --pair or --all")
    if pair and run_all:
        raise click.UsageError("--pair and --all are mutually exclusive")
    if (pair or run_all) and not bundle_path:
        raise click.UsageError("--pair and --all require --bundle")


def handle_predict(
    model: Path | None,
    bundle_path: Path | None,
    pair: str | None,
    run_all: bool,
    raw: bool,
    pod5: Path,
    bam: Path,
    output: Path,
    device: str,
    base_justify: str,
    no_reverse_signal: bool,
    anchor: str,
    reference_fasta: Path | None,
    motif: str | None,
    motif_offset: int,
    batch_size: int,
    min_mapq: int,
    workers: int,
    aggregation: str = "naive",
) -> None:
    """
    Handle the predict command logic.

    Args:
        model: Trained model checkpoint directory
        bundle_path: Model bundle .pt file
        pair: Single pair name from bundle
        run_all: Run every model in the bundle
        raw: Write per-pair probabilities with --all
        pod5: POD5 file with raw signal
        bam: BAM file with alignments
        output: Output BAM with predictions
        device: Device for inference
        base_justify: Signal chunk centering
        no_reverse_signal: Disable signal reversal
        anchor: Anchor mode ("basecall" or "reference")
        reference_fasta: Reference FASTA for reference-anchored mode
        motif: Motif to search for
        motif_offset: Offset within motif
        batch_size: Chunks per forward pass
        min_mapq: Minimum mapping quality
        workers: Parallel chunk extraction workers
    """
    from leech.inference import run_bundle_inference, run_inference
    from leech.util import load_model_from_bundle

    validate_predict_args(model, bundle_path, pair, run_all)

    reverse_signal = not no_reverse_signal

    if bundle_path and run_all:
        # Multi-model inference: run all models in bundle
        logger.info(f"Running multi-model inference with bundle: {bundle_path}")
        run_bundle_inference(
            bundle_path=bundle_path,
            pod5_path=pod5,
            bam_path=bam,
            output_path=output,
            device=device,
            min_mapq=min_mapq,
            motif=motif,
            motif_offset=motif_offset,
            base_justify=base_justify,
            reverse_signal=reverse_signal,
            raw=raw,
            aggregation=aggregation,
            anchor=anchor,
            reference_fasta=reference_fasta,
            batch_size=batch_size,
            num_workers=workers,
        )
    else:
        # Single-model inference (auto-detects leech vs Remora)
        if bundle_path and pair:
            loaded_model, config = load_model_from_bundle(bundle_path, pair, device=device)
            logger.info(f"Running inference with pair '{pair}' from bundle: {bundle_path}")
            model_and_config = (loaded_model, config)
            model_path_arg = None
        elif model is not None:
            # Use load_model_auto for auto-detection of leech vs Remora
            model_and_config = None
            model_path_arg = Path(model)
        else:
            raise click.UsageError("Either --model or --bundle is required")

        logger.info(f"Input: {pod5}, {bam}")
        logger.info(f"Output: {output}")

        run_inference(
            model_and_config=model_and_config,
            model_path=model_path_arg,
            pod5_path=pod5,
            bam_path=bam,
            output_path=output,
            device=device,
            min_mapq=min_mapq,
            motif=motif,
            motif_offset=motif_offset,
            batch_size=batch_size,
            base_justify=base_justify,
            reverse_signal=reverse_signal,
            num_workers=workers,
            anchor=anchor,
        )

    console.print("[bold green]Inference complete![/bold green]")
    logger.info(f"Predictions saved to {output}")
