"""
Command-line interface for leech.

Designed for Snakemake integration with clear input/output paths.
"""

import json
import logging
from pathlib import Path

import rich_click as click
from rich.table import Table

from leech.cli_config import configure_rich_click, console
from leech.cli_options import MODEL_CHOICES, training_hyperparams
from leech.constants import DEFAULT_DEVICE, DEFAULT_SEED
from leech.logging_config import setup_logging

# Setup logging for CLI
logger = logging.getLogger("leech.cli")

# Apply rich-click styling
configure_rich_click()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version="0.1.0", prog_name="leech")
def cli():
    """LEECH - Learning Enhanced Electrical Classifiers from Hanopore signals

    A workflow-based CLI for nanopore signal classification:

    • leech data     - Prepare and process training data
    • leech model    - Train and optimize models
    • leech eval     - Evaluate and analyze models
    • leech predict  - Run inference on new data
    """
    setup_logging(level=logging.INFO)


# ============================================================================
# DATA PREPARATION COMMANDS
# ============================================================================


@cli.group()
def data():
    """Prepare and process training data from POD5/BAM files."""
    pass


@data.command()
@click.option(
    "--pod5",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="POD5 file with raw signal",
)
@click.option(
    "--bam",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="BAM file with alignments and mv tags",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for training chunks",
)
@click.option(
    "--motif",
    type=str,
    default=None,
    help='Sequence motif to extract (e.g., "CCAGGC" for tRNA 3\' end)',
)
@click.option(
    "--motif-offset",
    type=int,
    default=0,
    help="Offset within motif for focus base",
)
@click.option(
    "--motif-reference",
    type=click.Choice(["fasta", "bam"]),
    default="fasta",
    help='Where to search for motif: "fasta" (reference sequence, recommended) or "bam" (basecalled sequence)',
)
@click.option(
    "--reference-fasta",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="External reference FASTA file (if not embedded in BAM @SQ header)",
)
@click.option(
    "--skip-motif-indels",
    is_flag=True,
    default=True,
    help="Skip reads with indels in motif region (for motif-reference=fasta)",
)
@click.option(
    "--label",
    type=str,
    default=None,
    help="Label identifier for this sample (e.g., 'Ala', 'Gly', 'charged', 'uncharged'). Numeric labels (0/1) are assigned during merge-and-split for pairwise comparisons.",
)
@click.option(
    "--min-mapq",
    type=int,
    default=0,
    help="Minimum mapping quality",
)
@click.option(
    "--feature-set",
    type=click.Choice(["signal", "signal+dwell", "signal+levels", "signal+dwell+levels"]),
    default="signal+dwell+levels",
    help="Feature set to extract",
)
@click.option(
    "--train-split",
    type=float,
    default=0.7,
    help="Fraction of data for training",
)
@click.option(
    "--val-split",
    type=float,
    default=0.15,
    help="Fraction of data for validation",
)
@click.option(
    "--seed",
    type=int,
    default=DEFAULT_SEED,
    help="Random seed for reproducibility",
)
@click.option(
    "--no-split",
    is_flag=True,
    default=False,
    help="Extract chunks without splitting (for later merge-then-split workflow)",
)
@click.option(
    "--workers",
    type=int,
    default=8,
    help="Number of parallel workers for data processing (1=sequential, >1=parallel)",
)
@click.option(
    "--chunk-size",
    type=int,
    default=100,
    help="Number of reads to process per worker batch (for parallel processing)",
)
@click.option(
    "--base-justify",
    type=click.Choice(["start", "center", "end"]),
    default="center",
    help='Where to center signal chunk within the focus base: "start" (first sample), "center" (midpoint, default), or "end" (last sample, useful for 3\' modifications)',
)
@click.option(
    "--dwell-margin",
    type=int,
    default=0,
    help="Extra bases on each side of dwell/feature arrays for runtime dwell_offset tuning (default: 0, use 15 for grid search over dwell offsets)",
)
@click.option(
    "--no-reverse-signal",
    is_flag=True,
    default=False,
    help="Do NOT reverse the raw signal. By default, signal is reversed for direct RNA sequencing (POD5 stores 3'→5', basecaller expects 5'→3'). Use this flag for DNA data.",
)
@click.option(
    "--anchor",
    type=click.Choice(["basecall", "reference"]),
    default="basecall",
    help='Anchor mode: "basecall" uses basecalled sequence, "reference" uses ref sequence + ref->signal mapping via CIGAR and trims signal to aligned region',
)
@click.option(
    "--signal-norm",
    type=click.Choice(["median_mad", "zscore", "quantile", "pa_scaling"]),
    default="median_mad",
    help="Signal normalization method",
)
@click.option(
    "--pa-mean",
    type=float,
    default=None,
    help="Global shift for pa_scaling normalization (from basecaller model)",
)
@click.option(
    "--pa-stdev",
    type=float,
    default=None,
    help="Global scale for pa_scaling normalization (from basecaller model)",
)
@click.option(
    "--refine-signal-map",
    is_flag=True,
    default=False,
    help="Apply signal map refinement using kmer level tables (improves base boundary accuracy)",
)
@click.option(
    "--kmer-table",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to kmer level table for signal map refinement (e.g., rna004_9mer_levels.txt)",
)
def prepare(
    pod5,
    bam,
    output_dir,
    motif,
    motif_offset,
    motif_reference,
    reference_fasta,
    skip_motif_indels,
    label,
    min_mapq,
    feature_set,
    train_split,
    val_split,
    seed,
    no_split,
    workers,
    chunk_size,
    base_justify,
    dwell_margin,
    no_reverse_signal,
    anchor,
    signal_norm,
    pa_mean,
    pa_stdev,
    refine_signal_map,
    kmer_table,
):
    """Prepare training data from POD5 and BAM files."""
    from leech.commands import handle_prepare

    # Validate pa_scaling params
    if signal_norm == "pa_scaling" and (pa_mean is None or pa_stdev is None):
        raise click.UsageError("--signal-norm pa_scaling requires --pa-mean and --pa-stdev")
    if refine_signal_map and kmer_table is None:
        raise click.UsageError("--refine-signal-map requires --kmer-table")

    # display_logo()

    handle_prepare(
        pod5=pod5,
        bam=bam,
        output_dir=output_dir,
        motif=motif,
        motif_offset=motif_offset,
        motif_reference=motif_reference,
        reference_fasta=reference_fasta,
        skip_motif_indels=skip_motif_indels,
        label=label,
        min_mapq=min_mapq,
        feature_set=feature_set,
        train_split=train_split,
        val_split=val_split,
        seed=seed,
        no_split=no_split,
        workers=workers,
        chunk_size=chunk_size,
        base_justify=base_justify,
        dwell_margin=dwell_margin,
        reverse_signal=not no_reverse_signal,
        anchor=anchor,
        signal_norm=signal_norm,
        pa_mean=pa_mean,
        pa_stdev=pa_stdev,
        refine_signal_map=refine_signal_map,
        kmer_table=kmer_table,
    )


# ============================================================================
# MODEL TRAINING COMMANDS
# ============================================================================


@cli.group()
def model():
    """Train and optimize models."""
    pass


@data.command()
@click.option(
    "--input-chunks",
    "-i",
    required=True,
    multiple=True,
    type=str,
    help="Input chunk files with labels. Format: label=file.npz (e.g., -i Ala=ala.npz -i Gly=gly.npz or -i basic=lys.npz -i basic=arg.npz -i acidic=asp.npz)",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for split chunks",
)
@click.option(
    "--train-split",
    type=float,
    default=0.7,
    help="Fraction of reads for training",
)
@click.option(
    "--val-split",
    type=float,
    default=0.15,
    help="Fraction of reads for validation",
)
@click.option(
    "--seed",
    type=int,
    default=DEFAULT_SEED,
    help="Random seed for reproducibility",
)
@click.option(
    "--comparison-spec",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="TSV file with comparison specifications (4 columns: meta_label1, label_set1, meta_label2, label_set2). When provided, input-chunks should be directories.",
)
def merge(input_chunks, output_dir, train_split, val_split, seed, comparison_spec):
    """Merge multiple chunk files and split at read level to prevent data leakage.

    This command implements the correct workflow for multi-sample datasets:
    1. Merge all chunks from different samples
    2. Split merged data at the READ level into train/val/test

    This prevents data leakage that can occur when splitting each sample
    independently and then merging the splits.

    Examples:
        # Pairwise comparison with single labels
        leech data merge -i Ala=ala.npz -i Gly=gly.npz -o merged/

        # Multi-label comparison (chemical properties)
        leech data merge -i basic=lys.npz -i basic=arg.npz -i acidic=asp.npz -o merged/

        # Batch processing from TSV spec
        leech data merge -i chunks/dir1 -i chunks/dir2 --comparison-spec spec.tsv -o merged/
    """
    from leech.commands import handle_merge_and_split

    handle_merge_and_split(
        input_chunks=input_chunks,
        output_dir=output_dir,
        train_split=train_split,
        val_split=val_split,
        seed=seed,
        comparison_spec=comparison_spec,
    )


@model.command()
@click.option(
    "--train-data",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Training dataset config (JSON)",
)
@click.option(
    "--val-data",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Validation dataset config (JSON)",
)
@click.option(
    "--model",
    type=click.Choice(MODEL_CHOICES),
    default="ConvLSTMDwell",
    help="Model architecture",
)
@click.option(
    "--model-config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Model hyperparameters (JSON)",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for model and logs",
)
@training_hyperparams
@click.option(
    "--use-class-weights/--no-class-weights",
    default=True,
    help="Auto-compute class weights from training data to handle class imbalance (default: enabled)",
)
@click.option(
    "--pos-weight",
    type=float,
    default=None,
    help="Manual positive class weight for BCEWithLogitsLoss (overrides --use-class-weights)",
)
@click.option(
    "--resume",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Resume training from a checkpoint file (e.g., model_last.pt)",
)
@click.option(
    "--motif",
    type=str,
    default=None,
    help="Motif used for chunk extraction (recorded in config for provenance/inference)",
)
@click.option(
    "--motif-offset",
    type=int,
    default=0,
    help="Offset within motif for focus base (recorded in config)",
)
@click.option(
    "--base-justify",
    type=click.Choice(["start", "center", "end"]),
    default="center",
    help="Signal justification within focus base (recorded in config)",
)
@click.option(
    "--seq-encoding",
    type=click.Choice(["base_onehot", "signal_kmer"]),
    default="base_onehot",
    help="Sequence encoding type: base_onehot (4, kmer_len) or signal_kmer (36, signal_len)",
)
@click.option(
    "--num-workers",
    type=int,
    default=0,
    help="DataLoader workers (0=auto: 8 for GPU, 0 for CPU)",
)
def train(
    train_data,
    val_data,
    model,
    model_config,
    output_dir,
    epochs,
    batch_size,
    learning_rate,
    device,
    seed,
    early_stopping,
    use_class_weights,
    pos_weight,
    resume,
    weight_decay,
    max_grad_norm,
    scheduler,
    scheduler_patience,
    scheduler_factor,
    warmup_epochs,
    loss_type,
    focal_gamma,
    mixed_precision,
    augment_jitter,
    augment_scale_min,
    augment_scale_max,
    motif,
    motif_offset,
    base_justify,
    seq_encoding,
    num_workers,
):
    """Train a model on prepared data."""
    from leech.training import train_model

    # display_logo()

    logger.info(f"Training {model} model")
    logger.info(f"Train data: {train_data}")
    logger.info(f"Output: {output_dir}")

    # Load model config if provided
    model_kwargs = {}
    if model_config is not None:
        with open(model_config) as f:
            model_kwargs = json.load(f)

    # Train model
    history = train_model(
        train_data_path=train_data,
        val_data_path=val_data,
        model_name=model,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
        early_stopping_patience=early_stopping,
        use_class_weights=use_class_weights,
        pos_weight=pos_weight,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        scheduler=scheduler,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        warmup_epochs=warmup_epochs,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        mixed_precision=mixed_precision,
        augment_jitter=augment_jitter,
        augment_scale_min=augment_scale_min,
        augment_scale_max=augment_scale_max,
        resume_from=resume,
        num_workers=num_workers,
        motif=motif,
        motif_offset=motif_offset,
        base_justify=base_justify,
        seq_encoding=seq_encoding,
        **model_kwargs,
    )

    console.print("[bold green]Training complete![/bold green]")

    # Display final metrics in a table
    table = Table(title="Training Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    if "val_acc" in history and history["val_acc"]:
        table.add_row("Best Validation Accuracy", f"{history['val_acc'][-1]:.4f}")
    if "val_loss" in history and history["val_loss"]:
        table.add_row("Final Validation Loss", f"{history['val_loss'][-1]:.4f}")

    table.add_row("Model saved to", str(output_dir))

    console.print(table)


@model.command()
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Root dir containing pair subdirectories (each with model_best.pt + config.json)",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output bundle .pt file path",
)
@click.option(
    "--version",
    "-v",
    "bundle_version",
    required=True,
    type=str,
    help='Semantic version string (e.g., "0.1.0-alpha.1")',
)
@click.option(
    "--comparison-type",
    type=click.Choice(["pairwise", "one_vs_all"]),
    default="pairwise",
    help="Comparison type (default: pairwise)",
)
@click.option(
    "--torchscript/--no-torchscript",
    default=False,
    help="Bundle as TorchScript (standalone, no leech needed to load). Default: False.",
)
def bundle(model_dir, output, bundle_version, comparison_type, torchscript):
    """Bundle trained models into a single versioned file."""
    from leech.util import create_bundle, create_torchscript_bundle

    # Auto-discover pair subdirectories
    model_dirs = {}
    for subdir in sorted(model_dir.iterdir()):
        if (
            subdir.is_dir()
            and (subdir / "model_best.pt").exists()
            and (subdir / "config.json").exists()
        ):
            model_dirs[subdir.name] = subdir

    if not model_dirs:
        console.print(f"[bold red]No model directories found in {model_dir}[/bold red]")
        raise SystemExit(1)

    logger.info(f"Found {len(model_dirs)} model directories")

    if torchscript:
        bundle_path = create_torchscript_bundle(
            model_dirs=model_dirs,
            output_path=output,
            comparison_type=comparison_type,
            version=bundle_version,
        )
    else:
        bundle_path = create_bundle(
            model_dirs=model_dirs,
            output_path=output,
            comparison_type=comparison_type,
            version=bundle_version,
        )

    # Print summary table
    table = Table(title="Bundle Summary", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Version", bundle_version)
    table.add_row("Format", "TorchScript" if torchscript else "state_dict")
    table.add_row("Comparison type", comparison_type)
    table.add_row("Models", str(len(model_dirs)))
    table.add_row("Output", str(bundle_path))
    size_mb = bundle_path.stat().st_size / (1024 * 1024)
    table.add_row("File size", f"{size_mb:.1f} MB")

    console.print(table)
    console.print("[bold green]Bundle created![/bold green]")


@model.command(name="bundle-info")
@click.option(
    "--bundle",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to bundle .pt file",
)
def bundle_info(bundle):
    """Display metadata from a model bundle."""
    from leech.util import list_bundle_models

    metadata = list_bundle_models(bundle)

    table = Table(title="Bundle Info", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Bundle version", metadata.get("bundle_version", "unknown"))
    table.add_row("Format version", str(metadata.get("format_version", "unknown")))
    is_ts = metadata.get("torchscript", False)
    table.add_row("Format", "TorchScript" if is_ts else "state_dict")
    table.add_row("Architecture", metadata.get("architecture", "unknown"))
    table.add_row("Comparison type", metadata.get("comparison_type", "unknown"))
    table.add_row("Number of models", str(metadata.get("num_models", 0)))
    table.add_row("Created at", metadata.get("created_at", "unknown"))
    size_mb = Path(bundle).stat().st_size / (1024 * 1024)
    table.add_row("File size", f"{size_mb:.1f} MB")

    console.print(table)

    # Print pairs
    pairs = metadata.get("pairs", [])
    if pairs:
        pairs_table = Table(
            title=f"Models ({len(pairs)})", show_header=True, header_style="bold magenta"
        )
        pairs_table.add_column("#", style="dim", width=4)
        pairs_table.add_column("Pair", style="cyan")
        for i, pair in enumerate(pairs, 1):
            pairs_table.add_row(str(i), pair)
        console.print(pairs_table)


@model.command()
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Model checkpoint directory (with config.json and model_best.pt)",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output TorchScript .pt file path",
)
def export(model_dir, output):
    """Export a trained model as a standalone TorchScript file.

    The exported file is loadable with just torch.jit.load() — no leech
    codebase required. Model config is embedded in the file.
    """
    from leech.util import export_single_model

    output_path = export_single_model(model_dir, output)
    size_mb = output_path.stat().st_size / (1024 * 1024)

    table = Table(title="Export Summary", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Source", str(model_dir))
    table.add_row("Output", str(output_path))
    table.add_row("File size", f"{size_mb:.1f} MB")

    console.print(table)
    console.print("[bold green]TorchScript export complete![/bold green]")


@model.command()
@click.option(
    "--train-data",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Training dataset (.npz)",
)
@click.option(
    "--val-data",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Validation dataset (.npz)",
)
@click.option(
    "--model",
    type=click.Choice(MODEL_CHOICES),
    default="ConvLSTMDwell",
    help="Model architecture",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for grid results",
)
@click.option(
    "--context-grid",
    type=str,
    required=True,
    help="Context values to test. Comma-separated (e.g., '200,500,1000') or range as start:stop:step (e.g., '200:1000:200')",
)
@click.option(
    "--left-contexts",
    type=str,
    default=None,
    help="Override left contexts (comma-separated or start:stop:step). If not provided, uses --context-grid",
)
@click.option(
    "--right-contexts",
    type=str,
    default=None,
    help="Override right contexts (comma-separated or start:stop:step). If not provided, uses --context-grid",
)
@click.option(
    "--kmer-context",
    type=int,
    default=5,
    help="K-mer context for sequence encoding",
)
@training_hyperparams
@click.option(
    "--base-justify",
    type=click.Choice(["start", "center", "end"]),
    default="center",
    help='Where to center signal chunk within the focus base: "start" (first sample), "center" (midpoint, default), or "end" (last sample, useful for 3\' modifications)',
)
@click.option(
    "--dwell-offsets",
    type=str,
    default="0",
    help='Dwell offset values to search (comma-separated or start:stop:step). Shifts dwell/feature window toward 3\' end. Default: "0" (no offset). Requires chunks prepared with --dwell-margin >= max offset.',
)
@click.option(
    "--parallel",
    type=int,
    default=1,
    help="Number of grid points to run concurrently (default: 1, sequential). Each worker loads data independently.",
)
@click.option(
    "--num-workers",
    type=int,
    default=0,
    help="DataLoader workers (0=auto: 8 for GPU, 0 for CPU)",
)
def optimize(
    train_data,
    val_data,
    model,
    output_dir,
    context_grid,
    left_contexts,
    right_contexts,
    kmer_context,
    epochs,
    batch_size,
    learning_rate,
    device,
    seed,
    early_stopping,
    base_justify,
    dwell_offsets,
    parallel,
    weight_decay,
    max_grad_norm,
    scheduler,
    scheduler_patience,
    scheduler_factor,
    warmup_epochs,
    loss_type,
    focal_gamma,
    mixed_precision,
    augment_jitter,
    augment_scale_min,
    augment_scale_max,
    num_workers,
):
    """Optimize model hyperparameters using grid search over chunk contexts."""
    from leech.gridsearch import GridSearchConfig, parse_context_grid, parse_values, run_grid_search

    # Parse context grids
    left_contexts_list, right_contexts_list = parse_context_grid(
        context_grid, left_contexts, right_contexts
    )

    # Parse dwell offsets
    dwell_offsets_list = parse_values(dwell_offsets)

    logger.info(
        f"Starting grid search with {len(left_contexts_list)} x {len(right_contexts_list)} x {len(dwell_offsets_list)} grid points"
    )
    logger.info(f"Left contexts: {left_contexts_list}")
    logger.info(f"Right contexts: {right_contexts_list}")
    logger.info(f"Dwell offsets: {dwell_offsets_list}")

    # Create config
    config = GridSearchConfig(
        train_data_path=train_data,
        val_data_path=val_data,
        model_name=model,
        output_dir=output_dir,
        left_contexts=left_contexts_list,
        right_contexts=right_contexts_list,
        kmer_context=kmer_context,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
        early_stopping_patience=early_stopping,
        base_justify=base_justify,
        dwell_offsets=dwell_offsets_list,
        n_parallel=parallel,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        scheduler=scheduler,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        warmup_epochs=warmup_epochs,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        mixed_precision=mixed_precision,
        augment_jitter=augment_jitter,
        augment_scale_min=augment_scale_min,
        augment_scale_max=augment_scale_max,
        num_workers=num_workers,
    )

    # Run grid search
    summary_path = run_grid_search(config)

    console.print("[bold green]Grid search complete![/bold green]")
    logger.info(f"Results saved to: {summary_path}")


# ============================================================================
# MODEL EVALUATION COMMANDS
# ============================================================================


@cli.group()
def eval():
    """Evaluate and analyze trained models."""
    pass


@eval.command()
@click.option(
    "--model",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Trained model file (.pt)",
)
@click.option(
    "--test-data",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Test dataset config (JSON)",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output metrics file (JSON)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default=DEFAULT_DEVICE,
    help="Device for inference",
)
def test(model, test_data, output, device):
    """Test a trained model on a holdout test set."""
    from leech.evaluation import evaluate_model

    logger.info(f"Testing model: {model}")
    logger.info(f"Test data: {test_data}")
    logger.info(f"Output: {output}")

    # Run evaluation
    evaluate_model(
        model_path=model,
        test_data_path=test_data,
        output_path=output,
        device=device,
    )

    console.print("[bold green]Testing complete![/bold green]")
    logger.info(f"Results saved to {output}")


@eval.command()
@click.option(
    "--model-dirs",
    "-m",
    required=True,
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Model directories to compare (can specify multiple)",
)
@click.option(
    "--test-data",
    "-t",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Test dataset for evaluation",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for comparison results",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default=DEFAULT_DEVICE,
    help="Device for evaluation",
)
@click.option(
    "--no-plot",
    is_flag=True,
    help="Skip generating plots",
)
def compare(model_dirs, test_data, output_dir, device, no_plot):
    """Compare multiple trained models on the same test set."""
    from leech.commands.analyze import handle_compare

    handle_compare(
        model_dirs=list(model_dirs),
        test_data=test_data,
        output_dir=output_dir,
        device=device,
        plot=not no_plot,
    )

    console.print("[bold green]Model comparison complete![/bold green]")


@eval.command()
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to trained model checkpoint",
)
@click.option(
    "--test-data",
    "-t",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Test dataset for analysis",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for results",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default=DEFAULT_DEVICE,
    help="Device for computation",
)
@click.option(
    "--method",
    type=click.Choice(["gradient", "integrated_gradients"]),
    default="gradient",
    help="Feature importance method",
)
@click.option(
    "--no-plot",
    is_flag=True,
    help="Skip generating plots",
)
def importance(model, test_data, output_dir, device, method, no_plot):
    """Compute feature importance scores for a trained model."""
    from leech.commands.analyze import handle_feature_importance

    handle_feature_importance(
        model_path=model,
        test_data=test_data,
        output_dir=output_dir,
        device=device,
        method=method,
        plot=not no_plot,
    )

    console.print("[bold green]Feature importance analysis complete![/bold green]")


@eval.command()
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to trained model checkpoint",
)
@click.option(
    "--test-data",
    "-t",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Test dataset for analysis",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for results",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default=DEFAULT_DEVICE,
    help="Device for computation",
)
@click.option(
    "--no-plot",
    is_flag=True,
    help="Skip generating plots",
)
def ablation(model, test_data, output_dir, device, no_plot):
    """Test model performance with sequence ablation."""
    from leech.commands.analyze import handle_sequence_ablation

    handle_sequence_ablation(
        model_path=model,
        test_data=test_data,
        output_dir=output_dir,
        device=device,
        plot=not no_plot,
    )

    console.print("[bold green]Sequence ablation test complete![/bold green]")


# ============================================================================
# INFERENCE COMMAND
# ============================================================================


def _validate_predict_args(model, bundle_path, pair, run_all):
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


@cli.command()
@click.option(
    "--model",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Trained model checkpoint directory",
)
@click.option(
    "--bundle",
    "bundle_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Model bundle .pt file (mutually exclusive with --model)",
)
@click.option(
    "--pair",
    type=str,
    default=None,
    help="Run a single model from the bundle (requires --bundle)",
)
@click.option(
    "--all",
    "run_all",
    is_flag=True,
    default=False,
    help="Run every model in the bundle on each read (requires --bundle)",
)
@click.option(
    "--raw",
    is_flag=True,
    default=False,
    help="With --all, additionally write per-pair probabilities (pn/pp tags)",
)
@click.option(
    "--pod5",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="POD5 file with raw signal",
)
@click.option(
    "--bam",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="BAM file with alignments",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output BAM with predictions",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default=DEFAULT_DEVICE,
    help="Device for inference",
)
@click.option(
    "--base-justify",
    type=click.Choice(["start", "center", "end"]),
    default="center",
    help='Where to center signal chunk within the focus base: "start" (first sample), "center" (midpoint, default), or "end" (last sample, useful for 3\' modifications)',
)
@click.option(
    "--no-reverse-signal",
    is_flag=True,
    default=False,
    help="Do NOT reverse the raw signal. By default, signal is reversed for direct RNA sequencing (POD5 stores 3'→5', basecaller expects 5'→3'). Use this flag for DNA data.",
)
@click.option(
    "--reference-anchored",
    is_flag=True,
    default=False,
    help="Use reference-anchored mode: search motif in reference sequence and use ref->signal mapping via CIGAR. Matches Remora's --reference-anchored behavior.",
)
@click.option(
    "--motif",
    type=str,
    default=None,
    help="Motif to search for in reads (auto-read from model config if not provided; required for Remora models)",
)
@click.option(
    "--motif-offset",
    type=int,
    default=0,
    help="Offset within motif for the prediction position (0-indexed)",
)
@click.option(
    "--batch-size",
    type=int,
    default=256,
    help="Chunks per forward pass (default: 256)",
)
@click.option(
    "--min-mapq",
    type=int,
    default=0,
    help="Minimum mapping quality (default: 0)",
)
@click.option(
    "--workers",
    type=int,
    default=0,
    help="Parallel chunk extraction workers (0=sequential). Only useful with GPU inference; CPU-only is faster with 0.",
)
def predict(
    model,
    bundle_path,
    pair,
    run_all,
    raw,
    pod5,
    bam,
    output,
    device,
    base_justify,
    no_reverse_signal,
    reference_anchored,
    motif,
    motif_offset,
    batch_size,
    min_mapq,
    workers,
):
    """Run inference on new data to generate predictions."""
    from leech.inference import run_bundle_inference, run_inference
    from leech.util import load_model_from_bundle

    _validate_predict_args(model, bundle_path, pair, run_all)

    reverse_signal = not no_reverse_signal
    anchor = "reference" if reference_anchored else "basecall"

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


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
