"""
Command-line interface for leech.

Designed for Snakemake integration with clear input/output paths.
"""

import logging
from importlib.metadata import version as pkg_version
from pathlib import Path

import rich_click as click

from leech.cli_config import configure_rich_click, console
from leech.cli_options import MODEL_CHOICES, training_hyperparams
from leech.constants import DEFAULT_DEVICE, DEFAULT_SEED
from leech.logging_config import setup_logging

# Apply rich-click styling
configure_rich_click()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=pkg_version("leech"), prog_name="leech")
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
    help="Offset within motif for focus base (0-indexed)",
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
    help="Extra bases on each side of dwell/feature arrays (symmetric fallback, default: 0). Overridden by --dwell-margin-left/--dwell-margin-right if provided.",
)
@click.option(
    "--dwell-margin-left",
    type=int,
    default=None,
    help="Extra dwell bases toward tRNA body (left of focus). Keep small to avoid isoacceptor confounds.",
)
@click.option(
    "--dwell-margin-right",
    type=int,
    default=None,
    help="Extra dwell bases toward 3' adaptor (right of focus). Safe to be generous (constant sequence).",
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
    dwell_margin_left,
    dwell_margin_right,
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
        from leech.data import get_kmer_table

        kmer_table = get_kmer_table()
        if not kmer_table.exists():
            raise click.UsageError(
                "--refine-signal-map requires --kmer-table (bundled table not found)"
            )

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
        dwell_margin_left=dwell_margin_left,
        dwell_margin_right=dwell_margin_right,
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
    "--k-fold",
    type=int,
    default=1,
    help="Number of cross-validation folds. When > 1, creates k-fold splits instead of a single train/val/test split. Must be >= 3 when used.",
)
@click.option(
    "--comparison-spec",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="TSV file with comparison specifications (4 columns: meta_label1, label_set1, meta_label2, label_set2). When provided, input-chunks should be directories.",
)
def merge(input_chunks, output_dir, train_split, val_split, seed, k_fold, comparison_spec):
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

        # 5-fold cross-validation
        leech data merge -i Ala=ala.npz -i Gly=gly.npz -o kfold/ --k-fold 5
    """
    if k_fold > 1:
        from leech.commands import handle_merge_and_split_kfold

        handle_merge_and_split_kfold(
            input_chunks=input_chunks,
            output_dir=output_dir,
            k_fold=k_fold,
            seed=seed,
        )
    else:
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
    type=click.Path(path_type=Path),
    default=None,
    help="Resume training from a checkpoint file (e.g., model_last.pt). Ignored if file doesn't exist.",
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
    help="Offset within motif for focus base (0-indexed, recorded in config)",
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
    default="signal_kmer",
    help="Sequence encoding type: signal_kmer (36, signal_len) or base_onehot (4, kmer_len)",
)
@click.option(
    "--num-workers",
    type=int,
    default=0,
    help="DataLoader workers (0=auto: 8 for GPU, 0 for CPU)",
)
@click.option(
    "--balance-groups/--no-balance-groups",
    default=False,
    help="Balance sampling across source groups (e.g., per-AA) so each group contributes equally per epoch",
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
    balance_groups,
):
    """Train a model on prepared data."""
    from leech.commands.train import handle_train

    handle_train(
        train_data=train_data,
        val_data=val_data,
        model_name=model,
        model_config=model_config,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
        early_stopping=early_stopping,
        use_class_weights=use_class_weights,
        pos_weight=pos_weight,
        resume=resume,
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
        motif=motif,
        motif_offset=motif_offset,
        base_justify=base_justify,
        seq_encoding=seq_encoding,
        balance_groups=balance_groups,
    )


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
    type=click.Choice(["pairwise", "one_vs_all", "group"]),
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
    from leech.commands.bundle import handle_bundle

    handle_bundle(
        model_dir=model_dir,
        output=output,
        bundle_version=bundle_version,
        comparison_type=comparison_type,
        torchscript=torchscript,
    )


@model.command(name="bundle-info")
@click.option(
    "--bundle",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to bundle .pt file",
)
def bundle_info(bundle):
    """Display metadata from a model bundle."""
    from leech.commands.bundle import handle_bundle_info

    handle_bundle_info(bundle)


@model.command()
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Model directory (with config.json and model_best.pt), or parent with pair subdirs",
)
@click.option(
    "--val-data",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Validation data (.npz) used to learn Platt scaling parameters",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help="Device for inference (default: cpu)",
)
@click.option(
    "--batch-size",
    type=int,
    default=1024,
    help="Batch size for validation pass (default: 1024)",
)
@click.option(
    "--num-workers",
    type=int,
    default=0,
    help="DataLoader workers (default: 0)",
)
def calibrate(model_dir, val_data, device, batch_size, num_workers):
    """Learn post-hoc Platt scaling on the validation set.

    Fits two parameters (a, b) per model so that sigmoid(a*logit + b) is
    better calibrated. This handles both confidence scaling (a) and decision
    threshold shift (b) — critical when class imbalance biases the boundary.

    Writes platt.json to the model directory.

    For a parent directory with pair subdirs (e.g., one_vs_all/Ala_notAla/),
    calibrates each pair independently.
    """
    from leech.commands.calibrate import handle_calibrate

    handle_calibrate(
        model_dir=model_dir,
        val_data=val_data,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )


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
    from leech.commands.bundle import handle_export

    handle_export(model_dir=model_dir, output=output)


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
    required=False,
    default=None,
    help="Fallback context values when --left-contexts or --right-contexts are not provided. Comma-separated (e.g., '200,500,1000') or range as start:stop:step (e.g., '200:1000:200')",
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
@click.option(
    "--balance-groups/--no-balance-groups",
    default=False,
    help="Balance sampling across source groups (e.g., per-AA) so each group contributes equally per epoch",
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
    balance_groups,
):
    """Optimize model hyperparameters using grid search over chunk contexts."""
    from leech.commands.optimize import handle_optimize

    handle_optimize(
        train_data=train_data,
        val_data=val_data,
        model_name=model,
        output_dir=output_dir,
        context_grid=context_grid,
        left_contexts=left_contexts,
        right_contexts=right_contexts,
        kmer_context=kmer_context,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
        early_stopping=early_stopping,
        base_justify=base_justify,
        dwell_offsets=dwell_offsets,
        parallel=parallel,
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
        balance_groups=balance_groups,
    )


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
    from leech.commands.eval import handle_test

    handle_test(model=model, test_data=test_data, output=output, device=device)


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
    from leech.commands.predict import handle_predict

    handle_predict(
        model=model,
        bundle_path=bundle_path,
        pair=pair,
        run_all=run_all,
        raw=raw,
        pod5=pod5,
        bam=bam,
        output=output,
        device=device,
        base_justify=base_justify,
        no_reverse_signal=no_reverse_signal,
        reference_anchored=reference_anchored,
        motif=motif,
        motif_offset=motif_offset,
        batch_size=batch_size,
        min_mapq=min_mapq,
        workers=workers,
    )


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
