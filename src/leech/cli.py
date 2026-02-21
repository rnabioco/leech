"""
Command-line interface for leech.

Designed for Snakemake integration with clear input/output paths.
"""

import json
import logging
from pathlib import Path

import rich_click as click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from leech.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
)
from leech.logging_config import setup_logging

# Setup logging for CLI
logger = logging.getLogger("leech.cli")
console = Console()

# Configure rich-click for beautiful help display
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.MAX_WIDTH = 100

# Error styling
click.rich_click.STYLE_ERRORS_SUGGESTION = "magenta italic"
click.rich_click.ERRORS_SUGGESTION = "Try running the '--help' flag for more information."
click.rich_click.ERRORS_EPILOGUE = ""

# Color styling for help elements
click.rich_click.STYLE_OPTION = "bold cyan"
click.rich_click.STYLE_ARGUMENT = "bold yellow"
click.rich_click.STYLE_COMMAND = "bold green"
click.rich_click.STYLE_SWITCH = "bold blue"
click.rich_click.STYLE_METAVAR = "bold yellow"
click.rich_click.STYLE_METAVAR_SEPARATOR = "dim"
click.rich_click.STYLE_HEADER_TEXT = "bold magenta"
click.rich_click.STYLE_EPILOG_TEXT = "dim"
click.rich_click.STYLE_FOOTER_TEXT = "dim"
click.rich_click.STYLE_USAGE = "bold yellow"
click.rich_click.STYLE_USAGE_COMMAND = "bold"
click.rich_click.STYLE_DEPRECATED = "red"
click.rich_click.STYLE_HELPTEXT_FIRST_LINE = ""
click.rich_click.STYLE_HELPTEXT = "dim"
click.rich_click.STYLE_OPTION_HELP = ""
click.rich_click.STYLE_OPTION_DEFAULT = "dim italic"
click.rich_click.STYLE_REQUIRED_SHORT = "bold red"
click.rich_click.STYLE_REQUIRED_LONG = "bold red"

# Table and panel styling
click.rich_click.STYLE_OPTIONS_TABLE_LEADING = 1
click.rich_click.STYLE_OPTIONS_TABLE_PAD_EDGE = True
click.rich_click.STYLE_OPTIONS_TABLE_PADDING = (0, 1)
click.rich_click.STYLE_COMMANDS_TABLE_LEADING = 1
click.rich_click.STYLE_COMMANDS_TABLE_PAD_EDGE = True
click.rich_click.STYLE_COMMANDS_TABLE_PADDING = (0, 1)

# Additional features
click.rich_click.SHOW_METAVARS_COLUMN = True
click.rich_click.APPEND_METAVARS_HELP = True
click.rich_click.USE_CLICK_SHORT_HELP = True

# ASCII Logo
LOGO = """
   ___
  (O O)    LEECH
  <VVV>    Learning Enhanced Electrical
   |||     Classifiers from Hanopore signals
   |||
    V
"""

# Model choices for CLI
MODEL_CHOICES = [
    "ConvLSTMDwell",
    "ConvLSTMBase",
    "TransformerDwell",
    "ConvOnly",
    "TCNDwell",
    "ResNetDwell",
]


def display_logo():
    """Display the ASCII logo in a panel."""
    console.print(Panel(LOGO, border_style="cyan", padding=(0, 2)))


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
    default=10,
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
):
    """Prepare training data from POD5 and BAM files."""
    from leech.commands import handle_prepare

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
@click.option(
    "--epochs",
    type=int,
    default=DEFAULT_EPOCHS,
    help="Number of training epochs",
)
@click.option(
    "--batch-size",
    type=int,
    default=DEFAULT_BATCH_SIZE,
    help="Batch size",
)
@click.option(
    "--learning-rate",
    type=float,
    default=DEFAULT_LEARNING_RATE,
    help="Learning rate",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default=DEFAULT_DEVICE,
    help="Device for training",
)
@click.option(
    "--seed",
    type=int,
    default=DEFAULT_SEED,
    help="Random seed for reproducibility",
)
@click.option(
    "--early-stopping",
    type=int,
    default=10,
    help="Patience for early stopping: stop training if validation accuracy doesn't improve for N epochs (set to 0 to disable)",
)
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
    help="Comma-separated context values to test (e.g., '200,500,1000,2000,5000')",
)
@click.option(
    "--left-contexts",
    type=str,
    default=None,
    help="Override left contexts (comma-separated). If not provided, uses --context-grid",
)
@click.option(
    "--right-contexts",
    type=str,
    default=None,
    help="Override right contexts (comma-separated). If not provided, uses --context-grid",
)
@click.option(
    "--kmer-context",
    type=int,
    default=5,
    help="K-mer context for sequence encoding",
)
@click.option(
    "--epochs",
    type=int,
    default=DEFAULT_EPOCHS,
    help="Number of training epochs",
)
@click.option(
    "--batch-size",
    type=int,
    default=DEFAULT_BATCH_SIZE,
    help="Batch size",
)
@click.option(
    "--learning-rate",
    type=float,
    default=DEFAULT_LEARNING_RATE,
    help="Learning rate",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default=DEFAULT_DEVICE,
    help="Device for training",
)
@click.option(
    "--seed",
    type=int,
    default=DEFAULT_SEED,
    help="Random seed for reproducibility",
)
@click.option(
    "--early-stopping",
    type=int,
    default=10,
    help="Patience for early stopping: stop training if validation accuracy doesn't improve for N epochs (set to 0 to disable)",
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
):
    """Optimize model hyperparameters using grid search over chunk contexts."""
    from leech.gridsearch import GridSearchConfig, parse_context_grid, run_grid_search

    # Parse context grids
    left_contexts_list, right_contexts_list = parse_context_grid(
        context_grid, left_contexts, right_contexts
    )

    logger.info(
        f"Starting grid search with {len(left_contexts_list)} x {len(right_contexts_list)} grid points"
    )
    logger.info(f"Left contexts: {left_contexts_list}")
    logger.info(f"Right contexts: {right_contexts_list}")

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


@cli.command()
@click.option(
    "--model",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Trained model file (.pt)",
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
def predict(model, pod5, bam, output, device):
    """Run inference on new data to generate predictions."""
    from leech.inference import run_inference

    logger.info(f"Running inference with model: {model}")
    logger.info(f"Input: {pod5}, {bam}")
    logger.info(f"Output: {output}")

    # Run inference
    run_inference(
        model_path=model,
        pod5_path=pod5,
        bam_path=bam,
        output_path=output,
        device=device,
    )

    console.print("[bold green]Inference complete![/bold green]")
    logger.info(f"Predictions saved to {output}")


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
