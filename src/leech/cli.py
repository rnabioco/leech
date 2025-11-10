"""
Command-line interface for leech.

Designed for Snakemake integration with clear input/output paths.
"""

import json
import logging
import random
from pathlib import Path

import click
import numpy as np
import rich_click as rc
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from leech.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    generate_random_seed,
)
from leech.logging_config import setup_logging

# Setup logging for CLI
logger = logging.getLogger("leech.cli")
console = Console()

# Configure rich-click for beautiful help display
rc.USE_RICH_MARKUP = True
rc.SHOW_ARGUMENTS = True
rc.GROUP_ARGUMENTS_OPTIONS = True
rc.MAX_WIDTH = 100

# Error styling
rc.STYLE_ERRORS_SUGGESTION = "magenta italic"
rc.ERRORS_SUGGESTION = "Try running the '--help' flag for more information."
rc.ERRORS_EPILOGUE = ""

# Color styling for help elements
rc.STYLE_OPTION = "bold cyan"
rc.STYLE_ARGUMENT = "bold yellow"
rc.STYLE_COMMAND = "bold green"
rc.STYLE_SWITCH = "bold blue"
rc.STYLE_METAVAR = "bold yellow"
rc.STYLE_METAVAR_SEPARATOR = "dim"
rc.STYLE_HEADER_TEXT = "bold magenta"
rc.STYLE_EPILOG_TEXT = "dim"
rc.STYLE_FOOTER_TEXT = "dim"
rc.STYLE_USAGE = "bold yellow"
rc.STYLE_USAGE_COMMAND = "bold"
rc.STYLE_DEPRECATED = "red"
rc.STYLE_HELPTEXT_FIRST_LINE = ""
rc.STYLE_HELPTEXT = "dim"
rc.STYLE_OPTION_HELP = ""
rc.STYLE_OPTION_DEFAULT = "dim italic"
rc.STYLE_REQUIRED_SHORT = "bold red"
rc.STYLE_REQUIRED_LONG = "bold red"

# Table and panel styling
rc.STYLE_OPTIONS_TABLE_LEADING = 1
rc.STYLE_OPTIONS_TABLE_PAD_EDGE = True
rc.STYLE_OPTIONS_TABLE_PADDING = (0, 1)
rc.STYLE_COMMANDS_TABLE_LEADING = 1
rc.STYLE_COMMANDS_TABLE_PAD_EDGE = True
rc.STYLE_COMMANDS_TABLE_PADDING = (0, 1)

# Additional features
rc.SHOW_METAVARS_COLUMN = True
rc.APPEND_METAVARS_HELP = True
rc.USE_CLICK_SHORT_HELP = True

# ASCII Logo
LOGO = """
   ___
  (O O)    LEECH
  <VVV>    Learning Enhanced Aminoacylation
   |||     Classification from Hanopore signals
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
    """LEECH - Learning Enhanced Aminoacylation Classification from Hanopore signals

    Enhanced classification from nanopore signals for aa-tRNA-seq experiments.
    """
    setup_logging(level=logging.INFO)


@cli.command()
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
    "--label",
    type=int,
    default=0,
    help="Label for all chunks from this file (0=uncharged, 1=charged)",
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
def prepare(
    pod5,
    bam,
    output_dir,
    motif,
    motif_offset,
    label,
    min_mapq,
    feature_set,
    train_split,
    val_split,
    seed,
):
    """Prepare training data from POD5 and BAM files."""
    from leech.data_prep import extract_training_chunks, iter_bam_with_pod5, save_chunks

    display_logo()

    logger.info(f"Preparing data from {pod5} and {bam}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate random seed if not provided
    if seed is None:
        seed = generate_random_seed()
        logger.info(f"Generated random seed: {seed}")
    else:
        logger.info(f"Using provided seed: {seed}")

    # Save seed for reproducibility
    seed_file = output_dir / "seed.txt"
    with open(seed_file, "w") as f:
        f.write(f"{seed}\n")
    logger.info(f"Saved seed to {seed_file}")

    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Extract chunks with progress bar
    chunks = []
    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Extracting chunks...", total=None)

        for read in iter_bam_with_pod5(bam, pod5, min_mapq=min_mapq):
            read_chunks = extract_training_chunks(
                read, motif=motif, motif_offset=motif_offset, label=label
            )
            chunks.extend(read_chunks)
            progress.update(task, advance=1, description=f"[cyan]Extracted {len(chunks)} chunks...")

        progress.update(task, completed=True)

    console.print(f"[green]Extracted {len(chunks)} training chunks[/green]")

    # Shuffle chunks
    random.shuffle(chunks)

    # Split into train/val/test
    n_total = len(chunks)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_chunks = chunks[:n_train]
    val_chunks = chunks[n_train : n_train + n_val]
    test_chunks = chunks[n_train + n_val :]

    # Display split statistics in a table
    table = Table(title="Data Split", show_header=True, header_style="bold magenta")
    table.add_column("Split", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")

    table.add_row("Train", str(len(train_chunks)), f"{len(train_chunks) / n_total * 100:.1f}%")
    table.add_row("Validation", str(len(val_chunks)), f"{len(val_chunks) / n_total * 100:.1f}%")
    table.add_row("Test", str(len(test_chunks)), f"{len(test_chunks) / n_total * 100:.1f}%")
    table.add_row("Total", str(n_total), "100.0%", style="bold")

    console.print(table)

    # Save chunks
    if train_chunks:
        train_file = output_dir / "train.npz"
        save_chunks(train_chunks, train_file)
        logger.info(f"Saved train to {train_file}")

    if val_chunks:
        val_file = output_dir / "val.npz"
        save_chunks(val_chunks, val_file)
        logger.info(f"Saved val to {val_file}")

    if test_chunks:
        test_file = output_dir / "test.npz"
        save_chunks(test_chunks, test_file)
        logger.info(f"Saved test to {test_file}")

    console.print("[bold green]Data preparation complete![/bold green]")


@cli.command()
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
    help="Stop training if validation loss doesn't improve for N epochs",
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
):
    """Train a model on prepared data."""
    from leech.training import train_model

    display_logo()

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


@cli.command()
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
    """Test a trained model."""
    from leech.evaluation import evaluate_model

    display_logo()

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
def infer(model, pod5, bam, output, device):
    """Run inference on new data."""
    from leech.inference import run_inference

    display_logo()

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


@cli.command(name="grid-search")
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
    help="Stop training if validation loss doesn't improve for N epochs",
)
def grid_search(
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
    """Run grid search over chunk contexts for model optimization."""
    from leech.gridsearch import GridSearchConfig, run_grid_search

    display_logo()

    # Parse context grids
    if left_contexts is not None:
        left_contexts_list = [int(x.strip()) for x in left_contexts.split(",")]
    else:
        left_contexts_list = [int(x.strip()) for x in context_grid.split(",")]

    if right_contexts is not None:
        right_contexts_list = [int(x.strip()) for x in right_contexts.split(",")]
    else:
        right_contexts_list = [int(x.strip()) for x in context_grid.split(",")]

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


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
