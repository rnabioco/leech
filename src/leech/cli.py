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
from rich.progress import Progress
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
    from leech.data_prep import (
        get_reference_sequences,
        prepare_training_data_parallel,
        prepare_training_data_with_split,
        save_chunks,
        split_chunks_by_read,
    )
    from leech.util import setup_random_seed

    # display_logo()

    logger.info(f"Preparing data from {pod5} and {bam}")
    logger.info(f"Motif reference mode: {motif_reference}")
    if workers > 1:
        logger.info(f"Parallel mode: {workers} workers, {chunk_size} reads per batch")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference sequences if using reference-based motif search
    reference_sequences = None
    if motif_reference == "fasta":
        logger.info("Loading reference sequences for reference-based motif search")
        reference_sequences = get_reference_sequences(bam, reference_fasta)

    # Extract chunks (parallel or sequential)
    if workers > 1:
        # Parallel processing - extract chunks, then handle splitting/saving separately
        logger.info("Extracting chunks in parallel...")
        chunks, stats = prepare_training_data_parallel(
            bam_path=bam,
            pod5_path=pod5,
            motif=motif,
            motif_offset=motif_offset,
            label=label,
            label_int=None,  # Will be assigned during merge-and-split
            min_mapq=min_mapq,
            motif_reference=motif_reference,
            reference_sequences=reference_sequences,
            skip_motif_indels=skip_motif_indels,
            num_workers=workers,
            chunk_size=chunk_size,
        )

        # Setup seed and handle splitting/saving
        setup_random_seed(seed, output_dir)

        if no_split:
            # Save all chunks without splitting
            all_file = output_dir / "all.npz"
            save_chunks(chunks, all_file)
            logger.info(f"Saved all chunks to {all_file}")
            result = {
                "n_chunks": len(chunks),
                "n_train": 0,
                "n_val": 0,
                "n_test": 0,
            }
        else:
            # Split at read level
            train_chunks, val_chunks, test_chunks = split_chunks_by_read(
                chunks, train_frac=train_split, val_frac=val_split, seed=seed
            )

            # Save splits
            if train_chunks:
                train_file = output_dir / "train.npz"
                save_chunks(train_chunks, train_file)
                logger.info(f"Saved {len(train_chunks)} train chunks to {train_file}")

            if val_chunks:
                val_file = output_dir / "val.npz"
                save_chunks(val_chunks, val_file)
                logger.info(f"Saved {len(val_chunks)} val chunks to {val_file}")

            if test_chunks:
                test_file = output_dir / "test.npz"
                save_chunks(test_chunks, test_file)
                logger.info(f"Saved {len(test_chunks)} test chunks to {test_file}")

            result = {
                "n_chunks": len(chunks),
                "n_train": len(train_chunks),
                "n_val": len(val_chunks),
                "n_test": len(test_chunks),
            }
    else:
        # Sequential processing with refactored function
        progress_container = {"progress": None, "task": None}

        def update_progress(n_chunks):
            if progress_container["progress"] is not None:
                progress_container["progress"].update(
                    progress_container["task"],
                    advance=1,
                    description=f"[cyan]Extracted {n_chunks} chunks...",
                )

        with Progress(console=console) as progress:
            progress_container["progress"] = progress
            progress_container["task"] = progress.add_task("[cyan]Extracting chunks...", total=None)

            result = prepare_training_data_with_split(
                pod5_path=pod5,
                bam_path=bam,
                output_dir=output_dir,
                motif=motif,
                motif_offset=motif_offset,
                motif_reference=motif_reference,
                reference_fasta=reference_fasta,
                skip_motif_indels=skip_motif_indels,
                label=label,
                label_int=None,  # Will be assigned during merge-and-split
                min_mapq=min_mapq,
                feature_set=feature_set,
                train_split=train_split,
                val_split=val_split,
                seed=seed,
                no_split=no_split,
                progress_callback=update_progress,
            )

            progress.update(progress_container["task"], completed=True)

    # Display results
    console.print(f"[green]Extracted {result['n_chunks']} training chunks[/green]")

    if no_split:
        console.print(
            "[yellow]Skipped splitting (--no-split). All chunks saved to all.npz[/yellow]"
        )
    else:
        # Display split statistics in a table
        table = Table(
            title="Data Split (Read-Level)", show_header=True, header_style="bold magenta"
        )
        table.add_column("Split", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="yellow")

        n_total = result["n_chunks"]
        table.add_row("Train", str(result["n_train"]), f"{result['n_train'] / n_total * 100:.1f}%")
        table.add_row("Validation", str(result["n_val"]), f"{result['n_val'] / n_total * 100:.1f}%")
        table.add_row("Test", str(result["n_test"]), f"{result['n_test'] / n_total * 100:.1f}%")
        table.add_row("Total", str(n_total), "100.0%", style="bold")

        console.print(table)

    console.print("[bold green]Data preparation complete![/bold green]")


@cli.command()
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
def merge_and_split(input_chunks, output_dir, train_split, val_split, seed, comparison_spec):
    """Merge multiple chunk files and split at read level to prevent data leakage.

    This command implements the correct workflow for multi-sample datasets:
    1. Merge all chunks from different samples
    2. Split merged data at the READ level into train/val/test

    This prevents data leakage that can occur when splitting each sample
    independently and then merging the splits.

    Examples:
        # Pairwise comparison with single labels
        leech merge-and-split -i Ala=ala.npz -i Gly=gly.npz -o merged/

        # Multi-label comparison (chemical properties)
        leech merge-and-split -i basic=lys.npz -i basic=arg.npz -i acidic=asp.npz -o merged/

        # Batch processing from TSV spec
        leech merge-and-split -i chunks/dir1 -i chunks/dir2 --comparison-spec spec.tsv -o merged/
    """
    from leech.data_prep import merge_and_split_chunks, process_comparison_spec

    # Batch processing mode with comparison spec
    if comparison_spec:
        logger.info("Processing comparisons from spec file")

        # For batch mode, input_chunks should be directories (no label= prefix)
        chunk_dirs = [Path(c) for c in input_chunks]

        result = process_comparison_spec(
            chunk_dirs=chunk_dirs,
            comparison_spec=comparison_spec,
            output_dir=output_dir,
            train_frac=train_split,
            val_frac=val_split,
            seed=seed,
        )

        # Display summary
        console.print(
            f"\n[bold green]Processed {result['n_comparisons']} comparisons:[/bold green]"
        )
        for comp_name, stats in result["comparisons"].items():
            console.print(
                f"  {comp_name}: {stats['n_train']} train, {stats['n_val']} val, {stats['n_test']} test"
            )

        console.print(f"\n[bold green]All comparisons saved to {output_dir}/[/bold green]")
        return

    # Single comparison mode with label=file syntax
    logger.info("Merging and splitting chunks at read level")

    # Parse label=file format and group files by meta-label
    import numpy as np

    meta_to_files = {}
    meta_to_chunk_labels = {}
    meta_order = []  # Track order for label_int assignment

    for chunk_spec in input_chunks:
        if "=" not in chunk_spec:
            raise ValueError(
                f"Invalid input format: '{chunk_spec}'. "
                "Expected format: label=file.npz (e.g., Ala=ala.npz or basic=lys.npz)"
            )

        meta_label, file_path = chunk_spec.split("=", 1)
        meta_label = meta_label.strip()
        file_path = Path(file_path.strip())

        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        # Track meta-label order (first appearance)
        if meta_label not in meta_to_files:
            meta_to_files[meta_label] = []
            meta_to_chunk_labels[meta_label] = set()
            meta_order.append(meta_label)

        meta_to_files[meta_label].append(file_path)

        # Extract actual chunk labels from file (peek at first chunk)
        with np.load(file_path, allow_pickle=True) as data:
            # Get unique labels from this file
            if "labels" in data:
                chunk_labels = set(data["labels"])
                # Filter out None values
                chunk_labels = {
                    label for label in chunk_labels if label is not None and label != ""
                }
                meta_to_chunk_labels[meta_label].update(chunk_labels)

    # Validate we have exactly 2 groups for binary classification
    if len(meta_to_files) != 2:
        raise ValueError(
            f"Expected exactly 2 unique meta-labels for binary classification, got {len(meta_to_files)}: {list(meta_to_files.keys())}"
        )

    # Build relabel_pairwise tuple: (group1_chunk_labels, group2_chunk_labels)
    # First seen meta-label = 0, second = 1
    meta1, meta2 = meta_order[0], meta_order[1]
    group1_labels = sorted(meta_to_chunk_labels[meta1])
    group2_labels = sorted(meta_to_chunk_labels[meta2])

    # Collect all file paths in order
    all_files = meta_to_files[meta1] + meta_to_files[meta2]

    logger.info(
        f"Relabeling for comparison: {meta1} (labels={group1_labels}) = label_int 0, "
        f"{meta2} (labels={group2_labels}) = label_int 1"
    )

    # Build relabel tuple: use list for multi-label groups, string for single
    relabel_group1 = group1_labels[0] if len(group1_labels) == 1 else group1_labels
    relabel_group2 = group2_labels[0] if len(group2_labels) == 1 else group2_labels
    relabel_tuple = (relabel_group1, relabel_group2)

    # Merge and split at read level
    result = merge_and_split_chunks(
        input_paths=all_files,
        output_dir=output_dir,
        train_frac=train_split,
        val_frac=val_split,
        seed=seed,
        relabel_pairwise=relabel_tuple,
    )

    # Type narrowing: result is always a dict when output_dir is provided
    assert isinstance(result, dict)

    # Display statistics
    table = Table(
        title="Merged Data Split (Read-Level)", show_header=True, header_style="bold magenta"
    )
    table.add_column("Split", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")

    n_total = result["n_total"]
    table.add_row("Train", str(result["n_train"]), f"{result['n_train'] / n_total * 100:.1f}%")
    table.add_row("Validation", str(result["n_val"]), f"{result['n_val'] / n_total * 100:.1f}%")
    table.add_row("Test", str(result["n_test"]), f"{result['n_test'] / n_total * 100:.1f}%")
    table.add_row("Total", str(n_total), "100.0%", style="bold")

    console.print(table)

    console.print("[bold green]Merge and split complete![/bold green]")


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

    # display_logo()

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

    # display_logo()

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
    from leech.gridsearch import GridSearchConfig, parse_context_grid, run_grid_search

    # display_logo()

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


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
