"""
Grid search functionality for chunk context optimization.

Implements systematic grid search over signal window sizes (left/right context)
to find optimal model performance, as described in the leech training strategy.
"""

import csv
import itertools
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from leech.chunking import extract_training_chunks, save_chunks
from leech.preparation import iter_bam_with_pod5
from leech.training import train_model

logger = logging.getLogger("leech.gridsearch")
console = Console()


def parse_values(spec: str) -> list[int]:
    """Parse a value specification into a list of integers.

    Supports three formats:
    - Range: ``start:stop:step`` — expands to ``range(start, stop + 1, step)`` (inclusive stop)
    - Comma-separated: ``200,500,1000``
    - Single value: ``500``

    Args:
        spec: Value specification string.

    Returns:
        List of integer values.

    Raises:
        ValueError: If the specification is invalid.

    Examples:
        >>> parse_values("200:1000:200")
        [200, 400, 600, 800, 1000]
        >>> parse_values("200,500,1000")
        [200, 500, 1000]
        >>> parse_values("500")
        [500]
    """
    spec = spec.strip()

    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Range spec must be start:stop:step, got {len(parts)} parts: '{spec}'"
            )
        try:
            start, stop, step = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            raise ValueError(f"Range spec contains non-integer values: '{spec}'")
        if step <= 0:
            raise ValueError(f"Step must be positive, got {step}")
        if start > stop:
            raise ValueError(
                f"Start ({start}) must be <= stop ({stop})"
            )
        return list(range(start, stop + 1, step))

    if "," in spec:
        return [int(x.strip()) for x in spec.split(",")]

    return [int(spec)]


def parse_context_grid(
    context_grid: str,
    left_contexts: str | None = None,
    right_contexts: str | None = None,
) -> tuple[list[int], list[int]]:
    """Parse context grid strings into integer lists.

    Supports range syntax (``start:stop:step``), comma-separated lists,
    and single values.

    Args:
        context_grid: Context values (e.g., "200,500,1000" or "200:1000:200")
        left_contexts: Override left contexts, or None to use context_grid
        right_contexts: Override right contexts, or None to use context_grid

    Returns:
        Tuple of (left_contexts_list, right_contexts_list)

    Examples:
        >>> parse_context_grid("200,500,1000")
        ([200, 500, 1000], [200, 500, 1000])
        >>> parse_context_grid("200:1000:200")
        ([200, 400, 600, 800, 1000], [200, 400, 600, 800, 1000])
        >>> parse_context_grid("200,500", left_contexts="100,200", right_contexts="300,400")
        ([100, 200], [300, 400])
    """
    left_list = parse_values(left_contexts if left_contexts is not None else context_grid)
    right_list = parse_values(right_contexts if right_contexts is not None else context_grid)

    return left_list, right_list


@dataclass
class GridSearchConfig:
    """
    Configuration for chunk context grid search.

    Attributes:
        train_data_path: Path to training chunks or BAM/POD5 for preparation
        val_data_path: Path to validation chunks or BAM/POD5
        model_name: Model architecture to use
        output_dir: Base output directory for all grid results
        left_contexts: List of left context sizes to test
        right_contexts: List of right context sizes to test
        kmer_context: K-mer context for sequence encoding
        epochs: Number of training epochs per grid point
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device for training
        seed: Random seed
        early_stopping_patience: Stop training if validation loss doesn't improve for N epochs
        motif: Optional motif for chunk extraction
        motif_offset: Offset within motif
    """

    train_data_path: Path
    val_data_path: Path | None
    model_name: str
    output_dir: Path
    left_contexts: list[int]
    right_contexts: list[int]
    kmer_context: int = 5
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.001
    device: str = "cuda"
    seed: int | None = None  # None = generate random seed
    early_stopping_patience: int = 10
    motif: str | None = None
    motif_offset: int = 0


def prepare_chunks_with_context(
    bam_path: Path,
    pod5_path: Path,
    output_path: Path,
    left_context: int,
    right_context: int,
    kmer_context: int = 5,
    motif: str | None = None,
    motif_offset: int = 0,
    label: str | None = None,
    label_int: int = 0,
    min_mapq: int = 10,
) -> None:
    """
    Prepare training chunks with specific signal context.

    Args:
        bam_path: Path to BAM file
        pod5_path: Path to POD5 file
        output_path: Output path for chunks
        left_context: Left signal context (samples)
        right_context: Right signal context (samples)
        kmer_context: K-mer context for sequence
        motif: Optional motif for filtering
        motif_offset: Offset within motif
        label: String label for chunks (e.g., 'charged', 'uncharged')
        label_int: Numeric label for chunks (0 or 1)
        min_mapq: Minimum mapping quality
    """
    chunks = []

    for read in iter_bam_with_pod5(bam_path, pod5_path, min_mapq=min_mapq):
        # Temporarily modify get_chunk to use custom context
        original_get_chunk = read.get_chunk

        def custom_get_chunk(
            base_idx: int,
            signal_context: tuple[int, int] = (left_context, right_context),
            kmer_context: int = kmer_context,
            _original_get_chunk=original_get_chunk,
        ):
            return _original_get_chunk(
                base_idx, signal_context=signal_context, kmer_context=kmer_context
            )

        read.get_chunk = custom_get_chunk  # type: ignore[method-assign]

        # Extract chunks
        read_chunks = extract_training_chunks(
            read, motif=motif, motif_offset=motif_offset, label=label, label_int=label_int
        )
        chunks.extend(read_chunks)

    # Save
    save_chunks(chunks, output_path)
    logger.info(f"Prepared {len(chunks)} chunks with context ({left_context}, {right_context})")


def run_grid_point(
    train_data_path: Path,
    val_data_path: Path | None,
    model_name: str,
    output_dir: Path,
    left_context: int,
    right_context: int,
    kmer_len: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    seed: int,
    early_stopping_patience: int = 10,
) -> dict:
    """
    Train model for a single grid point.

    Args:
        train_data_path: Path to training data
        val_data_path: Path to validation data
        model_name: Model architecture
        output_dir: Output directory for this grid point
        left_context: Left signal context
        right_context: Right signal context
        kmer_len: K-mer length
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device
        seed: Random seed
        early_stopping_patience: Stop training if validation loss doesn't improve for N epochs

    Returns:
        Dictionary with grid point results
    """
    signal_len = left_context + right_context

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Training grid point: left={left_context}, right={right_context}")
    logger.info(f"Signal length: {signal_len}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'=' * 80}\n")

    start_time = time.time()

    try:
        history = train_model(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            model_name=model_name,
            output_dir=output_dir,
            signal_len=signal_len,
            kmer_len=kmer_len,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            seed=seed,
            early_stopping_patience=early_stopping_patience,
        )

        train_time = time.time() - start_time

        # Extract best metrics
        best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
        best_val_auc = max(history["val_auc"]) if history["val_auc"] else 0.0
        best_epoch = int(np.argmax(history["val_acc"]) + 1) if history["val_acc"] else epochs

        result = {
            "left_context": left_context,
            "right_context": right_context,
            "signal_len": signal_len,
            "best_val_acc": best_val_acc,
            "best_val_auc": best_val_auc,
            "best_val_loss": min(history["val_loss"]) if history["val_loss"] else 0.0,
            "best_epoch": best_epoch,
            "final_train_acc": history["train_acc"][-1] if history["train_acc"] else 0.0,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0.0,
            "train_time_sec": train_time,
            "model_path": str(output_dir / "model_best.pt"),
            "status": "success",
        }

    except Exception as e:
        logger.exception(f"Training failed for grid point: {e}")
        result = {
            "left_context": left_context,
            "right_context": right_context,
            "signal_len": left_context + right_context,
            "status": "failed",
            "error": str(e),
        }

    return result


def run_grid_search(config: GridSearchConfig) -> Path:
    """
    Run grid search over chunk contexts.

    Args:
        config: Grid search configuration

    Returns:
        Path to grid search summary CSV
    """
    from leech.constants import generate_random_seed

    # Generate random seed if not provided
    if config.seed is None:
        seed = generate_random_seed()
        logger.info(f"Generated random seed: {seed}")
        config.seed = seed
    else:
        seed = config.seed
        logger.info(f"Using provided seed: {seed}")

    logger.info("=" * 80)
    logger.info("Starting Grid Search")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Left contexts: {config.left_contexts}")
    logger.info(f"Right contexts: {config.right_contexts}")
    logger.info(f"Total grid points: {len(config.left_contexts) * len(config.right_contexts)}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Random seed: {seed}")
    logger.info("=" * 80)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save seed to file
    seed_file = config.output_dir / "grid_search_seed.txt"
    with open(seed_file, "w") as f:
        f.write(f"{seed}\n")

    # Save grid search config
    config_dict = {
        "model_name": config.model_name,
        "left_contexts": config.left_contexts,
        "right_contexts": config.right_contexts,
        "kmer_context": config.kmer_context,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "device": config.device,
        "seed": seed,
    }

    with open(config.output_dir / "grid_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Generate grid
    grid_points = list(itertools.product(config.left_contexts, config.right_contexts))
    results = []

    # Run each grid point
    for i, (left, right) in enumerate(grid_points, 1):
        logger.info(f"\n\nGrid point {i}/{len(grid_points)}")

        # Create output directory for this grid point
        grid_output_dir = config.output_dir / f"left_{left}_right_{right}"

        # Train model
        result = run_grid_point(
            train_data_path=config.train_data_path,
            val_data_path=config.val_data_path,
            model_name=config.model_name,
            output_dir=grid_output_dir,
            left_context=left,
            right_context=right,
            kmer_len=2 * config.kmer_context + 1,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            device=config.device,
            seed=config.seed,
            early_stopping_patience=config.early_stopping_patience,
        )

        results.append(result)

        # Save intermediate results
        summary_path = config.output_dir / "grid_summary.csv"
        save_grid_summary(results, summary_path)

    # Print summary with Rich tables
    console.print("\n[bold green]Grid Search Complete![/bold green]\n")

    successful_results = [r for r in results if r.get("status") == "success"]

    if successful_results:
        # Create results table
        table = Table(title="Grid Search Results", show_header=True, header_style="bold magenta")
        table.add_column("Left Context", justify="right", style="cyan")
        table.add_column("Right Context", justify="right", style="cyan")
        table.add_column("Val Accuracy", justify="right", style="green")
        table.add_column("Val AUC", justify="right", style="yellow")
        table.add_column("Best Epoch", justify="right", style="blue")
        table.add_column("Training Time", justify="right", style="white")

        # Sort by validation accuracy
        sorted_results = sorted(
            successful_results, key=lambda x: x.get("best_val_acc", 0), reverse=True
        )

        for r in sorted_results[:10]:  # Show top 10
            table.add_row(
                str(r["left_context"]),
                str(r["right_context"]),
                f"{r.get('best_val_acc', 0):.4f}",
                f"{r.get('best_val_auc', 0):.4f}",
                str(r.get("best_epoch", 0)),
                f"{r.get('training_time_sec', 0):.1f}s",
                style="bold" if r == sorted_results[0] else None,
            )

        console.print(table)

        # Best configuration summary
        best_result = sorted_results[0]
        summary_table = Table(
            title="Best Configuration", show_header=True, header_style="bold magenta"
        )
        summary_table.add_column("Parameter", style="cyan")
        summary_table.add_column("Value", justify="right", style="green")

        summary_table.add_row("Left Context", str(best_result["left_context"]))
        summary_table.add_row("Right Context", str(best_result["right_context"]))
        summary_table.add_row("Validation Accuracy", f"{best_result['best_val_acc']:.4f}")
        summary_table.add_row("Validation AUC", f"{best_result.get('best_val_auc', 0):.4f}")
        summary_table.add_row("Best Epoch", str(best_result.get("best_epoch", 0)))
        summary_table.add_row("Model Path", str(best_result["model_path"]))

        console.print(summary_table)

        best_params_path = config.output_dir / "best_params.json"
        with open(best_params_path, "w") as f:
            json.dump(
                {
                    "left_context": best_result["left_context"],
                    "right_context": best_result["right_context"],
                },
                f,
                indent=2,
            )
        console.print(f"[bold]Best params saved to:[/bold] {best_params_path}")

    console.print(f"\n[bold]Results saved to:[/bold] {summary_path}")

    return summary_path


def save_grid_summary(results: list[dict], output_path: Path) -> None:
    """
    Save grid search results to CSV.

    Args:
        results: List of result dictionaries
        output_path: Output CSV path
    """
    if not results:
        return

    # Define columns
    columns = [
        "left_context",
        "right_context",
        "signal_len",
        "best_val_acc",
        "best_val_auc",
        "best_val_loss",
        "best_epoch",
        "final_train_acc",
        "final_train_loss",
        "train_time_sec",
        "model_path",
        "status",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
