"""
Handler for the 'merge-and-split' command.

This module contains the business logic for merging multiple chunk files
and splitting at read level to prevent data leakage.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from leech.constants import DEFAULT_SEED

logger = logging.getLogger("leech.commands.merge_split")
console = Console()


def _extract_file_paths(parsed: Any) -> list[Path]:
    """Extract file paths from _parse_and_validate_inputs result."""
    if isinstance(parsed, tuple) and len(parsed) == 2:
        # Multi-class: (files, labels)
        return parsed[0]
    elif isinstance(parsed, tuple) and len(parsed) == 3:
        # Pairwise: (files, relabel, meta_labels)
        return parsed[0]
    return []


def _propagate_prepare_config(input_paths: list[Path], output_dir: Path) -> None:
    """Collect prepare_config.json from input directories, validate consistency, and write to output.

    If no sidecar is found (old data), silently skips.
    If multiple sidecars disagree on critical fields, logs a warning but writes the first one found.
    """
    configs: list[dict] = []
    for p in input_paths:
        sidecar = p.parent / "prepare_config.json"
        if sidecar.exists():
            with open(sidecar) as f:
                configs.append(json.load(f))

    if not configs:
        return

    # Validate consistency on critical fields
    critical_keys = [
        "anchor",
        "refine_signal_map",
        "signal_norm",
        "reverse_signal",
        "motif_reference",
    ]
    first = configs[0]
    for i, cfg in enumerate(configs[1:], start=1):
        for key in critical_keys:
            if cfg.get(key) != first.get(key):
                logger.warning(
                    f"prepare_config.json mismatch on '{key}': "
                    f"input 0 has {first.get(key)!r}, input {i} has {cfg.get(key)!r}. "
                    f"Using values from first input."
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "prepare_config.json", "w") as f:
        json.dump(first, f, indent=2)
    logger.info(f"Propagated prepare_config.json to {output_dir}")


def handle_merge_and_split(
    input_chunks: tuple[str, ...],
    output_dir: Path,
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int | None = DEFAULT_SEED,
    comparison_spec: Path | None = None,
) -> dict[str, Any]:
    """
    Handle the merge-and-split command logic.

    Args:
        input_chunks: Tuple of input specifications (label=file.npz format or directories)
        output_dir: Output directory for split chunks
        train_split: Fraction of reads for training
        val_split: Fraction of reads for validation
        seed: Random seed for reproducibility
        comparison_spec: Optional TSV file with comparison specifications

    Returns:
        Dictionary with merge and split statistics
    """
    from leech.splitting import merge_and_split_chunks, process_comparison_spec

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
        _display_batch_results(result, output_dir)
        return result

    # Single comparison mode with label=file syntax
    logger.info("Merging and splitting chunks at read level")

    # Parse and validate inputs
    parsed = _parse_and_validate_inputs(input_chunks)

    # Propagate prepare_config.json sidecar (extract file paths for propagation)
    _all_input_files = _extract_file_paths(parsed)
    _propagate_prepare_config(_all_input_files, output_dir)

    # Multi-class mode (N > 2 unique labels)
    if (
        isinstance(parsed, tuple)
        and len(parsed) == 2
        and isinstance(parsed[1], list)
        and isinstance(parsed[1][0], str)
    ):
        all_files, all_labels = parsed
        logger.info(f"Multi-class mode: {len(set(all_labels))} classes")
        from leech.splitting import merge_and_split_multiclass

        result = merge_and_split_multiclass(
            input_paths=all_files,
            labels=all_labels,
            output_dir=output_dir,
            train_frac=train_split,
            val_frac=val_split,
            seed=seed,
        )
        _display_single_comparison_results(result)
        console.print("[bold green]Multi-class merge and split complete![/bold green]")
        return result

    all_files, relabel_tuple, meta_labels = parsed

    logger.info(
        f"Relabeling for comparison: {meta_labels[0]} = label_int 0, {meta_labels[1]} = label_int 1"
    )

    # Merge and split at read level
    split_result = merge_and_split_chunks(
        input_paths=all_files,
        output_dir=output_dir,
        train_frac=train_split,
        val_frac=val_split,
        seed=seed,
        relabel_pairwise=relabel_tuple,
    )

    # Type narrowing: result is always a dict when output_dir is provided
    assert isinstance(split_result, dict)
    result = split_result

    # Display statistics
    _display_single_comparison_results(result)

    console.print("[bold green]Merge and split complete![/bold green]")

    return result


def _parse_and_validate_inputs(
    input_chunks: tuple[str, ...],
) -> tuple[list[Path], tuple[str | list[str], str | list[str]], tuple[str, str]]:
    """
    Parse and validate input chunk specifications.

    Args:
        input_chunks: Tuple of label=file.npz specifications

    Returns:
        Tuple of (all_files, relabel_tuple, meta_labels)
        - all_files: List of file paths in order
        - relabel_tuple: Tuple of (group1_labels, group2_labels) for relabeling
        - meta_labels: Tuple of (meta_label1, meta_label2)

    Raises:
        ValueError: If input format is invalid or not exactly 2 meta-labels
        FileNotFoundError: If any input file doesn't exist
    """
    meta_to_files: dict[str, list[Path]] = {}
    meta_to_chunk_labels: dict[str, set[Any]] = {}
    meta_order = []  # Track order for label_int assignment

    for chunk_spec in input_chunks:
        if "=" not in chunk_spec:
            raise ValueError(
                f"Invalid input format: '{chunk_spec}'. "
                "Expected format: label=file.npz (e.g., Ala=ala.npz or basic=lys.npz)"
            )

        meta_label, file_path_str = chunk_spec.split("=", 1)
        meta_label = meta_label.strip()
        file_path = Path(file_path_str.strip())

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

    # Validate we have at least 2 groups
    if len(meta_to_files) < 2:
        raise ValueError(
            f"Expected at least 2 unique meta-labels, "
            f"got {len(meta_to_files)}: {list(meta_to_files.keys())}"
        )

    # For N > 2 groups, return multi-class format instead of pairwise
    if len(meta_to_files) > 2:
        all_files = []
        all_labels = []
        for meta_label in meta_order:
            for f in meta_to_files[meta_label]:
                all_files.append(f)
                all_labels.append(meta_label)
        return all_files, all_labels  # type: ignore[return-value]

    # Build relabel_pairwise tuple: (group1_chunk_labels, group2_chunk_labels)
    # First seen meta-label = 0, second = 1
    meta1, meta2 = meta_order[0], meta_order[1]
    group1_labels = sorted(meta_to_chunk_labels[meta1])
    group2_labels = sorted(meta_to_chunk_labels[meta2])

    # Collect all file paths in order
    all_files = meta_to_files[meta1] + meta_to_files[meta2]

    # Build relabel tuple: use list for multi-label groups, string for single
    relabel_group1 = group1_labels[0] if len(group1_labels) == 1 else group1_labels
    relabel_group2 = group2_labels[0] if len(group2_labels) == 1 else group2_labels
    relabel_tuple = (relabel_group1, relabel_group2)

    return all_files, relabel_tuple, (meta1, meta2)


def handle_merge_and_split_kfold(
    input_chunks: tuple[str, ...],
    output_dir: Path,
    k_fold: int,
    seed: int | None = DEFAULT_SEED,
) -> dict[str, Any]:
    """
    Handle the merge-and-split command with k-fold cross-validation.

    Args:
        input_chunks: Tuple of input specifications (label=file.npz format)
        output_dir: Output directory for fold splits
        k_fold: Number of folds for cross-validation
        seed: Random seed for reproducibility

    Returns:
        Dictionary with k-fold split statistics
    """
    from leech.splitting import merge_and_kfold_split_chunks

    logger.info(f"Merging and splitting chunks with {k_fold}-fold cross-validation")

    # Parse and validate inputs
    parsed = _parse_and_validate_inputs(input_chunks)

    # Propagate prepare_config.json sidecar
    _all_input_files = _extract_file_paths(parsed)
    _propagate_prepare_config(_all_input_files, output_dir)

    # Multi-class mode (N > 2 unique labels)
    if (
        isinstance(parsed, tuple)
        and len(parsed) == 2
        and isinstance(parsed[1], list)
        and isinstance(parsed[1][0], str)
    ):
        all_files, all_labels = parsed
        logger.info(f"Multi-class {k_fold}-fold mode: {len(set(all_labels))} classes")
        from leech.splitting import merge_and_kfold_split_multiclass

        result = merge_and_kfold_split_multiclass(
            input_paths=all_files,
            labels=all_labels,
            output_dir=output_dir,
            k_fold=k_fold,
            seed=seed,
        )
        _display_kfold_results(result)
        console.print("[bold green]Multi-class k-fold merge and split complete![/bold green]")
        return result

    all_files, relabel_tuple, meta_labels = parsed

    logger.info(
        f"Relabeling for comparison: {meta_labels[0]} = label_int 0, {meta_labels[1]} = label_int 1"
    )

    # Merge and k-fold split at read level
    result = merge_and_kfold_split_chunks(
        input_paths=all_files,
        output_dir=output_dir,
        k_fold=k_fold,
        seed=seed,
        relabel_pairwise=relabel_tuple,
    )

    # Display per-fold statistics
    _display_kfold_results(result)

    console.print("[bold green]K-fold merge and split complete![/bold green]")

    return result


def _display_kfold_results(result: dict[str, Any]) -> None:
    """Display results for k-fold cross-validation split."""
    table = Table(
        title=f"{result['k_fold']}-Fold Cross-Validation Split (Read-Level)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Fold", style="cyan")
    table.add_column("Train", justify="right", style="green")
    table.add_column("Val", justify="right", style="yellow")
    table.add_column("Test", justify="right", style="red")

    for i, fold in enumerate(result["folds"]):
        table.add_row(
            str(i),
            str(fold["n_train"]),
            str(fold["n_val"]),
            str(fold["n_test"]),
        )

    console.print(table)


def _display_single_comparison_results(result: dict[str, Any]) -> None:
    """Display results for a single comparison."""
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


def _display_batch_results(result: dict[str, Any], output_dir: Path) -> None:
    """Display results for batch processing with comparison spec."""
    console.print(f"\n[bold green]Processed {result['n_comparisons']} comparisons:[/bold green]")
    for comp_name, stats in result["comparisons"].items():
        console.print(
            f"  {comp_name}: {stats['n_train']} train, {stats['n_val']} val, {stats['n_test']} test"
        )

    console.print(f"\n[bold green]All comparisons saved to {output_dir}/[/bold green]")
