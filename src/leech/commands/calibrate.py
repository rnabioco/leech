"""
Handler for the 'calibrate' command.

This module contains the business logic for post-hoc Platt scaling calibration.
"""

import logging
from pathlib import Path

from rich.console import Console

from leech.commands._utils import discover_model_dirs

logger = logging.getLogger("leech.commands.calibrate")
console = Console()


def handle_calibrate(
    model_dir: Path,
    val_data: Path,
    device: str = "cpu",
    batch_size: int = 1024,
    num_workers: int = 0,
) -> int:
    """
    Handle the calibrate command logic.

    Fits Platt scaling parameters (a, b) per model on the validation set.
    For a parent directory with pair subdirs, calibrates each pair independently.

    Args:
        model_dir: Model directory or parent with pair subdirs
        val_data: Validation data (.npz)
        device: Device for inference
        batch_size: Batch size for validation pass
        num_workers: DataLoader workers

    Returns:
        Number of models calibrated
    """
    from leech.calibration import calibrate_model

    # Check if model_dir is a single model or a parent with pair subdirs
    if (model_dir / "config.json").exists():
        # Single model directory
        a, b = calibrate_model(
            model_dir,
            val_data,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        console.print(f"[green]Platt: a={a:.4f}, b={b:.4f}[/green]")
        return 1

    # Parent directory — calibrate each pair subdir
    console.print(f"[cyan]Scanning {model_dir} for model subdirectories...[/cyan]")
    pair_dirs = discover_model_dirs(model_dir)

    for name, pair_path in sorted(pair_dirs.items()):
        console.print(f"  [cyan]{name}[/cyan]", end=" ")
        a, b = calibrate_model(
            pair_path,
            val_data,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        console.print(f"[green]a={a:.4f}, b={b:.4f}[/green]")

    if not pair_dirs:
        console.print("[bold red]No model directories found[/bold red]")
        raise SystemExit(1)
    console.print(f"[bold green]Calibrated {len(pair_dirs)} models[/bold green]")
    return len(pair_dirs)
