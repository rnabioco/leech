"""
Handler for the 'calibrate' command.

This module contains the business logic for post-hoc Platt scaling calibration.
"""

import logging
from pathlib import Path

from rich.console import Console

from leech.commands.bundle import pick_best_fold

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
    pairs_done = 0
    for subdir in sorted(model_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Direct model
        if (subdir / "config.json").exists():
            console.print(f"  [cyan]{subdir.name}[/cyan]", end=" ")
            a, b = calibrate_model(
                subdir,
                val_data,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            console.print(f"[green]a={a:.4f}, b={b:.4f}[/green]")
            pairs_done += 1
            continue
        # K-fold: calibrate best fold
        fold_dirs = sorted(subdir.glob("fold_*/"))
        valid_folds = [
            f for f in fold_dirs if (f / "model_best.pt").exists() and (f / "config.json").exists()
        ]
        if valid_folds:
            best_fold = pick_best_fold(valid_folds)
            console.print(f"  [cyan]{subdir.name}/{best_fold.name}[/cyan]", end=" ")
            a, b = calibrate_model(
                best_fold,
                val_data,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            console.print(f"[green]a={a:.4f}, b={b:.4f}[/green]")
            pairs_done += 1

    if pairs_done == 0:
        console.print("[bold red]No model directories found[/bold red]")
        raise SystemExit(1)
    console.print(f"[bold green]Calibrated {pairs_done} models[/bold green]")
    return pairs_done
