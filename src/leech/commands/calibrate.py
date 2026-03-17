"""
Handler for the 'calibrate' command.

This module contains the business logic for post-hoc Platt scaling calibration.
"""

import logging
from pathlib import Path

from leech.cli_config import make_console
from leech.commands.bundle import pick_best_fold

logger = logging.getLogger("leech.commands.calibrate")
console = make_console()


def handle_calibrate(
    model_dir: Path,
    val_data: Path,
    device: str = "cpu",
    batch_size: int = 1024,
    num_workers: int = 0,
    method: str = "temperature",
    reg_lambda: float = 0.01,
    output: Path | None = None,
) -> int:
    """
    Handle the calibrate command logic.

    Binary models: Platt scaling (method param ignored).
    Multiclass models: temperature, matrix, or dirichlet scaling.

    For a parent directory with pair subdirs, calibrates each pair independently.

    Args:
        model_dir: Model directory or parent with pair subdirs
        val_data: Validation data (.npz)
        device: Device for inference
        batch_size: Batch size for validation pass
        num_workers: DataLoader workers
        method: Multiclass calibration method
        reg_lambda: L2 regularization for matrix/dirichlet
        output: Explicit output path for calibration JSON (single model only)

    Returns:
        Number of models calibrated
    """
    import json

    from leech.calibration import calibrate_model, calibrate_model_multiclass

    def _is_multiclass(config_path: Path) -> bool:
        """Check if model config indicates multiclass (num_out > 2)."""
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("num_out", 1) > 2

    def _calibrate_single(mdir: Path, output_path: Path | None = None) -> str:
        """Calibrate a single model dir, auto-detecting binary vs multiclass."""
        config_path = mdir / "config.json"
        if _is_multiclass(config_path):
            result = calibrate_model_multiclass(
                mdir,
                val_data,
                method=method,
                reg_lambda=reg_lambda,
                output_path=output_path,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            m = result["method"]
            ece_b = result["ece_before"]
            ece_a = result["ece_after"]
            return f"{m}: ECE {ece_b:.4f} -> {ece_a:.4f}"
        else:
            a, b = calibrate_model(
                mdir,
                val_data,
                output_path=output_path,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            return f"Platt: a={a:.4f}, b={b:.4f}"

    # Check if model_dir is a single model or a parent with pair subdirs
    if (model_dir / "config.json").exists():
        # Single model directory
        result_str = _calibrate_single(model_dir, output_path=output)
        console.print(f"[green]{result_str}[/green]")
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
            result_str = _calibrate_single(subdir)
            console.print(f"[green]{result_str}[/green]")
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
            result_str = _calibrate_single(best_fold)
            console.print(f"[green]{result_str}[/green]")
            pairs_done += 1

    if pairs_done == 0:
        console.print("[bold red]No model directories found[/bold red]")
        raise SystemExit(1)
    console.print(f"[bold green]Calibrated {pairs_done} models[/bold green]")
    return pairs_done
