"""
Handlers for model bundle, bundle-info, and export commands.

This module contains the business logic for bundling, inspecting, and
exporting trained models.
"""

import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

logger = logging.getLogger("leech.commands.bundle")
console = Console()


def pick_best_fold(fold_dirs: list[Path]) -> Path:
    """Select fold with lowest final_val_loss to avoid overfitting.

    Falls back to highest best_val_f1 if final_val_loss unavailable.
    """
    candidates = []
    for fold_dir in fold_dirs:
        summary_path = fold_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            candidates.append(
                (
                    summary.get("final_val_loss", float("inf")),
                    -summary.get("best_val_f1", -1.0),
                    fold_dir,
                )
            )
        else:
            candidates.append((float("inf"), 0.0, fold_dir))

    candidates.sort(key=lambda x: (x[0], x[1]))
    best = candidates[0]
    logger.info(
        f"Selected {best[2].name} (final_val_loss={best[0]:.4f}, best_val_f1={-best[1]:.4f})"
    )
    return best[2]


def handle_bundle(
    model_dir: Path,
    output: Path,
    bundle_version: str,
    comparison_type: str = "pairwise",
    torchscript: bool = False,
) -> Path:
    """
    Handle the bundle command logic.

    Args:
        model_dir: Root dir containing pair subdirectories
        output: Output bundle .pt file path
        bundle_version: Semantic version string
        comparison_type: Comparison type (pairwise, one_vs_all, group)
        torchscript: Whether to bundle as TorchScript

    Returns:
        Path to created bundle file
    """
    from leech.util import create_bundle, create_torchscript_bundle

    # Auto-discover pair subdirectories
    model_dirs = {}
    for subdir in sorted(model_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Direct model (no folds)
        if (subdir / "model_best.pt").exists() and (subdir / "config.json").exists():
            model_dirs[subdir.name] = subdir
            continue
        # K-fold: pick best fold by val F1
        fold_dirs = sorted(subdir.glob("fold_*/"))
        valid_folds = [
            f for f in fold_dirs if (f / "model_best.pt").exists() and (f / "config.json").exists()
        ]
        if valid_folds:
            best_fold = pick_best_fold(valid_folds)
            model_dirs[subdir.name] = best_fold
            logger.info(f"{subdir.name}: selected {best_fold.name} (lowest val loss)")

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

    return bundle_path


def handle_bundle_info(bundle: Path) -> dict[str, Any]:
    """
    Handle the bundle-info command logic.

    Args:
        bundle: Path to bundle .pt file

    Returns:
        Bundle metadata dictionary
    """
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

    return metadata


def handle_export(model_dir: Path, output: Path) -> Path:
    """
    Handle the export command logic.

    Args:
        model_dir: Model checkpoint directory
        output: Output TorchScript .pt file path

    Returns:
        Path to exported file
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

    return output_path
