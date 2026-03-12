"""
Shared utilities for CLI command handlers.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("leech.commands")


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


def discover_model_dirs(parent_dir: Path) -> dict[str, Path]:
    """Discover model directories, auto-selecting best fold for k-fold dirs.

    Scans subdirectories of ``parent_dir`` for either:
    - Direct model dirs (containing model_best.pt + config.json)
    - K-fold dirs (containing fold_*/ subdirs with model_best.pt + config.json)

    For k-fold dirs, the best fold is selected via :func:`pick_best_fold`.

    Args:
        parent_dir: Root directory containing pair subdirectories

    Returns:
        Dict mapping pair name -> model directory path
    """
    model_dirs: dict[str, Path] = {}
    for subdir in sorted(parent_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Direct model (no folds)
        if (subdir / "model_best.pt").exists() and (subdir / "config.json").exists():
            model_dirs[subdir.name] = subdir
            continue
        # K-fold: pick best fold
        fold_dirs = sorted(subdir.glob("fold_*/"))
        valid_folds = [
            f for f in fold_dirs if (f / "model_best.pt").exists() and (f / "config.json").exists()
        ]
        if valid_folds:
            best_fold = pick_best_fold(valid_folds)
            model_dirs[subdir.name] = best_fold
            logger.info(f"{subdir.name}: selected {best_fold.name} (lowest val loss)")

    return model_dirs
