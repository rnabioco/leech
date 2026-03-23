"""Classification metrics computation, saving, and display."""

import json
import logging
from pathlib import Path

import numpy as np
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from leech.cli_config import make_console

logger = logging.getLogger("leech.metrics")
console = make_console()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true: True labels (binary)
        y_pred: Predicted labels (binary)
        y_prob: Predicted probabilities (0-1)

    Returns:
        Dictionary with metrics:
        - accuracy: Overall accuracy
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - auroc: ROC AUC score (area under receiver operating characteristic curve)
        - auprc: Average precision score (area under precision-recall curve)
        - confusion_matrix: 2x2 confusion matrix as list

    Raises:
        ValueError: If input arrays are empty or have mismatched lengths
    """
    # Validate inputs
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_prob) == 0:
        raise ValueError("Cannot compute metrics on empty arrays")

    if len(y_true) != len(y_pred) or len(y_true) != len(y_prob):
        raise ValueError(
            f"Array length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, y_prob={len(y_prob)}"
        )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # AUROC and AUPRC only if we have both classes
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["auroc"] = 0.0
        metrics["auprc"] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def save_metrics(metrics: dict, output_path: Path) -> None:
    """
    Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {output_path}")


def print_metrics(metrics: dict) -> None:
    """
    Pretty print metrics to console using Rich tables.

    Args:
        metrics: Dictionary of metrics
    """
    # Main metrics table
    table = Table(title="Evaluation Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", justify="right", style="green", width=15)

    table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
    table.add_row("Precision", f"{metrics['precision']:.4f}")
    table.add_row("Recall", f"{metrics['recall']:.4f}")
    table.add_row("F1 Score", f"{metrics['f1']:.4f}")

    # Handle both old (auc) and new (auroc) formats
    if "auroc" in metrics:
        table.add_row("AUROC", f"{metrics['auroc']:.4f}")
        if "auprc" in metrics:
            table.add_row("AUPRC", f"{metrics['auprc']:.4f}")
    elif "auc" in metrics:
        # Backward compatibility with old format
        table.add_row("ROC AUC", f"{metrics['auc']:.4f}")

    console.print(table)

    # Confusion matrix table
    if "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        cm_table = Table(title="Confusion Matrix", show_header=True, header_style="bold magenta")
        cm_table.add_column("", style="cyan", width=10)
        cm_table.add_column("Predicted Neg", justify="right", style="yellow", width=15)
        cm_table.add_column("Predicted Pos", justify="right", style="yellow", width=15)

        cm_table.add_row("Actual Neg", str(cm[0][0]), str(cm[0][1]))
        cm_table.add_row("Actual Pos", str(cm[1][0]), str(cm[1][1]))

        console.print(cm_table)
