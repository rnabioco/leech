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
    roc_curve,
)

from leech.cli_config import make_console

logger = logging.getLogger("leech.metrics")
console = make_console()


def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Best operating points over every distinct score, not just 0.5.

    A confusion matrix reports ONE threshold. That is a choice, not a property
    of the model, and it is the wrong choice whenever training and evaluation
    see different class ratios: a model trained with ``--oversample-minority``
    emits scores calibrated for a 50/50 prior and is then thresholded on data
    that is not 50/50, which shows up as a collapsed precision that says more
    about the class ratio than about the model.

    Three criteria, because they disagree and the disagreement is the point:

    * ``at_youden`` maximises TPR - FPR. Both terms condition on a single true
      class, so it is INVARIANT to prevalence -- the right default when the
      deployment class ratio is unknown, varies per sample, or is itself the
      quantity being measured.
    * ``at_mcc`` and ``at_f1`` maximise prevalence-DEPENDENT quantities, so the
      threshold each picks is a property of this test set's class ratio as much
      as of the model. They are reported because they are what most callers
      compare against, with ``prevalence`` beside them so the dependence is
      visible rather than implied.

    Every point is exact: ``roc_curve`` already enumerates the distinct
    thresholds, and the counts follow from (TPR, FPR) and the class totals, so
    nothing here is a grid approximation.

    Returns ``{}`` when only one class is present, where no threshold is
    defined.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    pos = int(y_true.sum())
    neg = int(y_true.size - pos)
    if pos == 0 or neg == 0:
        return {}

    fpr, tpr, thr = roc_curve(y_true, y_prob)
    # roc_curve prepends a "call everything negative" point whose threshold is
    # infinite. It is a real operating point but not a reportable score.
    if thr.size and not np.isfinite(thr[0]):
        fpr, tpr, thr = fpr[1:], tpr[1:], thr[1:]
    if thr.size == 0:
        return {}

    tp = tpr * pos
    fn = pos - tp
    fp = fpr * neg
    tn = neg - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(den > 0, (tp * tn - fp * fn) / den, 0.0)
        f1_den = 2 * tp + fp + fn
        f1 = np.where(f1_den > 0, 2 * tp / f1_den, 0.0)
    youden = tpr - fpr

    def _at(i: int) -> dict:
        i = int(i)
        return {
            "threshold": float(thr[i]),
            "tpr": float(tpr[i]),
            "fpr": float(fpr[i]),
            "mcc": float(mcc[i]),
            "f1": float(f1[i]),
            "youden_j": float(youden[i]),
            "called_positive_frac": float((tp[i] + fp[i]) / y_true.size),
        }

    return {
        "prevalence": pos / y_true.size,
        "at_youden": _at(np.argmax(youden)),
        "at_mcc": _at(np.argmax(mcc)),
        "at_f1": _at(np.argmax(f1)),
    }


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
        # The metrics above this line are all read at one fixed threshold.
        metrics["threshold_sweep"] = sweep_thresholds(y_true, y_prob)
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
