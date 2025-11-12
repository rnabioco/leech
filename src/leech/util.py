"""
Utility functions for leech.

Includes model loading/saving helpers, metrics computation, and logging utilities.
"""

import inspect
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from leech.constants import generate_random_seed
from leech.models import get_model

logger = logging.getLogger("leech.util")
console = Console()


def setup_random_seed(seed: int | None, output_dir: Path | None = None) -> int:
    """Setup random seed for reproducibility and optionally save to file.

    Args:
        seed: Random seed value, or None to generate one
        output_dir: Directory to save seed.txt file, or None to skip saving

    Returns:
        The seed value used
    """
    # Generate if needed
    if seed is None:
        seed = generate_random_seed()
        logger.info(f"Generated random seed: {seed}")
    else:
        logger.info(f"Using provided seed: {seed}")

    # Set for all libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Save if requested
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        seed_file = output_dir / "seed.txt"
        with open(seed_file, "w") as f:
            f.write(f"{seed}\n")
        logger.info(f"Saved seed to {seed_file}")

    return seed


def load_model_from_checkpoint(
    checkpoint_path: Path, device: str = "cuda", checkpoint_name: str = "model_best.pt"
) -> tuple[nn.Module, dict]:
    """
    Load a trained model from checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory (contains config.json and .pt files)
        device: Device to load model on
        checkpoint_name: Name of checkpoint file (default: model_best.pt)

    Returns:
        Tuple of (model, config_dict)

    Raises:
        FileNotFoundError: If config.json or checkpoint file not found
        ValueError: If config is invalid
    """
    checkpoint_path = Path(checkpoint_path)

    # Load config
    config_file = checkpoint_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file) as f:
        config = json.load(f)

    # Get model parameters
    model_name = config["model_name"]
    signal_len = config["signal_len"]
    kmer_len = config["kmer_len"]

    # Training-specific parameters that should NOT be passed to models
    training_params = {
        "epochs",
        "batch_size",
        "learning_rate",
        "device",
        "seed",
        "val_split",
        "patience",
        "min_delta",
        "save_dir",
        "log_dir",
        "num_workers",
        "pin_memory",
        "prefetch_factor",
        "use_class_weights",
        "pos_weight",
    }

    # Extract model-specific kwargs (filter out training params)
    model_kwargs = {
        k: v
        for k, v in config.items()
        if k not in ["model_name", "signal_len", "kmer_len"] and k not in training_params
    }

    # Filter kwargs to only include parameters the model actually accepts
    from leech.models import MODEL_REGISTRY

    model_class = MODEL_REGISTRY[model_name]
    model_signature = inspect.signature(model_class)
    valid_params = set(model_signature.parameters.keys()) - {"self"}

    filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in valid_params}

    # Create model with only valid parameters
    model = get_model(
        model_name,
        signal_len=signal_len,
        kmer_len=kmer_len,
        **filtered_kwargs,
    )

    # Load checkpoint
    checkpoint_file = checkpoint_path / checkpoint_name
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


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
        - auc: ROC AUC score
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

    # AUC only if we have both classes
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auc"] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
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
