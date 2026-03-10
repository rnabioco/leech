"""
Utility functions for leech.

Includes model loading/saving helpers, metrics computation, and logging utilities.
"""

import inspect
import io
import json
import logging
import os
import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
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

from leech.constants import generate_random_seed
from leech.models import MODEL_REGISTRY, get_model

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

    # Create model from config (filters training params and validates constructor args)
    model = _instantiate_model(config)

    # Load checkpoint
    checkpoint_file = checkpoint_path / checkpoint_name
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location=device)

    # Strip _orig_mod. prefix added by torch.compile()
    state_dict = checkpoint["model_state_dict"]
    state_dict = {
        k.removeprefix("_orig_mod."): v for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, config


# Training-specific parameters that should NOT be in bundle configs
_TRAINING_PARAMS = {
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
    "scheduler",
    "scheduler_patience",
    "scheduler_factor",
    "max_grad_norm",
    "weight_decay",
    "mixed_precision",
    "warmup_epochs",
    "loss_type",
    "focal_gamma",
    "augment_jitter",
    "augment_scale_min",
    "augment_scale_max",
    "resume_from",
}


def _architecture_config(config: dict) -> dict:
    """Extract architecture-only parameters from a full training config."""
    return {k: v for k, v in config.items() if k not in _TRAINING_PARAMS}


def _instantiate_model(config: dict) -> nn.Module:
    """Instantiate a model from a config dict, filtering to valid constructor params.

    Strips training-specific keys and any kwargs not accepted by the model class,
    so it works with both full training configs and bundle architecture configs.
    """
    model_name = config["model_name"]
    signal_len = config["signal_len"]
    kmer_len = config["kmer_len"]

    model_kwargs = {
        k: v
        for k, v in config.items()
        if k not in {"model_name", "signal_len", "kmer_len"} and k not in _TRAINING_PARAMS
    }

    model_class = MODEL_REGISTRY[model_name]
    valid_params = set(inspect.signature(model_class).parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in valid_params}

    return get_model(model_name, signal_len=signal_len, kmer_len=kmer_len, **filtered_kwargs)


# ============================================================================
# TorchScript export helpers
# ============================================================================


def trace_model(model: nn.Module, config: dict) -> torch.jit.ScriptModule:
    """Trace a leech model into a TorchScript ScriptModule.

    Args:
        model: PyTorch model (must be in eval mode or will be set to eval)
        config: Model config dict with signal_len, kmer_len, etc.

    Returns:
        Traced ScriptModule
    """
    from leech.models.inference_wrapper import ModelInferenceWrapper

    model.eval()

    signal_len = config["signal_len"]
    kmer_len = config["kmer_len"]
    seq_encoding = config.get("seq_encoding", "base_onehot")
    signal_kmer_context = config.get("signal_kmer_context", [4, 4])

    # Determine sequence tensor shape based on encoding
    if seq_encoding == "signal_kmer":
        seq_channels = sum(signal_kmer_context) * 4 + 4  # 36 for default (4,4)
        seq_len = signal_len
    else:
        seq_channels = 4
        seq_len = kmer_len

    # Build example inputs
    signal = torch.randn(1, signal_len)
    sequence = torch.randn(1, seq_channels, seq_len)

    model_name = config.get("model_name", "")
    requires_features = model_name in ModelInferenceWrapper.FEATURE_MODELS

    if requires_features:
        num_features = config.get("num_features", 5)
        features = torch.randn(1, num_features, kmer_len)
        example_inputs = (signal, sequence, features)
    else:
        example_inputs = (signal, sequence)

    with torch.no_grad():
        traced = torch.jit.trace(model, example_inputs)

    return traced


def serialize_traced_model(traced: torch.jit.ScriptModule) -> bytes:
    """Serialize a traced model to bytes."""
    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    return buf.getvalue()


def deserialize_traced_model(data: bytes, device: str = "cpu") -> torch.jit.ScriptModule:
    """Deserialize a traced model from bytes."""
    buf = io.BytesIO(data)
    return torch.jit.load(buf, map_location=device)


def export_single_model(model_dir: Path, output_path: Path) -> Path:
    """Export a single trained model as a standalone TorchScript .pt file.

    The exported file is loadable with just ``torch.jit.load()`` — no leech
    required.  Model config is embedded via ``_extra_files``.

    Args:
        model_dir: Directory with config.json and model_best.pt
        output_path: Where to write the TorchScript .pt file

    Returns:
        Path to the saved file
    """
    model, config = load_model_from_checkpoint(model_dir, device="cpu")
    traced = trace_model(model, config)

    meta = json.dumps(config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(output_path), _extra_files={"leech_meta.txt": meta})

    logger.info(f"Exported TorchScript model to {output_path}")
    return output_path


def create_bundle(
    model_dirs: dict[str, Path],
    output_path: Path,
    comparison_type: str,
    version: str,
) -> Path:
    """
    Bundle multiple trained models into a single versioned file.

    Args:
        model_dirs: Mapping of pair name -> model directory (with config.json and model_best.pt)
        output_path: Output .pt file path
        comparison_type: "pairwise" or "one_vs_all"
        version: Semantic version string (e.g., "0.1.0-alpha.1")

    Returns:
        Path to the saved bundle file

    Raises:
        FileNotFoundError: If config.json or model_best.pt missing in any dir
        ValueError: If architecture configs don't match across models
    """
    if not model_dirs:
        raise ValueError("model_dirs must not be empty")

    pairs = sorted(model_dirs.keys())

    # Load first model's config as reference
    first_dir = model_dirs[pairs[0]]
    ref_config_path = first_dir / "config.json"
    if not ref_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {ref_config_path}")
    with open(ref_config_path) as f:
        ref_full_config = json.load(f)
    ref_arch_config = _architecture_config(ref_full_config)

    architecture = ref_full_config["model_name"]

    models_dict: dict[str, dict] = {}
    for pair in pairs:
        model_dir = Path(model_dirs[pair])

        # Validate config matches
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            pair_config = json.load(f)
        pair_arch_config = _architecture_config(pair_config)

        if pair_arch_config != ref_arch_config:
            raise ValueError(
                f"Architecture config mismatch for {pair}. "
                f"Expected {ref_arch_config}, got {pair_arch_config}"
            )

        # Load checkpoint
        checkpoint_path = model_dir / "model_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        models_dict[pair] = {
            "state_dict": checkpoint["model_state_dict"],
            "best_val_acc": checkpoint.get("best_val_acc"),
            "best_epoch": checkpoint.get("best_epoch"),
        }

    bundle = {
        "metadata": {
            "format_version": 1,
            "bundle_version": version,
            "architecture": architecture,
            "comparison_type": comparison_type,
            "num_models": len(models_dict),
            "pairs": pairs,
            "created_at": datetime.now(UTC).isoformat(),
        },
        "config": ref_arch_config,
        "models": models_dict,
    }

    # Atomic save: write to temp file then rename
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".pt.tmp")
    try:
        os.close(fd)
        torch.save(bundle, tmp_path)
        os.rename(tmp_path, output_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    logger.info(f"Bundle saved to {output_path} ({len(models_dict)} models, version {version})")
    return output_path


def create_torchscript_bundle(
    model_dirs: dict[str, Path],
    output_path: Path,
    comparison_type: str,
    version: str,
) -> Path:
    """Bundle multiple trained models as TorchScript into a single versioned file.

    Like :func:`create_bundle` but stores traced model bytes instead of raw
    state dicts.  The resulting bundle is loadable without the leech model
    registry — each pair is a self-contained TorchScript graph.

    Args:
        model_dirs: Mapping of pair name -> model directory
        output_path: Output .pt file path
        comparison_type: "pairwise" or "one_vs_all"
        version: Semantic version string

    Returns:
        Path to the saved bundle file
    """
    from leech.models.inference_wrapper import ModelInferenceWrapper

    if not model_dirs:
        raise ValueError("model_dirs must not be empty")

    pairs = sorted(model_dirs.keys())

    # Load first model's config as reference
    first_dir = model_dirs[pairs[0]]
    ref_config_path = first_dir / "config.json"
    if not ref_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {ref_config_path}")
    with open(ref_config_path) as f:
        ref_full_config = json.load(f)
    ref_arch_config = _architecture_config(ref_full_config)

    architecture = ref_full_config["model_name"]
    requires_features = architecture in ModelInferenceWrapper.FEATURE_MODELS

    models_dict: dict[str, dict] = {}
    for pair in pairs:
        model_dir = Path(model_dirs[pair])

        # Validate config matches
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            pair_config = json.load(f)
        pair_arch_config = _architecture_config(pair_config)

        if pair_arch_config != ref_arch_config:
            raise ValueError(
                f"Architecture config mismatch for {pair}. "
                f"Expected {ref_arch_config}, got {pair_arch_config}"
            )

        # Load checkpoint, instantiate, trace
        checkpoint_path = model_dir / "model_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Strip _orig_mod. prefix added by torch.compile()
        state_dict = checkpoint["model_state_dict"]
        state_dict = {
            k.removeprefix("_orig_mod."): v for k, v in state_dict.items()
        }

        model = _instantiate_model(ref_arch_config)
        model.load_state_dict(state_dict)
        traced = trace_model(model, ref_arch_config)
        traced_bytes = serialize_traced_model(traced)

        models_dict[pair] = {
            "traced_bytes": traced_bytes,
            "best_val_acc": checkpoint.get("best_val_acc"),
            "best_epoch": checkpoint.get("best_epoch"),
        }

    bundle = {
        "metadata": {
            "format_version": 2,
            "bundle_version": version,
            "architecture": architecture,
            "comparison_type": comparison_type,
            "num_models": len(models_dict),
            "pairs": pairs,
            "created_at": datetime.now(UTC).isoformat(),
            "torchscript": True,
            "requires_features": requires_features,
        },
        "config": ref_arch_config,
        "models": models_dict,
    }

    # Atomic save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".pt.tmp")
    try:
        os.close(fd)
        torch.save(bundle, tmp_path)
        os.rename(tmp_path, output_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    logger.info(
        f"TorchScript bundle saved to {output_path} ({len(models_dict)} models, version {version})"
    )
    return output_path


def load_model_from_bundle(
    bundle_path: Path,
    pair: str,
    device: str = "cuda",
) -> tuple[nn.Module, dict]:
    """
    Load a single model from a bundle file.

    Supports both format_version 1 (state_dict) and format_version 2
    (TorchScript) bundles transparently.

    Args:
        bundle_path: Path to bundle .pt file
        pair: Pair name to load (e.g., "Asn_Gln")
        device: Device to load model on

    Returns:
        Tuple of (model, config_dict) — same interface as load_model_from_checkpoint

    Raises:
        KeyError: If pair not found in bundle
    """
    bundle = torch.load(bundle_path, map_location=device)
    models = bundle["models"]

    if pair not in models:
        available = ", ".join(sorted(models.keys()))
        raise KeyError(f"Pair '{pair}' not in bundle. Available: {available}")

    config = bundle["config"]
    metadata = bundle.get("metadata", {})

    if metadata.get("torchscript", False):
        # TorchScript bundle (format_version 2)
        model = deserialize_traced_model(models[pair]["traced_bytes"], device=device)
        model.eval()
    else:
        # Legacy state_dict bundle (format_version 1)
        model = _instantiate_model(config)
        model.load_state_dict(models[pair]["state_dict"])
        model = model.to(device)
        model.eval()

    return model, config


def list_bundle_models(bundle_path: Path) -> dict:
    """
    List metadata from a bundle file.

    Args:
        bundle_path: Path to bundle .pt file

    Returns:
        Metadata dict with keys: format_version, bundle_version, architecture,
        comparison_type, num_models, pairs, created_at
    """
    bundle = torch.load(bundle_path, map_location="cpu")
    return bundle["metadata"]


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
