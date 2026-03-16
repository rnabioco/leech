"""
Post-hoc Platt scaling for model calibration.

Learns two parameters (a, b) per model on the validation set such that
sigmoid(a * logit + b) is better calibrated. Compared to temperature scaling
(1 param), Platt scaling can also shift the decision boundary — critical when
class imbalance during training biases the threshold.

This is essential for one-vs-all bundles where 20 models trained with
different positive-class rates must produce comparable probabilities for
argmax aggregation.

Reference: Platt, "Probabilistic Outputs for SVMs" (1999)
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import LBFGS
from torch.utils.data import DataLoader

from leech.chunking import load_chunks
from leech.dataset import LeechDataset, collate_fn
from leech.models import get_model
from leech.models.inference_wrapper import ModelInferenceWrapper

logger = logging.getLogger("leech.calibration")


def collect_logits(
    model_wrapper: ModelInferenceWrapper,
    val_loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect raw logits and labels from validation set."""
    all_logits = []
    all_labels = []

    model_wrapper.model.eval()
    with torch.no_grad():
        for batch in val_loader:
            logits = model_wrapper.forward_batch(batch, device)
            labels = batch["label"].float().to(device)
            all_logits.append(logits.flatten())
            all_labels.append(labels.flatten())

    return torch.cat(all_logits), torch.cat(all_labels)


def learn_platt(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
) -> tuple[float, float]:
    """
    Learn Platt scaling parameters (a, b) via NLL minimization.

    Optimizes: a, b = argmin BCE(sigmoid(a * logit + b), labels)
    using L-BFGS. Initialized at a=1.0, b=0.0 (identity transform).

    Args:
        logits: Raw model logits from validation set [N]
        labels: Binary labels [N]
        max_iter: Maximum L-BFGS iterations

    Returns:
        Tuple of (a, b) — scale and bias parameters
    """
    a = nn.Parameter(torch.ones(1, device=logits.device))
    b = nn.Parameter(torch.zeros(1, device=logits.device))
    criterion = nn.BCEWithLogitsLoss()

    optimizer = LBFGS([a, b], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = a * logits + b
        loss = criterion(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    a_val = a.item()
    b_val = b.item()

    # a must be positive (monotonicity); clamp to reasonable range
    a_val = max(a_val, 0.01)
    if a_val > 10.0:
        logger.warning(
            f"Platt a={a_val:.4f} exceeds 10.0 — clamping. "
            "This may indicate overfitting or degenerate logits."
        )
        a_val = 10.0
    if abs(b_val) < 0.01:
        logger.warning(
            f"Platt b={b_val:.4f} is near zero — possible degenerate fit. "
            "Check that validation set has sufficient class diversity."
        )

    logger.info(f"Learned Platt params: a={a_val:.4f}, b={b_val:.4f}")
    return a_val, b_val


def calibrate_model(
    model_dir: Path,
    val_data_path: Path,
    output_path: Path | None = None,
    device: str = "cpu",
    batch_size: int = 1024,
    num_workers: int = 0,
) -> tuple[float, float]:
    """
    Learn Platt scaling for a trained model.

    Loads the model and validation data, collects logits, optimizes (a, b),
    and saves to platt.json in the model directory.

    Args:
        model_dir: Directory with model_best.pt and config.json
        val_data_path: Path to validation data (.npz)
        output_path: Output JSON path (default: model_dir/platt.json)
        device: Device for inference
        batch_size: Batch size for validation pass
        num_workers: DataLoader workers

    Returns:
        Tuple of (a, b) — Platt scaling parameters
    """
    if output_path is None:
        output_path = model_dir / "platt.json"

    # Load model config
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    model_name = config["model_name"]
    signal_len = config["signal_len"]
    kmer_len = config["kmer_len"]
    seq_encoding = config.get("seq_encoding", "base_onehot")
    signal_kmer_context = tuple(config.get("signal_kmer_context", (4, 4)))

    # Build model init kwargs
    model_kwargs = {
        "signal_len": signal_len,
        "kmer_len": kmer_len,
        "seq_encoding": seq_encoding,
        "signal_kmer_context": signal_kmer_context,
    }
    no_feature_models = {
        "ConvLSTMBase",
        "ConvLSTMBaseBN",
        "ConvLSTMBaseAttn",
        "ConvLSTMBaseBNAttn",
        "ConvLSTMRemoraBase",
    }
    if model_name not in no_feature_models and "num_features" in config:
        model_kwargs["num_features"] = config["num_features"]

    model = get_model(model_name, **model_kwargs)

    # Load weights (always load to CPU first, then move to device)
    checkpoint_path = model_dir / "model_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    from leech.util import _migrate_state_dict_keys

    state_dict = _migrate_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    wrapper = ModelInferenceWrapper(model, model_name)

    # Load validation data
    val_chunks = load_chunks(val_data_path)
    val_dataset = LeechDataset(
        chunks=val_chunks,
        model_type=model_name,
        signal_len=signal_len,
        kmer_len=kmer_len,
        seq_encoding=seq_encoding,
        signal_kmer_context=signal_kmer_context,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    logger.info(f"Collecting logits from {len(val_dataset)} validation samples...")
    logits, labels = collect_logits(wrapper, val_loader, device)

    # Compute pre-calibration metrics
    pre_probs = torch.sigmoid(logits)
    pre_ece = _expected_calibration_error(pre_probs, labels)

    a, b = learn_platt(logits, labels)

    # Compute post-calibration metrics
    post_probs = torch.sigmoid(a * logits + b)
    post_ece = _expected_calibration_error(post_probs, labels)

    if post_ece >= pre_ece:
        logger.warning(
            f"Calibration did not improve ECE: {pre_ece:.4f} -> {post_ece:.4f}. "
            "Platt params may be unreliable for this model."
        )
    logger.info(f"ECE: {pre_ece:.4f} -> {post_ece:.4f} (a={a:.4f}, b={b:.4f})")

    result = {
        "platt_a": a,
        "platt_b": b,
        "ece_before": pre_ece,
        "ece_after": post_ece,
        "n_val_samples": len(val_dataset),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Platt scaling saved to {output_path}")
    return a, b


def collect_logits_multiclass(
    model_wrapper: ModelInferenceWrapper,
    val_loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect raw logits (N, K) and integer labels (N,) from validation set."""
    all_logits = []
    all_labels = []

    model_wrapper.model.eval()
    with torch.no_grad():
        for batch in val_loader:
            logits = model_wrapper.forward_batch(batch, device)
            labels = batch["label"].long().to(device)
            all_logits.append(logits)
            all_labels.append(labels.flatten())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels)


def learn_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
) -> float:
    """
    Learn temperature scaling parameter T via NLL minimization.

    Optimizes: T = argmin CrossEntropy(logits / T, labels) using L-BFGS.
    Initialized at T=1.5 (slight over-confidence is typical for neural nets).

    Args:
        logits: Raw model logits from validation set (N, K)
        labels: Integer class labels (N,)
        max_iter: Maximum L-BFGS iterations

    Returns:
        Learned temperature T
    """
    # Use log(T) as the parameter to keep T positive without clamping during optimization
    log_t = nn.Parameter(torch.tensor([0.4055], device=logits.device))  # log(1.5)
    criterion = nn.CrossEntropyLoss()

    optimizer = LBFGS([log_t], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        t = log_t.exp()
        scaled = logits / t
        loss = criterion(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    t_val = log_t.exp().item()

    # Clamp to reasonable range
    t_val = max(t_val, 0.1)
    if t_val > 10.0:
        logger.warning(
            f"Temperature T={t_val:.4f} exceeds 10.0 — clamping. "
            "This may indicate degenerate logits."
        )
        t_val = 10.0

    logger.info(f"Learned temperature: T={t_val:.4f}")
    return t_val


def calibrate_model_temperature(
    model_dir: Path,
    val_data_path: Path,
    output_path: Path | None = None,
    device: str = "cpu",
    batch_size: int = 1024,
    num_workers: int = 0,
) -> float:
    """
    Learn temperature scaling for a trained multiclass model.

    Loads the model and validation data, collects logits, optimizes T,
    and saves to temperature.json in the model directory.

    Args:
        model_dir: Directory with model_best.pt and config.json
        val_data_path: Path to validation data (.npz)
        output_path: Output JSON path (default: model_dir/temperature.json)
        device: Device for inference
        batch_size: Batch size for validation pass
        num_workers: DataLoader workers

    Returns:
        Learned temperature T
    """
    if output_path is None:
        output_path = model_dir / "temperature.json"

    # Load model config
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    model_name = config["model_name"]
    signal_len = config["signal_len"]
    kmer_len = config["kmer_len"]
    num_out = config.get("num_out", 1)
    seq_encoding = config.get("seq_encoding", "base_onehot")
    signal_kmer_context = tuple(config.get("signal_kmer_context", (4, 4)))

    # Build model init kwargs
    model_kwargs: dict = {
        "signal_len": signal_len,
        "kmer_len": kmer_len,
        "seq_encoding": seq_encoding,
        "signal_kmer_context": signal_kmer_context,
    }
    no_feature_models = {
        "ConvLSTMBase",
        "ConvLSTMBaseBN",
        "ConvLSTMBaseAttn",
        "ConvLSTMBaseBNAttn",
        "ConvLSTMRemoraBase",
    }
    if model_name not in no_feature_models and "num_features" in config:
        model_kwargs["num_features"] = config["num_features"]

    # Pass num_out for multiclass models (same set as training.py)
    num_out_models = {
        "ConvLSTMDwell",
        "ConvLSTMDwellBN",
        "ConvLSTMDwellBNAttn",
        "ConvLSTMRemora",
        "ConvLSTMRemoraBase",
        "TCNDwell",
        "TCNDwellGN",
        "TCNDwellLN",
        "TCNDwellResidual",
        "TCNDwellResidualGN",
        "TCNDwellResidualLN",
    }
    if model_name in num_out_models:
        model_kwargs["num_out"] = num_out

    model = get_model(model_name, **model_kwargs)

    # Load weights (always load to CPU first, then move to device)
    checkpoint_path = model_dir / "model_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    from leech.util import _migrate_state_dict_keys

    state_dict = _migrate_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    wrapper = ModelInferenceWrapper(model, model_name)

    # Load validation data
    val_chunks = load_chunks(val_data_path)
    val_dataset = LeechDataset(
        chunks=val_chunks,
        model_type=model_name,
        signal_len=signal_len,
        kmer_len=kmer_len,
        seq_encoding=seq_encoding,
        signal_kmer_context=signal_kmer_context,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    logger.info(f"Collecting logits from {len(val_dataset)} validation samples...")
    logits, labels = collect_logits_multiclass(wrapper, val_loader, device)

    # Compute pre-calibration ECE
    pre_probs = torch.softmax(logits, dim=-1)
    pre_ece = _expected_calibration_error_multiclass(pre_probs, labels)

    t = learn_temperature(logits, labels)

    # Compute post-calibration ECE
    post_probs = torch.softmax(logits / t, dim=-1)
    post_ece = _expected_calibration_error_multiclass(post_probs, labels)

    if post_ece >= pre_ece:
        logger.warning(
            f"Calibration did not improve ECE: {pre_ece:.4f} -> {post_ece:.4f}. "
            "Temperature may be unreliable for this model."
        )
    logger.info(f"ECE: {pre_ece:.4f} -> {post_ece:.4f} (T={t:.4f})")

    result = {
        "temperature": t,
        "ece_before": pre_ece,
        "ece_after": post_ece,
        "n_val_samples": len(val_dataset),
        "num_classes": num_out,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Temperature scaling saved to {output_path}")
    return t


def _expected_calibration_error(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> float:
    """Compute Expected Calibration Error (ECE) for binary predictions."""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean().item()
        bin_acc = labels[mask].mean().item()
        ece += mask.sum().item() / len(probs) * abs(bin_conf - bin_acc)
    return ece


def _expected_calibration_error_multiclass(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> float:
    """Compute top-1 confidence ECE for multiclass predictions.

    Bins samples by max(softmax), measures |mean_confidence - accuracy| per bin.
    """
    confidences, preds = probs.max(dim=-1)
    accuracies = (preds == labels).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean().item()
        bin_acc = accuracies[mask].mean().item()
        ece += mask.sum().item() / n * abs(bin_conf - bin_acc)
    return ece
