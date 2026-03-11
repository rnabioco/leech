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
        "ConvLSTMBase", "ConvLSTMBaseBN", "ConvLSTMBaseAttn",
        "ConvLSTMBaseBNAttn", "ConvLSTMRemoraBase",
    }
    if model_name not in no_feature_models and "num_features" in config:
        model_kwargs["num_features"] = config["num_features"]

    model = get_model(model_name, **model_kwargs)

    # Load weights
    checkpoint_path = model_dir / "model_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
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


def _expected_calibration_error(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> float:
    """Compute Expected Calibration Error (ECE)."""
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
