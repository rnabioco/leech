"""
Feature importance analysis using gradient-based methods.

Compute feature importance scores to understand which input features
(signal, sequence, dwell times, signal levels) contribute most to predictions.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger("leech.analysis.feature_importance")


def compute_feature_importance(
    model_path: Path,
    test_data_path: Path,
    output_path: Path,
    device: str = "cuda",
    method: str = "gradient",
) -> dict[str, np.ndarray]:
    """
    Compute feature importance using gradient-based methods.

    Args:
        model_path: Path to trained model checkpoint
        test_data_path: Path to test dataset
        output_path: Path to save importance scores
        device: Device for computation
        method: Method to use ("gradient", "integrated_gradients")

    Returns:
        Dictionary with importance scores for each input type
    """
    from leech.dataset import LeechDataset, collate_fn
    from leech.util import load_model_from_checkpoint

    logger.info(f"Computing feature importance using {method} method")

    # Load model
    model, model_info = load_model_from_checkpoint(model_path)
    model.to(device)
    model.eval()

    # Load test data
    test_dataset = LeechDataset(
        test_data_path,
        signal_len=model_info["signal_len"],
        kmer_len=model_info["kmer_len"],
        model_type=model_info["model_name"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Compute importance scores
    if method == "gradient":
        importance = _gradient_importance(model, test_loader, device, model_info["model_name"])
    elif method == "integrated_gradients":
        importance = _integrated_gradients(model, test_loader, device, model_info["model_name"])
    else:
        raise ValueError(f"Unknown method: {method}")

    # Save results
    np.savez(output_path, **importance)  # type: ignore[arg-type]
    logger.info(f"Feature importance saved to {output_path}")

    return importance


def _gradient_importance(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    model_type: str,
) -> dict[str, np.ndarray]:
    """
    Compute importance using gradient magnitudes.

    Args:
        model: Trained model
        data_loader: Test data loader
        device: Device for computation
        model_type: Model architecture name

    Returns:
        Dictionary with gradient-based importance scores
    """
    from leech.dataset import SIGNAL_FEATURES_MODELS

    signal_grads = []
    sequence_grads = []
    features_grads = []

    for batch in data_loader:
        signal = batch["signal"].to(device).requires_grad_(True)

        # Handle different model types
        if model_type in SIGNAL_FEATURES_MODELS:
            # Signal + Features models
            features = batch["features"].to(device).requires_grad_(True)
            logits = model(signal, features)

            # Backward pass
            loss = logits.sum()
            loss.backward()

            signal_grads.append(signal.grad.abs().cpu().numpy())
            features_grads.append(features.grad.abs().cpu().numpy())

        else:
            # Models with sequence branch
            sequence = batch["sequence"].to(device).requires_grad_(True)

            if "features" in batch:
                # Signal + Sequence + Features
                features = batch["features"].to(device).requires_grad_(True)
                logits = model(signal, sequence, features)
            else:
                # Signal + Sequence only
                logits = model(signal, sequence)

            # Backward pass
            loss = logits.sum()
            loss.backward()

            signal_grads.append(signal.grad.abs().cpu().numpy())
            sequence_grads.append(sequence.grad.abs().cpu().numpy())

            if "features" in batch:
                features_grads.append(features.grad.abs().cpu().numpy())

    # Aggregate importance scores
    importance = {
        "signal_importance": np.concatenate(signal_grads, axis=0).mean(axis=0),
    }

    if sequence_grads:
        importance["sequence_importance"] = np.concatenate(sequence_grads, axis=0).mean(axis=0)

    if features_grads:
        importance["features_importance"] = np.concatenate(features_grads, axis=0).mean(axis=0)

    return importance


def _integrated_gradients(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    model_type: str,
    steps: int = 50,
) -> dict[str, np.ndarray]:
    """
    Compute importance using integrated gradients.

    Integrated gradients provides better attribution by integrating
    gradients along the path from a baseline to the input.

    Args:
        model: Trained model
        data_loader: Test data loader
        device: Device for computation
        model_type: Model architecture name
        steps: Number of integration steps

    Returns:
        Dictionary with integrated gradient importance scores
    """
    from leech.dataset import SIGNAL_FEATURES_MODELS

    signal_ig = []

    for batch in data_loader:
        signal = batch["signal"].to(device)

        # Create baseline (zeros)
        signal_baseline = torch.zeros_like(signal)

        # Interpolate between baseline and input
        alphas = torch.linspace(0, 1, steps).to(device)

        signal_grads_along_path = []

        for alpha in alphas:
            signal_interp = signal_baseline + alpha * (signal - signal_baseline)
            signal_interp.requires_grad_(True)

            if model_type in SIGNAL_FEATURES_MODELS:
                features = batch["features"].to(device)
                features_baseline = torch.zeros_like(features)
                features_interp = features_baseline + alpha * (features - features_baseline)
                features_interp.requires_grad_(True)

                logits = model(signal_interp, features_interp)
            else:
                sequence = batch["sequence"].to(device)
                sequence_baseline = torch.zeros_like(sequence)
                sequence_interp = sequence_baseline + alpha * (sequence - sequence_baseline)
                sequence_interp.requires_grad_(True)

                if "features" in batch:
                    features = batch["features"].to(device)
                    features_baseline = torch.zeros_like(features)
                    features_interp = features_baseline + alpha * (features - features_baseline)
                    features_interp.requires_grad_(True)
                    logits = model(signal_interp, sequence_interp, features_interp)
                else:
                    logits = model(signal_interp, sequence_interp)

            # Backward pass
            loss = logits.sum()
            loss.backward()

            signal_grads_along_path.append(signal_interp.grad.cpu().numpy())

        # Integrate gradients
        signal_integrated = np.array(signal_grads_along_path).mean(axis=0)
        signal_ig.append((signal.cpu().numpy() - signal_baseline.cpu().numpy()) * signal_integrated)

    # Aggregate importance scores
    importance = {
        "signal_importance": np.abs(np.concatenate(signal_ig, axis=0)).mean(axis=0),
    }

    return importance
