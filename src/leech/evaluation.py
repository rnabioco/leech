"""
Model evaluation and testing utilities.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from leech.dataset import LeechDataset, collate_fn
from leech.util import compute_metrics, load_model_from_checkpoint, print_metrics, save_metrics


def evaluate_model(
    model_path: Path,
    test_data_path: Path,
    output_path: Path,
    signal_len: int | None = None,
    kmer_len: int | None = None,
    batch_size: int = 128,
    device: str = "cuda",
) -> dict:
    """
    Evaluate a trained model on test data.

    Args:
        model_path: Path to model checkpoint directory
        test_data_path: Path to test dataset (.npz)
        output_path: Path to save evaluation metrics (JSON)
        signal_len: Signal length (if None, read from model config)
        kmer_len: K-mer length (if None, read from model config)
        batch_size: Batch size for evaluation
        device: Device for inference

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Loading model from {model_path}")

    # Load model and config
    model, config = load_model_from_checkpoint(model_path, device=device)

    # Use config values if not provided
    if signal_len is None:
        signal_len = config["signal_len"]
    if kmer_len is None:
        kmer_len = config["kmer_len"]

    model_type = config["model_name"]

    print(f"Model: {model_type}")
    print(f"Signal length: {signal_len}")
    print(f"K-mer length: {kmer_len}")

    # Load test dataset
    print(f"\nLoading test data from {test_data_path}")
    test_dataset = LeechDataset(
        test_data_path, signal_len=signal_len, kmer_len=kmer_len, model_type=model_type
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2
    )

    print(f"Test samples: {len(test_dataset)}")

    # Run evaluation
    print("\nRunning evaluation...")
    all_labels = []
    all_probs = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move to device
            signal = batch["signal"].to(device)
            sequence = batch["sequence"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            if "features" in batch:
                features = batch["features"].to(device)
                logits = model(signal, sequence, features)
            else:
                logits = model(signal, sequence)

            # Get predictions
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())

    # Compute metrics
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    metrics = compute_metrics(all_labels, all_preds, all_probs)

    # Add metadata
    metrics["model_path"] = str(model_path)
    metrics["test_data_path"] = str(test_data_path)
    metrics["num_samples"] = len(all_labels)
    metrics["model_type"] = model_type

    # Print and save
    print_metrics(metrics)
    save_metrics(metrics, output_path)

    return metrics
