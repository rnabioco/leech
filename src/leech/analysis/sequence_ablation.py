"""
Sequence ablation testing.

Test model performance when the sequence branch is removed or randomized
to quantify the contribution of sequence information.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger("leech.analysis.sequence_ablation")


def test_sequence_ablation(
    model_path: Path,
    test_data_path: Path,
    output_path: Path,
    device: str = "cuda",
) -> dict:
    """
    Test model performance with sequence ablation.

    Compares:
    1. Normal performance (with sequence)
    2. Zero-sequence performance (all zeros)
    3. Random-sequence performance (random bases)

    Args:
        model_path: Path to trained model
        test_data_path: Path to test dataset
        output_path: Path to save results
        device: Device for computation

    Returns:
        Dictionary with ablation results
    """
    from leech.dataset import SIGNAL_FEATURES_MODELS, LeechDataset, collate_fn
    from leech.util import load_model_from_checkpoint

    logger.info("Running sequence ablation test...")

    # Load model
    model, model_info = load_model_from_checkpoint(model_path)
    model.to(device)
    model.eval()

    model_type = model_info["model_name"]

    # Check if model uses sequence
    if model_type in SIGNAL_FEATURES_MODELS:
        logger.warning(f"{model_type} does not use sequence input - ablation not applicable")
        return {"error": "Model does not use sequence input"}

    # Load test data
    test_dataset = LeechDataset(
        test_data_path,
        signal_len=model_info["signal_len"],
        kmer_len=model_info["kmer_len"],
        model_type=model_type,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Test conditions
    conditions = {
        "normal": lambda seq: seq,  # Original sequence
        "zeros": lambda seq: torch.zeros_like(seq),  # All zeros
        "random": lambda seq: _random_sequence(seq),  # Random bases
    }

    results = {}

    for condition_name, transform_fn in conditions.items():
        logger.info(f"Testing condition: {condition_name}")

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                signal = batch["signal"].to(device)
                sequence = batch["sequence"].to(device)
                labels = batch["label"].to(device)

                # Apply transformation
                sequence_transformed = transform_fn(sequence)

                # Forward pass
                if "features" in batch:
                    features = batch["features"].to(device)
                    logits = model(signal, sequence_transformed, features)
                else:
                    logits = model(signal, sequence_transformed)

                preds = torch.sigmoid(logits) > 0.5

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Compute metrics
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        results[condition_name] = {
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
            "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        }

    # Compute performance drops
    for condition_name in ["zeros", "random"]:
        drop = {}
        for metric in ["accuracy", "precision", "recall", "f1"]:
            normal_val = results["normal"][metric]
            ablated_val = results[condition_name][metric]
            drop[f"{metric}_drop"] = normal_val - ablated_val
            drop[f"{metric}_drop_pct"] = (
                100 * (normal_val - ablated_val) / normal_val if normal_val > 0 else 0.0
            )
        results[f"{condition_name}_drop"] = drop

    # Save results
    import json

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Ablation results saved to {output_path}")

    # Print summary
    _print_ablation_summary(results)

    return results


def _random_sequence(sequence_tensor: torch.Tensor) -> torch.Tensor:
    """
    Generate random one-hot encoded sequences.

    Args:
        sequence_tensor: Original sequence tensor (batch, 4, kmer_len)

    Returns:
        Random sequence tensor with same shape
    """
    batch_size, num_bases, kmer_len = sequence_tensor.shape
    device = sequence_tensor.device

    # Generate random base indices (0-3 for A, C, G, T)
    random_indices = torch.randint(0, num_bases, (batch_size, kmer_len), device=device)

    # Convert to one-hot
    random_seq = torch.nn.functional.one_hot(random_indices, num_classes=num_bases).float()

    # Transpose to (batch, 4, kmer_len)
    random_seq = random_seq.transpose(1, 2)

    return random_seq


def _print_ablation_summary(results: dict) -> None:
    """Print formatted summary of ablation results."""
    print("\nSequence Ablation Test Results")
    print("=" * 80)
    print(f"\n{'Condition':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 80)

    for condition in ["normal", "zeros", "random"]:
        if condition in results:
            r = results[condition]
            print(
                f"{condition:<15} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
                f"{r['recall']:>10.4f} {r['f1']:>10.4f}"
            )

    print("\nPerformance Drops:")
    print("-" * 80)

    for condition in ["zeros", "random"]:
        drop_key = f"{condition}_drop"
        if drop_key in results:
            d = results[drop_key]
            print(f"\n{condition.upper()} ablation:")
            print(f"  Accuracy drop: {d['accuracy_drop']:.4f} ({d['accuracy_drop_pct']:.2f}%)")
            print(f"  F1 drop:       {d['f1_drop']:.4f} ({d['f1_drop_pct']:.2f}%)")

    print()
