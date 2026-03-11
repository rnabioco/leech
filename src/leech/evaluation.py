"""
Model evaluation and testing utilities.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from rich.progress import Progress
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
)
from sklearn.metrics import (
    f1_score as sk_f1_score,
)
from torch.utils.data import DataLoader

from leech.dataset import LeechDataset, collate_fn
from leech.models.inference_wrapper import ModelInferenceWrapper
from leech.util import compute_metrics, load_model_from_checkpoint, print_metrics, save_metrics

logger = logging.getLogger("leech.evaluation")


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
    logger.info(f"Loading model from {model_path}")

    # Load model and config
    model, config = load_model_from_checkpoint(model_path, device=device)

    # Use config values if not provided
    if signal_len is None:
        signal_len = config["signal_len"]
    if kmer_len is None:
        kmer_len = config["kmer_len"]

    model_type = config["model_name"]
    num_out = config.get("num_out", 1)
    left_context = config.get("left_context")
    right_context = config.get("right_context")
    seq_encoding = config.get("seq_encoding", "signal_kmer")
    dwell_offset = config.get("dwell_offset", 0)
    signal_kmer_context = tuple(config.get("signal_kmer_context", (4, 4)))

    # Wrap model for unified forward pass
    model_wrapper = ModelInferenceWrapper(model, model_type)

    logger.info(f"Model: {model_type}")
    logger.info(f"Signal length: {signal_len}")
    logger.info(f"K-mer length: {kmer_len}")
    logger.info(f"seq_encoding: {seq_encoding}, dwell_offset: {dwell_offset}")

    # Load test dataset
    logger.info(f"\nLoading test data from {test_data_path}")
    test_dataset = LeechDataset(
        test_data_path,
        signal_len=signal_len,
        kmer_len=kmer_len,
        model_type=model_type,
        dwell_offset=dwell_offset,
        left_context=left_context,
        right_context=right_context,
        seq_encoding=seq_encoding,
        signal_kmer_context=signal_kmer_context,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # Run evaluation
    logger.info("\nRunning evaluation...")

    is_multiclass = num_out > 2

    if is_multiclass:
        all_labels_int: list[int] = []
        all_preds_int: list[int] = []
        all_logits_list: list[np.ndarray] = []

        model.eval()
        with torch.no_grad():
            with Progress() as progress:
                task = progress.add_task("[cyan]Evaluating...", total=len(test_loader))
                for batch in test_loader:
                    labels = batch["label"].to(device)
                    logits = model_wrapper.forward_batch(batch, device)

                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    all_preds_int.extend(preds.flatten().tolist())
                    all_labels_int.extend(labels.cpu().numpy().flatten().tolist())
                    all_logits_list.append(logits.cpu().numpy())
                    progress.update(task, advance=1)

        all_labels_arr = np.array(all_labels_int)
        all_preds_arr = np.array(all_preds_int)

        accuracy = accuracy_score(all_labels_arr, all_preds_arr)
        macro_f1 = sk_f1_score(all_labels_arr, all_preds_arr, average="macro", zero_division=0.0)
        weighted_f1 = sk_f1_score(
            all_labels_arr, all_preds_arr, average="weighted", zero_division=0.0
        )
        cm = sk_confusion_matrix(all_labels_arr, all_preds_arr).tolist()

        # Per-class report
        report = classification_report(
            all_labels_arr, all_preds_arr, output_dict=True, zero_division=0.0
        )

        # Top-K accuracy (K=3, 5)
        all_logits_combined = np.concatenate(all_logits_list, axis=0)
        top3_acc = _topk_accuracy(all_logits_combined, all_labels_arr, k=3)
        top5_acc = _topk_accuracy(all_logits_combined, all_labels_arr, k=5)

        metrics: dict = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "top3_accuracy": top3_acc,
            "top5_accuracy": top5_acc,
            "confusion_matrix": cm,
            "classification_report": report,
            "num_classes": num_out,
        }

        # Add metadata
        metrics["model_path"] = str(model_path)
        metrics["test_data_path"] = str(test_data_path)
        metrics["num_samples"] = len(all_labels_arr)
        metrics["model_type"] = model_type

        # Print summary
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Macro F1: {macro_f1:.4f}")
        logger.info(f"Weighted F1: {weighted_f1:.4f}")
        logger.info(f"Top-3 accuracy: {top3_acc:.4f}")
        logger.info(f"Top-5 accuracy: {top5_acc:.4f}")

        save_metrics(metrics, output_path)
        return metrics

    # Binary evaluation path
    all_labels: list[float] = []
    all_probs: list[float] = []
    all_preds: list[int] = []

    model.eval()
    with torch.no_grad():
        with Progress() as progress:
            task = progress.add_task("[cyan]Evaluating...", total=len(test_loader))

            for batch in test_loader:
                # Move labels to device
                labels = batch["label"].to(device)

                # Forward pass (wrapper handles moving tensors and calling model correctly)
                logits = model_wrapper.forward_batch(batch, device)

                # Get predictions
                if num_out > 1:
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                else:
                    probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_labels.extend(labels.cpu().numpy().flatten())
                all_probs.extend(probs.flatten())
                all_preds.extend(preds.flatten())

                progress.update(task, advance=1)

    # Compute metrics
    all_labels_arr = np.array(all_labels)
    all_probs_arr = np.array(all_probs)
    all_preds_arr = np.array(all_preds)

    metrics = compute_metrics(all_labels_arr, all_preds_arr, all_probs_arr)

    # Add metadata
    metrics["model_path"] = str(model_path)
    metrics["test_data_path"] = str(test_data_path)
    metrics["num_samples"] = len(all_labels_arr)
    metrics["model_type"] = model_type

    # Print and save
    print_metrics(metrics)
    save_metrics(metrics, output_path)

    return metrics


def _topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Compute top-K accuracy from logits and integer labels."""
    if logits.shape[1] < k:
        k = logits.shape[1]
    top_k_preds = np.argsort(logits, axis=1)[:, -k:]
    correct = np.array([labels[i] in top_k_preds[i] for i in range(len(labels))])
    return float(correct.mean())
