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

from leech.dataset import LeechDataset, collate_fn, resolve_dataloader_workers
from leech.metrics import compute_metrics, print_metrics, save_metrics
from leech.model_loading import load_model_from_checkpoint
from leech.models.inference_wrapper import ModelInferenceWrapper

logger = logging.getLogger("leech.evaluation")


def _save_scores(
    path: Path,
    test_data_path: Path,
    labels: np.ndarray,
    probs: np.ndarray,
) -> None:
    """Write per-chunk scores, keyed by read id where the test set has them.

    Order is the join. The evaluation loader is built with ``shuffle=False``
    and no ``drop_last``, so row i of ``probs`` is chunk i of the test ``.npz``
    and lines up with its ``read_ids``. That assumption is CHECKED rather than
    trusted: a length mismatch means the dataset filtered or reordered chunks,
    which would silently mislabel every score, so it raises instead.
    """
    read_ids = None
    try:
        with np.load(test_data_path, allow_pickle=False) as data:
            if "read_ids" in data:
                read_ids = data["read_ids"]
    except (OSError, ValueError) as exc:  # not an npz, or unreadable
        logger.warning(f"Could not read read_ids from {test_data_path}: {exc}")

    if read_ids is not None and len(read_ids) != len(probs):
        raise ValueError(
            f"Scored {len(probs)} chunks but {test_data_path} holds "
            f"{len(read_ids)} read_ids. Row order is the only key here, so a "
            "mismatch would misattribute every score."
        )

    arrays = {"labels": labels, "probs": probs}
    if read_ids is not None:
        arrays["read_ids"] = read_ids
    else:
        logger.warning(f"{test_data_path} has no read_ids; scores are positional only")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    logger.info(f"Wrote {len(probs)} per-chunk scores to {path}")


def evaluate_model(
    model_path: Path,
    test_data_path: Path,
    output_path: Path,
    signal_len: int | None = None,
    kmer_len: int | None = None,
    batch_size: int = 512,
    device: str = "cuda",
    num_workers: int = 0,
    emit_scores: Path | None = None,
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
        num_workers: DataLoader workers; 0 means auto (see
            ``resolve_dataloader_workers``) -- up to 8 on CUDA, 0 on CPU
        emit_scores: If set, also write per-chunk scores to this .npz. A
            confusion matrix is a summary at ONE threshold; the scores behind
            it answer questions the summary cannot -- per-group error
            breakdowns, paired model comparisons, calibration, and any
            operating point other than the one that was reported. They are
            computed either way, so this only decides whether they survive.

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
    signal_mode = config.get("signal_mode", "both")

    # GPU optimizations (forward-pass only, no training stability concerns)
    if device != "cpu":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if hasattr(torch, "compile"):
        try:
            compile_mode = "reduce-overhead" if device != "cpu" else None
            model = torch.compile(model, mode=compile_mode)
        except Exception as e:
            logger.debug(f"torch.compile failed, using eager mode: {e}")

    # Wrap model for unified forward pass
    model_wrapper = ModelInferenceWrapper(model, model_type)

    logger.info(f"Model: {model_type}")
    logger.info(f"Signal length: {signal_len}")
    logger.info(f"K-mer length: {kmer_len}")
    logger.info(f"seq_encoding: {seq_encoding}, dwell_offset: {dwell_offset}")

    # Load test dataset. Forward dwell_template_table from training config so
    # the feature tensor has the same channel count as training — otherwise
    # the feature-branch Conv1d crashes with a channel mismatch on models
    # that were trained with per-AA dwell templates.
    logger.info(f"\nLoading test data from {test_data_path}")
    dwell_template_table = config.get("dwell_template_table") or None
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
        # The encoding is the trained model's, not a preference: its sequence
        # branch has a fixed channel count, so falling back to base_onehot
        # cannot produce a working run — only a channel-count RuntimeError
        # several steps later, naming neither the corpus nor the encoding (#230).
        allow_encoding_fallback=False,
        signal_mode=signal_mode,
        dwell_template_table=dwell_template_table,
    )

    # Collate, the host-to-device copy and the forward pass run serially in
    # whichever process owns the loader, so a worker-less loader on a GPU means
    # one core feeding an accelerator that then waits (issue #205).
    effective_workers = resolve_dataloader_workers(num_workers, device)
    loader_kwargs: dict = {"num_workers": effective_workers}
    if device != "cpu":
        loader_kwargs["pin_memory"] = True
    if effective_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, **loader_kwargs
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # Run evaluation
    logger.info("\nRunning evaluation...")

    is_multiclass = num_out > 2
    use_autocast = device != "cpu"

    if is_multiclass:
        all_labels_int: list[int] = []
        all_preds_int: list[int] = []
        all_logits_list: list[np.ndarray] = []

        model.eval()
        with torch.inference_mode():
            with Progress() as progress:
                task = progress.add_task("[cyan]Evaluating...", total=len(test_loader))
                for batch in test_loader:
                    labels = batch["label"].to(device)
                    with torch.amp.autocast("cuda", enabled=use_autocast):
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

        # ECE (top-1 confidence)
        logits_t = torch.from_numpy(all_logits_combined).float()
        labels_t = torch.from_numpy(all_labels_arr).long()
        probs_t = torch.softmax(logits_t, dim=-1)
        ece = _expected_calibration_error_multiclass(probs_t, labels_t)
        logger.info(f"ECE (uncalibrated): {ece:.4f}")

        # Calibrated ECE if calibration params exist
        ece_calibrated = None
        from leech.calibration import apply_calibration, load_calibration

        cal_params = load_calibration(model_path)
        if cal_params is not None:
            cal_logits = apply_calibration(logits_t, cal_params)
            cal_probs_t = torch.softmax(cal_logits, dim=-1)
            ece_calibrated = _expected_calibration_error_multiclass(cal_probs_t, labels_t)
            cal_method = cal_params.get("method", "unknown")
            logger.info(f"ECE (calibrated, method={cal_method}): {ece_calibrated:.4f}")

        metrics: dict = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "top3_accuracy": top3_acc,
            "top5_accuracy": top5_acc,
            "ece": ece,
            "confusion_matrix": cm,
            "classification_report": report,
            "num_classes": num_out,
        }
        if ece_calibrated is not None:
            metrics["ece_calibrated"] = ece_calibrated

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

        if emit_scores is not None:
            _save_scores(
                emit_scores,
                test_data_path,
                all_labels_arr,
                torch.softmax(logits_t, dim=-1).numpy(),
            )

        save_metrics(metrics, output_path)
        return metrics

    # Binary evaluation path
    all_labels: list[float] = []
    all_probs: list[float] = []
    all_preds: list[int] = []

    model.eval()
    with torch.inference_mode():
        with Progress() as progress:
            task = progress.add_task("[cyan]Evaluating...", total=len(test_loader))

            for batch in test_loader:
                # Move labels to device
                labels = batch["label"].to(device)

                # Forward pass (wrapper handles moving tensors and calling model correctly)
                with torch.amp.autocast("cuda", enabled=use_autocast):
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

    if emit_scores is not None:
        _save_scores(emit_scores, test_data_path, all_labels_arr, all_probs_arr)

    # Print and save
    print_metrics(metrics)
    save_metrics(metrics, output_path)

    return metrics


def _topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Compute top-K accuracy from logits and integer labels."""
    if logits.shape[1] < k:
        k = logits.shape[1]
    top_k_preds = np.argsort(logits, axis=1)[:, -k:]
    correct = (top_k_preds == labels[:, None]).any(axis=1)
    return float(correct.mean())


def _expected_calibration_error_multiclass(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> float:
    """Compute top-1 confidence ECE for multiclass predictions."""
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
