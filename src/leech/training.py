"""
Training loop and utilities for leech models.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from leech.dataset import LeechDataset, collate_fn
from leech.losses import FocalBCEWithLogitsLoss
from leech.models import get_model
from leech.models.inference_wrapper import ModelInferenceWrapper

logger = logging.getLogger("leech.training")
console = Console()


def compute_class_weights(dataset: LeechDataset) -> torch.Tensor | None:
    """
    Compute pos_weight for BCEWithLogitsLoss from dataset.

    Args:
        dataset: Dataset with binary labels (0/1)

    Returns:
        pos_weight tensor for BCEWithLogitsLoss, or None if balanced
    """
    # Count class occurrences
    labels = []
    for i in range(len(dataset)):
        chunk = dataset.chunks[i]
        labels.append(chunk["label_int"])

    labels_array = np.array(labels)
    unique, counts = np.unique(labels_array, return_counts=True)

    if len(unique) != 2:
        logger.warning(f"Expected 2 classes, found {len(unique)}. Skipping class weighting.")
        return None

    # Count negatives (0) and positives (1)
    label_counts = dict(zip(unique, counts, strict=True))
    neg_count = label_counts.get(0, 0)
    pos_count = label_counts.get(1, 0)

    if pos_count == 0:
        logger.warning("No positive samples found. Skipping class weighting.")
        return None

    # Calculate pos_weight = neg_count / pos_count
    pos_weight = neg_count / pos_count

    logger.info(f"Class distribution: negative={neg_count}, positive={pos_count}")
    logger.info(f"Using pos_weight={pos_weight:.4f} for class weighting")

    return torch.tensor([pos_weight], dtype=torch.float32)


class Trainer:
    """
    Trainer for leech models.

    Handles training loop, validation, checkpointing, and metrics logging.
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: str = "cuda",
        learning_rate: float = 0.001,
        output_dir: Path | None = None,
        pos_weight: torch.Tensor | None = None,
        weight_decay: float = 0.0,
        max_grad_norm: float = 0.0,
        scheduler_type: str = "none",
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        warmup_epochs: int = 0,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
        use_mixed_precision: bool = False,
        resume_checkpoint: Path | None = None,
        num_out: int | None = None,
    ):
        # Wrap model with inference wrapper for unified forward pass
        self.model_wrapper = ModelInferenceWrapper(model, model_type)
        self.model = self.model_wrapper.model  # Keep reference to underlying model
        self.model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.max_grad_norm = max_grad_norm
        self.warmup_epochs = warmup_epochs
        self.base_lr = learning_rate

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Setup loss
        self.loss_type = loss_type
        self._num_out = num_out if num_out is not None else 1
        pw = pos_weight.to(device) if pos_weight is not None else None
        if loss_type == "cross_entropy":
            # CrossEntropyLoss expects (B, num_classes) logits and (B,) integer labels
            if pw is not None and self._num_out <= 2:
                # Convert pos_weight to per-class weights for CE (binary case)
                ce_weights = torch.tensor([1.0, pw.item()], dtype=torch.float32).to(device)
                self.criterion = nn.CrossEntropyLoss(weight=ce_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
            logger.info(f"Using CrossEntropyLoss ({self._num_out}-class)")
        elif loss_type == "focal":
            self.criterion = FocalBCEWithLogitsLoss(gamma=focal_gamma, pos_weight=pw)
            logger.info(f"Using focal loss (gamma={focal_gamma})")
        else:
            if pw is not None:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
                logger.info("Training without class weighting")

        # Setup LR scheduler
        self.scheduler = None
        if scheduler_type == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=scheduler_patience,
                factor=scheduler_factor,
            )
            logger.info(
                f"Using ReduceLROnPlateau scheduler (patience={scheduler_patience}, factor={scheduler_factor})"
            )

        # Mixed precision (only on CUDA)
        self.use_mixed_precision = use_mixed_precision and device != "cpu"
        self.scaler = None
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler("cuda")
            logger.info("Mixed precision training enabled")

        # Track best model
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.start_epoch = 1

        # History
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_auc": [],
            "val_f1": [],
        }

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resume from checkpoint (skip if file doesn't exist)
        if resume_checkpoint is not None:
            if resume_checkpoint.exists():
                self._resume_from_checkpoint(resume_checkpoint)
            else:
                logger.info(f"Checkpoint not found, starting fresh: {resume_checkpoint}")

    def _resume_from_checkpoint(self, checkpoint_path: Path) -> None:
        """Restore training state from a checkpoint."""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.best_val_f1 = checkpoint.get("best_val_f1", 0.0)
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.start_epoch = checkpoint.get("epoch", 0) + 1

        # Restore scheduler state if available
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore scaler state if available
        if self.scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(
            f"Resumed from epoch {self.start_epoch - 1} "
            f"(best_val_acc={self.best_val_acc:.4f}, best_val_f1={self.best_val_f1:.4f} "
            f"at epoch {self.best_epoch})"
        )

    def train_epoch(
        self, progress: Progress | None = None, task_id: TaskID | None = None
    ) -> tuple[float, float]:
        """
        Train for one epoch.

        Args:
            progress: Rich Progress instance (optional)
            task_id: Progress task ID (optional)

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        all_preds: list[float] = []
        all_labels: list[float] = []

        for batch in self.train_loader:
            # Move labels to device
            labels = batch["label"].to(self.device)

            # Adapt labels for CrossEntropyLoss
            if self.loss_type == "cross_entropy":
                ce_labels = labels.squeeze(-1).long()
            else:
                ce_labels = None

            # Forward pass (wrapper handles moving tensors and calling model correctly)
            self.optimizer.zero_grad()

            if self.use_mixed_precision:
                with torch.amp.autocast("cuda"):
                    logits = self.model_wrapper.forward_batch(batch, self.device)
                    if ce_labels is not None:
                        loss = self.criterion(logits, ce_labels)
                    else:
                        loss = self.criterion(logits, labels)

                # Scaled backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping with mixed precision
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model_wrapper.forward_batch(batch, self.device)
                if ce_labels is not None:
                    loss = self.criterion(logits, ce_labels)
                else:
                    loss = self.criterion(logits, labels)

                loss.backward()

                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            if self.loss_type == "cross_entropy" and self._num_out > 2:
                # Multi-class: argmax predictions
                preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                all_preds.extend(preds.flatten())
            elif self.loss_type == "cross_entropy":
                # Binary CE: probabilities via softmax, take class 1
                probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                all_preds.extend(probs.flatten())
            else:
                preds = torch.sigmoid(logits).detach().cpu().numpy()
                all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

            # Update progress if provided
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)

        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        if self._num_out > 2:
            # Multi-class: preds are already class indices
            accuracy = accuracy_score(all_labels, all_preds)
        else:
            all_preds_binary = (np.array(all_preds) > 0.5).astype(int)
            accuracy = accuracy_score(all_labels, all_preds_binary)

        return avg_loss, accuracy

    def validate(
        self, progress: Progress | None = None, task_id: TaskID | None = None
    ) -> tuple[float, float, float, float]:
        """
        Validate model.

        Args:
            progress: Rich Progress instance (optional)
            task_id: Progress task ID (optional)

        Returns:
            Tuple of (average_loss, accuracy, roc_auc, f1)
        """
        if self.val_loader is None:
            return 0.0, 0.0, 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        all_preds: list[float] = []
        all_labels: list[float] = []

        with torch.inference_mode():
            for batch in self.val_loader:
                # Move labels to device
                labels = batch["label"].to(self.device)

                # Adapt labels for CrossEntropyLoss
                if self.loss_type == "cross_entropy":
                    ce_labels = labels.squeeze(-1).long()
                else:
                    ce_labels = None

                # Forward pass (wrapper handles moving tensors and calling model correctly)
                if self.use_mixed_precision:
                    with torch.amp.autocast("cuda"):
                        logits = self.model_wrapper.forward_batch(batch, self.device)
                        if ce_labels is not None:
                            loss = self.criterion(logits, ce_labels)
                        else:
                            loss = self.criterion(logits, labels)
                else:
                    logits = self.model_wrapper.forward_batch(batch, self.device)
                    if ce_labels is not None:
                        loss = self.criterion(logits, ce_labels)
                    else:
                        loss = self.criterion(logits, labels)

                # Track metrics
                total_loss += loss.item()
                if self.loss_type == "cross_entropy" and self._num_out > 2:
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    all_preds.extend(preds.flatten())
                elif self.loss_type == "cross_entropy":
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    all_preds.extend(probs.flatten())
                else:
                    preds = torch.sigmoid(logits).cpu().numpy()
                    all_preds.extend(preds.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

                # Update progress if provided
                if progress is not None and task_id is not None:
                    progress.update(task_id, advance=1)

        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        if self._num_out > 2:
            # Multi-class: preds are already class indices
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0.0)
            auc = 0.0  # ROC AUC needs probability estimates per class
        else:
            all_preds_binary = (np.array(all_preds) > 0.5).astype(int)
            accuracy = accuracy_score(all_labels, all_preds_binary)
            auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0
            f1 = f1_score(all_labels, all_preds_binary, zero_division=0.0)

        return avg_loss, accuracy, auc, f1

    def train(self, epochs: int, early_stopping_patience: int = 10) -> dict[str, Any]:
        """
        Train model for multiple epochs.

        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Stop if validation accuracy doesn't improve for N epochs (0 to disable)

        Returns:
            Training history dictionary
        """
        patience_counter = 0
        total_epochs = max(0, epochs - self.start_epoch + 1)
        last_epoch = self.start_epoch - 1

        # If training already completed (resume past final epoch), save best and exit
        if self.start_epoch > epochs:
            logger.info(
                f"Training already complete (resumed at epoch {self.start_epoch - 1}, "
                f"requested {epochs}). Saving checkpoints and exiting."
            )
            if self.output_dir:
                self.save_checkpoint("model_best.pt", epoch=self.best_epoch)
                self.save_checkpoint("model_last.pt", epoch=self.start_epoch - 1)
                self.save_history()
            return self.history

        # Create progress bars
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            epoch_task = progress.add_task("[cyan]Training epochs...", total=total_epochs)

            for epoch in range(self.start_epoch, epochs + 1):
                last_epoch = epoch
                # LR warmup
                if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                    warmup_lr = self.base_lr * (epoch / self.warmup_epochs)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = warmup_lr

                # Create tasks for this epoch
                train_task = progress.add_task(
                    f"[green]Epoch {epoch}/{epochs} - Training", total=len(self.train_loader)
                )

                # Train
                train_loss, train_acc = self.train_epoch(progress, train_task)
                self.history["train_loss"].append(train_loss)
                self.history["train_acc"].append(train_acc)

                progress.remove_task(train_task)

                # Validate
                if self.val_loader is not None:
                    val_task = progress.add_task(
                        f"[yellow]Epoch {epoch}/{epochs} - Validation", total=len(self.val_loader)
                    )
                    val_loss, val_acc, val_auc, val_f1 = self.validate(progress, val_task)
                    self.history["val_loss"].append(val_loss)
                    self.history["val_acc"].append(val_acc)
                    self.history["val_auc"].append(val_auc)
                    self.history["val_f1"].append(val_f1)

                    progress.remove_task(val_task)

                    # LR scheduler step (only after warmup)
                    if self.scheduler is not None and epoch > self.warmup_epochs:
                        old_lr = self.optimizer.param_groups[0]["lr"]
                        self.scheduler.step(val_loss)
                        new_lr = self.optimizer.param_groups[0]["lr"]
                        if new_lr != old_lr:
                            logger.info(f"LR reduced: {old_lr:.6f} -> {new_lr:.6f}")

                    # Display metrics
                    lr_str = ""
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    if current_lr != self.base_lr:
                        lr_str = f" LR: {current_lr:.6f}"
                    console.print(
                        f"[cyan]Epoch {epoch}/{epochs}[/cyan] | "
                        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
                        f"F1: {val_f1:.4f} AUC: {val_auc:.4f}"
                        f"{lr_str}"
                    )

                    # Save best model
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        self.best_val_f1 = val_f1
                        self.best_epoch = epoch
                        patience_counter = 0

                        if self.output_dir:
                            self.save_checkpoint("model_best.pt", epoch=epoch)
                            console.print(
                                f"[bold green]✓ Saved best model "
                                f"(val_acc: {val_acc:.4f}, val_f1: {val_f1:.4f})[/bold green]"
                            )
                    else:
                        patience_counter += 1

                    # Early stopping (disabled if patience is 0)
                    if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                        console.print(f"[yellow]Early stopping at epoch {epoch}[/yellow]")
                        break
                else:
                    console.print(
                        f"[cyan]Epoch {epoch}/{epochs}[/cyan] | "
                        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}"
                    )

                progress.update(epoch_task, advance=1)

        # Save final model
        if self.output_dir:
            self.save_checkpoint("model_last.pt", epoch=last_epoch)
            self.save_history()

        return self.history

    def save_checkpoint(self, filename: str, epoch: int = 0) -> None:
        """Save model checkpoint."""
        if self.output_dir is None:
            return

        checkpoint_path = self.output_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "best_val_f1": self.best_val_f1,
            "best_epoch": self.best_epoch,
            "epoch": epoch,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
        }
        torch.save(checkpoint, checkpoint_path)

    def save_history(self) -> None:
        """Save training history to JSON."""
        if self.output_dir is None:
            return

        history_path = self.output_dir / "metrics.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # Save summary (only if training occurred)
        if len(self.history["train_loss"]) == 0:
            return

        summary = {
            "best_val_acc": self.best_val_acc,
            "best_val_f1": self.best_val_f1,
            "best_epoch": self.best_epoch,
            "final_train_loss": self.history["train_loss"][-1],
            "final_train_acc": self.history["train_acc"][-1],
        }

        if self.val_loader is not None and len(self.history["val_loss"]) > 0:
            summary.update(
                {
                    "final_val_loss": self.history["val_loss"][-1],
                    "final_val_acc": self.history["val_acc"][-1],
                    "final_val_auc": self.history["val_auc"][-1],
                    "final_val_f1": self.history["val_f1"][-1],
                }
            )

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)


def train_model(
    train_data_path: Path,
    val_data_path: Path | None,
    model_name: str,
    output_dir: Path,
    signal_len: int = 400,
    kmer_len: int = 11,
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    device: str = "cuda",
    seed: int | None = None,
    early_stopping_patience: int = 10,
    use_class_weights: bool = True,
    pos_weight: float | None = None,
    train_chunks: list[dict] | None = None,
    val_chunks: list[dict] | None = None,
    weight_decay: float = 0.0,
    max_grad_norm: float = 0.0,
    scheduler: str = "none",
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    warmup_epochs: int = 0,
    loss_type: str = "bce",
    focal_gamma: float = 2.0,
    mixed_precision: bool = False,
    augment_jitter: float = 0.0,
    augment_scale_min: float = 1.0,
    augment_scale_max: float = 1.0,
    resume_from: Path | None = None,
    num_workers: int = 0,
    motif: str | None = None,
    motif_offset: int = 0,
    base_justify: str = "center",
    seq_encoding: str = "signal_kmer",
    signal_kmer_context: tuple[int, int] = (4, 4),
    left_context: int | None = None,
    right_context: int | None = None,
    balance_groups: bool = False,
    label_map: dict[str, int] | None = None,
    num_out: int = 1,
    **model_kwargs: Any,
) -> dict[str, Any]:
    """
    High-level training function.

    Args:
        train_data_path: Path to training chunks (.npz)
        val_data_path: Path to validation chunks (.npz)
        model_name: Model architecture name
        output_dir: Output directory for models and logs
        signal_len: Signal length for model input
        kmer_len: K-mer length for model input
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device for training
        seed: Random seed (None = generate random seed)
        early_stopping_patience: Stop training if validation loss doesn't improve for N epochs
        use_class_weights: Auto-compute class weights from training data (default: True)
        pos_weight: Manual positive class weight (overrides use_class_weights if provided)
        train_chunks: Pre-loaded training chunks (skips loading from train_data_path)
        val_chunks: Pre-loaded validation chunks (skips loading from val_data_path)
        weight_decay: L2 regularization weight (0 = disabled)
        max_grad_norm: Max gradient norm for clipping (0 = disabled)
        scheduler: LR scheduler type ("none" or "reduce_on_plateau")
        scheduler_patience: Epochs to wait before reducing LR
        scheduler_factor: Factor to reduce LR by
        warmup_epochs: Number of LR warmup epochs (0 = disabled)
        loss_type: Loss function type ("bce" or "focal")
        focal_gamma: Focal loss gamma parameter
        mixed_precision: Enable mixed precision training (CUDA only)
        augment_jitter: Signal jitter noise std dev (0 = disabled)
        augment_scale_min: Min random scale factor for signal augmentation
        augment_scale_max: Max random scale factor for signal augmentation
        resume_from: Path to checkpoint to resume training from
        motif: Motif used for chunk extraction (recorded in config for provenance)
        motif_offset: Offset within motif for focus base (recorded in config)
        base_justify: Signal justification within focus base (recorded in config)
        **model_kwargs: Additional model parameters (passed to model constructor)

    Returns:
        Training history dictionary with metrics
    """
    from leech.constants import generate_random_seed

    # Generate random seed if not provided
    if seed is None:
        seed = generate_random_seed()
        logger.info(f"Generated random seed: {seed}")
    else:
        logger.info(f"Using provided seed: {seed}")

    # Save seed to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_file = output_dir / "training_seed.txt"
    with open(seed_file, "w") as f:
        f.write(f"{seed}\n")
    logger.info(f"Saved seed to {seed_file}")

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Override signal_len from asymmetric context if both provided
    if left_context is not None and right_context is not None:
        signal_len = left_context + right_context

    # Extract dwell_offset from model_kwargs (grid search param, not model init param)
    dwell_offset = model_kwargs.pop("dwell_offset", 0)

    # Build augmentation config for training dataset
    augmentation = None
    if augment_jitter > 0 or (augment_scale_min, augment_scale_max) != (1.0, 1.0):
        augmentation = {
            "jitter_std": augment_jitter,
            "scale_range": (augment_scale_min, augment_scale_max),
        }
        logger.info(f"Signal augmentation enabled: {augmentation}")

    # Create datasets (use pre-loaded chunks if provided)
    train_dataset = LeechDataset(
        chunk_path=train_data_path,
        signal_len=signal_len,
        kmer_len=kmer_len,
        model_type=model_name,
        dwell_offset=dwell_offset,
        chunks=train_chunks,
        augmentation=augmentation,
        seq_encoding=seq_encoding,
        signal_kmer_context=signal_kmer_context,
        left_context=left_context,
        right_context=right_context,
    )

    val_dataset = None
    if val_chunks is not None or val_data_path is not None:
        val_dataset = LeechDataset(
            chunk_path=val_data_path,
            signal_len=signal_len,
            kmer_len=kmer_len,
            model_type=model_name,
            dwell_offset=dwell_offset,
            chunks=val_chunks,
            seq_encoding=seq_encoding,
            signal_kmer_context=signal_kmer_context,
            left_context=left_context,
            right_context=right_context,
        )

    # Create data loaders
    # Daemon processes (e.g. multiprocessing pool workers) cannot spawn children,
    # so num_workers must be 0. On CPU, workers compete for CPU time with training;
    # with pre-tensorized data __getitem__ is trivially fast, so workers add overhead.
    import multiprocessing

    is_daemon = multiprocessing.current_process().daemon
    if is_daemon:
        effective_workers = 0
    elif num_workers > 0:
        effective_workers = num_workers
    elif device == "cpu":
        effective_workers = 0
    else:
        effective_workers = 8  # auto default for CUDA

    logger.info(
        f"DataLoader workers: {effective_workers} "
        f"(requested={num_workers}, daemon={is_daemon}, device={device})"
    )

    loader_kwargs: dict = {
        "collate_fn": collate_fn,
        "num_workers": effective_workers,
    }
    if device != "cpu":
        loader_kwargs["pin_memory"] = True
    if effective_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    # Cap batch_size so drop_last doesn't discard all data
    effective_batch_size = min(batch_size, len(train_dataset))
    if effective_batch_size < batch_size:
        logger.warning(
            f"Batch size {batch_size} > dataset size {len(train_dataset)}, "
            f"reducing to {effective_batch_size}"
        )

    # Build balanced sampler if requested
    train_sampler = None
    if balance_groups:
        # Compute per-chunk weights so each source group is equally represented
        group_counts: dict[str, int] = {}
        for chunk in train_dataset.chunks:
            sg = chunk.get("source_group") or "unknown"
            group_counts[sg] = group_counts.get(sg, 0) + 1

        if len(group_counts) > 1:
            weights = []
            for chunk in train_dataset.chunks:
                sg = chunk.get("source_group") or "unknown"
                weights.append(1.0 / group_counts[sg])
            train_sampler = WeightedRandomSampler(
                weights, num_samples=len(train_dataset), replacement=True
            )
            logger.info(f"Balanced sampling enabled across {len(group_counts)} source groups:")
            for sg, count in sorted(group_counts.items(), key=lambda x: -x[1]):
                logger.info(f"  {sg}: {count} chunks, weight={1.0 / count:.6f}")
        else:
            logger.warning(
                f"balance_groups enabled but only 1 source group found "
                f"({list(group_counts.keys())}). Falling back to shuffle."
            )

    # Use drop_last=True for training to avoid BatchNorm issues with batch_size=1
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )

    val_loader = None
    if val_dataset is not None:
        # No need to drop last for validation since model is in eval mode
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    # Determine num_features and signal_channels from first batch
    first_batch = next(iter(train_loader))
    num_features = first_batch.get("features", torch.zeros(1, 1, kmer_len)).shape[1]
    signal_shape = first_batch["signal"].shape
    signal_in_channels = signal_shape[1] if len(signal_shape) == 3 else 1

    # Compute class weights if requested
    pos_weight_tensor = None
    if pos_weight is not None:
        # Manual pos_weight provided
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
        logger.info(f"Using manual pos_weight={pos_weight:.4f}")
    elif use_class_weights:
        # Auto-compute from training data
        pos_weight_tensor = compute_class_weights(train_dataset)

    # Create model
    # Only pass num_features to models that need it
    model_init_kwargs = {
        "signal_len": signal_len,
        "kmer_len": kmer_len,
        "seq_encoding": seq_encoding,
        "signal_kmer_context": signal_kmer_context,
        **model_kwargs,
    }
    # Models without a feature branch don't take num_features
    no_feature_models = {
        "ConvLSTMBase",
        "ConvLSTMBaseBN",
        "ConvLSTMBaseAttn",
        "ConvLSTMBaseBNAttn",
        "ConvLSTMRemoraBase",
    }
    if model_name not in no_feature_models:
        model_init_kwargs["num_features"] = num_features

    # Models that accept signal_in_channels for multi-channel signal input
    signal_channels_models = {"TCNDwellResidual"}
    if model_name in signal_channels_models:
        model_init_kwargs["signal_in_channels"] = signal_in_channels

    # Only pass num_out to models whose __init__ accepts it
    num_out_models = {
        "ConvLSTMDwell",
        "ConvLSTMDwellBN",
        "ConvLSTMDwellBNAttn",
        "ConvLSTMRemora",
        "ConvLSTMRemoraBase",
    }
    if model_name in num_out_models:
        model_init_kwargs["num_out"] = num_out

    model = get_model(model_name, **model_init_kwargs)

    # Enable cuDNN autotuner for fixed-size inputs (finds fastest conv algorithms)
    if device != "cpu":
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark enabled")

    # Compile model with torch.compile for graph-level optimizations (PyTorch 2+)
    if device != "cpu" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            logger.info("torch.compile enabled")
        except Exception as e:
            logger.warning(f"torch.compile failed, falling back to eager mode: {e}")

    # Auto-detect cross-entropy models (num_out > 1)
    if num_out > 1 and loss_type != "cross_entropy":
        loss_type = "cross_entropy"
        logger.info(f"Model {model_name} has num_out={num_out}, switching to cross_entropy loss")

    # Introspect dwell margin from raw training data (source of truth for feature width)
    _dwell_margin_left = 0
    _dwell_margin_right = 0
    _raw_chunk = train_dataset.chunks[0]
    _raw_features = _raw_chunk.get("features")
    if _raw_features is not None and _raw_features.ndim > 1 and _raw_features.shape[1] > kmer_len:
        _feat_width = _raw_features.shape[1]
        _dwell_margin_left = int(_raw_chunk.get("dwell_margin_left", (_feat_width - kmer_len) // 2))
        _dwell_margin_right = _feat_width - kmer_len - _dwell_margin_left

    # Save config
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_name": model_name,
        "signal_len": signal_len,
        "kmer_len": kmer_len,
        "num_features": num_features,
        "signal_in_channels": signal_in_channels,
        "dwell_offset": dwell_offset,
        "dwell_margin_left": _dwell_margin_left,
        "dwell_margin_right": _dwell_margin_right,
        "motif": motif,
        "motif_offset": motif_offset,
        "base_justify": base_justify,
        "left_context": left_context,
        "right_context": right_context,
        "seq_encoding": seq_encoding,
        "signal_kmer_context": list(signal_kmer_context),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": device,
        "seed": seed,
        "use_class_weights": use_class_weights,
        "pos_weight": pos_weight
        if pos_weight is not None
        else (pos_weight_tensor.item() if pos_weight_tensor is not None else None),
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "scheduler": scheduler,
        "scheduler_patience": scheduler_patience,
        "scheduler_factor": scheduler_factor,
        "warmup_epochs": warmup_epochs,
        "loss_type": loss_type,
        "num_out": num_out,
        "focal_gamma": focal_gamma,
        "mixed_precision": mixed_precision,
        "augment_jitter": augment_jitter,
        "augment_scale_min": augment_scale_min,
        "augment_scale_max": augment_scale_max,
        "balance_groups": balance_groups,
        "label_map": label_map,
        **model_kwargs,
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create trainer
    trainer = Trainer(
        model=model,
        model_type=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        output_dir=output_dir,
        pos_weight=pos_weight_tensor,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        scheduler_type=scheduler,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        warmup_epochs=warmup_epochs,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        use_mixed_precision=mixed_precision,
        resume_checkpoint=resume_from,
        num_out=num_out,
    )

    # Train
    history = trainer.train(epochs=epochs, early_stopping_patience=early_stopping_patience)

    return history
