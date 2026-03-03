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
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from leech.dataset import LeechDataset, collate_fn
from leech.models import get_model
from leech.models.inference_wrapper import ModelInferenceWrapper

logger = logging.getLogger("leech.training")
console = Console()


def compute_class_weights(dataset: Any) -> torch.Tensor | None:
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
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            model_type: Model architecture name (e.g., "ConvLSTMDwell")
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Device for training ("cuda" or "cpu")
            learning_rate: Learning rate for optimizer
            output_dir: Directory for saving models and logs
            pos_weight: Weight for positive class in BCEWithLogitsLoss (None = no weighting)
        """
        # Wrap model with inference wrapper for unified forward pass
        self.model_wrapper = ModelInferenceWrapper(model, model_type)
        self.model = self.model_wrapper.model  # Keep reference to underlying model
        self.model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir

        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Setup loss with optional class weighting
        if pos_weight is not None:
            pos_weight = pos_weight.to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            logger.info("Training without class weighting")

        # Track best model
        self.best_val_acc = 0.0
        self.best_epoch = 0

        # History
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_auc": [],
        }

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

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

            # Forward pass (wrapper handles moving tensors and calling model correctly)
            self.optimizer.zero_grad()
            logits = self.model_wrapper.forward_batch(batch, self.device)

            # Compute loss
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

            # Update progress if provided
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1)

        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        all_preds_binary = (np.array(all_preds) > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, all_preds_binary)

        return avg_loss, accuracy

    def validate(
        self, progress: Progress | None = None, task_id: TaskID | None = None
    ) -> tuple[float, float, float]:
        """
        Validate model.

        Args:
            progress: Rich Progress instance (optional)
            task_id: Progress task ID (optional)

        Returns:
            Tuple of (average_loss, accuracy, roc_auc)
        """
        if self.val_loader is None:
            return 0.0, 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        all_preds: list[float] = []
        all_labels: list[float] = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Move labels to device
                labels = batch["label"].to(self.device)

                # Forward pass (wrapper handles moving tensors and calling model correctly)
                logits = self.model_wrapper.forward_batch(batch, self.device)

                # Compute loss
                loss = self.criterion(logits, labels)

                # Track metrics
                total_loss += loss.item()
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

                # Update progress if provided
                if progress is not None and task_id is not None:
                    progress.update(task_id, advance=1)

        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        all_preds_binary = (np.array(all_preds) > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, all_preds_binary)
        auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0

        return avg_loss, accuracy, auc

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

        # Create progress bars
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            epoch_task = progress.add_task("[cyan]Training epochs...", total=epochs)

            for epoch in range(1, epochs + 1):
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
                    val_loss, val_acc, val_auc = self.validate(progress, val_task)
                    self.history["val_loss"].append(val_loss)
                    self.history["val_acc"].append(val_acc)
                    self.history["val_auc"].append(val_auc)

                    progress.remove_task(val_task)

                    # Display metrics
                    console.print(
                        f"[cyan]Epoch {epoch}/{epochs}[/cyan] | "
                        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}"
                    )

                    # Save best model
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        self.best_epoch = epoch
                        patience_counter = 0

                        if self.output_dir:
                            self.save_checkpoint("model_best.pt")
                            console.print(
                                f"[bold green]✓ Saved best model (val_acc: {val_acc:.4f})[/bold green]"
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
            self.save_checkpoint("model_last.pt")
            self.save_history()

        return self.history

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        if self.output_dir is None:
            return

        checkpoint_path = self.output_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_acc": self.best_val_acc,
                "best_epoch": self.best_epoch,
            },
            checkpoint_path,
        )

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

    # Extract dwell_offset from model_kwargs (grid search param, not model init param)
    dwell_offset = model_kwargs.pop("dwell_offset", 0)

    # Create datasets (use pre-loaded chunks if provided)
    train_dataset = LeechDataset(
        chunk_path=train_data_path,
        signal_len=signal_len,
        kmer_len=kmer_len,
        model_type=model_name,
        dwell_offset=dwell_offset,
        chunks=train_chunks,
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
        )

    # Create data loaders
    # On CPU, workers compete for CPU time with training; with pre-tensorized
    # data __getitem__ is trivially fast, so workers add overhead without benefit.
    import multiprocessing

    if device == "cpu" or multiprocessing.current_process().daemon:
        num_workers = 0
    else:
        num_workers = 2

    loader_kwargs: dict = {
        "collate_fn": collate_fn,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    # Use drop_last=True for training to avoid BatchNorm issues with batch_size=1
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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

    # Determine num_features from first batch
    first_batch = next(iter(train_loader))
    num_features = first_batch.get("features", torch.zeros(1, 1, kmer_len)).shape[1]

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
    # Remove grid-search context parameters (not model init params)
    model_kwargs.pop("left_context", None)
    model_kwargs.pop("right_context", None)

    # Only pass num_features to models that need it (not ConvLSTMBase)
    model_init_kwargs = {
        "signal_len": signal_len,
        "kmer_len": kmer_len,
        **model_kwargs,
    }
    if model_name != "ConvLSTMBase":
        model_init_kwargs["num_features"] = num_features

    model = get_model(model_name, **model_init_kwargs)

    # Save config
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_name": model_name,
        "signal_len": signal_len,
        "kmer_len": kmer_len,
        "num_features": num_features,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": device,
        "seed": seed,
        "use_class_weights": use_class_weights,
        "pos_weight": pos_weight
        if pos_weight is not None
        else (pos_weight_tensor.item() if pos_weight_tensor is not None else None),
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
    )

    # Train
    history = trainer.train(epochs=epochs, early_stopping_patience=early_stopping_patience)

    return history
