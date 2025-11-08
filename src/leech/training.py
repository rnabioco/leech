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
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from leech.dataset import LeechDataset, collate_fn
from leech.models import get_model

logger = logging.getLogger("leech.training")


class Trainer:
    """
    Trainer for leech models.

    Handles training loop, validation, checkpointing, and metrics logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: str = "cuda",
        learning_rate: float = 0.001,
        output_dir: Path | None = None,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Device for training ("cuda" or "cpu")
            learning_rate: Learning rate for optimizer
            output_dir: Directory for saving models and logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir

        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

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

    def train_epoch(self) -> tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        all_preds: list[float] = []
        all_labels: list[float] = []

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            # Move to device
            signal = batch["signal"].to(self.device)
            sequence = batch["sequence"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if "features" in batch:
                # ConvLSTMDwell
                features = batch["features"].to(self.device)
                logits = self.model(signal, sequence, features)
            else:
                # ConvLSTMBase
                logits = self.model(signal, sequence)

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

        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        all_preds_binary = (np.array(all_preds) > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, all_preds_binary)

        return avg_loss, accuracy

    def validate(self) -> tuple[float, float, float]:
        """
        Validate model.

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
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move to device
                signal = batch["signal"].to(self.device)
                sequence = batch["sequence"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward pass
                if "features" in batch:
                    features = batch["features"].to(self.device)
                    logits = self.model(signal, sequence, features)
                else:
                    logits = self.model(signal, sequence)

                # Compute loss
                loss = self.criterion(logits, labels)

                # Track metrics
                total_loss += loss.item()
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

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
            early_stopping_patience: Stop if no improvement for N epochs

        Returns:
            Training history dictionary
        """
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")

            # Train
            train_loss, train_acc = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

            # Validate
            if self.val_loader is not None:
                val_loss, val_acc, val_auc = self.validate()
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history["val_auc"].append(val_auc)

                logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

                # Save best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    patience_counter = 0

                    if self.output_dir:
                        self.save_checkpoint("model_best.pt")
                        logger.info(f"Saved best model (val_acc: {val_acc:.4f})")
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

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

        # Save summary
        summary = {
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "final_train_loss": self.history["train_loss"][-1],
            "final_train_acc": self.history["train_acc"][-1],
        }

        if self.val_loader is not None:
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
    seed: int = 42,
    **model_kwargs,
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
        seed: Random seed
        **model_kwargs: Additional model parameters

    Returns:
        Training history dictionary
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create datasets
    train_dataset = LeechDataset(
        train_data_path, signal_len=signal_len, kmer_len=kmer_len, model_type=model_name
    )

    val_dataset = None
    if val_data_path is not None:
        val_dataset = LeechDataset(
            val_data_path, signal_len=signal_len, kmer_len=kmer_len, model_type=model_name
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2
        )

    # Determine num_features from first batch
    first_batch = next(iter(train_loader))
    num_features = first_batch.get("features", torch.zeros(1, 1, kmer_len)).shape[1]

    # Create model
    model = get_model(
        model_name,
        signal_len=signal_len,
        kmer_len=kmer_len,
        num_features=num_features,
        **model_kwargs,
    )

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
        **model_kwargs,
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        output_dir=output_dir,
    )

    # Train
    history = trainer.train(epochs=epochs)

    return history
