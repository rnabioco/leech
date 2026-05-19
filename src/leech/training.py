"""
Training loop and utilities for leech models.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
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

import leech
from leech.cli_config import make_console
from leech.dataset import LeechDataset, collate_fn
from leech.losses import AdversarialHead, FocalBCEWithLogitsLoss, RegressionHead
from leech.models import get_model
from leech.models.inference_wrapper import ModelInferenceWrapper

logger = logging.getLogger("leech.training")
console = make_console()


def compute_class_weights(dataset: LeechDataset) -> torch.Tensor | None:
    """
    Compute class weights from dataset label distribution.

    For binary (2-class): returns pos_weight tensor for BCEWithLogitsLoss.
    For multiclass (>2 classes): returns per-class weight tensor (inverse sqrt
    frequency) for CrossEntropyLoss.

    Args:
        dataset: Dataset with integer labels

    Returns:
        Weight tensor, or None if balanced / single class
    """
    # Count class occurrences
    labels = []
    for i in range(len(dataset)):
        chunk = dataset.chunks[i]
        labels.append(chunk["label_int"])

    labels_array = np.array(labels)
    unique, counts = np.unique(labels_array, return_counts=True)

    if len(unique) < 2:
        logger.warning(f"Only {len(unique)} class(es) found. Skipping class weighting.")
        return None

    if len(unique) == 2:
        # Binary: pos_weight = neg_count / pos_count
        label_counts = dict(zip(unique, counts, strict=True))
        neg_count = label_counts.get(0, 0)
        pos_count = label_counts.get(1, 0)

        if pos_count == 0:
            logger.warning("No positive samples found. Skipping class weighting.")
            return None

        pos_weight = neg_count / pos_count
        logger.info(f"Class distribution: negative={neg_count}, positive={pos_count}")
        logger.info(f"Using pos_weight={pos_weight:.4f} for class weighting")
        return torch.tensor([pos_weight], dtype=torch.float32)

    # Multiclass: inverse sqrt frequency weights
    # w_c = sqrt(N / (K * n_c)) — balances rare classes without over-weighting
    num_classes = int(unique.max()) + 1
    total = counts.sum()
    weights = torch.ones(num_classes, dtype=torch.float32)
    for cls, count in zip(unique, counts, strict=True):
        weights[cls] = float(np.sqrt(total / (num_classes * count)))

    logger.info(
        f"Multiclass class weights ({num_classes} classes): "
        f"min={weights.min():.3f}, max={weights.max():.3f}, "
        f"range={weights.max() / weights.min():.1f}x"
    )
    for cls, count in zip(unique, counts, strict=True):
        logger.info(f"  class {cls}: n={count}, weight={weights[cls]:.3f}")

    return weights


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
        label_smoothing: float = 0.0,
        use_mixed_precision: bool = False,
        resume_checkpoint: Path | None = None,
        num_out: int | None = None,
        epochs: int = 50,
        adversarial_lambda: float = 0.0,
        adversarial_num_classes: int = 4,
        adversarial_anneal_epochs: int = 0,
        cl_regression: bool = False,
        cl_lambda: float = 1.0,
        checkpoint_metric: str = "auto",
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

        # Adversarial training (gradient reversal for confound invariance)
        self.adversarial_head: AdversarialHead | None = None
        self.adversarial_criterion: nn.CrossEntropyLoss | None = None
        self.adversarial_lambda = adversarial_lambda
        self.adversarial_anneal_epochs = adversarial_anneal_epochs

        if adversarial_lambda > 0:
            repr_dim = self.model_wrapper.enable_repr_capture()
            self.adversarial_head = AdversarialHead(
                input_dim=repr_dim,
                num_classes=adversarial_num_classes,
                lambda_=0.0 if adversarial_anneal_epochs > 0 else adversarial_lambda,
            )
            self.adversarial_head.to(device)
            # ignore_index=-1 skips samples with unknown confound (e.g. uncharged)
            self.adversarial_criterion = nn.CrossEntropyLoss(ignore_index=-1)
            logger.info(
                f"Adversarial training enabled: lambda={adversarial_lambda}, "
                f"classes={adversarial_num_classes}, anneal_epochs={adversarial_anneal_epochs}, "
                f"repr_dim={repr_dim}"
            )

        # CL regression head (continuous charging-level prediction)
        self.cl_regression_head: RegressionHead | None = None
        self.cl_lambda = cl_lambda

        if cl_regression:
            # Enable repr capture if not already active for adversarial
            if self.adversarial_head is None:
                repr_dim = self.model_wrapper.enable_repr_capture()
            else:
                repr_dim = self.model_wrapper.enable_repr_capture()
            self.cl_regression_head = RegressionHead(input_dim=repr_dim)
            self.cl_regression_head.to(device)
            logger.info(f"CL regression head enabled: repr_dim={repr_dim}, cl_lambda={cl_lambda}")

        # Setup optimizer (include adversarial + CL regression head params if present)
        all_params = list(model.parameters())
        if self.adversarial_head is not None:
            all_params += list(self.adversarial_head.parameters())
        if self.cl_regression_head is not None:
            all_params += list(self.cl_regression_head.parameters())
        # AdamW decouples weight_decay from the gradient (vs. Adam's L2-on-grad).
        # With Adam, adaptive per-param LRs almost cancel L2 reg for low-grad
        # params, so weight_decay loses bite as cosine LR decays toward 0. AdamW
        # applies decay directly to weights and preserves regularization through
        # the LR schedule. State dict is compatible with Adam checkpoints for
        # resume — only the step rule changes.
        self.optimizer = torch.optim.AdamW(all_params, lr=learning_rate, weight_decay=weight_decay)

        # Setup loss
        self.loss_type = loss_type
        self._num_out = num_out if num_out is not None else 1
        self.label_smoothing = label_smoothing
        pw = pos_weight.to(device) if pos_weight is not None else None
        if loss_type == "cross_entropy":
            # CrossEntropyLoss expects (B, num_classes) logits and (B,) integer labels
            if pw is not None and self._num_out <= 2:
                # Convert pos_weight to per-class weights for CE (binary case)
                ce_weights = torch.tensor([1.0, pw.item()], dtype=torch.float32).to(device)
                self.criterion = nn.CrossEntropyLoss(
                    weight=ce_weights, label_smoothing=label_smoothing
                )
            elif pw is not None and self._num_out > 2:
                # Multiclass: pw is already a per-class weight tensor
                self.criterion = nn.CrossEntropyLoss(
                    weight=pw.to(device), label_smoothing=label_smoothing
                )
                logger.info(f"Using class weights for {self._num_out}-class CE")
            else:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            logger.info(f"Using CrossEntropyLoss ({self._num_out}-class)")
        elif loss_type == "focal":
            self.criterion = FocalBCEWithLogitsLoss(gamma=focal_gamma, pos_weight=pw)
            if label_smoothing > 0:
                logger.info(
                    f"Using focal loss (gamma={focal_gamma}) with label smoothing={label_smoothing}"
                )
            else:
                logger.info(f"Using focal loss (gamma={focal_gamma})")
        else:
            if pw is not None:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
                logger.info("Training without class weighting")
        if label_smoothing > 0 and loss_type != "cross_entropy":
            logger.info(f"Label smoothing={label_smoothing} (applied to binary targets)")

        # Setup LR scheduler
        self.scheduler = None
        self.scheduler_type = scheduler_type
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
        elif scheduler_type == "cosine":
            # T_max = total epochs minus warmup; eta_min = 1e-6 floor
            effective_epochs = max(1, epochs - warmup_epochs)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=effective_epochs,
                eta_min=1e-6,
            )
            logger.info(
                f"Using CosineAnnealingLR scheduler (T_max={effective_epochs}, eta_min=1e-6)"
            )

        # Mixed precision (only on CUDA)
        self.use_mixed_precision = use_mixed_precision and device != "cpu"
        self.scaler = None
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler("cuda")
            logger.info("Mixed precision training enabled")

        # Track best model — checkpoint criterion is configurable.
        # "auto" maps to macro-F1 for multiclass and AUROC for binary (immune to
        # class-imbalance gaming, unlike raw accuracy on one-vs-all heads).
        self._checkpoint_metric = self._resolve_checkpoint_metric(checkpoint_metric)
        logger.info(f"Checkpoint criterion: {self._checkpoint_metric}")
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.start_epoch = 1
        self._best_model_state: dict[str, Any] | None = None

        # History
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_auc": [],
            "val_f1": [],
        }
        if self.adversarial_head is not None:
            self.history["train_adv_loss"] = []
            self.history["train_adv_acc"] = []
        if self.cl_regression_head is not None:
            self.history["train_cl_loss"] = []
            self.history["val_cl_loss"] = []

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resume from checkpoint (skip if file doesn't exist)
        if resume_checkpoint is not None:
            if resume_checkpoint.exists():
                self._resume_from_checkpoint(resume_checkpoint)
            else:
                logger.info(f"Checkpoint not found, starting fresh: {resume_checkpoint}")

    _VALID_CHECKPOINT_METRICS = ("auto", "val_acc", "val_f1", "val_auc")

    def _resolve_checkpoint_metric(self, metric: str) -> str:
        if metric == "auto":
            return "val_f1" if self._num_out > 2 else "val_auc"
        if metric not in self._VALID_CHECKPOINT_METRICS:
            raise ValueError(
                f"checkpoint_metric must be one of {self._VALID_CHECKPOINT_METRICS}, got {metric!r}"
            )
        return metric

    def _resume_from_checkpoint(self, checkpoint_path: Path) -> None:
        """Restore training state from a checkpoint."""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.best_val_f1 = checkpoint.get("best_val_f1", 0.0)
        self.best_val_auc = checkpoint.get("best_val_auc", 0.0)
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.start_epoch = checkpoint.get("epoch", 0) + 1

        # Restore best model state dict (for ensuring model_best.pt can be saved)
        self._best_model_state = checkpoint.get("best_model_state_dict")

        # Restore scheduler state if available
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore scaler state if available
        if self.scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore CL regression head if available
        if (
            self.cl_regression_head is not None
            and checkpoint.get("cl_regression_head_state_dict") is not None
        ):
            self.cl_regression_head.load_state_dict(checkpoint["cl_regression_head_state_dict"])

        logger.info(
            f"Resumed from epoch {self.start_epoch - 1} "
            f"(best_val_acc={self.best_val_acc:.4f}, best_val_f1={self.best_val_f1:.4f} "
            f"at epoch {self.best_epoch})"
        )

    def _adversarial_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute adversarial loss from captured representation.

        Returns:
            Tuple of (loss, adv_preds, confound_labels) — all on device.
            Loss is zero if no confound labels are in the batch or capture failed.
        """
        assert self.adversarial_head is not None
        assert self.adversarial_criterion is not None
        repr_vec = self.model_wrapper.captured_repr
        confound_labels = batch["confound_label"].to(self.device)
        if repr_vec is None:
            return (
                torch.tensor(0.0, device=self.device),
                confound_labels,
                confound_labels,
            )
        adv_logits = self.adversarial_head(repr_vec)
        adv_loss = self.adversarial_criterion(adv_logits, confound_labels)
        adv_preds = torch.argmax(adv_logits, dim=-1)
        return adv_loss, adv_preds, confound_labels

    def _cl_regression_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute CL regression loss from captured representation.

        Only valid samples (cl_target >= 0) contribute to the loss.
        Returns zero if no valid targets exist or capture failed.
        """
        assert self.cl_regression_head is not None
        repr_vec = self.model_wrapper.captured_repr
        cl_targets = batch["cl_target"].to(self.device)
        if repr_vec is None:
            return torch.tensor(0.0, device=self.device)
        mask = cl_targets >= 0
        if not mask.any():
            return torch.tensor(0.0, device=self.device)
        preds = self.cl_regression_head(repr_vec[mask])
        return nn.functional.mse_loss(preds, cl_targets[mask])

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
        if self.adversarial_head is not None:
            self.adversarial_head.train()
        if self.cl_regression_head is not None:
            self.cl_regression_head.train()
        total_loss = 0.0
        total_adv_loss = 0.0
        total_cl_loss = 0.0
        all_preds: list[float] = []
        all_labels: list[float] = []
        all_adv_preds: list[int] = []
        all_adv_labels: list[int] = []

        for batch in self.train_loader:
            # Move labels to device
            labels = batch["label"].to(self.device)

            # Apply label smoothing to binary targets (BCE/focal only;
            # CrossEntropyLoss handles its own smoothing via constructor arg).
            # Keep original labels for metrics; smoothed targets are for loss only.
            if self.label_smoothing > 0 and self.loss_type != "cross_entropy":
                loss_targets = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            else:
                loss_targets = labels

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
                        main_loss = self.criterion(logits, ce_labels)
                    else:
                        main_loss = self.criterion(logits, loss_targets)

                    # Adversarial loss (GRL reverses gradients to encoder)
                    if self.adversarial_head is not None and "confound_label" in batch:
                        adv_loss, adv_preds, adv_labels = self._adversarial_loss(batch)
                        loss = main_loss + adv_loss
                        total_adv_loss += adv_loss.item()
                        mask = adv_labels != -1
                        if mask.any():
                            all_adv_preds.extend(adv_preds[mask].cpu().numpy().tolist())
                            all_adv_labels.extend(adv_labels[mask].cpu().numpy().tolist())
                    else:
                        loss = main_loss

                    # CL regression loss
                    if self.cl_regression_head is not None and "cl_target" in batch:
                        cl_loss = self._cl_regression_loss(batch)
                        loss = loss + self.cl_lambda * cl_loss
                        total_cl_loss += cl_loss.item()

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
                    main_loss = self.criterion(logits, ce_labels)
                else:
                    main_loss = self.criterion(logits, loss_targets)

                # Adversarial loss (GRL reverses gradients to encoder)
                if self.adversarial_head is not None and "confound_label" in batch:
                    adv_loss, adv_preds, adv_labels = self._adversarial_loss(batch)
                    loss = main_loss + adv_loss
                    total_adv_loss += adv_loss.item()
                    mask = adv_labels != -1
                    if mask.any():
                        all_adv_preds.extend(adv_preds[mask].cpu().numpy().tolist())
                        all_adv_labels.extend(adv_labels[mask].cpu().numpy().tolist())
                else:
                    loss = main_loss

                # CL regression loss
                if self.cl_regression_head is not None and "cl_target" in batch:
                    cl_loss = self._cl_regression_loss(batch)
                    loss = loss + self.cl_lambda * cl_loss
                    total_cl_loss += cl_loss.item()

                loss.backward()

                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()

            # Track metrics
            total_loss += main_loss.item()
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

        # Adversarial metrics
        if self.adversarial_head is not None:
            avg_adv_loss = total_adv_loss / max(1, len(self.train_loader))
            adv_acc = accuracy_score(all_adv_labels, all_adv_preds) if all_adv_labels else 0.0
            self.history["train_adv_loss"].append(avg_adv_loss)
            self.history["train_adv_acc"].append(adv_acc)

        # CL regression metrics
        if self.cl_regression_head is not None:
            avg_cl_loss = total_cl_loss / max(1, len(self.train_loader))
            self.history["train_cl_loss"].append(avg_cl_loss)

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
        if self.cl_regression_head is not None:
            self.cl_regression_head.eval()
        total_loss = 0.0
        total_cl_loss = 0.0
        all_preds: list[float] = []
        all_labels: list[float] = []
        # For multiclass: keep full softmax probabilities so we can compute
        # macro one-vs-rest AUROC (argmax-only preds lose that information).
        all_probs_mc: list[np.ndarray] = []

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

                # CL regression validation loss
                if (
                    self.cl_regression_head is not None
                    and "cl_target" in batch
                    and self.model_wrapper.captured_repr is not None
                ):
                    cl_targets = batch["cl_target"].to(self.device)
                    cl_mask = cl_targets >= 0
                    if cl_mask.any():
                        cl_preds = self.cl_regression_head(
                            self.model_wrapper.captured_repr[cl_mask]
                        )
                        total_cl_loss += nn.functional.mse_loss(
                            cl_preds, cl_targets[cl_mask]
                        ).item()

                # Track metrics
                total_loss += loss.item()
                if self.loss_type == "cross_entropy" and self._num_out > 2:
                    probs_mc = torch.softmax(logits, dim=-1).cpu().numpy()
                    all_probs_mc.append(probs_mc)
                    preds = probs_mc.argmax(axis=-1)
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
            # Multi-class: preds are class indices; probs are needed for AUROC.
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0.0)
            labels_arr = np.asarray(all_labels)
            present = np.unique(labels_arr)
            # Macro one-vs-rest AUROC. Requires >=2 classes present in this
            # validation set; otherwise undefined and reported as 0.
            if len(present) > 1 and all_probs_mc:
                probs_all = np.concatenate(all_probs_mc, axis=0)
                # Restrict columns to classes that actually appear, otherwise
                # roc_auc_score raises on absent-class columns.
                cols = present.astype(int)
                try:
                    auc = roc_auc_score(
                        labels_arr,
                        probs_all[:, cols],
                        multi_class="ovr",
                        average="macro",
                        labels=cols,
                    )
                except ValueError:
                    auc = 0.0
            else:
                auc = 0.0
        else:
            all_preds_binary = (np.array(all_preds) > 0.5).astype(int)
            accuracy = accuracy_score(all_labels, all_preds_binary)
            auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0
            f1 = f1_score(all_labels, all_preds_binary, zero_division=0.0)

        # CL regression validation metrics
        if self.cl_regression_head is not None:
            avg_cl_loss = total_cl_loss / max(1, len(self.val_loader))
            self.history["val_cl_loss"].append(avg_cl_loss)

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
                self._ensure_best_checkpoint()
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

                # Adversarial lambda annealing (linear ramp)
                if self.adversarial_head is not None and self.adversarial_anneal_epochs > 0:
                    progress_frac = min(1.0, epoch / self.adversarial_anneal_epochs)
                    current_lambda = self.adversarial_lambda * progress_frac
                    self.adversarial_head.set_lambda(current_lambda)

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

                    # LR scheduler step (only after warmup). For reduce_on_plateau
                    # we step on the *selection metric*, not val_loss, so the
                    # plateau detection matches what early-stopping and best-
                    # model checkpointing react to. If val_loss were used while
                    # selection runs off F1/AUC, LR can drop because loss
                    # plateaued on easy examples even though the ranking metric
                    # is still improving — burning the schedule prematurely.
                    if self.scheduler is not None and epoch > self.warmup_epochs:
                        old_lr = self.optimizer.param_groups[0]["lr"]
                        if self.scheduler_type == "reduce_on_plateau":
                            if self._checkpoint_metric == "val_f1":
                                plateau_signal = -val_f1
                            elif self._checkpoint_metric == "val_auc":
                                plateau_signal = -val_auc
                            elif self._checkpoint_metric == "val_acc":
                                plateau_signal = -val_acc
                            else:
                                plateau_signal = val_loss
                            self.scheduler.step(plateau_signal)
                        else:
                            # Epoch-based schedulers (cosine, etc.)
                            self.scheduler.step()
                        new_lr = self.optimizer.param_groups[0]["lr"]
                        if new_lr != old_lr:
                            logger.info(f"LR reduced: {old_lr:.6f} -> {new_lr:.6f}")

                    # Display metrics — [*] marks the checkpoint criterion
                    lr_str = ""
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    if current_lr != self.base_lr:
                        lr_str = f" LR: {current_lr:.6f}"
                    acc_label = "Acc[*]" if self._checkpoint_metric == "val_acc" else "Acc"
                    f1_label = "F1[*]" if self._checkpoint_metric == "val_f1" else "F1"
                    auc_label = "AUC[*]" if self._checkpoint_metric == "val_auc" else "AUC"
                    adv_str = ""
                    if self.adversarial_head is not None and self.history["train_adv_loss"]:
                        _adv_l = self.history["train_adv_loss"][-1]
                        _adv_a = self.history["train_adv_acc"][-1]
                        _lam = self.adversarial_head.grl.lambda_
                        adv_str = f" | Adv: L={_adv_l:.3f} Acc={_adv_a:.3f} λ={_lam:.3f}"
                    cl_str = ""
                    if self.cl_regression_head is not None and self.history["train_cl_loss"]:
                        _tcl = self.history["train_cl_loss"][-1]
                        _vcl = (
                            self.history["val_cl_loss"][-1] if self.history["val_cl_loss"] else 0.0
                        )
                        cl_str = f" | CL: train={_tcl:.4f} val={_vcl:.4f}"
                    console.print(
                        f"[cyan]Epoch {epoch}/{epochs}[/cyan] | "
                        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f} {acc_label}: {val_acc:.4f} "
                        f"{f1_label}: {val_f1:.4f} {auc_label}: {val_auc:.4f}"
                        f"{lr_str}{adv_str}{cl_str}"
                    )

                    # Save best model per configured checkpoint metric
                    if self._checkpoint_metric == "val_f1":
                        improved = val_f1 > self.best_val_f1
                    elif self._checkpoint_metric == "val_auc":
                        improved = val_auc > self.best_val_auc
                    else:
                        improved = val_acc > self.best_val_acc
                    if improved:
                        self.best_val_acc = val_acc
                        self.best_val_f1 = val_f1
                        self.best_val_auc = val_auc
                        self.best_epoch = epoch
                        self._best_model_state = copy.deepcopy(self.model.state_dict())
                        patience_counter = 0

                        if self.output_dir:
                            self.save_checkpoint("model_best.pt", epoch=epoch)
                            console.print(
                                f"[bold green]✓ Saved best model "
                                f"(val_acc: {val_acc:.4f}, val_f1: {val_f1:.4f}, "
                                f"val_auc: {val_auc:.4f})[/bold green]"
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
            self._ensure_best_checkpoint()
            self.save_history()

        return self.history

    def _ensure_best_checkpoint(self) -> None:
        """Ensure model_best.pt exists, creating it from stored best weights if needed."""
        if self.output_dir is None:
            return

        best_path = self.output_dir / "model_best.pt"
        if best_path.exists():
            return

        if self._best_model_state is not None:
            # Save best model from stored state dict
            best_checkpoint = {
                "model_state_dict": self._best_model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_acc": self.best_val_acc,
                "best_val_f1": self.best_val_f1,
                "best_val_auc": self.best_val_auc,
                "best_epoch": self.best_epoch,
                "epoch": self.best_epoch,
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                "best_model_state_dict": self._best_model_state,
            }
            torch.save(best_checkpoint, best_path)
            logger.info(
                f"Restored model_best.pt from stored best weights (epoch {self.best_epoch})"
            )
        else:
            # Fallback: save current model (best available)
            self.save_checkpoint("model_best.pt", epoch=self.best_epoch)
            logger.info("Saved model_best.pt from current model weights (no stored best)")

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
            "best_val_auc": self.best_val_auc,
            "best_epoch": self.best_epoch,
            "epoch": epoch,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_model_state_dict": self._best_model_state,
        }
        if self.cl_regression_head is not None:
            checkpoint["cl_regression_head_state_dict"] = self.cl_regression_head.state_dict()
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
            "best_val_auc": self.best_val_auc,
            "best_epoch": self.best_epoch,
            "checkpoint_metric": self._checkpoint_metric,
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
    label_smoothing: float = 0.0,
    augment_jitter: float = 0.0,
    augment_scale_min: float = 1.0,
    augment_scale_max: float = 1.0,
    augment_time_mask_bases: int = 0,
    augment_time_mask_count: int = 1,
    augment_shift_max_bases: float = 0.0,
    augment_feature_noise_scale: float = 0.0,
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
    oversample_minority: bool = False,
    label_map: dict[str, int] | None = None,
    num_out: int = 1,
    adversarial_lambda: float = 0.0,
    adversarial_anneal_epochs: int = 0,
    confound: str | None = None,
    cl_regression: bool = False,
    cl_lambda: float = 1.0,
    signal_mode: str = "both",
    checkpoint_metric: str = "auto",
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

    # Validate required provenance fields
    if motif is None:
        raise ValueError(
            "motif must not be None. Pass --motif on the CLI or motif= in code. "
            "A null motif causes inference to predict at every position, producing noise."
        )

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
    dwell_template_table = model_kwargs.pop("dwell_template_table", None)

    # Build augmentation config for training dataset.
    # Per-channel dicts (e.g. {"signal": 0.02, "signal_residual": 0.001})
    # can be provided via --model-config JSON; CLI flags give uniform scalars.
    augmentation = None
    jitter_cfg = model_kwargs.pop("augment_jitter", augment_jitter)
    scale_cfg = model_kwargs.pop("augment_scale_range", (augment_scale_min, augment_scale_max))

    has_jitter = (isinstance(jitter_cfg, dict) and any(v > 0 for v in jitter_cfg.values())) or (
        isinstance(jitter_cfg, (int, float)) and jitter_cfg > 0
    )
    has_scale = isinstance(scale_cfg, dict) or (
        isinstance(scale_cfg, (list, tuple)) and tuple(scale_cfg) != (1.0, 1.0)
    )

    if has_jitter or has_scale:
        augmentation = {
            "jitter_std": jitter_cfg,
            "scale_range": scale_cfg,
        }
        logger.info(f"Signal augmentation enabled: {augmentation}")

    # Build confound map for adversarial training
    confound_map: dict[int, int] | None = None
    ref_confound_map: dict[str, int] | None = None
    adversarial_num_classes = 4  # default: A/C/G/T

    if adversarial_lambda > 0 and confound == "disc_base":
        from leech.confounds import (
            NUM_DISC_BASES,
            build_confound_map,
            load_disc_base_map,
        )

        _lm = label_map
        if _lm is None:
            _lm_path = train_data_path.parent / "label_map.json"
            if not _lm_path.exists():
                _lm_path = train_data_path.parent.parent / "label_map.json"
            if _lm_path.exists():
                with open(_lm_path) as f:
                    _lm = json.load(f)
        if _lm is not None:
            _dbm = load_disc_base_map(train_data_path.parent)
            confound_map = build_confound_map(_lm, disc_base_map=_dbm)
            adversarial_num_classes = NUM_DISC_BASES
            logger.info(f"Adversarial confound 'disc_base': {confound_map}")
        else:
            logger.warning(
                "adversarial_lambda > 0 with confound='disc_base' but no label_map found; "
                "adversarial training disabled"
            )
            adversarial_lambda = 0.0

    elif adversarial_lambda > 0 and confound == "trna_id":
        # Full tRNA-isoacceptor identity confound. Each unique reference name
        # (e.g. tRNA-Ser-CGA-1-1) becomes a distinct adversarial class,
        # directly penalising the model for encoding ANY tRNA-body information.
        from leech.confounds import build_trna_identity_map

        _ref_names = None
        if train_chunks is not None:
            _ref_names = [c.get("reference_name", "") for c in train_chunks]
        elif train_data_path is not None:
            _npz = np.load(train_data_path, allow_pickle=True)
            if "reference_names" in _npz:
                _ref_names = _npz["reference_names"].tolist()
        if _ref_names is not None:
            ref_confound_map, adversarial_num_classes = build_trna_identity_map(_ref_names)
            logger.info(f"Adversarial confound 'trna_id': {adversarial_num_classes} classes")
        else:
            logger.warning(
                "adversarial_lambda > 0 with confound='trna_id' but no "
                "reference_names in training data; adversarial training disabled"
            )
            adversarial_lambda = 0.0

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
        confound_map=confound_map,
        ref_confound_map=ref_confound_map,
        cl_regression=cl_regression,
        signal_mode=signal_mode,
        time_mask_bases=augment_time_mask_bases,
        time_mask_count=augment_time_mask_count,
        shift_max_bases=augment_shift_max_bases,
        feature_noise_scale=augment_feature_noise_scale,
        dwell_template_table=dwell_template_table,
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
            confound_map=confound_map,
            ref_confound_map=ref_confound_map,
            cl_regression=cl_regression,
            signal_mode=signal_mode,
            dwell_template_table=dwell_template_table,
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

    # Validate mutually exclusive sampling strategies
    if balance_groups and oversample_minority:
        raise ValueError(
            "--balance-groups and --oversample-minority are mutually exclusive. "
            "Use --balance-groups to equalize source groups, or "
            "--oversample-minority to equalize class labels."
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
    elif oversample_minority:
        # Compute per-sample weights inversely proportional to class frequency
        label_counts: dict[int, int] = {}
        for chunk in train_dataset.chunks:
            lbl = chunk["label_int"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        if len(label_counts) > 1:
            weights = []
            for chunk in train_dataset.chunks:
                lbl = chunk["label_int"]
                weights.append(1.0 / label_counts[lbl])
            train_sampler = WeightedRandomSampler(
                weights, num_samples=len(train_dataset), replacement=True
            )
            logger.info(f"Minority oversampling enabled across {len(label_counts)} classes:")
            for lbl, count in sorted(label_counts.items()):
                logger.info(f"  class {lbl}: {count} chunks, weight={1.0 / count:.6f}")
        else:
            logger.warning(
                "oversample_minority enabled but only 1 class found. Falling back to shuffle."
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
        # Validation __getitem__ is trivially fast (no augmentation, pre-tensorized
        # lookups only), so workers add memory overhead without benefit.  Dropping
        # val workers halves the total process count and avoids OOM-triggered
        # segfaults on large multiclass datasets.
        val_loader_kwargs: dict = {
            "collate_fn": collate_fn,
            "num_workers": 0,
        }
        if device != "cpu":
            val_loader_kwargs["pin_memory"] = True
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **val_loader_kwargs,
        )

    # Determine num_features and signal_channels from first batch
    first_batch = next(iter(train_loader))
    num_features = first_batch.get("features", torch.zeros(1, 1, kmer_len)).shape[1]
    signal_shape = first_batch["signal"].shape
    signal_in_channels = signal_shape[1] if len(signal_shape) == 3 else 1

    # Auto-detect num_out from training data when not explicitly set
    if num_out <= 1:
        max_label = max(int(c["label_int"]) for c in train_dataset.chunks)
        if max_label > 1:
            num_out = max_label + 1
            logger.info(f"Auto-detected multi-class: num_out={num_out}")
            # Load label_map from sidecar if available
            if label_map is None:
                label_map_path = train_data_path.parent / "label_map.json"
                if not label_map_path.exists():
                    # k-fold: data is in fold_N/ subdir, label_map is one level up
                    label_map_path = train_data_path.parent.parent / "label_map.json"
                if label_map_path.exists():
                    with open(label_map_path) as f:
                        label_map = json.load(f)
                    logger.info(f"Loaded label_map from {label_map_path}: {label_map}")

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
    model_init_kwargs["num_features"] = num_features

    # All models accept signal_in_channels for multi-channel signal input
    model_init_kwargs["signal_in_channels"] = signal_in_channels

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

    # Introspect feature_start/feature_end from raw training data
    _raw_chunk = train_dataset.chunks[0]
    _kmer_context = kmer_len // 2
    if "feature_start" in _raw_chunk:
        _feature_start = int(_raw_chunk["feature_start"])
    elif "feature_left" in _raw_chunk:
        _feature_start = -int(_raw_chunk["feature_left"])
    elif "dwell_margin_left" in _raw_chunk:
        _feature_start = -(_kmer_context + int(_raw_chunk["dwell_margin_left"]))
    else:
        _feature_start = -_kmer_context
    if "feature_end" in _raw_chunk:
        _feature_end = int(_raw_chunk["feature_end"])
    elif "feature_right" in _raw_chunk:
        _feature_end = int(_raw_chunk["feature_right"])
    elif "dwell_margin_right" in _raw_chunk:
        _raw_features = _raw_chunk.get("features")
        if _raw_features is not None and _raw_features.ndim > 1:
            _feat_width = _raw_features.shape[1]
            _feature_end = _feat_width - 1 + _feature_start
        else:
            _feature_end = _kmer_context
    else:
        _feature_end = _kmer_context

    # Read preparation config sidecar if available (for provenance in config.json)
    prepare_metadata: dict[str, Any] = {}
    _sidecar_path = train_data_path.parent / "prepare_config.json"
    if _sidecar_path.exists():
        with open(_sidecar_path) as f:
            prepare_metadata = json.load(f)
        logger.info(f"Read preparation metadata from {_sidecar_path}")
    else:
        # Use correct defaults for old data without sidecar
        prepare_metadata = {
            "anchor": "reference",
            "signal_norm": "median_mad",
            "reverse_signal": True,
            "refine_signal_map": True,
            "motif_reference": "fasta",
        }
        logger.info("No prepare_config.json found, using correct defaults for preparation metadata")

    # Save config
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_name": model_name,
        "signal_len": signal_len,
        "kmer_len": kmer_len,
        "num_features": num_features,
        "signal_in_channels": signal_in_channels,
        "dwell_offset": dwell_offset,
        "feature_start": _feature_start,
        "feature_end": _feature_end,
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
        else (pos_weight_tensor.tolist() if pos_weight_tensor is not None else None),
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
        "label_smoothing": label_smoothing,
        "augment_jitter": jitter_cfg,
        "augment_scale_min": augment_scale_min,
        "augment_scale_max": augment_scale_max,
        "augment_scale_range": scale_cfg if isinstance(scale_cfg, dict) else None,
        "augment_time_mask_bases": augment_time_mask_bases,
        "augment_time_mask_count": augment_time_mask_count,
        "augment_shift_max_bases": augment_shift_max_bases,
        "augment_feature_noise_scale": augment_feature_noise_scale,
        "balance_groups": balance_groups,
        "label_map": label_map,
        "adversarial_lambda": adversarial_lambda,
        "adversarial_anneal_epochs": adversarial_anneal_epochs,
        "confound": confound,
        "cl_regression": cl_regression,
        "cl_lambda": cl_lambda,
        "signal_mode": signal_mode,
        "checkpoint_metric": checkpoint_metric,
        "dwell_template_table": str(dwell_template_table) if dwell_template_table else None,
        # Preparation metadata (from prepare_config.json sidecar)
        "reference_fasta": prepare_metadata.get("reference_fasta"),
        "anchor": prepare_metadata.get("anchor", "reference"),
        "signal_norm": prepare_metadata.get("signal_norm", "median_mad"),
        "reverse_signal": prepare_metadata.get("reverse_signal", True),
        "refine_signal_map": prepare_metadata.get("refine_signal_map", True),
        "refine_scale_iters": prepare_metadata.get("refine_scale_iters", 2),
        "motif_reference": prepare_metadata.get("motif_reference", "fasta"),
        "pa_mean": prepare_metadata.get("pa_mean"),
        "pa_stdev": prepare_metadata.get("pa_stdev"),
        "skip_motif_indels": prepare_metadata.get("skip_motif_indels", False),
        "refine_half_bandwidth": prepare_metadata.get("refine_half_bandwidth", 5),
        "refine_do_rough_rescale": prepare_metadata.get("refine_do_rough_rescale", True),
        "refine_kmer_center_idx": prepare_metadata.get("refine_kmer_center_idx", -1),
        "recover_softclip_signal": prepare_metadata.get("recover_softclip_signal", False),
        "kmer_table_sha256": prepare_metadata.get("kmer_table_sha256"),
        # Provenance
        "leech_version": leech.__version__,
        "git_commit": leech._get_git_revision(),
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
        label_smoothing=label_smoothing,
        use_mixed_precision=mixed_precision,
        resume_checkpoint=resume_from,
        num_out=num_out,
        epochs=epochs,
        adversarial_lambda=adversarial_lambda,
        adversarial_num_classes=adversarial_num_classes,
        adversarial_anneal_epochs=adversarial_anneal_epochs,
        cl_regression=cl_regression,
        cl_lambda=cl_lambda,
        checkpoint_metric=checkpoint_metric,
    )

    # Train
    history = trainer.train(epochs=epochs, early_stopping_patience=early_stopping_patience)

    return history
