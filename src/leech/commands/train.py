"""
Handler for the 'train' command.

This module contains the business logic for training a model on prepared data.
"""

import json
import logging
from pathlib import Path
from typing import Any

from rich.table import Table

from leech.cli_config import make_console

logger = logging.getLogger("leech.commands.train")
console = make_console()


def handle_train(
    train_data: Path,
    val_data: Path | None,
    model_name: str,
    model_config: Path | None,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    seed: int,
    early_stopping: int,
    use_class_weights: bool,
    pos_weight: float | None,
    resume: Path | None,
    weight_decay: float,
    max_grad_norm: float,
    scheduler: str,
    scheduler_patience: int,
    scheduler_factor: float,
    warmup_epochs: int,
    loss_type: str,
    focal_gamma: float,
    label_smoothing: float,
    mixed_precision: bool,
    augment_jitter: float,
    augment_scale_min: float,
    augment_scale_max: float,
    augment_time_mask_bases: int,
    augment_time_mask_count: int,
    augment_shift_max_bases: float,
    augment_feature_noise_scale: float,
    num_workers: int,
    quantile_grad_clip: bool = False,
    grad_accum_split: int = 1,
    save_optim_every: int = 1,
    motif: str | None = None,
    motif_offset: int = 0,
    base_justify: str = "center",
    seq_encoding: str = "signal_kmer",
    balance_groups: bool = False,
    oversample_minority: bool = False,
    adversarial_lambda: float = 0.0,
    adversarial_anneal_epochs: int = 0,
    confound: str | None = None,
    cl_regression: bool = False,
    cl_lambda: float = 1.0,
    signal_mode: str = "both",
    dwell_template_table: str | None = None,
    checkpoint_metric: str = "auto",
    **model_kwargs: Any,
) -> dict[str, Any]:
    """
    Handle the train command logic.

    Args:
        train_data: Training dataset config (JSON)
        val_data: Validation dataset config (JSON)
        model_name: Model architecture name
        model_config: Optional JSON file with model hyperparameters
        output_dir: Output directory for model and logs
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        device: Device for training (cuda/cpu)
        seed: Random seed
        early_stopping: Patience for early stopping (0 = disable)
        use_class_weights: Auto-compute class weights
        pos_weight: Manual positive class weight
        resume: Resume from checkpoint file
        weight_decay: L2 weight decay
        max_grad_norm: Gradient clipping norm (0 = disabled)
        quantile_grad_clip: Clip at a quantile of recent gradient norms (bonito ClipGrad)
        grad_accum_split: Sub-batches per optimizer step (1 = no accumulation)
        save_optim_every: Write optimizer state every N epochs (1 = every save)
        scheduler: LR scheduler type
        scheduler_patience: LR scheduler patience
        scheduler_factor: LR scheduler factor
        warmup_epochs: Linear warmup epochs
        loss_type: Loss function type
        focal_gamma: Focal loss gamma
        mixed_precision: Enable mixed precision training
        augment_jitter: Signal jitter noise std dev
        augment_scale_min: Min random scale factor
        augment_scale_max: Max random scale factor
        num_workers: DataLoader workers
        motif: Motif used for chunk extraction (provenance)
        motif_offset: Offset within motif (provenance)
        base_justify: Signal justification (provenance)
        seq_encoding: Sequence encoding type
        balance_groups: Balance sampling across source groups
        **model_kwargs: Additional model-specific parameters

    Returns:
        Training history dictionary
    """
    from leech.training import train_model

    logger.info(f"Training {model_name} model")
    logger.info(f"Train data: {train_data}")
    logger.info(f"Output: {output_dir}")

    # Load model config if provided
    extra_kwargs = dict(model_kwargs)
    if model_config is not None:
        with open(model_config) as f:
            extra_kwargs.update(json.load(f))

    # Remove keys that are already passed as explicit arguments to avoid
    # "got multiple values for keyword argument" errors
    _explicit_keys = {
        "loss_type",
        "scheduler",
        "scheduler_patience",
        "scheduler_factor",
        "warmup_epochs",
        "weight_decay",
        "max_grad_norm",
        "quantile_grad_clip",
        "grad_accum_split",
        "save_optim_every",
        "learning_rate",
        "epochs",
        "batch_size",
        "early_stopping_patience",
        "use_class_weights",
        "pos_weight",
        "mixed_precision",
        "focal_gamma",
        "label_smoothing",
        "augment_jitter",
        "augment_scale_min",
        "augment_scale_max",
        "augment_time_mask_bases",
        "augment_time_mask_count",
        "augment_shift_max_bases",
        "augment_feature_noise_scale",
        "num_workers",
        "seq_encoding",
        "balance_groups",
        "oversample_minority",
        "device",
        "seed",
        "motif",
        "motif_offset",
        "base_justify",
        "num_out",
        "adversarial_lambda",
        "adversarial_anneal_epochs",
        "confound",
        "cl_regression",
        "cl_lambda",
        "signal_mode",
        "dwell_template_table",
        "checkpoint_metric",
    }
    for key in _explicit_keys:
        extra_kwargs.pop(key, None)

    # Train model
    history = train_model(
        train_data_path=train_data,
        val_data_path=val_data,
        model_name=model_name,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
        early_stopping_patience=early_stopping,
        use_class_weights=use_class_weights,
        pos_weight=pos_weight,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        quantile_grad_clip=quantile_grad_clip,
        grad_accum_split=grad_accum_split,
        save_optim_every=save_optim_every,
        scheduler=scheduler,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        warmup_epochs=warmup_epochs,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
        mixed_precision=mixed_precision,
        augment_jitter=augment_jitter,
        augment_scale_min=augment_scale_min,
        augment_scale_max=augment_scale_max,
        augment_time_mask_bases=augment_time_mask_bases,
        augment_time_mask_count=augment_time_mask_count,
        augment_shift_max_bases=augment_shift_max_bases,
        augment_feature_noise_scale=augment_feature_noise_scale,
        resume_from=resume,
        num_workers=num_workers,
        motif=motif,
        motif_offset=motif_offset,
        base_justify=base_justify,
        seq_encoding=seq_encoding,
        balance_groups=balance_groups,
        oversample_minority=oversample_minority,
        adversarial_lambda=adversarial_lambda,
        adversarial_anneal_epochs=adversarial_anneal_epochs,
        confound=confound,
        cl_regression=cl_regression,
        cl_lambda=cl_lambda,
        signal_mode=signal_mode,
        dwell_template_table=dwell_template_table,
        checkpoint_metric=checkpoint_metric,
        **extra_kwargs,
    )

    console.print("[bold green]Training complete![/bold green]")

    # Display final metrics in a table
    table = Table(title="Training Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    if "val_acc" in history and history["val_acc"]:
        table.add_row("Best Validation Accuracy", f"{max(history['val_acc']):.4f}")
    if "val_loss" in history and history["val_loss"]:
        table.add_row("Final Validation Loss", f"{history['val_loss'][-1]:.4f}")
    if "val_f1" in history and history["val_f1"]:
        table.add_row("Best Validation F1", f"{max(history['val_f1']):.4f}")

    table.add_row("Model saved to", str(output_dir))

    console.print(table)

    return history
