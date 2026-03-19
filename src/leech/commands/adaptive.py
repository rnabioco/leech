"""
Handler for the 'adaptive' command.

Trains a model with learnable signal context window boundaries.
"""

import json
import logging
from pathlib import Path
from typing import Any

from rich.table import Table

from leech.cli_config import make_console

logger = logging.getLogger("leech.commands.adaptive")
console = make_console()


def handle_adaptive(
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
    motif: str | None = None,
    motif_offset: int = 0,
    base_justify: str = "center",
    seq_encoding: str = "signal_kmer",
    balance_groups: bool = False,
    oversample_minority: bool = False,
    num_out: int = 1,
    adversarial_lambda: float = 0.0,
    adversarial_anneal_epochs: int = 0,
    confound: str | None = None,
    cl_regression: bool = False,
    cl_lambda: float = 1.0,
    # Adaptive-specific parameters
    window_mode: str = "adaptive",
    window_sharpness: float = 0.1,
    window_lr_multiplier: float = 10.0,
    init_left_context: int | None = None,
    init_right_context: int | None = None,
    independent_residual_window: bool = False,
    **model_kwargs: Any,
) -> dict[str, Any]:
    """Handle the adaptive training command.

    Trains a model with learnable signal context window boundaries.
    Uses BranchBoundaryMask (sigmoid mode) or BranchGate (content-dependent)
    to discover the optimal context window during training.

    Returns:
        Training history dictionary.
    """
    from leech.training import train_model

    logger.info(f"Adaptive training {model_name} model (window_mode={window_mode})")
    logger.info(f"Train data: {train_data}")
    logger.info(f"Output: {output_dir}")

    # Load model config if provided
    extra_kwargs = dict(model_kwargs)
    if model_config is not None:
        with open(model_config) as f:
            extra_kwargs.update(json.load(f))

    # Remove keys that are already passed as explicit arguments
    _explicit_keys = {
        "loss_type",
        "scheduler",
        "scheduler_patience",
        "scheduler_factor",
        "warmup_epochs",
        "weight_decay",
        "max_grad_norm",
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
        # Adaptive-specific keys
        "learnable_window",
        "window_mode",
        "window_sharpness",
        "window_lr_multiplier",
        "init_left_context",
        "init_right_context",
        "independent_residual_window",
    }
    for key in _explicit_keys:
        extra_kwargs.pop(key, None)

    # Train model with learnable window
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
        num_out=num_out,
        adversarial_lambda=adversarial_lambda,
        adversarial_anneal_epochs=adversarial_anneal_epochs,
        confound=confound,
        cl_regression=cl_regression,
        cl_lambda=cl_lambda,
        # Adaptive-specific parameters
        learnable_window=True,
        window_mode=window_mode,
        window_sharpness=window_sharpness,
        window_lr_multiplier=window_lr_multiplier,
        init_left_context=init_left_context,
        init_right_context=init_right_context,
        independent_residual_window=independent_residual_window,
        **extra_kwargs,
    )

    console.print("[bold green]Adaptive training complete![/bold green]")

    # Display final metrics in a table
    table = Table(title="Adaptive Training Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    if "val_acc" in history and history["val_acc"]:
        table.add_row("Best Validation Accuracy", f"{max(history['val_acc']):.4f}")
    if "val_loss" in history and history["val_loss"]:
        table.add_row("Final Validation Loss", f"{history['val_loss'][-1]:.4f}")
    if "val_f1" in history and history["val_f1"]:
        table.add_row("Best Validation F1", f"{max(history['val_f1']):.4f}")

    # Display learned window info
    if "learned_signal_left" in history and history["learned_signal_left"]:
        table.add_row("Final Signal Left", str(history["learned_signal_left"][-1]))
    if "learned_signal_right" in history and history["learned_signal_right"]:
        table.add_row("Final Signal Right", str(history["learned_signal_right"][-1]))

    table.add_row("Window Mode", window_mode)
    table.add_row("Model saved to", str(output_dir))

    console.print(table)

    return history
