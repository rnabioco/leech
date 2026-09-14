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
    augment_time_stretch: tuple[float, float] = (1.0, 1.0),
    gpus: int = 1,
    quantile_grad_clip: bool = False,
    grad_accum_split: int = 1,
    save_optim_every: int = 1,
    motif: str | None = None,
    motif_offset: int = 0,
    base_justify: str = "center",
    seq_encoding: str = "signal_kmer",
    allow_encoding_fallback: bool = True,
    strict_window: bool = False,
    balance_groups: bool = False,
    oversample_minority: bool = False,
    sample_weight_field: str | None = None,
    adversarial_lambda: float = 0.0,
    adversarial_anneal_epochs: int = 0,
    confound: str | None = None,
    label_noise_rates: dict[str, float] | None = None,
    noise_sink_class: str | None = None,
    cl_regression: bool = False,
    cl_lambda: float = 1.0,
    signal_mode: str = "both",
    dwell_template_table: str | None = None,
    checkpoint_metric: str = "auto",
    focal_neg_gamma: float | None = None,
    standardize_features: bool = False,
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
        augment_time_stretch: (min, max) per-sample time-stretch factor range
            ((1.0, 1.0) = disabled)
        num_workers: DataLoader workers
        gpus: Data-parallel ranks (1 = single device); batch_size is the global
            batch and is split across them
        motif: Motif used for chunk extraction (provenance)
        motif_offset: Offset within motif (provenance)
        base_justify: Signal justification (provenance)
        seq_encoding: Sequence encoding type
        allow_encoding_fallback: Permit signal_kmer to degrade to base_onehot
            when the corpus carries no base-to-signal maps (False raises
            instead; a partially covered corpus raises either way)
        strict_window: Raise instead of zero-padding when the asymmetric crop
            (left_context/right_context, reached via --model-config) extends
            outside the stored chunk. Default False logs one warning per
            dataset the first time it happens.
        balance_groups: Balance sampling across source groups
        label_noise_rates: ``{source_group: flip_rate}`` for
            ``loss_type="noise_corrected_bce"``; unmapped groups get rate 0.
            At ``num_out > 1`` keys are class labels instead -- see
            ``leech.losses.build_class_noise_rates``.
        noise_sink_class: Sink class name/index for
            ``loss_type="noise_corrected_bce"`` at ``num_out > 1``; unused
            for binary
        sample_weight_field: Chunk metadata field to inverse-frequency weight
            sampling by (mutually exclusive with balance_groups and
            oversample_minority), e.g. "junction_indel" to over-sample the
            disrupted-junction population
        standardize_features: Compute per-channel feature mean/std once over
            the training corpus and freeze them into the feature branch as
            an affine layer (off by default).
        **model_kwargs: Additional model-specific parameters

    Returns:
        Training history dictionary
    """
    from leech.configs import (
        AugmentConfig,
        AuxHeadConfig,
        OptimConfig,
        SchedulerConfig,
        TrainConfig,
    )
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
        "focal_neg_gamma",
        "label_smoothing",
        "augment_jitter",
        "augment_scale_min",
        "augment_scale_max",
        "augment_time_mask_bases",
        "augment_time_mask_count",
        "augment_shift_max_bases",
        "augment_feature_noise_scale",
        "augment_time_stretch",
        # config.json (train_model's output) records the resolved range as
        # two scalar keys, not the single "augment_time_stretch" tuple param
        # above -- reusing a prior run's config.json via --model-config
        # (test_label_map_survives_model_config's pattern) would otherwise
        # leave these two unpopped, flowing through **extra_kwargs into
        # get_model(...)'s init kwargs and raising TypeError on every
        # TOML-declared architecture that doesn't accept them.
        "augment_time_stretch_min",
        "augment_time_stretch_max",
        "num_workers",
        "seq_encoding",
        "allow_encoding_fallback",
        "strict_window",
        "balance_groups",
        "oversample_minority",
        "sample_weight_field",
        "device",
        "seed",
        "motif",
        "motif_offset",
        "base_justify",
        "num_out",
        "adversarial_lambda",
        "adversarial_anneal_epochs",
        "confound",
        "label_noise_rates",
        "noise_sink_class",
        "cl_regression",
        "cl_lambda",
        "signal_mode",
        "dwell_template_table",
        "checkpoint_metric",
        "standardize_features",
        "signal_kmer_context",
        "left_context",
        "right_context",
        "label_map",
    }
    # These four reach this function only via **model_kwargs / --model-config
    # (grid search's best_params.json, for instance, supplies left_context/
    # right_context this way) -- pop them out before the rest of
    # _explicit_keys strips the set, and thread them into `cfg` explicitly.
    # Leaving them in extra_kwargs while cfg carries its own (default) value
    # for the same name doesn't raise, because train_model's cfg-priority
    # unpack overwrites its own parameter with cfg's default unconditionally
    # -- so the explicit value from --model-config would be silently
    # reverted to the TrainConfig default instead of erroring or applying.
    _train_config_defaults = TrainConfig()
    signal_kmer_context = extra_kwargs.pop(
        "signal_kmer_context", _train_config_defaults.signal_kmer_context
    )
    left_context = extra_kwargs.pop("left_context", _train_config_defaults.left_context)
    right_context = extra_kwargs.pop("right_context", _train_config_defaults.right_context)
    label_map = extra_kwargs.pop("label_map", _train_config_defaults.label_map)
    for key in _explicit_keys:
        extra_kwargs.pop(key, None)

    # Pre-existing gap, not introduced by #270: "num_out" is in
    # _explicit_keys above (so a `--num-out` value reaching this function via
    # **model_kwargs gets popped from extra_kwargs here) but this function
    # has no `num_out` parameter of its own to forward it through instead --
    # so `--num-out` on the CLI is silently dropped, and train_model always
    # falls back to auto-detecting it from the training data's label column.
    # Left as-is: fixing it is a choice about what `--num-out` should mean
    # (an explicit override vs. only ever auto-detected) and is out of scope
    # for this refactor, which preserves existing behavior exactly.

    # One recipe object (#270) instead of re-listing every option a fourth
    # time (cli.py, this function's signature, and train_model/Trainer are
    # the other three). dwell_template_table and gpus/num_workers/device/seed
    # aren't part of the recipe -- they're provenance/runtime knobs threaded
    # through separately, same as train_data/output_dir.
    cfg = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping,
        use_class_weights=use_class_weights,
        pos_weight=pos_weight,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        focal_neg_gamma=focal_neg_gamma,
        label_smoothing=label_smoothing,
        mixed_precision=mixed_precision,
        checkpoint_metric=checkpoint_metric,
        standardize_features=standardize_features,
        signal_mode=signal_mode,
        motif=motif,
        motif_offset=motif_offset,
        base_justify=base_justify,
        seq_encoding=seq_encoding,
        signal_kmer_context=signal_kmer_context,
        allow_encoding_fallback=allow_encoding_fallback,
        left_context=left_context,
        right_context=right_context,
        strict_window=strict_window,
        balance_groups=balance_groups,
        oversample_minority=oversample_minority,
        sample_weight_field=sample_weight_field,
        label_map=label_map,
        confound=confound,
        label_noise_rates=label_noise_rates,
        noise_sink_class=noise_sink_class,
        optim=OptimConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            quantile_grad_clip=quantile_grad_clip,
            grad_accum_split=grad_accum_split,
            save_optim_every=save_optim_every,
        ),
        scheduler=SchedulerConfig(
            scheduler_type=scheduler,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
            warmup_epochs=warmup_epochs,
        ),
        augment=AugmentConfig(
            jitter=augment_jitter,
            scale_min=augment_scale_min,
            scale_max=augment_scale_max,
            time_mask_bases=augment_time_mask_bases,
            time_mask_count=augment_time_mask_count,
            shift_max_bases=augment_shift_max_bases,
            feature_noise_scale=augment_feature_noise_scale,
            time_stretch=augment_time_stretch,
        ),
        aux_head=AuxHeadConfig(
            adversarial_lambda=adversarial_lambda,
            adversarial_anneal_epochs=adversarial_anneal_epochs,
            cl_regression=cl_regression,
            cl_lambda=cl_lambda,
        ),
    )

    # Train model
    history = train_model(
        train_data_path=train_data,
        val_data_path=val_data,
        model_name=model_name,
        output_dir=output_dir,
        device=device,
        seed=seed,
        resume_from=resume,
        num_workers=num_workers,
        gpus=gpus,
        cfg=cfg,
        dwell_template_table=dwell_template_table,
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
