"""
Handler for the 'optimize' command.

This module contains the business logic for hyperparameter optimization
via grid search over chunk contexts.
"""

import logging
from pathlib import Path

import rich_click as click

from leech.cli_config import make_console

logger = logging.getLogger("leech.commands.optimize")
console = make_console()


def handle_optimize(
    train_data: Path,
    val_data: Path | None,
    model_name: str,
    output_dir: Path,
    context_grid: str | None,
    left_contexts: str | None,
    right_contexts: str | None,
    kmer_context: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    seed: int,
    early_stopping: int,
    base_justify: str,
    dwell_offsets: str,
    parallel: int,
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
    num_workers: int,
    balance_groups: bool = False,
    motif: str = "",
    motif_offset: int = 0,
    adversarial_lambda: float = 0.0,
    adversarial_anneal_epochs: int = 0,
    confound: str | None = None,
    cl_regression: bool = False,
    cl_lambda: float = 1.0,
) -> Path:
    """
    Handle the optimize command logic.

    Args:
        train_data: Training dataset (.npz)
        val_data: Validation dataset (.npz)
        model_name: Model architecture name
        output_dir: Output directory for grid results
        context_grid: Fallback context values (comma-separated or start:stop:step)
        left_contexts: Override left context grid
        right_contexts: Override right context grid
        kmer_context: K-mer context for sequence encoding
        epochs: Training epochs per grid point
        batch_size: Training batch size
        learning_rate: Learning rate
        device: Device for training
        seed: Random seed
        early_stopping: Early stopping patience
        base_justify: Signal chunk centering
        dwell_offsets: Dwell offset values to search
        parallel: Number of grid points to run concurrently
        weight_decay: L2 weight decay
        max_grad_norm: Gradient clipping norm
        scheduler: LR scheduler type
        scheduler_patience: LR scheduler patience
        scheduler_factor: LR scheduler factor
        warmup_epochs: Linear warmup epochs
        loss_type: Loss function type
        focal_gamma: Focal loss gamma
        mixed_precision: Enable mixed precision
        augment_jitter: Signal jitter noise std dev
        augment_scale_min: Min random scale factor
        augment_scale_max: Max random scale factor
        num_workers: DataLoader workers
        balance_groups: Balance sampling across source groups

    Returns:
        Path to grid search summary file
    """
    from leech.gridsearch import GridSearchConfig, parse_context_grid, parse_values, run_grid_search

    # Validate: need context_grid as fallback if left/right not both provided
    if context_grid is None and (left_contexts is None or right_contexts is None):
        raise click.UsageError(
            "--context-grid is required when --left-contexts or --right-contexts is not provided"
        )

    # Parse context grids
    left_contexts_list, right_contexts_list = parse_context_grid(
        context_grid, left_contexts, right_contexts
    )

    # Parse dwell offsets
    dwell_offsets_list = parse_values(dwell_offsets)

    logger.info(
        f"Starting grid search with {len(left_contexts_list)} x {len(right_contexts_list)} x {len(dwell_offsets_list)} grid points"
    )
    logger.info(f"Left contexts: {left_contexts_list}")
    logger.info(f"Right contexts: {right_contexts_list}")
    logger.info(f"Dwell offsets: {dwell_offsets_list}")

    # Create config
    config = GridSearchConfig(
        train_data_path=train_data,
        val_data_path=val_data,
        model_name=model_name,
        output_dir=output_dir,
        left_contexts=left_contexts_list,
        right_contexts=right_contexts_list,
        kmer_context=kmer_context,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
        early_stopping_patience=early_stopping,
        base_justify=base_justify,
        dwell_offsets=dwell_offsets_list,
        n_parallel=parallel,
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
        num_workers=num_workers,
        balance_groups=balance_groups,
        motif=motif,
        motif_offset=motif_offset,
        adversarial_lambda=adversarial_lambda,
        adversarial_anneal_epochs=adversarial_anneal_epochs,
        confound=confound,
        cl_regression=cl_regression,
        cl_lambda=cl_lambda,
    )

    # Run grid search
    summary_path = run_grid_search(config)

    console.print("[bold green]Grid search complete![/bold green]")
    logger.info(f"Results saved to: {summary_path}")

    return summary_path
