"""
Shared CLI option decorators for the leech CLI.

Defines reusable decorator groups for options shared across multiple commands.
"""

import rich_click as click

from leech.constants import (
    DEFAULT_AUGMENT_FEATURE_NOISE_SCALE,
    DEFAULT_AUGMENT_JITTER,
    DEFAULT_AUGMENT_SCALE_MAX,
    DEFAULT_AUGMENT_SCALE_MIN,
    DEFAULT_AUGMENT_SHIFT_MAX_BASES,
    DEFAULT_AUGMENT_TIME_MASK_BASES,
    DEFAULT_AUGMENT_TIME_MASK_COUNT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_LABEL_SMOOTHING,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOSS_TYPE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MIXED_PRECISION,
    DEFAULT_SCHEDULER,
    DEFAULT_SCHEDULER_FACTOR,
    DEFAULT_SCHEDULER_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
)


# Model choices for CLI. leech.models is torch-free at import time and its
# MODEL_REGISTRY resolves classes lazily, so listing the names (to render
# `--model` help / validate choices) does not import torch (~10s).
def get_model_choices() -> list[str]:
    """Return sorted model names without importing torch."""
    from leech.models import MODEL_REGISTRY

    return sorted(MODEL_REGISTRY.keys())


class FloatOrDict(click.ParamType):
    """Click type that accepts a plain float or ``key=val,key=val`` per-channel dict.

    Examples::

        0.02                                    → 0.02
        signal=0.02,signal_residual=0.001       → {"signal": 0.02, "signal_residual": 0.001}
    """

    name = "FLOAT_OR_DICT"

    def convert(self, value, param, ctx):
        if isinstance(value, (int, float, dict)):
            return value
        try:
            return float(value)
        except ValueError:
            pass
        # Parse key=val,key=val
        try:
            result = {}
            for part in value.split(","):
                k, v = part.strip().split("=", 1)
                result[k.strip()] = float(v.strip())
            return result
        except (ValueError, AttributeError):
            self.fail(f"'{value}' is not a float or key=value,... dict", param, ctx)


FLOAT_OR_DICT = FloatOrDict()


class LazyChoice(click.Choice):
    """A click.Choice that defers resolving its choices until first access.

    This avoids importing heavy dependencies (e.g., torch via MODEL_REGISTRY)
    at CLI decoration time, keeping ``leech --help`` fast.
    """

    def __init__(self, choices_fn, case_sensitive=True):
        self._choices_fn = choices_fn
        self._resolved = False
        self.case_sensitive = case_sensitive
        self.name = "CHOICE"

    @property
    def choices(self):
        if not self._resolved:
            self._choices = self._choices_fn()
            self._resolved = True
        return self._choices

    @choices.setter
    def choices(self, value):
        self._choices = value


def training_hyperparams(f):
    """Training hyperparameter options shared by ``train`` and ``optimize``.

    Includes: epochs, batch-size, learning-rate, device, seed, early-stopping,
    weight-decay, max-grad-norm, scheduler options, loss options, mixed-precision,
    and data augmentation options.
    """
    # Applied in reverse order so --help display matches the original ordering.
    f = click.option(
        "--augment-feature-noise-scale",
        type=float,
        default=DEFAULT_AUGMENT_FEATURE_NOISE_SCALE,
        help="Per-channel Gaussian noise scale for features (0 = disabled)",
    )(f)
    f = click.option(
        "--augment-shift-max-bases",
        type=float,
        default=DEFAULT_AUGMENT_SHIFT_MAX_BASES,
        help="Max cross-layer shift in bases; float for sub-base resolution (0 = disabled)",
    )(f)
    f = click.option(
        "--augment-time-mask-count",
        type=int,
        default=DEFAULT_AUGMENT_TIME_MASK_COUNT,
        help="Number of time masks to apply (default: 1)",
    )(f)
    f = click.option(
        "--augment-time-mask-bases",
        type=int,
        default=DEFAULT_AUGMENT_TIME_MASK_BASES,
        help="Max width in bases for time masking (0 = disabled)",
    )(f)
    f = click.option(
        "--augment-scale-max",
        type=float,
        default=DEFAULT_AUGMENT_SCALE_MAX,
        help="Max random scale factor for signal augmentation",
    )(f)
    f = click.option(
        "--augment-scale-min",
        type=float,
        default=DEFAULT_AUGMENT_SCALE_MIN,
        help="Min random scale factor for signal augmentation",
    )(f)
    f = click.option(
        "--augment-jitter",
        type=FLOAT_OR_DICT,
        default=DEFAULT_AUGMENT_JITTER,
        help="Signal jitter noise std dev (0 = disabled). "
        "Per-channel: signal=0.02,signal_residual=0.001",
    )(f)
    f = click.option(
        "--mixed-precision/--no-mixed-precision",
        default=DEFAULT_MIXED_PRECISION,
        help="Enable mixed precision training (CUDA only)",
    )(f)
    f = click.option(
        "--label-smoothing",
        type=float,
        default=DEFAULT_LABEL_SMOOTHING,
        help="Label smoothing factor (0 = disabled; e.g., 0.05 softens 0/1 targets)",
    )(f)
    f = click.option(
        "--focal-gamma",
        type=float,
        default=DEFAULT_FOCAL_GAMMA,
        help="Focal loss gamma parameter (only used with --loss focal)",
    )(f)
    f = click.option(
        "--loss",
        "loss_type",
        type=click.Choice(["bce", "focal", "cross_entropy"]),
        default=DEFAULT_LOSS_TYPE,
        help="Loss function type",
    )(f)
    f = click.option(
        "--warmup-epochs",
        type=int,
        default=DEFAULT_WARMUP_EPOCHS,
        help="Number of LR warmup epochs (0 = disabled)",
    )(f)
    f = click.option(
        "--scheduler-factor",
        type=float,
        default=DEFAULT_SCHEDULER_FACTOR,
        help="Factor to reduce LR by (for reduce_on_plateau)",
    )(f)
    f = click.option(
        "--scheduler-patience",
        type=int,
        default=DEFAULT_SCHEDULER_PATIENCE,
        help="Epochs to wait before reducing LR (for reduce_on_plateau)",
    )(f)
    f = click.option(
        "--scheduler",
        type=click.Choice(["none", "reduce_on_plateau", "cosine"]),
        default=DEFAULT_SCHEDULER,
        help="Learning rate scheduler",
    )(f)
    f = click.option(
        "--max-grad-norm",
        type=float,
        default=DEFAULT_MAX_GRAD_NORM,
        help="Max gradient norm for clipping (0 = disabled)",
    )(f)
    f = click.option(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="L2 weight decay for optimizer (0 = disabled)",
    )(f)
    f = click.option(
        "--early-stopping",
        type=int,
        default=10,
        help="Patience for early stopping: stop training if validation accuracy doesn't improve for N epochs (set to 0 to disable)",
    )(f)
    f = click.option(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility",
    )(f)
    f = click.option(
        "--device",
        type=click.Choice(["cuda", "cpu"]),
        default=DEFAULT_DEVICE,
        help="Device for training",
    )(f)
    f = click.option(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate",
    )(f)
    f = click.option(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size",
    )(f)
    f = click.option(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs",
    )(f)
    return f


def model_provenance(f):
    """Model provenance options shared by ``train`` and ``optimize``.

    Includes: motif (required), motif-offset, base-justify.
    These are recorded in config.json for inference reproducibility.
    """
    # Applied in reverse order so --help display matches logical ordering.
    f = click.option(
        "--base-justify",
        type=click.Choice(["start", "center", "end"]),
        default="center",
        help="Signal justification within focus base (recorded in config)",
    )(f)
    f = click.option(
        "--motif-offset",
        type=int,
        default=0,
        help="Offset within motif for focus base (0-indexed, recorded in config)",
    )(f)
    f = click.option(
        "--motif",
        type=str,
        required=True,
        help="Motif used for chunk extraction (recorded in config for provenance/inference)",
    )(f)
    return f
