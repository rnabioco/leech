"""
Pydantic configuration models for leech.

Centralizes configuration management with validation and type safety.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from leech.constants import (
    DEFAULT_AUGMENT_JITTER,
    DEFAULT_AUGMENT_SCALE_MAX,
    DEFAULT_AUGMENT_SCALE_MIN,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_KMER_CONTEXT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOSS_TYPE,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MIXED_PRECISION,
    DEFAULT_SCHEDULER,
    DEFAULT_SCHEDULER_FACTOR,
    DEFAULT_SCHEDULER_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_SIGNAL_CONTEXT,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
)


class DataPrepConfig(BaseModel):
    """Configuration for data preparation."""

    # Input files
    pod5_path: Path
    bam_path: Path
    output_dir: Path

    # Motif search
    motif: str | None = None
    motif_offset: int = 0
    motif_reference: Literal["bam", "fasta"] = "fasta"
    reference_fasta: Path | None = None
    skip_motif_indels: bool = True

    # Labels
    label: str | None = None
    label_int: int | None = None

    # Quality filtering
    min_mapq: int = 10

    # Feature extraction
    feature_set: str = "signal+dwell+levels"

    # Splitting
    train_split: float = 0.7
    val_split: float = 0.15
    seed: int | None = DEFAULT_SEED
    no_split: bool = False

    # Parallel processing
    workers: int = 1
    chunk_size: int = 100

    @field_validator("train_split", "val_split")
    @classmethod
    def validate_split_fraction(cls, v: float) -> float:
        """Ensure split fractions are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Split fraction must be between 0 and 1, got {v}")
        return v

    @field_validator("workers")
    @classmethod
    def validate_workers(cls, v: int) -> int:
        """Ensure workers is positive."""
        if v < 1:
            raise ValueError(f"Workers must be >= 1, got {v}")
        return v

    def validate_splits(self) -> None:
        """Validate that split fractions sum to <= 1.0."""
        if self.train_split + self.val_split > 1.0:
            raise ValueError(
                f"train_split ({self.train_split}) + val_split ({self.val_split}) must be <= 1.0"
            )


class MergeAndSplitConfig(BaseModel):
    """Configuration for merge-and-split operation."""

    # Input files (list of label=file pairs or directories)
    input_chunks: list[str]
    output_dir: Path

    # Splitting
    train_split: float = 0.7
    val_split: float = 0.15
    seed: int | None = DEFAULT_SEED

    # Comparison spec (for batch processing)
    comparison_spec: Path | None = None

    @field_validator("train_split", "val_split")
    @classmethod
    def validate_split_fraction(cls, v: float) -> float:
        """Ensure split fractions are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Split fraction must be between 0 and 1, got {v}")
        return v


class ModelConfig(BaseModel):
    """Configuration for model architecture."""

    # Model architecture
    model_name: str = "ConvLSTMDwell"
    signal_len: int = DEFAULT_SIGNAL_CONTEXT[0] + DEFAULT_SIGNAL_CONTEXT[1]
    kmer_len: int = 2 * DEFAULT_KMER_CONTEXT + 1
    num_features: int | None = None

    # Model-specific hyperparameters (flexible dict for model variants)
    model_kwargs: dict = Field(default_factory=dict)

    @field_validator("signal_len", "kmer_len")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Ensure lengths are positive."""
        if v <= 0:
            raise ValueError(f"Length must be positive, got {v}")
        return v


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    # Data paths
    train_data_path: Path
    val_data_path: Path | None = None
    output_dir: Path

    # Model config
    model: ModelConfig = Field(default_factory=ModelConfig)

    # Training hyperparameters
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    device: str = DEFAULT_DEVICE
    seed: int | None = DEFAULT_SEED
    early_stopping_patience: int = 10

    # Advanced training
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    scheduler: str = DEFAULT_SCHEDULER
    scheduler_patience: int = DEFAULT_SCHEDULER_PATIENCE
    scheduler_factor: float = DEFAULT_SCHEDULER_FACTOR
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS
    loss_type: str = DEFAULT_LOSS_TYPE
    focal_gamma: float = DEFAULT_FOCAL_GAMMA
    mixed_precision: bool = DEFAULT_MIXED_PRECISION

    # Data augmentation
    augment_jitter: float = DEFAULT_AUGMENT_JITTER
    augment_scale_min: float = DEFAULT_AUGMENT_SCALE_MIN
    augment_scale_max: float = DEFAULT_AUGMENT_SCALE_MAX

    # Checkpoint recovery
    resume_from: Path | None = None

    @field_validator("epochs", "batch_size")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Ensure positive values."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Ensure learning rate is positive."""
        if v <= 0:
            raise ValueError(f"Learning rate must be positive, got {v}")
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Ensure device is valid."""
        if v not in ["cuda", "cpu"]:
            raise ValueError(f"Device must be 'cuda' or 'cpu', got {v}")
        return v


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""

    # Paths
    model_path: Path
    test_data_path: Path
    output_path: Path

    # Device
    device: str = DEFAULT_DEVICE

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Ensure device is valid."""
        if v not in ["cuda", "cpu"]:
            raise ValueError(f"Device must be 'cuda' or 'cpu', got {v}")
        return v


class InferenceConfig(BaseModel):
    """Configuration for model inference."""

    # Paths
    model_path: Path
    pod5_path: Path
    bam_path: Path
    output_path: Path

    # Device
    device: str = DEFAULT_DEVICE

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Ensure device is valid."""
        if v not in ["cuda", "cpu"]:
            raise ValueError(f"Device must be 'cuda' or 'cpu', got {v}")
        return v


class GridSearchConfig(BaseModel):
    """Configuration for grid search."""

    # Data paths
    train_data_path: Path
    val_data_path: Path | None = None
    output_dir: Path

    # Model
    model_name: str = "ConvLSTMDwell"

    # Grid parameters
    left_contexts: list[int]
    right_contexts: list[int]
    kmer_context: int = DEFAULT_KMER_CONTEXT

    # Training parameters
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    device: str = DEFAULT_DEVICE
    seed: int | None = DEFAULT_SEED
    early_stopping_patience: int = 10

    @field_validator("left_contexts", "right_contexts")
    @classmethod
    def validate_contexts(cls, v: list[int]) -> list[int]:
        """Ensure all context values are positive."""
        if not all(c > 0 for c in v):
            raise ValueError(f"All context values must be positive, got {v}")
        return v
