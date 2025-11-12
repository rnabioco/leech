"""
Pydantic configuration models for leech.

Centralizes configuration management with validation and type safety.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from leech.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_KMER_CONTEXT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_SIGNAL_CONTEXT,
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
    seed: int = DEFAULT_SEED
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
    seed: int = DEFAULT_SEED

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
    seed: int = DEFAULT_SEED
    early_stopping_patience: int = 10

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
    seed: int = DEFAULT_SEED
    early_stopping_patience: int = 10

    @field_validator("left_contexts", "right_contexts")
    @classmethod
    def validate_contexts(cls, v: list[int]) -> list[int]:
        """Ensure all context values are positive."""
        if not all(c > 0 for c in v):
            raise ValueError(f"All context values must be positive, got {v}")
        return v
