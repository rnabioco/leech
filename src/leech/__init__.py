"""
leech: Learning Enhanced Aminoacylation Classification from Hanopore signals

A Python library for training deep learning models on nanopore signal data to
detect RNA modifications, specifically designed for aa-tRNA-seq experiments.

Key Features:
    - Dwell time feature extraction from BAM move tables
    - Signal level statistics (mean, median, std, range) per base
    - Multiple model architectures (Conv-LSTM, Transformer, ResNet, TCN)
    - Integration with ONT POD5 and BAM file formats
    - Training, evaluation, and inference pipelines

Quick Start:
    >>> import leech
    >>> # Train a model
    >>> leech.train_model(train_data_path, val_data_path, model_name="ConvLSTMDwell")
    >>> # Run inference
    >>> leech.run_inference(model_path, pod5_path, bam_path, output_path)

For full documentation, see https://github.com/rnabioco/leech
"""

import subprocess
from pathlib import Path


def _get_git_revision():
    """Get the current git commit hash for version tracking."""
    try:
        repo_dir = Path(__file__).parent.parent.parent
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=1,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


__version__ = f"0.1.0-alpha+{_get_git_revision()}"

from leech.evaluation import evaluate_model
from leech.features import (
    compute_dwell_times,
    compute_signal_levels,
    extract_move_table,
)
from leech.inference import run_inference
from leech.training import train_model
from leech.util import load_model_from_checkpoint

__all__ = [
    "__version__",
    "compute_dwell_times",
    "compute_signal_levels",
    "extract_move_table",
    "train_model",
    "evaluate_model",
    "run_inference",
    "load_model_from_checkpoint",
]
