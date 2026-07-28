"""
leech: Learning Enhanced Electrical Classifiers from Hanopore signals

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
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


# Lazy imports (PEP 562) — heavy modules (torch, sklearn, etc.) are only
# loaded when their public API symbols are actually accessed, not at
# package import time.  This cuts CLI startup from ~9s to ~0.2s.
_LAZY_IMPORTS = {
    "evaluate_model": "leech.evaluation",
    "compute_dwell_times": "leech.features",
    "compute_signal_levels": "leech.features",
    "extract_move_table": "leech.features",
    "run_inference": "leech.inference",
    "train_model": "leech.training",
    "load_model_from_checkpoint": "leech.model_loading",
}

_version = None


def __getattr__(name: str):
    if name == "__version__":
        global _version
        if _version is None:
            from importlib.metadata import PackageNotFoundError, version

            try:
                _version = version("leech")
            except PackageNotFoundError:
                _version = f"0.0.0+{_get_git_revision()}"
        return _version
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'leech' has no attribute {name!r}")


def __dir__():
    return __all__
