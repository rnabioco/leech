"""
Model architectures for leech.

Available models (24 total):
- ConvLSTMBase: Baseline model (signal + sequence only)
- ConvLSTMBaseAttn: ConvLSTMBase with attention pooling
- ConvLSTMBaseBN: ConvLSTMBase with batch normalization
- ConvLSTMBaseBNAttn: ConvLSTMBase with batch normalization and attention
- ConvLSTMDwell: Full model with dwell time features (recommended)
- ConvLSTMDwellAttn: ConvLSTMDwell with attention pooling
- ConvLSTMDwellBN: ConvLSTMDwell with batch normalization
- ConvLSTMDwellBNAttn: ConvLSTMDwell with batch normalization and attention
- ConvLSTMDwellGNAttn: ConvLSTMDwell with group normalization and attention
- ConvLSTMDwellLNAttn: ConvLSTMDwell with layer normalization and attention
- ConvLSTMRemora: Remora-compatible architecture with dwell features
- ConvLSTMRemoraBase: Remora-compatible architecture without dwell features
- TransformerDwell: Transformer-based model with self-attention
- TransformerDwellResidual: TransformerDwell with 2-channel signal (raw + kmer residual)
- ConvOnly: Pure CNN baseline with multi-scale convolutions
- TCNDwell: Temporal Convolutional Network with dilated convolutions
- TCNDwellGN: TCNDwell with group normalization
- TCNDwellLN: TCNDwell with layer normalization
- TCNDwellResidual: TCNDwell with 2-channel signal (raw + kmer residual)
- TCNDwellResidualGN: TCNDwellResidual with group normalization
- TCNDwellResidualLN: TCNDwellResidual with layer normalization
- TCNDwellSplitResidual: TCN with separate branches for raw signal and kmer residual
- TCNDwellSplitResidualLN: TCNDwellSplitResidual with layer normalization
- ResNetDwell: Residual Network with skip connections
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

# ── Single source of truth ──────────────────────────────────────────────────
# name -> (submodule, class name). Adding an architecture here makes it appear
# in MODEL_REGISTRY, get_model(), the CLI `--model` choices, and `from
# leech.models import <Name>` — all derived, nothing to keep in sync.
#
# This module is deliberately torch-free at import time: the classes (and thus
# torch) load only when a model is actually accessed. That keeps `leech --help`
# and CLI choice rendering fast (importing this package used to pull in torch,
# ~10s).
_MODEL_SPECS: dict[str, tuple[str, str]] = {
    "ConvLSTMBase": ("leech.models.conv_lstm", "ConvLSTMBase"),
    "ConvLSTMBaseAttn": ("leech.models.conv_lstm_attn", "ConvLSTMBaseAttn"),
    "ConvLSTMBaseBN": ("leech.models.conv_lstm", "ConvLSTMBaseBN"),
    "ConvLSTMBaseBNAttn": ("leech.models.conv_lstm_attn", "ConvLSTMBaseBNAttn"),
    "ConvLSTMDwell": ("leech.models.conv_lstm", "ConvLSTMDwell"),
    "ConvLSTMDwellAttn": ("leech.models.conv_lstm_attn", "ConvLSTMDwellAttn"),
    "ConvLSTMDwellBN": ("leech.models.conv_lstm", "ConvLSTMDwellBN"),
    "ConvLSTMDwellBNAttn": ("leech.models.conv_lstm_attn", "ConvLSTMDwellBNAttn"),
    "ConvLSTMDwellGNAttn": ("leech.models.conv_lstm_attn", "ConvLSTMDwellGNAttn"),
    "ConvLSTMDwellLNAttn": ("leech.models.conv_lstm_attn", "ConvLSTMDwellLNAttn"),
    "ConvLSTMRemora": ("leech.models.conv_lstm_remora", "ConvLSTMRemora"),
    "ConvLSTMRemoraBase": ("leech.models.conv_lstm_remora", "ConvLSTMRemoraBase"),
    "TransformerDwell": ("leech.models.transformer_dwell", "TransformerDwell"),
    "TransformerDwellResidual": ("leech.models.transformer_dwell", "TransformerDwellResidual"),
    "ConvOnly": ("leech.models.conv_only", "ConvOnly"),
    "TCNDwell": ("leech.models.tcn_dwell", "TCNDwell"),
    "TCNDwellGN": ("leech.models.tcn_dwell", "TCNDwellGN"),
    "TCNDwellLN": ("leech.models.tcn_dwell", "TCNDwellLN"),
    "TCNDwellResidual": ("leech.models.tcn_dwell_residual", "TCNDwellResidual"),
    "TCNDwellResidualGN": ("leech.models.tcn_dwell_residual", "TCNDwellResidualGN"),
    "TCNDwellResidualLN": ("leech.models.tcn_dwell_residual", "TCNDwellResidualLN"),
    "TCNDwellSplitResidual": ("leech.models.tcn_dwell_split_residual", "TCNDwellSplitResidual"),
    "TCNDwellSplitResidualLN": ("leech.models.tcn_dwell_split_residual", "TCNDwellSplitResidualLN"),
    "ResNetDwell": ("leech.models.resnet_dwell", "ResNetDwell"),
}

# Non-registry classes also re-exported from this package (inference helpers).
_EXTRA_EXPORTS: dict[str, tuple[str, str]] = {
    "ModelInferenceWrapper": ("leech.models.inference_wrapper", "ModelInferenceWrapper"),
    "TracedModelWrapper": ("leech.models.inference_wrapper", "TracedModelWrapper"),
    "RemoraModelWrapper": ("leech.models.remora_compat", "RemoraModelWrapper"),
}

__all__ = [*_MODEL_SPECS, *_EXTRA_EXPORTS]


def _load(spec: tuple[str, str]) -> Any:
    module, cls = spec
    return getattr(importlib.import_module(module), cls)


class _LazyModelRegistry(Mapping):
    """Mapping of model name -> class that imports each class on first access.

    ``keys()``, ``in``, and ``len()`` are torch-free (they only touch the spec
    table); indexing (``registry[name]``) imports and returns the actual class.
    """

    def __init__(self, specs: dict[str, tuple[str, str]]):
        self._specs = specs

    def __getitem__(self, name: str) -> Any:
        return _load(self._specs[name])

    def __iter__(self):
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs


MODEL_REGISTRY = _LazyModelRegistry(_MODEL_SPECS)


def __getattr__(name: str) -> Any:
    # PEP 562: resolve `from leech.models import ConvLSTMDwell` (and the extra
    # wrapper exports) lazily so importing this package stays torch-free.
    if name in _MODEL_SPECS:
        return _load(_MODEL_SPECS[name])
    if name in _EXTRA_EXPORTS:
        return _load(_EXTRA_EXPORTS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])


def get_model(model_name: str, **kwargs: Any) -> nn.Module:
    """
    Get model by name.

    Args:
        model_name: Name of model architecture
        **kwargs: Model-specific parameters (passed to model constructor)

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_name not in registry
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    return MODEL_REGISTRY[model_name](**kwargs)
