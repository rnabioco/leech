"""
Model architectures for leech.

Architectures come from two sources, both reachable through ``get_model()``
and ``MODEL_REGISTRY``:

1. **TOML configs** under ``leech/models/configs/`` (see
   :mod:`leech.models.config_loader`): fully declarative architectures wired
   from the layer registry in :mod:`leech.models.nn`.  This is where the
   ConvLSTM and TCN families live, one config per family with a
   ``[[variants]]`` block per registry name.
2. **Hand-written classes** listed in ``_MODEL_SPECS`` below, for
   architectures that have not been converted yet.

Registered models:
- ConvLSTMBase: Baseline model (signal + sequence only)                [config]
- ConvLSTMBaseAttn: ConvLSTMBase with attention pooling                [config]
- ConvLSTMBaseBN: ConvLSTMBase with batch normalization                [config]
- ConvLSTMBaseBNAttn: ConvLSTMBase with batch norm and attention       [config]
- ConvLSTMDwell: Full model with dwell time features (recommended)     [config]
- ConvLSTMDwellAttn: ConvLSTMDwell with attention pooling              [config]
- ConvLSTMDwellBN: ConvLSTMDwell with batch normalization              [config]
- ConvLSTMDwellBNAttn: ConvLSTMDwell with batch norm and attention     [config]
- ConvLSTMDwellGNAttn: ConvLSTMDwell with group norm and attention     [config]
- ConvLSTMDwellLNAttn: ConvLSTMDwell with layer norm and attention     [config]
- ConvLSTMRemora: Remora-compatible architecture with dwell features
- ConvLSTMRemoraBase: Remora-compatible architecture without dwell features
- TransformerDwell: Transformer-based model with self-attention
- TransformerDwellResidual: TransformerDwell with 2-channel signal
- ConvOnly: Pure CNN baseline with multi-scale convolutions
- TCNDwell: Temporal Convolutional Network with dilated convolutions    [config]
- TCNDwellGN: TCNDwell with group normalization                        [config]
- TCNDwellLN: TCNDwell with layer normalization                        [config]
- TCNDwellResidual: TCNDwell with 2-channel signal input               [config]
- TCNDwellResidualGN: TCNDwellResidual with group normalization        [config]
- TCNDwellResidualLN: TCNDwellResidual with layer normalization        [config]
- TCNDwellResidualMotor: TCNDwellResidual + motor-region pooling       [config]
- TCNDwellResidualLNMotor: as above with layer normalization           [config]
- TCNDwellResidualDwellAttn: TCNDwellResidual + dwell-only attention   [config]
- TCNDwellResidualLNDwellAttn: as above with layer normalization       [config]
- TCNDwellSplitResidual: separate raw-signal / residual branches       [config]
- TCNDwellSplitResidualLN: TCNDwellSplitResidual with layer norm       [config]
- ResNetDwell: Residual Network with skip connections
- SignalCNN: Signal-only 1D-CNN classifier (ignores sequence/dwell inputs)
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

# ── Hand-written architectures ──────────────────────────────────────────────
# name -> (submodule, class name). Adding an architecture here makes it appear
# in MODEL_REGISTRY, get_model(), the CLI `--model` choices, and `from
# leech.models import <Name>` — all derived, nothing to keep in sync.
#
# This module is deliberately torch-free at import time: the classes (and thus
# torch) load only when a model is actually accessed. That keeps `leech --help`
# and CLI choice rendering fast (importing this package used to pull in torch,
# ~10s). Config discovery is likewise torch-free — it only parses TOML.
_MODEL_SPECS: dict[str, tuple[str, str]] = {
    "ConvLSTMRemora": ("leech.models.conv_lstm_remora", "ConvLSTMRemora"),
    "ConvLSTMRemoraBase": ("leech.models.conv_lstm_remora", "ConvLSTMRemoraBase"),
    "TransformerDwell": ("leech.models.transformer_dwell", "TransformerDwell"),
    "TransformerDwellResidual": ("leech.models.transformer_dwell", "TransformerDwellResidual"),
    "ConvOnly": ("leech.models.conv_only", "ConvOnly"),
    "ResNetDwell": ("leech.models.resnet_dwell", "ResNetDwell"),
    "SignalCNN": ("leech.models.signal_cnn", "SignalCNN"),
}

# Non-registry classes also re-exported from this package (inference helpers).
_EXTRA_EXPORTS: dict[str, tuple[str, str]] = {
    "ModelInferenceWrapper": ("leech.models.inference_wrapper", "ModelInferenceWrapper"),
    "TracedModelWrapper": ("leech.models.inference_wrapper", "TracedModelWrapper"),
    "RemoraModelWrapper": ("leech.models.remora_compat", "RemoraModelWrapper"),
}


def _config_names() -> dict[str, tuple[str, str | None]]:
    """Names declared by TOML configs (torch-free: parses TOML only)."""
    from leech.models.config_loader import discover_configs

    return discover_configs()


def _load(spec: tuple[str, str]) -> Any:
    module, cls = spec
    return getattr(importlib.import_module(module), cls)


def _resolve(name: str) -> Any:
    """Resolve a registry name to its class, importing torch on demand."""
    if name in _MODEL_SPECS:
        return _load(_MODEL_SPECS[name])
    from leech.models.config_loader import build_model_class

    return build_model_class(name)


class _LazyModelRegistry(Mapping):
    """Mapping of model name -> class that builds each class on first access.

    ``keys()``, ``in``, and ``len()`` are torch-free (they only touch the spec
    table and the TOML configs); indexing (``registry[name]``) imports/builds
    and returns the actual class.
    """

    def __init__(self, specs: dict[str, tuple[str, str]]):
        self._specs = specs

    def _names(self) -> list[str]:
        return [*self._specs, *_config_names()]

    def __getitem__(self, name: str) -> Any:
        if name not in self._specs and name not in _config_names():
            raise KeyError(name)
        return _resolve(name)

    def __iter__(self):
        return iter(self._names())

    def __len__(self) -> int:
        return len(self._specs) + len(_config_names())

    def __contains__(self, name: object) -> bool:
        return name in self._specs or name in _config_names()


MODEL_REGISTRY = _LazyModelRegistry(_MODEL_SPECS)


def __getattr__(name: str) -> Any:
    # PEP 562: resolve `from leech.models import ConvLSTMDwell` (and the extra
    # wrapper exports) lazily so importing this package stays torch-free.
    if name == "__all__":
        return [*_MODEL_SPECS, *_config_names(), *_EXTRA_EXPORTS]
    if name in _EXTRA_EXPORTS:
        return _load(_EXTRA_EXPORTS[name])
    if name in _MODEL_SPECS or name in _config_names():
        return _resolve(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_MODEL_SPECS, *_config_names(), *_EXTRA_EXPORTS])


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
