"""
Model architectures for leech.

Available models (14 total):
- ConvLSTMBase: Baseline model (signal + sequence only)
- ConvLSTMBaseAttn: ConvLSTMBase with attention pooling
- ConvLSTMBaseBN: ConvLSTMBase with batch normalization
- ConvLSTMBaseBNAttn: ConvLSTMBase with batch normalization and attention
- ConvLSTMDwell: Full model with dwell time features (recommended)
- ConvLSTMDwellAttn: ConvLSTMDwell with attention pooling
- ConvLSTMDwellBN: ConvLSTMDwell with batch normalization
- ConvLSTMDwellBNAttn: ConvLSTMDwell with batch normalization and attention
- ConvLSTMRemora: Remora-compatible architecture with dwell features
- ConvLSTMRemoraBase: Remora-compatible architecture without dwell features
- TransformerDwell: Transformer-based model with self-attention
- ConvOnly: Pure CNN baseline with multi-scale convolutions
- TCNDwell: Temporal Convolutional Network with dilated convolutions
- ResNetDwell: Residual Network with skip connections
"""

from typing import Any

import torch.nn as nn

from leech.models.conv_lstm import (
    ConvLSTMBase,
    ConvLSTMBaseAttn,
    ConvLSTMBaseBN,
    ConvLSTMBaseBNAttn,
    ConvLSTMCore,
    ConvLSTMDwell,
    ConvLSTMDwellAttn,
    ConvLSTMDwellBN,
    ConvLSTMDwellBNAttn,
)
from leech.models.conv_lstm_remora import ConvLSTMRemora, ConvLSTMRemoraBase
from leech.models.conv_only import ConvOnly
from leech.models.inference_wrapper import ModelInferenceWrapper, TracedModelWrapper
from leech.models.remora_compat import RemoraModelWrapper
from leech.models.resnet_dwell import ResNetDwell
from leech.models.tcn_dwell import TCNDwell
from leech.models.transformer_dwell import TransformerDwell

__all__ = [
    "ConvLSTMCore",
    "ConvLSTMBase",
    "ConvLSTMBaseAttn",
    "ConvLSTMBaseBN",
    "ConvLSTMBaseBNAttn",
    "ConvLSTMDwell",
    "ConvLSTMDwellAttn",
    "ConvLSTMDwellBN",
    "ConvLSTMDwellBNAttn",
    "ConvLSTMRemora",
    "ConvLSTMRemoraBase",
    "TransformerDwell",
    "ConvOnly",
    "TCNDwell",
    "ResNetDwell",
    "ModelInferenceWrapper",
    "TracedModelWrapper",
    "RemoraModelWrapper",
]


# Model registry for dynamic loading
MODEL_REGISTRY = {
    "ConvLSTMBase": ConvLSTMBase,
    "ConvLSTMBaseAttn": ConvLSTMBaseAttn,
    "ConvLSTMBaseBN": ConvLSTMBaseBN,
    "ConvLSTMBaseBNAttn": ConvLSTMBaseBNAttn,
    "ConvLSTMDwell": ConvLSTMDwell,
    "ConvLSTMDwellAttn": ConvLSTMDwellAttn,
    "ConvLSTMDwellBN": ConvLSTMDwellBN,
    "ConvLSTMDwellBNAttn": ConvLSTMDwellBNAttn,
    "ConvLSTMRemora": ConvLSTMRemora,
    "ConvLSTMRemoraBase": ConvLSTMRemoraBase,
    "TransformerDwell": TransformerDwell,
    "ConvOnly": ConvOnly,
    "TCNDwell": TCNDwell,
    "ResNetDwell": ResNetDwell,
}


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
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    return MODEL_REGISTRY[model_name](**kwargs)
