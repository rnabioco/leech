"""
Model architectures for leech.

Available models:
- ConvLSTMBase: Baseline model (signal + sequence only)
- ConvLSTMDwell: Full model with dwell time features
- TransformerDwell: Transformer-based model with self-attention (Priority 1)
- ConvOnly: Pure CNN baseline with multi-scale convolutions (Priority 1)
- TCNDwell: Temporal Convolutional Network with dilated convolutions (Priority 2)
- ResNetDwell: Residual Network with skip connections (Priority 2)
"""

from typing import Any

import torch.nn as nn

from leech.models.conv_lstm_base import ConvLSTMBase
from leech.models.conv_lstm_base_attn import ConvLSTMBaseAttn
from leech.models.conv_lstm_base_bn import ConvLSTMBaseBN
from leech.models.conv_lstm_dwell import ConvLSTMDwell
from leech.models.conv_lstm_dwell_attn import ConvLSTMDwellAttn
from leech.models.conv_lstm_dwell_bn import ConvLSTMDwellBN
from leech.models.conv_lstm_remora import ConvLSTMRemora, ConvLSTMRemoraBase
from leech.models.conv_only import ConvOnly
from leech.models.inference_wrapper import ModelInferenceWrapper, TracedModelWrapper
from leech.models.remora_compat import RemoraModelWrapper
from leech.models.resnet_dwell import ResNetDwell
from leech.models.tcn_dwell import TCNDwell
from leech.models.transformer_dwell import TransformerDwell

__all__ = [
    "ConvLSTMBase",
    "ConvLSTMBaseAttn",
    "ConvLSTMBaseBN",
    "ConvLSTMDwell",
    "ConvLSTMDwellAttn",
    "ConvLSTMDwellBN",
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
    "ConvLSTMDwell": ConvLSTMDwell,
    "ConvLSTMDwellAttn": ConvLSTMDwellAttn,
    "ConvLSTMDwellBN": ConvLSTMDwellBN,
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
