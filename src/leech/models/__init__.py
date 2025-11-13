"""
Model architectures for leech.

Available models:
- ConvLSTMBase: Baseline model (signal + sequence only)
- ConvLSTMDwell: Full model with dwell time features
- ConvLSTMSignalFeatures: Signal + features only (optimized for constant sequences)
- TransformerDwell: Transformer-based model with self-attention (Priority 1)
- ConvOnly: Pure CNN baseline with multi-scale convolutions (Priority 1)
- TCNDwell: Temporal Convolutional Network with dilated convolutions (Priority 2)
- TCNSignalFeatures: TCN with signal + features only (optimized for constant sequences)
- ResNetDwell: Residual Network with skip connections (Priority 2)
"""

import torch.nn as nn

from leech.models.conv_lstm_base import ConvLSTMBase
from leech.models.conv_lstm_dwell import ConvLSTMDwell
from leech.models.conv_lstm_signal_features import ConvLSTMSignalFeatures
from leech.models.conv_only import ConvOnly
from leech.models.inference_wrapper import ModelInferenceWrapper
from leech.models.resnet_dwell import ResNetDwell
from leech.models.tcn_dwell import TCNDwell
from leech.models.tcn_signal_features import TCNSignalFeatures
from leech.models.transformer_dwell import TransformerDwell

__all__ = [
    "ConvLSTMBase",
    "ConvLSTMDwell",
    "ConvLSTMSignalFeatures",
    "TransformerDwell",
    "ConvOnly",
    "TCNDwell",
    "TCNSignalFeatures",
    "ResNetDwell",
    "ModelInferenceWrapper",
]


# Model registry for dynamic loading
MODEL_REGISTRY = {
    "ConvLSTMBase": ConvLSTMBase,
    "ConvLSTMDwell": ConvLSTMDwell,
    "ConvLSTMSignalFeatures": ConvLSTMSignalFeatures,
    "TransformerDwell": TransformerDwell,
    "ConvOnly": ConvOnly,
    "TCNDwell": TCNDwell,
    "TCNSignalFeatures": TCNSignalFeatures,
    "ResNetDwell": ResNetDwell,
}


def get_model(model_name: str, **kwargs) -> nn.Module:
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

    return MODEL_REGISTRY[model_name](**kwargs)  # type: ignore[no-any-return]
