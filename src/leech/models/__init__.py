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

from leech.models.conv_lstm_base import ConvLSTMBase
from leech.models.conv_lstm_dwell import ConvLSTMDwell
from leech.models.conv_only import ConvOnly
from leech.models.resnet_dwell import ResNetDwell
from leech.models.tcn_dwell import TCNDwell
from leech.models.transformer_dwell import TransformerDwell

__all__ = [
    "ConvLSTMBase",
    "ConvLSTMDwell",
    "TransformerDwell",
    "ConvOnly",
    "TCNDwell",
    "ResNetDwell",
]


# Model registry for dynamic loading
MODEL_REGISTRY = {
    "ConvLSTMBase": ConvLSTMBase,
    "ConvLSTMDwell": ConvLSTMDwell,
    "TransformerDwell": TransformerDwell,
    "ConvOnly": ConvOnly,
    "TCNDwell": TCNDwell,
    "ResNetDwell": ResNetDwell,
}


def get_model(model_name: str, **kwargs):
    """
    Get model by name.

    Args:
        model_name: Name of model architecture
        **kwargs: Model-specific parameters

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_name not in registry
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    return MODEL_REGISTRY[model_name](**kwargs)
