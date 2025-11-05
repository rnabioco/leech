"""
Model architectures for leech.

Available models:
- ConvLSTMBase: Baseline model (signal + sequence only)
- ConvLSTMDwell: Full model with dwell time features
"""

from leech.models.conv_lstm_base import ConvLSTMBase
from leech.models.conv_lstm_dwell import ConvLSTMDwell

__all__ = ["ConvLSTMBase", "ConvLSTMDwell"]


# Model registry for dynamic loading
MODEL_REGISTRY = {
    "ConvLSTMBase": ConvLSTMBase,
    "ConvLSTMDwell": ConvLSTMDwell,
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
