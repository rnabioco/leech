"""
Model architectures for leech.

Available models (20 total):
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
- ResNetDwell: Residual Network with skip connections
"""

from typing import Any

import torch.nn as nn

from leech.models.conv_lstm_base import ConvLSTMBase
from leech.models.conv_lstm_base_attn import ConvLSTMBaseAttn
from leech.models.conv_lstm_base_bn import ConvLSTMBaseBN
from leech.models.conv_lstm_base_bn_attn import ConvLSTMBaseBNAttn
from leech.models.conv_lstm_dwell import ConvLSTMDwell
from leech.models.conv_lstm_dwell_attn import ConvLSTMDwellAttn
from leech.models.conv_lstm_dwell_bn import ConvLSTMDwellBN
from leech.models.conv_lstm_dwell_bn_attn import ConvLSTMDwellBNAttn
from leech.models.conv_lstm_dwell_gn_attn import ConvLSTMDwellGNAttn
from leech.models.conv_lstm_dwell_ln_attn import ConvLSTMDwellLNAttn
from leech.models.conv_lstm_remora import ConvLSTMRemora, ConvLSTMRemoraBase
from leech.models.conv_only import ConvOnly
from leech.models.inference_wrapper import ModelInferenceWrapper, TracedModelWrapper
from leech.models.remora_compat import RemoraModelWrapper
from leech.models.resnet_dwell import ResNetDwell
from leech.models.tcn_dwell import TCNDwell
from leech.models.tcn_dwell_gn import TCNDwellGN
from leech.models.tcn_dwell_ln import TCNDwellLN
from leech.models.tcn_dwell_residual import TCNDwellResidual
from leech.models.tcn_dwell_residual_gn import TCNDwellResidualGN
from leech.models.tcn_dwell_residual_ln import TCNDwellResidualLN
from leech.models.tcn_dwell_split_residual import TCNDwellSplitResidual
from leech.models.tcn_dwell_split_residual_ln import TCNDwellSplitResidualLN
from leech.models.transformer_dwell import TransformerDwell
from leech.models.transformer_dwell_residual import TransformerDwellResidual

__all__ = [
    "ConvLSTMBase",
    "ConvLSTMBaseAttn",
    "ConvLSTMBaseBN",
    "ConvLSTMBaseBNAttn",
    "ConvLSTMDwell",
    "ConvLSTMDwellAttn",
    "ConvLSTMDwellBN",
    "ConvLSTMDwellBNAttn",
    "ConvLSTMDwellGNAttn",
    "ConvLSTMDwellLNAttn",
    "ConvLSTMRemora",
    "ConvLSTMRemoraBase",
    "TransformerDwell",
    "TransformerDwellResidual",
    "ConvOnly",
    "TCNDwell",
    "TCNDwellGN",
    "TCNDwellLN",
    "TCNDwellResidual",
    "TCNDwellResidualGN",
    "TCNDwellResidualLN",
    "TCNDwellSplitResidual",
    "TCNDwellSplitResidualLN",
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
    "ConvLSTMDwellGNAttn": ConvLSTMDwellGNAttn,
    "ConvLSTMDwellLNAttn": ConvLSTMDwellLNAttn,
    "ConvLSTMRemora": ConvLSTMRemora,
    "ConvLSTMRemoraBase": ConvLSTMRemoraBase,
    "TransformerDwell": TransformerDwell,
    "TransformerDwellResidual": TransformerDwellResidual,
    "ConvOnly": ConvOnly,
    "TCNDwell": TCNDwell,
    "TCNDwellGN": TCNDwellGN,
    "TCNDwellLN": TCNDwellLN,
    "TCNDwellResidual": TCNDwellResidual,
    "TCNDwellResidualGN": TCNDwellResidualGN,
    "TCNDwellResidualLN": TCNDwellResidualLN,
    "TCNDwellSplitResidual": TCNDwellSplitResidual,
    "TCNDwellSplitResidualLN": TCNDwellSplitResidualLN,
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
