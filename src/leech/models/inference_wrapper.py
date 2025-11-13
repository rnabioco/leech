"""
Inference wrapper for unified model forward passes.

Eliminates conditional logic for models with/without feature inputs.
"""

import torch
import torch.nn as nn


class ModelInferenceWrapper:
    """
    Wrapper that provides unified forward pass interface for all model types.

    This eliminates the need for conditional if/else blocks when calling models
    that have different input signatures (signal+sequence vs signal+sequence+features).

    Example:
        # Instead of:
        if "features" in batch:
            logits = model(signal, sequence, features)
        else:
            logits = model(signal, sequence)

        # Use:
        wrapper = ModelInferenceWrapper(model, model_type)
        logits = wrapper.forward_batch(batch, device)
    """

    # Models that require dwell/signal features as input
    FEATURE_MODELS = {
        "ConvLSTMDwell",
        "TransformerDwell",
        "ConvOnly",
        "TCNDwell",
        "ResNetDwell",
        "ConvLSTMSignalFeatures",
        "TCNSignalFeatures",
    }

    # Models that do NOT require sequence input (signal + features only)
    SIGNAL_FEATURES_MODELS = {
        "ConvLSTMSignalFeatures",
        "TCNSignalFeatures",
    }

    def __init__(self, model: nn.Module, model_type: str):
        """
        Initialize wrapper.

        Args:
            model: PyTorch model to wrap
            model_type: Model architecture name (e.g., "ConvLSTMDwell")
        """
        self.model = model
        self.model_type = model_type
        self.requires_features = model_type in self.FEATURE_MODELS
        self.requires_sequence = model_type not in self.SIGNAL_FEATURES_MODELS

    def forward_batch(self, batch: dict, device: str) -> torch.Tensor:
        """
        Forward pass from batch dictionary.

        Automatically moves tensors to device and calls model with correct arguments.

        Args:
            batch: Batch dict with "signal", optionally "sequence", and optionally "features"
            device: Device to move tensors to

        Returns:
            Model logits
        """
        signal = batch["signal"].to(device)

        output: torch.Tensor

        # Signal + Features only models (no sequence)
        if not self.requires_sequence:
            features = batch["features"].to(device)
            output = self.model(signal, features)
        # Signal + Sequence + Features models
        elif self.requires_features:
            sequence = batch["sequence"].to(device)
            features = batch["features"].to(device)
            output = self.model(signal, sequence, features)
        # Signal + Sequence only models (baseline)
        else:
            sequence = batch["sequence"].to(device)
            output = self.model(signal, sequence)

        return output

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Direct call delegates to underlying model.

        This allows the wrapper to be used as a drop-in replacement in most contexts.
        """
        output: torch.Tensor = self.model(*args, **kwargs)
        return output

    def train(self) -> None:
        """Set model to training mode."""
        self.model.train()

    def eval(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()

    def to(self, device: str) -> "ModelInferenceWrapper":
        """
        Move model to device.

        Args:
            device: Device to move to

        Returns:
            Self for chaining
        """
        self.model.to(device)
        return self

    @property
    def parameters(self):
        """Access model parameters for optimizer."""
        return self.model.parameters()

    def state_dict(self):
        """Get model state dict for checkpointing."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        """Load model state dict from checkpoint."""
        self.model.load_state_dict(state_dict)
