"""
Inference wrapper for unified model forward passes.

Eliminates conditional logic for models with/without feature inputs.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("leech.models.inference_wrapper")


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

    # Models that require dwell/signal features as third input
    FEATURE_MODELS = {
        "ConvLSTMDwell",
        "ConvLSTMDwellAttn",
        "ConvLSTMDwellBN",
        "ConvLSTMDwellBNAttn",
        "ConvLSTMDwellGNAttn",
        "ConvLSTMDwellLNAttn",
        "ConvLSTMRemora",
        "TransformerDwell",
        "TransformerDwellResidual",
        "ConvOnly",
        "TCNDwell",
        "TCNDwellGN",
        "TCNDwellLN",
        "TCNDwellResidual",
        "TCNDwellResidualGN",
        "TCNDwellResidualLN",
        "TCNDwellResidualMotor",
        "TCNDwellResidualLNMotor",
        "TCNDwellResidualDwellAttn",
        "TCNDwellResidualLNDwellAttn",
        "TCNDwellSplitResidual",
        "TCNDwellSplitResidualLN",
        "ResNetDwell",
    }

    # Models that receive the full dwell margin (no dwell_offset slicing)
    # Cross-attention lets each signal position attend to all dwell positions,
    # learning the physical motor-sensor offset
    WIDE_FEATURE_MODELS = {
        "ConvLSTMDwellAttn",
        "ConvLSTMDwellBNAttn",
        "ConvLSTMDwellGNAttn",
        "ConvLSTMDwellLNAttn",
        "TransformerDwell",
        "TransformerDwellResidual",
        "TCNDwell",
        "TCNDwellGN",
        "TCNDwellLN",
        "TCNDwellResidual",
        "TCNDwellResidualGN",
        "TCNDwellResidualLN",
        "TCNDwellResidualMotor",
        "TCNDwellResidualLNMotor",
        "TCNDwellResidualDwellAttn",
        "TCNDwellResidualLNDwellAttn",
        "TCNDwellSplitResidual",
        "TCNDwellSplitResidualLN",
        "ResNetDwell",
        "ConvOnly",
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
        self.captured_repr: torch.Tensor | None = None
        self._repr_hook = None

    def enable_repr_capture(self) -> int:
        """Register a hook to capture the penultimate representation.

        The hook intercepts the input to the classifier/fc head so that
        ``self.captured_repr`` holds the representation after each forward
        pass.  This is used by the adversarial and CL-regression heads.

        Returns:
            Dimension of the captured representation vector.
        """
        # Find the classification head (Sequential named 'classifier' or 'fc')
        head: nn.Module | None = None
        head_name = ""
        for name in ("classifier", "fc"):
            head = getattr(self.model, name, None)
            if head is not None:
                head_name = name
                break
        if head is None:
            raise RuntimeError(
                f"Cannot find 'classifier' or 'fc' attribute on "
                f"{type(self.model).__name__}; enable_repr_capture requires "
                f"a model with a named classification head."
            )

        # Infer repr_dim from the first Linear layer in the head
        repr_dim: int | None = None
        for mod in head.modules():
            if isinstance(mod, nn.Linear):
                repr_dim = mod.in_features
                break
        if repr_dim is None:
            raise RuntimeError(
                f"No nn.Linear found in classification head of {type(self.model).__name__}"
            )

        # Register pre-hook to capture the head's input
        def _capture_hook(module, args):
            self.captured_repr = args[0]

        self._repr_hook = head.register_forward_pre_hook(_capture_hook)
        logger.info(
            f"Repr capture enabled on {type(self.model).__name__}.{head_name}, dim={repr_dim}"
        )
        return repr_dim

    def forward_batch(self, batch: dict, device: str) -> torch.Tensor:
        """
        Forward pass from batch dictionary.

        Automatically moves tensors to device and calls model with correct arguments.

        Args:
            batch: Batch dict with "signal", "sequence", and optionally "features"
            device: Device to move tensors to

        Returns:
            Model logits
        """
        signal = batch["signal"].to(device)
        sequence = batch["sequence"].to(device)

        output: torch.Tensor
        if self.requires_features:
            features = batch["features"].to(device)
            output = self.model(signal, sequence, features)
        else:
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


class TracedModelWrapper:
    """Wrapper for exported/traced models (torch.export or TorchScript).

    Provides the same ``forward_batch`` interface as
    :class:`ModelInferenceWrapper` and :class:`RemoraModelWrapper` so that
    exported models can be used interchangeably in the inference engine.
    """

    def __init__(self, traced_model: torch.nn.Module, requires_features: bool):
        self.model = traced_model
        self.requires_features = requires_features
        self.is_traced = True

    def forward_batch(self, batch: dict, device: str) -> torch.Tensor:
        """Forward pass from batch dictionary."""
        signal = batch["signal"].to(device)
        sequence = batch["sequence"].to(device)

        output: torch.Tensor
        if self.requires_features:
            features = batch["features"].to(device)
            output = self.model(signal, sequence, features)
        else:
            output = self.model(signal, sequence)
        return output

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        output: torch.Tensor = self.model(*args, **kwargs)
        return output

    def eval(self) -> None:
        if hasattr(self.model, "eval"):
            try:
                self.model.eval()
            except NotImplementedError:
                logger.debug("Model does not support eval() (torch.export GraphModule)")

    def to(self, device: str) -> "TracedModelWrapper":
        self.model.to(device)
        return self
