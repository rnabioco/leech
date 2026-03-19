"""
Adaptive signal context window modules.

Provides learnable boundary masks and content-dependent gating for
automatically discovering optimal signal context windows during training.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger("leech.models.adaptive_window")


class BranchBoundaryMask(nn.Module):
    """Soft sigmoid mask with learnable left/right boundary parameters.

    The mask smoothly attenuates the signal outside the learned window,
    allowing gradient-based optimization of the context boundaries.

    Args:
        max_left: Maximum left context (signal samples before focus).
        max_right: Maximum right context (signal samples after focus).
        init_left: Initial left context (defaults to max_left).
        init_right: Initial right context (defaults to max_right).
        sharpness: Sigmoid steepness controlling transition width.
        min_context: Minimum context to enforce on each side.
    """

    def __init__(
        self,
        max_left: int,
        max_right: int,
        init_left: int | None = None,
        init_right: int | None = None,
        sharpness: float = 0.1,
        min_context: int = 36,
    ):
        super().__init__()
        self.max_left = max_left
        self.max_right = max_right
        self.sharpness = sharpness
        self.min_context = min_context

        init_l = float(init_left if init_left is not None else max_left)
        init_r = float(init_right if init_right is not None else max_right)

        self.left_param = nn.Parameter(torch.tensor(init_l))
        self.right_param = nn.Parameter(torch.tensor(init_r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply soft boundary mask to input signal.

        Args:
            x: Signal tensor of shape (batch, signal_len) or
               (batch, channels, signal_len).

        Returns:
            Masked signal with the same shape as input.
        """
        signal_len = x.shape[-1]
        positions = torch.arange(signal_len, dtype=x.dtype, device=x.device)

        # Focus point is at max_left (the center of the full window)
        focus = float(self.max_left)

        # Clamp parameters to [min_context, max]
        left_extent = self.left_param.clamp(self.min_context, self.max_left)
        right_extent = self.right_param.clamp(self.min_context, self.max_right)

        left_edge = focus - left_extent
        right_edge = focus + right_extent

        # Sigmoid mask: 1 inside the window, 0 outside
        mask = torch.sigmoid(self.sharpness * (positions - left_edge)) * torch.sigmoid(
            self.sharpness * (right_edge - positions)
        )

        return x * mask

    def get_window(self) -> tuple[int, int]:
        """Return current (left, right) context as integers."""
        left = int(self.left_param.clamp(self.min_context, self.max_left).item())
        right = int(self.right_param.clamp(self.min_context, self.max_right).item())
        return left, right


class BranchGate(nn.Module):
    """Content-dependent gating network.

    A lightweight convolutional network (~2k params) that learns a
    per-position gate based on signal content.

    Args:
        in_channels: Number of input signal channels.
        hidden_channels: Hidden layer width.
        kernel_size: Convolutional kernel size.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        kernel_size: int = 31,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply content-dependent gate to input signal.

        Args:
            x: Signal tensor of shape (batch, signal_len) or
               (batch, channels, signal_len).

        Returns:
            Gated signal with the same shape as input.
        """
        if x.dim() == 2:
            # (batch, signal_len) -> (batch, 1, signal_len)
            gate_input = x.unsqueeze(1)
        else:
            gate_input = x

        # gate_weights: (batch, 1, signal_len)
        gate_weights = self.gate(gate_input)

        if x.dim() == 2:
            # Squeeze channel dim back: (batch, 1, signal_len) -> (batch, signal_len)
            return x * gate_weights.squeeze(1)
        else:
            # Broadcast over channels: (batch, channels, signal_len)
            return x * gate_weights

    def get_effective_window(
        self, x_batch: torch.Tensor, threshold: float = 0.5
    ) -> tuple[int, int]:
        """Estimate effective (left, right) window from gate weights.

        Args:
            x_batch: A batch of signal tensors.
            threshold: Gate weight threshold for considering a position active.

        Returns:
            Tuple of (left_context, right_context) estimated from gate.
        """
        with torch.no_grad():
            if x_batch.dim() == 2:
                gate_input = x_batch.unsqueeze(1)
            else:
                gate_input = x_batch
            gate_weights = self.gate(gate_input).squeeze(1)  # (batch, signal_len)
            avg_weights = gate_weights.mean(dim=0)  # (signal_len,)
            active = (avg_weights > threshold).nonzero(as_tuple=True)[0]
            if len(active) == 0:
                return 0, 0
            center = avg_weights.shape[0] // 2
            left = int(center - active[0].item())
            right = int(active[-1].item() - center)
            return max(0, left), max(0, right)


class BoundaryMaskedModel(nn.Module):
    """Wraps any leech model with learnable signal boundary masks.

    Applies optional signal and residual masks before delegating to
    the inner model. Masks are stored as nn.Module submodules so their
    parameters are included in state_dict and optimizer param groups.

    Args:
        model: The inner leech model to wrap.
        signal_mask: Optional mask module for the signal branch.
        residual_mask: Optional separate mask for the residual channel
            (only used when signal has >1 channels and independent
            masking is desired).
    """

    def __init__(
        self,
        model: nn.Module,
        signal_mask: nn.Module | None = None,
        residual_mask: nn.Module | None = None,
    ):
        super().__init__()
        self.inner_model = model
        self.signal_mask = signal_mask
        self.residual_mask = residual_mask

    def forward(
        self,
        signal: torch.Tensor,
        sequence: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional boundary masking.

        Args:
            signal: Raw signal tensor. Shape: (batch, signal_len) for
                single-channel, or (batch, channels, signal_len) for
                multi-channel.
            sequence: Sequence encoding tensor.
            features: Optional feature tensor.

        Returns:
            Model logits.
        """
        if self.signal_mask is not None:
            if signal.dim() == 3 and self.residual_mask is not None:
                # Independent masks per channel
                channels = []
                for i in range(signal.shape[1]):
                    ch = signal[:, i : i + 1, :]  # (batch, 1, signal_len)
                    if i == 0:
                        ch = self.signal_mask(ch)
                    else:
                        ch = self.residual_mask(ch)
                    channels.append(ch)
                signal = torch.cat(channels, dim=1)
            else:
                # Shared mask (broadcasts over channel dim for 3D)
                signal = self.signal_mask(signal)

        if features is not None:
            return self.inner_model(signal, sequence, features)
        return self.inner_model(signal, sequence)

    def get_learned_windows(self) -> dict[str, Any]:
        """Return learned window parameters from all mask branches."""
        windows: dict[str, Any] = {}
        if self.signal_mask is not None:
            if isinstance(self.signal_mask, BranchBoundaryMask):
                left, right = self.signal_mask.get_window()
                windows["signal_left"] = left
                windows["signal_right"] = right
            elif isinstance(self.signal_mask, BranchGate):
                windows["signal_gate"] = "active"
        if self.residual_mask is not None and isinstance(self.residual_mask, BranchBoundaryMask):
            left, right = self.residual_mask.get_window()
            windows["residual_left"] = left
            windows["residual_right"] = right
        return windows

    def adaptive_params(self):
        """Yield only the mask parameters (for separate optimizer param group)."""
        if self.signal_mask is not None:
            yield from self.signal_mask.parameters()
        if self.residual_mask is not None:
            yield from self.residual_mask.parameters()
