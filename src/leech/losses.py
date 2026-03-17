"""
Custom loss functions for leech models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Reverse gradients during backward pass (Ganin et al. 2016)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Module wrapper for :class:`GradientReversalFunction`.

    Args:
        lambda_: Gradient reversal strength.  Higher values apply stronger
            invariance pressure to the upstream encoder.
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class AdversarialHead(nn.Module):
    """Auxiliary classifier with gradient reversal for confound-invariant representations.

    During training the main encoder receives *reversed* gradients from this
    head, pushing it to learn representations that are invariant to the
    confound (e.g. discriminator base identity).

    Args:
        input_dim: Dimension of the penultimate representation.
        num_classes: Number of confound classes (e.g. 4 for A/C/G/T).
        lambda_: Initial gradient reversal strength.
    """

    def __init__(self, input_dim: int, num_classes: int, lambda_: float = 0.1) -> None:
        super().__init__()
        self.grl = GradientReversalLayer(lambda_)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def set_lambda(self, lambda_: float) -> None:
        self.grl.set_lambda(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.grl(x))


class RegressionHead(nn.Module):
    """Auxiliary regression head for continuous targets (e.g. charging level).

    Unlike :class:`AdversarialHead`, gradients flow normally (no reversal) —
    the encoder is encouraged to capture information about the target.

    Args:
        input_dim: Dimension of the penultimate representation.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict scalar in [0, 1] for each sample."""
        return self.net(x).squeeze(-1)


class FocalBCEWithLogitsLoss(nn.Module):
    """Focal loss for binary classification with logits.

    Applies a modulating factor (1 - p_t)^gamma to the standard BCE loss,
    down-weighting well-classified examples and focusing on hard negatives.

    Args:
        gamma: Focusing parameter. Higher values increase focus on hard examples.
            gamma=0 is equivalent to standard BCE loss.
        pos_weight: Weight for positive class (same as BCEWithLogitsLoss).
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()
