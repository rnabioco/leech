"""
Custom loss functions for leech models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class _GradientReversalFunction(Function):
    """Gradient reversal layer (Ganin et al. 2016).

    Forward: identity. Backward: negate gradients and scale by lambda.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps _GradientReversalFunction as an nn.Module with mutable lambda."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_)


class AdversarialHead(nn.Module):
    """Adversarial classifier with gradient reversal for confound invariance.

    Args:
        input_dim: Dimension of the representation vector.
        num_classes: Number of confound classes.
        lambda_: Initial gradient reversal scaling factor.
    """

    def __init__(self, input_dim: int, num_classes: int, lambda_: float = 1.0):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.grl(x))

    def set_lambda(self, lambda_: float) -> None:
        self.grl.lambda_ = lambda_


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
