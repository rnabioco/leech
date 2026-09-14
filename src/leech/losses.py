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


def parse_label_noise_rate(token: str | None) -> dict[str, float] | None:
    """Parse a ``--label-noise-rate`` token into ``{source_group: flip_rate}``.

    Format: ``group=rate[,group=rate,...]``, e.g. ``gold=0.09,enzymatic=0.17``.
    Rates are label-flip probabilities measured upstream (leech does not
    estimate them -- see the module docstring of
    :class:`NoiseCorrectedBCEWithLogitsLoss`); a group not present in the
    mapping gets rate 0 (no correction).

    Returns ``None`` for an empty/``None`` token.
    """
    if not token or not token.strip():
        return None
    rates: dict[str, float] = {}
    for part in token.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Malformed --label-noise-rate entry '{part}' (expected 'group=rate')")
        group, _, rate_str = part.partition("=")
        group = group.strip()
        if not group:
            raise ValueError(f"--label-noise-rate entry '{part}' has an empty group name")
        try:
            rate = float(rate_str.strip())
        except ValueError:
            raise ValueError(
                f"--label-noise-rate rate for '{group}' is not a float: '{rate_str}'"
            ) from None
        if not (0.0 <= rate < 1.0):
            raise ValueError(f"--label-noise-rate rate for '{group}' must be in [0, 1), got {rate}")
        rates[group] = rate
    return rates or None


class NoiseCorrectedBCEWithLogitsLoss(nn.Module):
    """Forward-corrected BCE for class-conditional label noise with known,
    per-sample flip rates (forward correction; Patrini et al. 2017).

    Each sample carries a flip probability ``rho`` -- the chance its *observed*
    label differs from its true one -- typically looked up per chunk from its
    ``source_group`` via ``--label-noise-rate`` (see :func:`parse_label_noise_rate`).
    Within one ``source_group`` every chunk carries the same assigned label (a
    block-level enrichment purity, not a per-read vote), so only one flip
    direction is ever live for a given ``rho``, and the general 2x2 noise
    matrix

        T = [[1 - rho, rho    ],   # true=0 -> observed {0, 1}
             [rho,     1 - rho]]   # true=1 -> observed {0, 1}

    degenerates to the same scalar formula regardless of which label a sample
    carries::

        P(observed=1) = (1 - rho) * p + rho * (1 - p)     # p = sigmoid(logit)

    The model is fit against this corrected probability of the *observed*
    label rather than against the raw noisy target directly, so the logits it
    learns estimate the true (uncorrupted) class posterior.

    ``rho=0`` for a sample leaves it exactly as plain BCE would treat it. When
    no ``noise_rate`` is passed at all (the common case: no
    ``--label-noise-rate``, so the dataset never puts a ``noise_rate`` field
    in the batch), this short-circuits to a direct call to
    ``F.binary_cross_entropy_with_logits`` -- not merely a mathematically
    equivalent formula -- which is what gives bit-for-bit equality with the
    plain BCE path; a differently-ordered-but-equal computation would not
    guarantee that. This check is on ``noise_rate is None`` only (never on the
    tensor's values), because a value-dependent branch would force a
    host/device sync on every batch on CUDA -- exactly the per-step
    synchronization ``_DeviceTally`` in ``training.py`` was built to avoid
    elsewhere. A ``noise_rate`` tensor that happens to be all zero (every
    sample's group unmapped, or an explicit all-zero rate) still takes the
    general path below; the result agrees with plain BCE to floating-point
    precision, just not bit-for-bit.

    Reduces by element count (``.mean()``), like the other losses in this
    module, so it decomposes over equal DDP shards the same way they do --
    see ``Trainer._weighted_ce_global`` in ``training.py`` for the one loss in
    this codebase where a mean over a shard is *not* the global mean.

    Args:
        pos_weight: Optional positive-class weight, same convention as
            ``nn.BCEWithLogitsLoss``.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        noise_rate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise_rate is None:
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)

        rho = noise_rate.to(dtype=logits.dtype, device=logits.device).clamp(0.0, 1.0)
        log_rho = torch.log(rho)
        log_1m_rho = torch.log1p(-rho)
        log_p = F.logsigmoid(logits)
        log_1m_p = F.logsigmoid(-logits)

        # log P(observed=1) = logsumexp(log(1-rho) + log p, log(rho) + log(1-p));
        # log P(observed=0) is the same mixture with p and 1-p swapped.
        log_p_obs1 = torch.logsumexp(torch.stack([log_1m_rho + log_p, log_rho + log_1m_p]), dim=0)
        log_p_obs0 = torch.logsumexp(torch.stack([log_1m_rho + log_1m_p, log_rho + log_p]), dim=0)

        weight = 1.0 if self.pos_weight is None else self.pos_weight.to(logits.device)
        per_sample = -(weight * targets * log_p_obs1 + (1 - targets) * log_p_obs0)
        return per_sample.mean()


class FocalBCEWithLogitsLoss(nn.Module):
    """Focal loss for binary classification with logits.

    Applies a modulating factor (1 - p_t)^gamma to the standard BCE loss,
    down-weighting well-classified examples and focusing on hard negatives.

    Args:
        gamma: Focusing parameter. Higher values increase focus on hard examples.
            gamma=0 is equivalent to standard BCE loss.
        pos_weight: Weight for positive class (same as BCEWithLogitsLoss).
        neg_gamma: Optional separate focusing parameter for negative-labeled
            examples (``--focal-neg-gamma``), making the loss asymmetric.
            ``None`` (the default) keeps the single-``gamma`` formula below
            and is bit-for-bit identical to the pre-#280 loss; passing the
            SAME value as ``gamma`` takes the per-element branch instead but
            is *also* bit-for-bit identical, since every target is exactly
            0.0 or 1.0 so ``targets * gamma + (1 - targets) * gamma == gamma``
            with no rounding -- "rate 1.0" reproduces the current loss either
            way (issue #280). A ``neg_gamma`` larger than ``gamma`` down-
            weights easy negatives harder than easy positives, concentrating
            gradient on the hard negatives sitting near the decision
            boundary -- the population that sets the model's FPR at a given
            threshold -- without touching ``pos_weight``, which only scales
            the positive term's magnitude and says nothing about which
            negatives contribute gradient.

    Raises:
        ValueError: ``gamma`` or ``neg_gamma`` is negative. ``(1 - p_t)`` is
            in ``[0, 1]``, so a negative exponent sends the modulating factor
            toward infinity as a confidently-correct example's ``p_t``
            approaches 1 -- silently poisoning the batch loss/gradient with
            inf/nan rather than raising anywhere near the mistake.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
        neg_gamma: float | None = None,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if neg_gamma is not None and neg_gamma < 0:
            raise ValueError(f"neg_gamma must be >= 0, got {neg_gamma}")
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.neg_gamma = neg_gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        if self.neg_gamma is None:
            focal_weight = (1 - p_t) ** self.gamma
        else:
            gamma_t = targets * self.gamma + (1 - targets) * self.neg_gamma
            focal_weight = (1 - p_t) ** gamma_t
        return (focal_weight * bce).mean()
