"""
Remora TorchScript model wrapper for leech's inference engine.

Loads Remora-trained TorchScript (.pt) models and provides a unified
forward_batch() interface compatible with leech's ModelInferenceWrapper.
"""

from pathlib import Path

import torch
import torch.nn as nn


class RemoraModelWrapper:
    """
    Wraps a TorchScript Remora model for leech's inference engine.

    Remora models expect:
    - signal: (B, 1, signal_len) — note the channel dim
    - enc_kmer: (B, 36, signal_len) — signal-level kmer encoding

    And output (B, 2) logits for two-class classification.
    """

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        self.model: nn.Module = torch.jit.load(str(model_path), map_location=device)
        self.model.eval()
        self.requires_features = False
        self.is_remora = True

    def forward_batch(self, batch: dict, device: str) -> torch.Tensor:
        """
        Run batched inference.

        Args:
            batch: Dict with "signal" (B, signal_len) and "sequence" (B, 36, signal_len)
            device: Target device

        Returns:
            Logits of shape (B, 1)
        """
        signal = batch["signal"].to(device)
        enc_kmer = batch["sequence"].to(device)

        # Remora expects (B, 1, signal_len) for signal
        if signal.dim() == 2:
            signal = signal.unsqueeze(1)

        output = self.model(signal, enc_kmer)  # (B, 2) two-class logits

        # Convert to leech convention: single logit (B, 1)
        # P(class 1) via softmax, then convert to logit
        probs = torch.softmax(output, dim=1)[:, 1:2]
        logits = torch.logit(probs.clamp(1e-7, 1 - 1e-7))
        return logits

    def eval(self) -> None:
        self.model.eval()

    def to(self, device: str) -> "RemoraModelWrapper":
        self.model.to(device)
        return self
