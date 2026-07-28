"""SignalCNN: signal-only 1D-CNN classifier.

Unlike the ConvLSTM/TCN families (which fuse signal + sequence [+ dwell]), this
classifies from a raw signal window alone — for tasks where only signal carries
the label (e.g. barcode demultiplexing from the adapter signal). ``forward``
accepts the leech ``(signal, sequence, features)`` call contract but ignores
``sequence``/``features``, so it plugs into ``ModelInferenceWrapper`` and
``Trainer`` unchanged. It exposes an ``fc`` head for repr-capture compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SignalCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        signal_len: int = 256,
        channels: int = 32,
        signal_in_channels: int = 1,
        kernel_size: int = 7,
        seq_encoding: str = "base_onehot",
        **_: object,
    ) -> None:
        super().__init__()
        self.signal_len = signal_len
        # Stored for registry-contract compatibility; SignalCNN ignores sequence
        # input, so the encoding only affects models that consume it.
        self.seq_encoding = seq_encoding

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(signal_in_channels, channels),
            nn.MaxPool1d(2),
            block(channels, channels * 2),
            nn.MaxPool1d(2),
            block(channels * 2, channels * 2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(channels * 2, num_classes)

    def forward(
        self,
        signal: torch.Tensor,
        sequence: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.features(signal).squeeze(-1)
        return self.fc(h)
