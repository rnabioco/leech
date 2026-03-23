"""
ConvLSTM models without attention pooling.

Four variants parameterized by has_features (Base/Dwell) and norm_type:
- ConvLSTMBase: signal + sequence only, no normalization
- ConvLSTMBaseBN: signal + sequence only, BatchNorm
- ConvLSTMDwell: signal + sequence + dwell features, no normalization
- ConvLSTMDwellBN: signal + sequence + dwell features, BatchNorm

Architecture:
    signal  → SignalBranch → pool → ┐
    sequence → SequenceBranch →     ├ cat → BiLSTM → center pick → FC
    features → FeatureBranch →      ┘ (Dwell variants only)
"""

import torch
import torch.nn as nn

from leech.constants import (
    DEFAULT_CONV_CHANNELS,
    DEFAULT_DROPOUT,
    DEFAULT_FC_HIDDEN,
    DEFAULT_KMER_LEN,
    DEFAULT_LSTM_HIDDEN,
    DEFAULT_LSTM_LAYERS,
    DEFAULT_NUM_FEATURES,
    DEFAULT_SIGNAL_KMER_CONTEXT,
    DEFAULT_SIGNAL_LEN,
)
from leech.models.components import BaseModel, FeatureBranch, SequenceBranch, SignalBranch


class _ConvLSTMNoAttn(BaseModel):
    """Internal base for non-attention ConvLSTM variants.

    Parameterized by _has_features (whether to build a FeatureBranch)
    and _norm_type (normalization in conv branches).
    """

    def __init__(
        self,
        *,
        _has_features: bool,
        _norm_type: str = "none",
        signal_len: int = DEFAULT_SIGNAL_LEN,
        kmer_len: int = DEFAULT_KMER_LEN,
        num_features: int = DEFAULT_NUM_FEATURES,
        conv_channels: list[int] | None = None,
        lstm_hidden: int = DEFAULT_LSTM_HIDDEN,
        dropout: float = DEFAULT_DROPOUT,
        seq_encoding: str = "base_onehot",
        signal_kmer_context: tuple[int, int] = DEFAULT_SIGNAL_KMER_CONTEXT,
        num_out: int = 1,
        signal_in_channels: int = 1,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = DEFAULT_CONV_CHANNELS

        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.lstm_hidden = lstm_hidden
        self.seq_encoding = seq_encoding
        if _has_features:
            self.num_features = num_features
            self.num_out = num_out

        # Compute sequence input channels
        if seq_encoding == "signal_kmer":
            seq_in_channels = 4 * (signal_kmer_context[0] + signal_kmer_context[1] + 1)
        else:
            seq_in_channels = 4

        norm_kw = {"norm_type": _norm_type} if _norm_type != "none" else {}

        self.signal_branch = SignalBranch(
            conv_channels=conv_channels, in_channels=signal_in_channels, **norm_kw
        )
        self.sequence_branch = SequenceBranch(
            in_channels=seq_in_channels, conv_channels=conv_channels, **norm_kw
        )

        if _has_features:
            self.feature_branch = FeatureBranch(
                num_features=num_features, conv_channels=conv_channels, **norm_kw
            )

        self.signal_pool = nn.AdaptiveAvgPool1d(kmer_len)
        if seq_encoding == "signal_kmer":
            self.seq_pool = nn.AdaptiveAvgPool1d(kmer_len)

        n_branches = 3 if _has_features else 2
        self.lstm = nn.LSTM(
            input_size=conv_channels[2] * n_branches,
            hidden_size=lstm_hidden,
            num_layers=DEFAULT_LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, DEFAULT_FC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(DEFAULT_FC_HIDDEN, num_out),
        )

        self._has_features = _has_features

    def forward(
        self,
        signal: torch.Tensor,
        sequence: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        signal_feat = self.signal_branch(signal)
        signal_feat = self.signal_pool(signal_feat)

        seq_feat = self.sequence_branch(sequence)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)

        if self._has_features:
            feat_feat = self.feature_branch(features)
            merged = torch.cat([signal_feat, seq_feat, feat_feat], dim=1)
        else:
            merged = torch.cat([signal_feat, seq_feat], dim=1)

        merged = merged.transpose(1, 2)
        lstm_out, _ = self.lstm(merged)

        center_idx = self.kmer_len // 2
        center_out = lstm_out[:, center_idx, :]

        logits: torch.Tensor = self.fc(center_out)
        return logits


class ConvLSTMBase(_ConvLSTMNoAttn):
    """Baseline model with signal and sequence branches only."""

    def __init__(self, **kwargs):
        super().__init__(_has_features=False, _norm_type="none", **kwargs)

    def forward(self, signal: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        return super().forward(signal, sequence)


class ConvLSTMBaseBN(_ConvLSTMNoAttn):
    """Baseline model with BatchNorm after each Conv1d layer."""

    def __init__(self, **kwargs):
        super().__init__(_has_features=False, _norm_type="batchnorm", **kwargs)

    def forward(self, signal: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        return super().forward(signal, sequence)


class ConvLSTMDwell(_ConvLSTMNoAttn):
    """Full model with signal, sequence, and dwell feature branches."""

    def __init__(self, **kwargs):
        super().__init__(_has_features=True, _norm_type="none", **kwargs)


class ConvLSTMDwellBN(_ConvLSTMNoAttn):
    """Full model with dwell features and BatchNorm."""

    def __init__(self, **kwargs):
        super().__init__(_has_features=True, _norm_type="batchnorm", **kwargs)
