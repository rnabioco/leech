"""
ConvLSTMDwellBN: Full model with BatchNorm after each Conv1d.

Identical to ConvLSTMDwell but passes norm_type="batchnorm" to branches.
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


class ConvLSTMDwellBN(BaseModel):
    """
    Full model with BatchNorm after each Conv1d layer.

    Same architecture as ConvLSTMDwell but with BatchNorm1d inserted
    between each Conv1d and ReLU activation in all three branches.

    Args:
        signal_len: Length of input signal
        kmer_len: Length of k-mer sequence (e.g., 2*context+1)
        num_features: Number of feature channels (dwell + signal levels)
        conv_channels: List of channel sizes for conv layers (default: [4, 16, 256])
        lstm_hidden: Hidden size for BiLSTM (default: 96)
        dropout: Dropout probability (default: 0.1)
        seq_encoding: Sequence encoding type ("base_onehot" or "signal_kmer")
        signal_kmer_context: Kmer context for signal_kmer encoding (default: (4, 4))
    """

    def __init__(
        self,
        signal_len: int = DEFAULT_SIGNAL_LEN,
        kmer_len: int = DEFAULT_KMER_LEN,
        num_features: int = DEFAULT_NUM_FEATURES,
        conv_channels: list[int] | None = None,
        lstm_hidden: int = DEFAULT_LSTM_HIDDEN,
        dropout: float = DEFAULT_DROPOUT,
        seq_encoding: str = "base_onehot",
        signal_kmer_context: tuple[int, int] = DEFAULT_SIGNAL_KMER_CONTEXT,
        num_out: int = 1,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = DEFAULT_CONV_CHANNELS

        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.num_features = num_features
        self.lstm_hidden = lstm_hidden
        self.seq_encoding = seq_encoding
        self.num_out = num_out

        # Compute sequence input channels
        if seq_encoding == "signal_kmer":
            seq_in_channels = 4 * (signal_kmer_context[0] + signal_kmer_context[1] + 1)
        else:
            seq_in_channels = 4

        # Signal branch with BatchNorm
        self.signal_branch = SignalBranch(conv_channels=conv_channels, norm_type="batchnorm")

        # Sequence branch with BatchNorm
        self.sequence_branch = SequenceBranch(
            in_channels=seq_in_channels, conv_channels=conv_channels, norm_type="batchnorm"
        )

        # Feature branch with BatchNorm
        self.feature_branch = FeatureBranch(
            num_features=num_features, conv_channels=conv_channels, norm_type="batchnorm"
        )

        # Adaptive pooling to match dimensions
        self.signal_pool = nn.AdaptiveAvgPool1d(kmer_len)
        if seq_encoding == "signal_kmer":
            self.seq_pool = nn.AdaptiveAvgPool1d(kmer_len)

        # BiLSTM on merged features
        self.lstm = nn.LSTM(
            input_size=conv_channels[2] * 3,
            hidden_size=lstm_hidden,
            num_layers=DEFAULT_LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, DEFAULT_FC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(DEFAULT_FC_HIDDEN, num_out),
        )

    def forward(
        self, signal: torch.Tensor, sequence: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            signal: Raw signal (batch, signal_len)
            sequence: Encoded sequence — (batch, 4, kmer_len) for base_onehot
                or (batch, 36, signal_len) for signal_kmer
            features: Dwell + signal level features (batch, num_features, kmer_len)

        Returns:
            Logits (batch, num_out)
        """
        signal_feat = self.signal_branch(signal)
        signal_feat = self.signal_pool(signal_feat)

        seq_feat = self.sequence_branch(sequence)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)

        feat_feat = self.feature_branch(features)

        merged = torch.cat([signal_feat, seq_feat, feat_feat], dim=1)
        merged = merged.transpose(1, 2)

        lstm_out, _ = self.lstm(merged)

        center_idx = self.kmer_len // 2
        center_out = lstm_out[:, center_idx, :]

        logits: torch.Tensor = self.fc(center_out)
        return logits
