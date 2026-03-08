"""
ConvLSTMDwell: Full model with dwell time features.

Architecture:
- Signal branch: Conv1d on raw signal
- Sequence branch: Conv1d on encoded k-mers (base_onehot or signal_kmer)
- Feature branch: Conv1d on dwell+level features (NEW vs. ConvLSTMBase)
- Merge → BiLSTM → FC → binary classification
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


class ConvLSTMDwell(BaseModel):
    """
    Full model with signal, sequence, and dwell feature branches.

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
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = DEFAULT_CONV_CHANNELS

        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.num_features = num_features
        self.lstm_hidden = lstm_hidden
        self.seq_encoding = seq_encoding

        # Compute sequence input channels
        if seq_encoding == "signal_kmer":
            seq_in_channels = 4 * (signal_kmer_context[0] + signal_kmer_context[1] + 1)
        else:
            seq_in_channels = 4

        # Signal branch: Shared component for signal processing
        self.signal_branch = SignalBranch(conv_channels=conv_channels)

        # Sequence branch: Shared component for sequence processing
        self.sequence_branch = SequenceBranch(
            in_channels=seq_in_channels, conv_channels=conv_channels
        )

        # Feature branch: Shared component for feature processing
        self.feature_branch = FeatureBranch(num_features=num_features, conv_channels=conv_channels)

        # Adaptive pooling to match dimensions
        self.signal_pool = nn.AdaptiveAvgPool1d(kmer_len)
        if seq_encoding == "signal_kmer":
            self.seq_pool = nn.AdaptiveAvgPool1d(kmer_len)

        # BiLSTM on merged features
        # Input: (batch, kmer_len, 768) - concatenated signal + seq + feature
        self.lstm = nn.LSTM(
            input_size=conv_channels[2] * 3,  # Three branches concatenated
            hidden_size=lstm_hidden,
            num_layers=DEFAULT_LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        # Fully connected layers
        # Take center position from BiLSTM output
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, DEFAULT_FC_HIDDEN),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(DEFAULT_FC_HIDDEN, 1),  # Binary classification
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
            Logits for binary classification (batch, 1)
        """
        # Signal branch (handles unsqueeze internally)
        signal_feat = self.signal_branch(signal)  # (batch, 256, signal_len)
        signal_feat = self.signal_pool(signal_feat)  # (batch, 256, kmer_len)

        # Sequence branch
        seq_feat = self.sequence_branch(sequence)  # (batch, 256, seq_len)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)  # (batch, 256, kmer_len)

        # Feature branch
        feat_feat = self.feature_branch(features)  # (batch, 256, kmer_len)

        # Merge all three branches
        merged = torch.cat([signal_feat, seq_feat, feat_feat], dim=1)

        # Transpose for LSTM: (batch, 768, kmer_len) -> (batch, kmer_len, 768)
        merged = merged.transpose(1, 2)

        # BiLSTM
        lstm_out, _ = self.lstm(merged)  # (batch, kmer_len, lstm_hidden*2)

        # Take center position
        center_idx = self.kmer_len // 2
        center_out = lstm_out[:, center_idx, :]  # (batch, lstm_hidden*2)

        # FC layers
        logits: torch.Tensor = self.fc(center_out)  # (batch, 1)

        return logits

    # predict_proba() is inherited from BaseModel
