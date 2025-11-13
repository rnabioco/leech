"""
ConvLSTMSignalFeatures: Model with signal and features branches only (no sequence).

This architecture is optimized for applications where the sequence context is constant
(e.g., tRNA aminoacylation where all training examples have identical CCAGGC motif).
By removing the sequence branch, the model:
- Uses only discriminative features (signal + dwell/levels)
- Reduces parameters and training time
- Avoids overfitting to constant sequences
- Focuses on the nanopore signal properties that differ between classes

Architecture:
- Signal branch: Conv1d on raw signal → pooled to kmer_len
- Feature branch: Conv1d on dwell+level features (num_features, kmer_len)
- Merge → BiLSTM → FC → binary classification

For applications where sequence context varies (e.g., pairwise amino acid classification
at different genomic positions), use ConvLSTMDwell instead.
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
    DEFAULT_SIGNAL_LEN,
)
from leech.models.components import BaseModel, FeatureBranch, SignalBranch


class ConvLSTMSignalFeatures(BaseModel):
    """
    Two-branch model with signal and features only (no sequence branch).

    Optimized for constant-sequence applications like tRNA aminoacylation.

    Args:
        signal_len: Length of input signal
        kmer_len: Length of feature dimension (determines pooling target)
        num_features: Number of feature channels (dwell + signal levels)
        conv_channels: List of channel sizes for conv layers (default: [4, 16, 256])
        lstm_hidden: Hidden size for BiLSTM (default: 96)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        signal_len: int = DEFAULT_SIGNAL_LEN,
        kmer_len: int = DEFAULT_KMER_LEN,
        num_features: int = DEFAULT_NUM_FEATURES,
        conv_channels: list[int] | None = None,
        lstm_hidden: int = DEFAULT_LSTM_HIDDEN,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = DEFAULT_CONV_CHANNELS

        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.num_features = num_features
        self.lstm_hidden = lstm_hidden

        # Signal branch: Shared component for signal processing
        self.signal_branch = SignalBranch(conv_channels=conv_channels)

        # Feature branch: Shared component for feature processing
        self.feature_branch = FeatureBranch(num_features=num_features, conv_channels=conv_channels)

        # Adaptive pooling to match dimensions
        # Signal branch output: (batch, 256, signal_len)
        # Feature branch output: (batch, 256, kmer_len)
        # Target: (batch, 256, kmer_len) - pool signal to match kmer length
        self.signal_pool = nn.AdaptiveAvgPool1d(kmer_len)

        # BiLSTM on merged features
        # Input: (batch, kmer_len, 512) - concatenated signal + feature (2 branches)
        self.lstm = nn.LSTM(
            input_size=conv_channels[2] * 2,  # Two branches concatenated (not 3)
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

    def forward(self, signal: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            signal: Raw signal tensor (batch_size, signal_len)
            features: Engineered features (batch_size, num_features, kmer_len)

        Returns:
            Logits for binary classification (batch_size, 1)
        """
        # Process signal branch
        signal_features = self.signal_branch(signal)  # (batch, 256, signal_len)
        signal_pooled = self.signal_pool(signal_features)  # (batch, 256, kmer_len)

        # Process feature branch
        feature_features = self.feature_branch(features)  # (batch, 256, kmer_len)

        # Concatenate along channel dimension
        merged = torch.cat([signal_pooled, feature_features], dim=1)  # (batch, 512, kmer_len)

        # Transpose for LSTM: (batch, kmer_len, 512)
        merged = merged.transpose(1, 2)

        # BiLSTM
        lstm_out, _ = self.lstm(merged)  # (batch, kmer_len, 192) if lstm_hidden=96

        # Take center position (focus base)
        center_idx = self.kmer_len // 2
        center_output = lstm_out[:, center_idx, :]  # (batch, 192)

        # Fully connected layers
        output: torch.Tensor = self.fc(center_output)  # (batch, 1)

        return output
