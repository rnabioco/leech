"""
ConvLSTMDwell: Full model with dwell time features.

Architecture:
- Signal branch: Conv1d on raw signal
- Sequence branch: Conv1d on one-hot encoded k-mers
- Feature branch: Conv1d on dwell+level features (NEW vs. ConvLSTMBase)
- Merge → BiLSTM → FC → binary classification
"""

import torch
import torch.nn as nn


class ConvLSTMDwell(nn.Module):
    """
    Full model with signal, sequence, and dwell feature branches.

    Args:
        signal_len: Length of input signal
        kmer_len: Length of k-mer sequence (e.g., 2*context+1)
        num_features: Number of feature channels (dwell + signal levels)
        conv_channels: List of channel sizes for conv layers (default: [4, 16, 256])
        lstm_hidden: Hidden size for BiLSTM (default: 96)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        signal_len: int = 400,
        kmer_len: int = 11,
        num_features: int = 5,  # e.g., dwell + mean + median + std + range
        conv_channels: list[int] | None = None,
        lstm_hidden: int = 96,
        dropout: float = 0.1,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = [4, 16, 256]

        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.num_features = num_features
        self.lstm_hidden = lstm_hidden

        # Signal branch: Conv1d layers
        # Input: (batch, 1, signal_len)
        self.signal_conv = nn.Sequential(
            nn.Conv1d(1, conv_channels[0], kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=5, padding=2),
            nn.ReLU(),
        )

        # Sequence branch: Conv1d on one-hot encoded k-mers
        # Input: (batch, 4, kmer_len) - 4 nucleotides (A, C, G, T)
        self.seq_conv = nn.Sequential(
            nn.Conv1d(4, conv_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Feature branch: Conv1d on dwell+level features
        # Input: (batch, num_features, kmer_len)
        self.feature_conv = nn.Sequential(
            nn.Conv1d(num_features, conv_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Adaptive pooling to match dimensions
        # Signal branch output: (batch, 256, signal_len)
        # Seq branch output: (batch, 256, kmer_len)
        # Feature branch output: (batch, 256, kmer_len)
        # Target: (batch, 256, kmer_len) - pool signal to match kmer length
        self.signal_pool = nn.AdaptiveAvgPool1d(kmer_len)

        # BiLSTM on merged features
        # Input: (batch, kmer_len, 768) - concatenated signal + seq + feature
        self.lstm = nn.LSTM(
            input_size=conv_channels[2] * 3,  # Three branches concatenated
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        # Fully connected layers
        # Take center position from BiLSTM output
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # Binary classification
        )

    def forward(
        self, signal: torch.Tensor, sequence: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            signal: Raw signal (batch, signal_len)
            sequence: One-hot encoded sequence (batch, 4, kmer_len)
            features: Dwell + signal level features (batch, num_features, kmer_len)

        Returns:
            Logits for binary classification (batch, 1)
        """
        batch_size = signal.size(0)

        # Signal branch
        # (batch, signal_len) -> (batch, 1, signal_len)
        signal_in = signal.unsqueeze(1)
        signal_feat = self.signal_conv(signal_in)  # (batch, 256, signal_len)
        signal_feat = self.signal_pool(signal_feat)  # (batch, 256, kmer_len)

        # Sequence branch
        seq_feat = self.seq_conv(sequence)  # (batch, 256, kmer_len)

        # Feature branch
        feat_feat = self.feature_conv(features)  # (batch, 256, kmer_len)

        # Merge all three branches
        # (batch, 256, kmer_len) x 3 -> (batch, 768, kmer_len)
        merged = torch.cat([signal_feat, seq_feat, feat_feat], dim=1)

        # Transpose for LSTM: (batch, 768, kmer_len) -> (batch, kmer_len, 768)
        merged = merged.transpose(1, 2)

        # BiLSTM
        lstm_out, _ = self.lstm(merged)  # (batch, kmer_len, lstm_hidden*2)

        # Take center position
        center_idx = self.kmer_len // 2
        center_out = lstm_out[:, center_idx, :]  # (batch, lstm_hidden*2)

        # FC layers
        logits = self.fc(center_out)  # (batch, 1)

        return logits

    def predict_proba(
        self, signal: torch.Tensor, sequence: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get probability predictions.

        Args:
            signal: Raw signal (batch, signal_len)
            sequence: One-hot encoded sequence (batch, 4, kmer_len)
            features: Dwell + signal level features (batch, num_features, kmer_len)

        Returns:
            Probabilities (batch, 1)
        """
        logits = self.forward(signal, sequence, features)
        return torch.sigmoid(logits)
