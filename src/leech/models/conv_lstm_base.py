"""
ConvLSTMBase: Baseline model without dwell features.

Architecture:
- Signal branch: Conv1d on raw signal
- Sequence branch: Conv1d on one-hot encoded k-mers
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
    DEFAULT_SEQ_KERNEL,
    DEFAULT_SIGNAL_KERNEL,
    DEFAULT_SIGNAL_LEN,
)


class ConvLSTMBase(nn.Module):
    """
    Baseline model with signal and sequence branches only.

    Args:
        signal_len: Length of input signal
        kmer_len: Length of k-mer sequence (e.g., 2*context+1)
        conv_channels: List of channel sizes for conv layers (default: [4, 16, 256])
        lstm_hidden: Hidden size for BiLSTM (default: 96)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        signal_len: int = DEFAULT_SIGNAL_LEN,
        kmer_len: int = DEFAULT_KMER_LEN,
        conv_channels: list[int] | None = None,
        lstm_hidden: int = DEFAULT_LSTM_HIDDEN,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = DEFAULT_CONV_CHANNELS

        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.lstm_hidden = lstm_hidden

        # Signal branch: Conv1d layers
        # Input: (batch, 1, signal_len)
        self.signal_conv = nn.Sequential(
            nn.Conv1d(1, conv_channels[0], kernel_size=DEFAULT_SIGNAL_KERNEL, padding=DEFAULT_SIGNAL_KERNEL // 2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=DEFAULT_SIGNAL_KERNEL, padding=DEFAULT_SIGNAL_KERNEL // 2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=DEFAULT_SIGNAL_KERNEL, padding=DEFAULT_SIGNAL_KERNEL // 2),
            nn.ReLU(),
        )

        # Sequence branch: Conv1d on one-hot encoded k-mers
        # Input: (batch, 4, kmer_len) - 4 nucleotides (A, C, G, T)
        self.seq_conv = nn.Sequential(
            nn.Conv1d(4, conv_channels[0], kernel_size=DEFAULT_SEQ_KERNEL, padding=DEFAULT_SEQ_KERNEL // 2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=DEFAULT_SEQ_KERNEL, padding=DEFAULT_SEQ_KERNEL // 2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=DEFAULT_SEQ_KERNEL, padding=DEFAULT_SEQ_KERNEL // 2),
            nn.ReLU(),
        )

        # Adaptive pooling to match dimensions
        # Signal branch output: (batch, 256, signal_len)
        # Seq branch output: (batch, 256, kmer_len)
        # Target: (batch, 256, kmer_len) - pool signal to match kmer length
        self.signal_pool = nn.AdaptiveAvgPool1d(kmer_len)

        # BiLSTM on merged features
        # Input: (batch, kmer_len, 512) - concatenated signal + seq features
        self.lstm = nn.LSTM(
            input_size=conv_channels[2] * 2,  # Concatenated features
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

    def forward(self, signal: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            signal: Raw signal (batch, signal_len)
            sequence: One-hot encoded sequence (batch, 4, kmer_len)

        Returns:
            Logits for binary classification (batch, 1)
        """
        signal.size(0)

        # Signal branch
        # (batch, signal_len) -> (batch, 1, signal_len)
        signal_in = signal.unsqueeze(1)
        signal_feat = self.signal_conv(signal_in)  # (batch, 256, signal_len)
        signal_feat = self.signal_pool(signal_feat)  # (batch, 256, kmer_len)

        # Sequence branch
        seq_feat = self.seq_conv(sequence)  # (batch, 256, kmer_len)

        # Merge branches
        # (batch, 256, kmer_len) + (batch, 256, kmer_len) -> (batch, 512, kmer_len)
        merged = torch.cat([signal_feat, seq_feat], dim=1)

        # Transpose for LSTM: (batch, 512, kmer_len) -> (batch, kmer_len, 512)
        merged = merged.transpose(1, 2)

        # BiLSTM
        lstm_out, _ = self.lstm(merged)  # (batch, kmer_len, lstm_hidden*2)

        # Take center position
        center_idx = self.kmer_len // 2
        center_out = lstm_out[:, center_idx, :]  # (batch, lstm_hidden*2)

        # FC layers
        logits: torch.Tensor = self.fc(center_out)  # type: ignore[assignment]  # (batch, 1)

        return logits

    def predict_proba(self, signal: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.

        Args:
            signal: Raw signal (batch, signal_len)
            sequence: One-hot encoded sequence (batch, 4, kmer_len)

        Returns:
            Probabilities (batch, 1)
        """
        logits = self.forward(signal, sequence)
        return torch.sigmoid(logits)


# encode_kmer() has been moved to leech.data_prep for centralized access
# Import it from there if needed:
# from leech.data_prep import encode_kmer
