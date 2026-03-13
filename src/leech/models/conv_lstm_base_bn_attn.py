"""
ConvLSTMBaseBNAttn: Baseline model with BatchNorm + attention pooling.

Combines BatchNorm (from ConvLSTMBaseBN) with attention pooling over LSTM
positions (from ConvLSTMBaseAttn). BN provides universal F1 gains while
attention helps on hard classification tasks.
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
    DEFAULT_SIGNAL_KMER_CONTEXT,
    DEFAULT_SIGNAL_LEN,
)
from leech.models.components import BaseModel, SequenceBranch, SignalBranch


class ConvLSTMBaseBNAttn(BaseModel):
    """
    Baseline model with BatchNorm and learned attention pooling over LSTM positions.

    Same architecture as ConvLSTMBaseAttn but with BatchNorm in conv branches
    for more stable training and improved generalization.

    Args:
        signal_len: Length of input signal
        kmer_len: Length of k-mer sequence (e.g., 2*context+1)
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
        self.lstm_hidden = lstm_hidden
        self.seq_encoding = seq_encoding

        # Compute sequence input channels
        if seq_encoding == "signal_kmer":
            seq_in_channels = 4 * (signal_kmer_context[0] + signal_kmer_context[1] + 1)
        else:
            seq_in_channels = 4

        # Signal branch (with BatchNorm)
        self.signal_branch = SignalBranch(conv_channels=conv_channels, norm_type="batchnorm")

        # Sequence branch (with BatchNorm)
        self.sequence_branch = SequenceBranch(
            in_channels=seq_in_channels, conv_channels=conv_channels, norm_type="batchnorm"
        )

        # Adaptive pooling to match dimensions
        self.signal_pool = nn.AdaptiveAvgPool1d(kmer_len)
        if seq_encoding == "signal_kmer":
            self.seq_pool = nn.AdaptiveAvgPool1d(kmer_len)

        # BiLSTM on merged features
        self.lstm = nn.LSTM(
            input_size=conv_channels[2] * 2,
            hidden_size=lstm_hidden,
            num_layers=DEFAULT_LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        # Attention pooling over LSTM positions
        self.attn_linear = nn.Linear(lstm_hidden * 2, 1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, DEFAULT_FC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(DEFAULT_FC_HIDDEN, 1),
        )

    def forward(self, signal: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            signal: Raw signal (batch, signal_len)
            sequence: Encoded sequence — (batch, 4, kmer_len) for base_onehot
                or (batch, 36, signal_len) for signal_kmer

        Returns:
            Logits for binary classification (batch, 1)
        """
        signal_feat = self.signal_branch(signal)
        signal_feat = self.signal_pool(signal_feat)

        seq_feat = self.sequence_branch(sequence)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)

        merged = torch.cat([signal_feat, seq_feat], dim=1)
        merged = merged.transpose(1, 2)

        lstm_out, _ = self.lstm(merged)  # (batch, kmer_len, lstm_hidden*2)

        # Attention pooling over all positions
        attn_scores = self.attn_linear(lstm_out)  # (batch, kmer_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, kmer_len, 1)
        context_vector = (attn_weights * lstm_out).sum(dim=1)  # (batch, lstm_hidden*2)

        logits: torch.Tensor = self.fc(context_vector)
        return logits
