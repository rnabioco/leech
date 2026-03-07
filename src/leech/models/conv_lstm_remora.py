"""
Remora-style ConvLSTM model variants.

ConvLSTMRemoraBase: Pure reproduction of Remora's ConvLSTM architecture
    - Signal + Sequence only (no feature branch)
    - BatchNorm + SiLU activations
    - Strided convolutions (no AdaptiveAvgPool)
    - Merge convolution
    - Manual bidirectional LSTM (flip trick), take last position
    - CrossEntropyLoss compatible (num_out=2)

ConvLSTMRemora: Remora architecture + leech's dwell/signal feature branch
    - All of above + Feature branch with BatchNorm + SiLU
    - 3-branch merge convolution
"""

import torch
import torch.nn as nn

from leech.constants import DEFAULT_NUM_FEATURES
from leech.models.components import BaseModel

DEFAULT_REMORA_SIZE = 64
DEFAULT_REMORA_DROPOUT = 0.3


class _RemoraSignalBranch(nn.Module):
    """Remora-style signal branch: Conv+BN+SiLU with stride=3 on last layer."""

    def __init__(self, size: int = DEFAULT_REMORA_SIZE):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(4),
            nn.SiLU(),
            nn.Conv1d(4, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, size, kernel_size=9, stride=3, padding=4),
            nn.BatchNorm1d(size),
            nn.SiLU(),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.layers(signal.unsqueeze(1))


class _RemoraSequenceBranch(nn.Module):
    """Remora-style sequence branch: Conv+BN+SiLU with stride=3 on last layer."""

    def __init__(self, in_channels: int = 36, size: int = DEFAULT_REMORA_SIZE):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, size, kernel_size=13, stride=3, padding=6),
            nn.BatchNorm1d(size),
            nn.SiLU(),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.layers(sequence)


class _RemoraFeatureBranch(nn.Module):
    """Remora-style feature branch for dwell+signal features."""

    def __init__(self, num_features: int, size: int = DEFAULT_REMORA_SIZE):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(num_features, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, size, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm1d(size),
            nn.SiLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class _RemoraLSTM(nn.Module):
    """Remora-style bidirectional LSTM using flip trick, take last position."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm_fwd = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.lstm_rev = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        fwd_out, _ = self.lstm_fwd(x)  # (B, L, H)
        rev_out, _ = self.lstm_rev(x.flip(1))  # (B, L, H)

        # Take last position from each direction
        fwd_last = fwd_out[:, -1, :]  # (B, H)
        rev_last = rev_out[:, -1, :]  # (B, H)

        # Average (Remora convention)
        return (fwd_last + rev_last) / 2  # (B, H)


class ConvLSTMRemoraBase(BaseModel):
    """
    Pure Remora ConvLSTM reproduction (signal + sequence only).

    Architecture matches Remora's default ConvLSTM:
    - BatchNorm + SiLU activations
    - Strided convolutions for downsampling
    - Merge convolution to fuse branches
    - Manual bidirectional LSTM (flip trick)
    - Single linear output layer

    Args:
        signal_len: Length of input signal
        kmer_len: Length of k-mer sequence (unused, kept for API compat)
        size: Channel width throughout the network (default: 64)
        num_out: Number of output classes (default: 2 for CrossEntropyLoss)
        dropout: Dropout probability (default: 0.3)
        seq_encoding: Sequence encoding type ("signal_kmer" expected)
        signal_kmer_context: Kmer context for signal_kmer encoding
    """

    def __init__(
        self,
        signal_len: int = 400,
        kmer_len: int = 11,
        size: int = DEFAULT_REMORA_SIZE,
        num_out: int = 2,
        dropout: float = DEFAULT_REMORA_DROPOUT,
        seq_encoding: str = "signal_kmer",
        signal_kmer_context: tuple[int, int] = (4, 4),
    ):
        super().__init__()
        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.size = size
        self.num_out = num_out
        self.seq_encoding = seq_encoding

        if seq_encoding == "signal_kmer":
            seq_in_channels = 4 * (signal_kmer_context[0] + signal_kmer_context[1] + 1)
        else:
            seq_in_channels = 4

        self.signal_branch = _RemoraSignalBranch(size)
        self.seq_branch = _RemoraSequenceBranch(seq_in_channels, size)

        # Merge conv: fuse signal + sequence
        self.merge_conv = nn.Sequential(
            nn.Conv1d(size * 2, size, kernel_size=5, padding=2),
            nn.BatchNorm1d(size),
            nn.SiLU(),
        )

        self.lstm = _RemoraLSTM(size, size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(size, num_out)

    def forward(self, signal: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            signal: (B, signal_len)
            sequence: (B, C, signal_len) for signal_kmer or (B, 4, kmer_len) for base_onehot

        Returns:
            (B, num_out) logits
        """
        sig_feat = self.signal_branch(signal)
        seq_feat = self.seq_branch(sequence)

        # Match lengths (stride=3 may produce different lengths)
        min_len = min(sig_feat.shape[2], seq_feat.shape[2])
        sig_feat = sig_feat[:, :, :min_len]
        seq_feat = seq_feat[:, :, :min_len]

        merged = torch.cat([sig_feat, seq_feat], dim=1)
        merged = self.merge_conv(merged)

        # (B, C, L) -> (B, L, C) for LSTM
        merged = merged.transpose(1, 2)
        out = self.lstm(merged)  # (B, size)
        out = self.dropout(out)
        logits: torch.Tensor = self.fc(out)  # (B, num_out)
        return logits

    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        """Predict probability of positive class."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(*args, **kwargs)
            if self.num_out == 2:
                probs = torch.softmax(logits, dim=-1)[:, 1:2]
            else:
                probs = torch.sigmoid(logits)
        return probs


class ConvLSTMRemora(BaseModel):
    """
    Remora architecture + leech's dwell/signal feature branch.

    Same as ConvLSTMRemoraBase but with a third branch for engineered features
    (dwell times + signal level statistics).

    Args:
        signal_len: Length of input signal
        kmer_len: Length of k-mer sequence (unused, kept for API compat)
        num_features: Number of feature channels (dwell + signal levels)
        size: Channel width throughout the network (default: 64)
        num_out: Number of output classes (default: 2 for CrossEntropyLoss)
        dropout: Dropout probability (default: 0.3)
        seq_encoding: Sequence encoding type ("signal_kmer" expected)
        signal_kmer_context: Kmer context for signal_kmer encoding
    """

    def __init__(
        self,
        signal_len: int = 400,
        kmer_len: int = 11,
        num_features: int = DEFAULT_NUM_FEATURES,
        size: int = DEFAULT_REMORA_SIZE,
        num_out: int = 2,
        dropout: float = DEFAULT_REMORA_DROPOUT,
        seq_encoding: str = "signal_kmer",
        signal_kmer_context: tuple[int, int] = (4, 4),
    ):
        super().__init__()
        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.num_features = num_features
        self.size = size
        self.num_out = num_out
        self.seq_encoding = seq_encoding

        if seq_encoding == "signal_kmer":
            seq_in_channels = 4 * (signal_kmer_context[0] + signal_kmer_context[1] + 1)
        else:
            seq_in_channels = 4

        self.signal_branch = _RemoraSignalBranch(size)
        self.seq_branch = _RemoraSequenceBranch(seq_in_channels, size)
        self.feature_branch = _RemoraFeatureBranch(num_features, size)

        # Merge conv: fuse all three branches
        self.merge_conv = nn.Sequential(
            nn.Conv1d(size * 3, size, kernel_size=5, padding=2),
            nn.BatchNorm1d(size),
            nn.SiLU(),
        )

        self.lstm = _RemoraLSTM(size, size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(size, num_out)

    def forward(
        self, signal: torch.Tensor, sequence: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            signal: (B, signal_len)
            sequence: (B, C, signal_len) for signal_kmer or (B, 4, kmer_len) for base_onehot
            features: (B, num_features, kmer_len)

        Returns:
            (B, num_out) logits
        """
        sig_feat = self.signal_branch(signal)
        seq_feat = self.seq_branch(sequence)
        feat_feat = self.feature_branch(features)

        # Match lengths (stride=3 produces different lengths from different inputs)
        min_len = min(sig_feat.shape[2], seq_feat.shape[2], feat_feat.shape[2])
        sig_feat = sig_feat[:, :, :min_len]
        seq_feat = seq_feat[:, :, :min_len]
        feat_feat = feat_feat[:, :, :min_len]

        merged = torch.cat([sig_feat, seq_feat, feat_feat], dim=1)
        merged = self.merge_conv(merged)

        # (B, C, L) -> (B, L, C) for LSTM
        merged = merged.transpose(1, 2)
        out = self.lstm(merged)  # (B, size)
        out = self.dropout(out)
        logits: torch.Tensor = self.fc(out)  # (B, num_out)
        return logits

    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        """Predict probability of positive class."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(*args, **kwargs)
            if self.num_out == 2:
                probs = torch.softmax(logits, dim=-1)[:, 1:2]
            else:
                probs = torch.sigmoid(logits)
        return probs
