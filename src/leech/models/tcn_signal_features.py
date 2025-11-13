"""
TCNSignalFeatures: Temporal Convolutional Network with signal and features only (no sequence).

This architecture is optimized for applications where the sequence context is constant
(e.g., tRNA aminoacylation where all training examples have identical CCAGGC motif).
By removing the sequence branch, the model:
- Uses only discriminative features (signal + dwell/levels)
- Reduces parameters and training time
- Avoids overfitting to constant sequences
- Focuses on the nanopore signal properties that differ between classes

Architecture:
- Signal branch: Stack of dilated causal convolutions with residual connections
- Feature branch: Stack of dilated causal convolutions
- Global pooling (avg + max) → concatenate → dense classifier

Rationale:
- Dilated convolutions provide large receptive fields with fewer parameters
- Faster than RNNs/LSTMs and Transformers
- Causal convolutions preserve temporal ordering
- Good for variable-length sequences
- 2^k dilation rates capture multi-scale temporal patterns

For applications where sequence context varies, use TCNDwell instead.
"""

import torch
import torch.nn as nn

from leech.constants import (
    DEFAULT_DROPOUT,
    DEFAULT_KMER_LEN,
    DEFAULT_NUM_FEATURES,
    DEFAULT_SIGNAL_LEN,
)
from leech.models.components import BaseModel


class TemporalBlock(nn.Module):
    """
    Temporal convolutional block with dilated causal convolutions and residual connection.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        dilation: Dilation rate
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()

        # Padding to maintain sequence length
        # For causal convolutions: padding = (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation

        # Two convolutional layers with batch norm and dropout
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # We'll manually pad
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, length)

        Returns:
            Output tensor (batch, out_channels, length)
        """
        # Causal padding (left padding only)
        x_padded = nn.functional.pad(x, (self.padding, 0))

        # First conv block
        out = self.conv1(x_padded)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        # Causal padding again
        out = nn.functional.pad(out, (self.padding, 0))

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        result: torch.Tensor = self.relu(out + res)  # type: ignore[assignment]
        return result


class TCN(nn.Module):
    """
    Temporal Convolutional Network with stacked dilated convolutions.

    Args:
        in_channels: Number of input channels
        hidden_channels: Number of channels in each layer
        num_layers: Number of temporal blocks
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            dilation = 2**i  # Exponentially increasing dilation: 1, 2, 4, 8, 16, 32
            in_ch = in_channels if i == 0 else hidden_channels
            layers.append(
                TemporalBlock(
                    in_ch,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, length)

        Returns:
            Output tensor (batch, hidden_channels, length)
        """
        out: torch.Tensor = self.network(x)  # type: ignore[assignment]
        return out


class TCNSignalFeatures(BaseModel):
    """
    TCN model with signal and features branches only (no sequence branch).

    Optimized for constant-sequence applications like tRNA aminoacylation.

    Args:
        signal_len: Length of input signal
        kmer_len: Length of feature dimension (for pooling target)
        num_features: Number of feature channels (dwell + signal levels)
        hidden_channels: Number of channels in TCN layers (default: 64)
        num_layers: Number of temporal blocks per branch (default: 6)
        kernel_size: Convolution kernel size (default: 3)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        signal_len: int = DEFAULT_SIGNAL_LEN,
        kmer_len: int = DEFAULT_KMER_LEN,
        num_features: int = DEFAULT_NUM_FEATURES,
        hidden_channels: int = 64,
        num_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()

        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.num_features = num_features

        # Signal branch: TCN
        # Input: (batch, 1, signal_len)
        self.signal_tcn = TCN(
            in_channels=1,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        # Pool to match kmer length
        self.signal_pool = nn.AdaptiveAvgPool1d(kmer_len)

        # Feature branch: TCN
        # Input: (batch, num_features, kmer_len)
        self.feature_tcn = TCN(
            in_channels=num_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Global pooling (mean and max)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Total features: (avg + max) * 2 branches * hidden_channels
        # Note: 2 branches instead of 3 (no sequence branch)
        total_features = hidden_channels * 2 * 2

        # Dense layers for classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # Binary classification
        )

    def forward(self, signal: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            signal: Raw signal (batch, signal_len)
            features: Dwell + signal level features (batch, num_features, kmer_len)

        Returns:
            Logits for binary classification (batch, 1)
        """
        # Signal branch
        # (batch, signal_len) -> (batch, 1, signal_len)
        signal_feat = signal.unsqueeze(1)
        signal_feat = self.signal_tcn(signal_feat)  # (batch, hidden_channels, signal_len)
        signal_feat = self.signal_pool(signal_feat)  # (batch, hidden_channels, kmer_len)

        # Feature branch
        feat_feat = self.feature_tcn(features)  # (batch, hidden_channels, kmer_len)

        # Global pooling for each branch
        # Average pooling
        signal_avg = self.global_avg_pool(signal_feat).squeeze(-1)  # (batch, hidden_channels)
        feat_avg = self.global_avg_pool(feat_feat).squeeze(-1)

        # Max pooling
        signal_max = self.global_max_pool(signal_feat).squeeze(-1)
        feat_max = self.global_max_pool(feat_feat).squeeze(-1)

        # Concatenate all features (only 2 branches now)
        merged = torch.cat([signal_avg, signal_max, feat_avg, feat_max], dim=1)

        # Classifier
        logits: torch.Tensor = self.classifier(merged)  # type: ignore[assignment]  # (batch, 1)

        return logits

    # predict_proba() is inherited from BaseModel
