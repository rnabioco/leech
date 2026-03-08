"""
ResNetDwell: Residual Network with dwell time features.

Architecture:
- Signal branch: ResNet-18 style (8 residual blocks)
- Sequence branch: Smaller ResNet (4 blocks)
- Feature branch: Lightweight ResNet (3 blocks)
- Global pooling → concatenate → FC classifier

Rationale:
- Deep feature extraction with skip connections
- Proven in signal processing applications
- Can go deeper without vanishing gradients
- Good generalization through residual learning
"""

import torch
import torch.nn as nn

from leech.constants import (
    DEFAULT_DROPOUT,
    DEFAULT_KMER_LEN,
    DEFAULT_NUM_FEATURES,
    DEFAULT_SIGNAL_KMER_CONTEXT,
    DEFAULT_SIGNAL_LEN,
)
from leech.models.components import BaseModel


class ResidualBlock1D(nn.Module):
    """
    1D Residual block with skip connection.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Stride for downsampling
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()

        padding = kernel_size // 2

        # Main path
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection
        self.skip: nn.Sequential | nn.Identity
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, length)

        Returns:
            Output tensor (batch, out_channels, length//stride)
        """
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add skip connection
        out += self.skip(identity)
        result: torch.Tensor = self.relu(out)

        return result


class ResNet1D(nn.Module):
    """
    1D ResNet with configurable depth.

    Args:
        in_channels: Number of input channels
        base_channels: Base number of channels (doubles with each layer)
        num_blocks: Number of residual blocks
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual blocks
        layers = []
        in_ch = base_channels
        for i in range(num_blocks):
            # Double channels every 2 blocks, downsample
            out_ch = base_channels * (2 ** (i // 2))
            stride = 2 if i % 2 == 0 and i > 0 else 1
            layers.append(
                ResidualBlock1D(
                    in_ch, out_ch, kernel_size=kernel_size, stride=stride, dropout=dropout
                )
            )
            in_ch = out_ch

        self.res_blocks = nn.Sequential(*layers)
        self.final_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, length)

        Returns:
            Output tensor (batch, final_channels, reduced_length)
        """
        out = self.conv1(x)
        result: torch.Tensor = self.res_blocks(out)
        return result


class ResNetDwell(BaseModel):
    """
    ResNet model with signal, sequence, and dwell feature branches.

    Architecture:
    - Signal branch: 8 residual blocks (ResNet-18 style)
    - Sequence branch: 4 residual blocks
    - Feature branch: 3 residual blocks
    - Global pooling → concatenate → FC classifier

    Args:
        signal_len: Length of input signal
        kmer_len: Length of k-mer sequence (e.g., 2*context+1)
        num_features: Number of feature channels (dwell + signal levels)
        base_channels: Base number of channels (default: 64)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        signal_len: int = DEFAULT_SIGNAL_LEN,
        kmer_len: int = DEFAULT_KMER_LEN,
        num_features: int = DEFAULT_NUM_FEATURES,
        base_channels: int = 64,
        dropout: float = DEFAULT_DROPOUT,
        seq_encoding: str = "base_onehot",
        signal_kmer_context: tuple[int, int] = DEFAULT_SIGNAL_KMER_CONTEXT,
    ):
        super().__init__()

        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.num_features = num_features
        self.seq_encoding = seq_encoding

        # Compute sequence input channels
        if seq_encoding == "signal_kmer":
            seq_in_channels = 4 * (signal_kmer_context[0] + signal_kmer_context[1] + 1)
        else:
            seq_in_channels = 4

        # Signal branch: Deep ResNet (8 blocks, ResNet-18 style)
        # Input: (batch, 1, signal_len)
        self.signal_resnet = ResNet1D(
            in_channels=1,
            base_channels=base_channels,
            num_blocks=8,
            dropout=dropout,
        )

        # Sequence branch: Medium ResNet (4 blocks)
        self.seq_resnet = ResNet1D(
            in_channels=seq_in_channels,
            base_channels=base_channels // 2,  # Smaller for sequence
            num_blocks=4,
            dropout=dropout,
        )

        # Feature branch: Lightweight ResNet (3 blocks)
        # Input: (batch, num_features, kmer_len)
        self.feature_resnet = ResNet1D(
            in_channels=num_features,
            base_channels=base_channels // 4,  # Smallest for features
            num_blocks=3,
            dropout=dropout,
        )

        # Global pooling (mean and max)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Get final channel sizes from each ResNet branch
        # Each ResNet1D tracks its final_channels attribute
        signal_final_ch = self.signal_resnet.final_channels
        seq_final_ch = self.seq_resnet.final_channels
        feat_final_ch = self.feature_resnet.final_channels

        # Total features: (avg + max) * 3 branches
        total_features = (signal_final_ch + seq_final_ch + feat_final_ch) * 2

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 1),  # Binary classification
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
        # Signal branch
        # (batch, signal_len) -> (batch, 1, signal_len)
        signal_feat = signal.unsqueeze(1)
        signal_feat = self.signal_resnet(signal_feat)  # (batch, final_ch, reduced_len)

        # Sequence branch — for signal_kmer, ResNet downsampling handles it
        seq_feat = self.seq_resnet(sequence)  # (batch, final_ch, reduced_len)

        # Feature branch
        feat_feat = self.feature_resnet(features)  # (batch, final_ch, reduced_len)

        # Global pooling for each branch
        # Average pooling
        signal_avg = self.global_avg_pool(signal_feat).squeeze(-1)  # (batch, channels)
        seq_avg = self.global_avg_pool(seq_feat).squeeze(-1)
        feat_avg = self.global_avg_pool(feat_feat).squeeze(-1)

        # Max pooling
        signal_max = self.global_max_pool(signal_feat).squeeze(-1)
        seq_max = self.global_max_pool(seq_feat).squeeze(-1)
        feat_max = self.global_max_pool(feat_feat).squeeze(-1)

        # Concatenate all features
        merged = torch.cat([signal_avg, signal_max, seq_avg, seq_max, feat_avg, feat_max], dim=1)

        # Classifier
        logits: torch.Tensor = self.classifier(merged)  # (batch, 1)

        return logits

    # predict_proba() is inherited from BaseModel
