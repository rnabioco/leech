"""
ResNetDwell: Residual Network with dwell time features.

Architecture:
- Signal branch: ResNet-18 style (8 residual blocks)
- Sequence branch: Smaller ResNet (4 blocks)
- Feature branch: Conv1d (full-width) for cross-attention K/V
- Cross-attention: ResNet output (Q) attends to wide dwell features (K/V)
- Attention pooling → MLP classifier

The dwell feature branch receives the full-width feature array
(kmer_len + margin_left + margin_right positions) so cross-attention
can learn the physical motor-sensor offset.
"""

import torch
import torch.nn as nn

from leech.constants import (
    DEFAULT_CONV_CHANNELS,
    DEFAULT_DROPOUT,
    DEFAULT_DWELL_MARGIN,
    DEFAULT_KMER_LEN,
    DEFAULT_NUM_FEATURES,
    DEFAULT_SIGNAL_KMER_CONTEXT,
    DEFAULT_SIGNAL_LEN,
)
from leech.models.components import BaseModel, FeatureBranch


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
    ResNet model with cross-attention for learning motor-sensor offset.

    Signal and sequence branches use ResNet1D. Their outputs are pooled
    to kmer_len positions and serve as Q for cross-attention against
    full-width dwell features (K/V), allowing each position to attend
    to dwell features at any offset.

    Args:
        signal_len: Length of input signal
        kmer_len: Length of k-mer sequence (e.g., 2*context+1)
        num_features: Number of feature channels (dwell + signal levels)
        dwell_margin: Extra bases on each side of dwell window (default: 15)
        base_channels: Base number of channels (default: 64)
        num_attn_heads: Number of attention heads for cross-attention (default: 4)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        signal_len: int = DEFAULT_SIGNAL_LEN,
        kmer_len: int = DEFAULT_KMER_LEN,
        num_features: int = DEFAULT_NUM_FEATURES,
        dwell_margin: int = DEFAULT_DWELL_MARGIN,
        base_channels: int = 64,
        num_attn_heads: int = 4,
        dropout: float = DEFAULT_DROPOUT,
        seq_encoding: str = "base_onehot",
        signal_kmer_context: tuple[int, int] = DEFAULT_SIGNAL_KMER_CONTEXT,
        signal_in_channels: int = 1,
    ):
        super().__init__()

        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.num_features = num_features
        self.dwell_margin = dwell_margin
        self.seq_encoding = seq_encoding

        conv_channels = DEFAULT_CONV_CHANNELS

        # Compute sequence input channels
        if seq_encoding == "signal_kmer":
            seq_in_channels = 4 * (signal_kmer_context[0] + signal_kmer_context[1] + 1)
        else:
            seq_in_channels = 4

        # Signal branch: Deep ResNet (8 blocks, ResNet-18 style)
        self.signal_resnet = ResNet1D(
            in_channels=signal_in_channels,
            base_channels=base_channels,
            num_blocks=8,
            dropout=dropout,
        )

        # Sequence branch: Medium ResNet (4 blocks)
        self.seq_resnet = ResNet1D(
            in_channels=seq_in_channels,
            base_channels=base_channels // 2,
            num_blocks=4,
            dropout=dropout,
        )

        # Pool ResNet outputs to kmer_len positions for cross-attention Q
        signal_final_ch = self.signal_resnet.final_channels
        seq_final_ch = self.seq_resnet.final_channels
        self.signal_pool = nn.AdaptiveAvgPool1d(kmer_len)
        self.seq_pool = nn.AdaptiveAvgPool1d(kmer_len)

        # Feature branch: Conv1d on full-width features for cross-attention K/V
        self.feature_branch = FeatureBranch(
            num_features=num_features, conv_channels=conv_channels, use_batchnorm=True
        )

        # Merged signal+seq dimension
        merged_dim = signal_final_ch + seq_final_ch

        # Cross-attention: merged signal+seq (Q) attends to wide dwell features (K/V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=merged_dim,
            num_heads=num_attn_heads,
            kdim=conv_channels[-1],  # 256 from FeatureBranch
            vdim=conv_channels[-1],
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(merged_dim)

        # Attention pooling over positions
        self.pool_linear = nn.Linear(merged_dim, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(merged_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self, signal: torch.Tensor, sequence: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            signal: Raw signal (batch, signal_len)
            sequence: Encoded sequence (batch, 4, kmer_len) or (batch, 36, signal_len)
            features: Dwell + signal level features with full margin
                (batch, num_features, kmer_len + margin_left + margin_right)

        Returns:
            Logits for binary classification (batch, 1)
        """
        # Signal branch
        if signal.dim() == 2:
            signal_feat = signal.unsqueeze(1)  # (batch, 1, signal_len)
        else:
            signal_feat = signal  # (batch, signal_in_channels, signal_len)
        signal_feat = self.signal_resnet(signal_feat)  # (batch, final_ch, reduced_len)
        signal_feat = self.signal_pool(signal_feat)  # (batch, final_ch, kmer_len)

        # Sequence branch
        seq_feat = self.seq_resnet(sequence)  # (batch, final_ch, reduced_len)
        seq_feat = self.seq_pool(seq_feat)  # (batch, final_ch, kmer_len)

        # Feature branch on full-width features
        feat_out = self.feature_branch(features)  # (batch, 256, feat_len)
        feat_out = feat_out.transpose(1, 2)  # (batch, feat_len, 256)

        # Merge signal + sequence → (batch, kmer_len, merged_dim)
        merged = torch.cat([signal_feat, seq_feat], dim=1)  # (batch, merged_dim, kmer_len)
        merged = merged.transpose(1, 2)  # (batch, kmer_len, merged_dim)

        # Cross-attention: merged (kmer_len positions) queries
        # dwell features (kmer_len + margin positions)
        attn_out, _ = self.cross_attn(
            query=merged,    # (batch, kmer_len, merged_dim)
            key=feat_out,    # (batch, feat_len, 256)
            value=feat_out,  # (batch, feat_len, 256)
        )  # attn_out: (batch, kmer_len, merged_dim)

        # Residual connection + layer norm
        combined = self.attn_norm(merged + attn_out)  # (batch, kmer_len, merged_dim)

        # Attention pooling over all positions
        pool_scores = self.pool_linear(combined)  # (batch, kmer_len, 1)
        pool_weights = torch.softmax(pool_scores, dim=1)  # (batch, kmer_len, 1)
        context_vector = (pool_weights * combined).sum(dim=1)  # (batch, merged_dim)

        # Classifier
        logits: torch.Tensor = self.classifier(context_vector)  # (batch, 1)

        return logits

    # predict_proba() is inherited from BaseModel
