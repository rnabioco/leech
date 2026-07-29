"""
TCNDwell: Temporal Convolutional Network with dwell time features.

Architecture:
- Signal branch: Stack of dilated causal convolutions with residual connections
- Sequence branch: Stack of dilated causal convolutions
- Feature branch: Conv1d (full-width) for cross-attention K/V
- Cross-attention: TCN output (Q) attends to wide dwell features (K/V)
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
from leech.models.components import BaseModel, FeatureBranch, make_norm


class TemporalBlock(nn.Module):
    """
    Temporal convolutional block with dilated causal convolutions and residual connection.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        dilation: Dilation rate
        dropout: Dropout probability
        norm_type: Normalization type ("batchnorm", "groupnorm", "layernorm")
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = DEFAULT_DROPOUT,
        norm_type: str = "batchnorm",
    ):
        super().__init__()

        # Padding to maintain sequence length
        # For causal convolutions: padding = (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation

        # Two convolutional layers with normalization and dropout
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # We'll manually pad
        )
        self.norm1 = make_norm(norm_type, out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.norm2 = make_norm(norm_type, out_channels)
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
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        # Causal padding again
        out = nn.functional.pad(out, (self.padding, 0))

        # Second conv block
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        result: torch.Tensor = self.relu(out + res)
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
        norm_type: Normalization type ("batchnorm", "groupnorm", "layernorm")
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = DEFAULT_DROPOUT,
        norm_type: str = "batchnorm",
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
                    norm_type=norm_type,
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
        out: torch.Tensor = self.network(x)
        return out


class TCNDwell(BaseModel):
    """
    TCN model with cross-attention for learning motor-sensor offset.

    Signal and sequence branches use TCN (dilated causal convolutions).
    Their outputs are merged and serve as Q for cross-attention against
    full-width dwell features (K/V), allowing each position to attend
    to dwell features at any offset.

    Args:
        signal_len: Length of input signal
        kmer_len: Length of k-mer sequence (e.g., 2*context+1)
        num_features: Number of feature channels (dwell + signal levels)
        dwell_margin: Extra bases on each side of dwell window (default: 15)
        hidden_channels: Number of channels in TCN layers (default: 64)
        num_layers: Number of temporal blocks per branch (default: 6)
        kernel_size: Convolution kernel size (default: 3)
        num_attn_heads: Number of attention heads for cross-attention (default: 4)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        signal_len: int = DEFAULT_SIGNAL_LEN,
        kmer_len: int = DEFAULT_KMER_LEN,
        num_features: int = DEFAULT_NUM_FEATURES,
        dwell_margin: int = DEFAULT_DWELL_MARGIN,
        hidden_channels: int = 64,
        num_layers: int = 6,
        kernel_size: int = 3,
        num_attn_heads: int = 4,
        dropout: float = DEFAULT_DROPOUT,
        seq_encoding: str = "base_onehot",
        signal_kmer_context: tuple[int, int] = DEFAULT_SIGNAL_KMER_CONTEXT,
        norm_type: str = "batchnorm",
        signal_in_channels: int = 1,
        num_out: int = 1,
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

        # Signal branch: TCN
        self.signal_tcn = TCN(
            in_channels=signal_in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            norm_type=norm_type,
        )
        self.signal_pool = nn.AdaptiveAvgPool1d(kmer_len)

        # Sequence branch: TCN
        self.seq_tcn = TCN(
            in_channels=seq_in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            norm_type=norm_type,
        )
        if seq_encoding == "signal_kmer":
            self.seq_pool = nn.AdaptiveAvgPool1d(kmer_len)

        # Feature branch: Conv1d on full-width features for cross-attention K/V
        # Processes kmer_len + margin_left + margin_right positions
        self.feature_branch = FeatureBranch(
            num_features=num_features, conv_channels=conv_channels, norm_type=norm_type
        )

        # Merged signal+seq dimension
        merged_dim = hidden_channels * 2

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

        # Dense layers for classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(merged_dim, 256),
            nn.ReLU(),
            make_norm(norm_type, 256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            make_norm(norm_type, 64),
            nn.Dropout(dropout),
            nn.Linear(64, num_out),
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
            Logits (batch, num_out)
        """
        # Signal branch: handle both 1-channel and multi-channel input
        if signal.dim() == 2:
            signal_feat = signal.unsqueeze(1)  # (batch, 1, signal_len)
        else:
            signal_feat = signal  # (batch, signal_in_channels, signal_len)
        signal_feat = self.signal_tcn(signal_feat)  # (batch, hidden_channels, signal_len)
        signal_feat = self.signal_pool(signal_feat)  # (batch, hidden_channels, kmer_len)

        # Sequence branch
        seq_feat = self.seq_tcn(sequence)  # (batch, hidden_channels, seq_len)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)  # (batch, hidden_channels, kmer_len)

        # Feature branch on full-width features
        feat_out = self.feature_branch(features)  # (batch, 256, feat_len)
        feat_out = feat_out.transpose(1, 2)  # (batch, feat_len, 256)

        # Merge signal + sequence → (batch, kmer_len, hidden_channels*2)
        merged = torch.cat([signal_feat, seq_feat], dim=1)  # (batch, hidden_channels*2, kmer_len)
        merged = merged.transpose(1, 2)  # (batch, kmer_len, hidden_channels*2)

        # Cross-attention: merged (kmer_len positions) queries
        # dwell features (kmer_len + margin positions)
        attn_out, _ = self.cross_attn(
            query=merged,  # (batch, kmer_len, hidden_channels*2)
            key=feat_out,  # (batch, feat_len, 256)
            value=feat_out,  # (batch, feat_len, 256)
            need_weights=False,  # enables fused SDPA kernel; weights are unused
        )  # attn_out: (batch, kmer_len, hidden_channels*2)

        # Residual connection + layer norm
        combined = self.attn_norm(merged + attn_out)  # (batch, kmer_len, hidden_channels*2)

        # Attention pooling over all positions
        pool_scores = self.pool_linear(combined)  # (batch, kmer_len, 1)
        pool_weights = torch.softmax(pool_scores, dim=1)  # (batch, kmer_len, 1)
        context_vector = (pool_weights * combined).sum(dim=1)  # (batch, hidden_channels*2)

        # Classifier
        logits: torch.Tensor = self.classifier(context_vector)  # (batch, 1)

        return logits

    # predict_proba() is inherited from BaseModel

    # Normalization variants (TCNDwellGN, TCNDwellLN) are declared in
    # models/configs/tcn_dwell_variants.toml — norm_type is already a
    # constructor switch, so they need no subclass.
