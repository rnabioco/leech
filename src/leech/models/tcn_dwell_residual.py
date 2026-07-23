"""
TCNDwellResidual: TCNDwell variant with 2-channel signal input (raw + kmer residual).

Identical to TCNDwell except:
- Signal branch accepts signal_in_channels (default 2) input channels
- Forward handles both (batch, signal_len) and (batch, 2, signal_len) inputs

This model is designed for comparison against TCNDwell to measure the
impact of the per-signal-sample kmer residual channel.
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
from leech.models.tcn_dwell import TCN


class TCNDwellResidual(BaseModel):
    """
    TCN model with 2-channel signal input and cross-attention for dwell features.

    Same architecture as TCNDwell, but the signal branch accepts multiple input
    channels (raw signal + kmer residual). This gives the model direct access to
    per-sample deviations from expected kmer levels.

    Args:
        signal_len: Length of input signal
        kmer_len: Length of k-mer sequence (e.g., 2*context+1)
        num_features: Number of feature channels (dwell + signal levels + kmer residuals)
        dwell_margin: Extra bases on each side of dwell window (default: 15)
        signal_in_channels: Number of signal input channels (1=raw only, 2=raw+residual)
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
        signal_in_channels: int = 2,
        hidden_channels: int = 64,
        num_layers: int = 6,
        kernel_size: int = 3,
        num_attn_heads: int = 4,
        dropout: float = DEFAULT_DROPOUT,
        seq_encoding: str = "base_onehot",
        signal_kmer_context: tuple[int, int] = DEFAULT_SIGNAL_KMER_CONTEXT,
        norm_type: str = "batchnorm",
        num_out: int = 1,
    ):
        super().__init__()

        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.num_features = num_features
        self.dwell_margin = dwell_margin
        self.signal_in_channels = signal_in_channels
        self.seq_encoding = seq_encoding

        conv_channels = DEFAULT_CONV_CHANNELS

        # Compute sequence input channels
        if seq_encoding == "signal_kmer":
            seq_in_channels = 4 * (signal_kmer_context[0] + signal_kmer_context[1] + 1)
        else:
            seq_in_channels = 4

        # Signal branch: TCN with configurable input channels
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
        self.feature_branch = FeatureBranch(
            num_features=num_features, conv_channels=conv_channels, norm_type=norm_type
        )

        # Merged signal+seq dimension
        merged_dim = hidden_channels * 2

        # Cross-attention: merged signal+seq (Q) attends to wide dwell features (K/V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=merged_dim,
            num_heads=num_attn_heads,
            kdim=conv_channels[-1],
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
            signal: Raw signal (batch, signal_len) or (batch, 2, signal_len) for 2-channel
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
            signal_feat = signal  # (batch, in_channels, signal_len)
        signal_feat = self.signal_tcn(signal_feat)  # (batch, hidden_channels, signal_len)
        signal_feat = self.signal_pool(signal_feat)  # (batch, hidden_channels, kmer_len)

        # Sequence branch
        seq_feat = self.seq_tcn(sequence)  # (batch, hidden_channels, seq_len)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)  # (batch, hidden_channels, kmer_len)

        # Feature branch on full-width features
        feat_out = self.feature_branch(features)  # (batch, 256, feat_len)
        feat_out = feat_out.transpose(1, 2)  # (batch, feat_len, 256)

        # Merge signal + sequence
        merged = torch.cat([signal_feat, seq_feat], dim=1)  # (batch, hidden_channels*2, kmer_len)
        merged = merged.transpose(1, 2)  # (batch, kmer_len, hidden_channels*2)

        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=merged,
            key=feat_out,
            value=feat_out,
            need_weights=False,  # enables fused SDPA kernel; weights are unused
        )

        # Residual connection + layer norm
        combined = self.attn_norm(merged + attn_out)

        # Attention pooling over all positions
        pool_scores = self.pool_linear(combined)
        pool_weights = torch.softmax(pool_scores, dim=1)
        context_vector = (pool_weights * combined).sum(dim=1)

        # Classifier
        logits: torch.Tensor = self.classifier(context_vector)

        return logits


class TCNDwellResidualGN(TCNDwellResidual):
    """TCNDwellResidual with GroupNorm instead of BatchNorm.

    Inherits all architecture from TCNDwellResidual, overriding only the
    default norm_type to "groupnorm".
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("norm_type", "groupnorm")
        super().__init__(**kwargs)


class TCNDwellResidualLN(TCNDwellResidual):
    """TCNDwellResidual with LayerNorm instead of BatchNorm.

    Inherits all architecture from TCNDwellResidual, overriding only the
    default norm_type to "layernorm".
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("norm_type", "layernorm")
        super().__init__(**kwargs)
