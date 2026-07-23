"""
TCNDwellResidualDwellAttn: Separate dwell-only attention variant.

Adds a second cross-attention head that attends ONLY to dwell features
(channels 0-4), independent from the main cross-attention that sees all
features. This forces the model to learn dwell-specific patterns
(especially at motor positions +8-12) without competition from
signal/kmer features at constriction positions.
"""

import torch
from torch import nn

from leech.models.components import FeatureBranch
from leech.models.tcn_dwell_residual import TCNDwellResidual

# Default: first 5 channels are dwell features
NUM_DWELL_FEATURES = 5
DWELL_CONV_CHANNELS = [4, 16, 64]


class TCNDwellResidualDwellAttn(TCNDwellResidual):
    """TCNDwellResidual with a separate dwell-only cross-attention head.

    The main cross-attention attends to all features (dwell + signal + kmer).
    A second cross-attention attends only to dwell features, forcing the model
    to extract dwell patterns independently.
    """

    def __init__(
        self,
        num_dwell_features: int = NUM_DWELL_FEATURES,
        dwell_conv_channels: list[int] | None = None,
        num_dwell_attn_heads: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if dwell_conv_channels is None:
            dwell_conv_channels = list(DWELL_CONV_CHANNELS)

        norm_type = kwargs.get("norm_type", "batchnorm")
        dropout = kwargs.get("dropout", 0.1)
        merged_dim = kwargs.get("hidden_channels", 64) * 2

        # Separate feature branch for dwell-only features
        self.dwell_branch = FeatureBranch(
            num_features=num_dwell_features,
            conv_channels=dwell_conv_channels,
            norm_type=norm_type,
        )
        self.num_dwell_features = num_dwell_features

        # Second cross-attention: merged signal+seq (Q) attends to dwell-only (K/V)
        self.dwell_attn = nn.MultiheadAttention(
            embed_dim=merged_dim,
            num_heads=num_dwell_attn_heads,
            kdim=dwell_conv_channels[-1],
            vdim=dwell_conv_channels[-1],
            dropout=dropout,
            batch_first=True,
        )
        self.dwell_attn_norm = nn.LayerNorm(merged_dim)

    def forward(
        self, signal: torch.Tensor, sequence: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        # Signal branch
        if signal.dim() == 2:
            signal_feat = signal.unsqueeze(1)
        else:
            signal_feat = signal
        signal_feat = self.signal_tcn(signal_feat)
        signal_feat = self.signal_pool(signal_feat)

        # Sequence branch
        seq_feat = self.seq_tcn(sequence)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)

        # Full feature branch (all channels) for main cross-attention
        feat_out = self.feature_branch(features)  # (batch, 256, feat_len)
        feat_kv = feat_out.transpose(1, 2)  # (batch, feat_len, 256)

        # Dwell-only feature branch (first 5 channels)
        dwell_feat = self.dwell_branch(
            features[:, : self.num_dwell_features, :]
        )  # (batch, 64, feat_len)
        dwell_kv = dwell_feat.transpose(1, 2)  # (batch, feat_len, 64)

        # Merge signal + sequence
        merged = torch.cat([signal_feat, seq_feat], dim=1)
        merged = merged.transpose(1, 2)  # (batch, kmer_len, merged_dim)

        # Main cross-attention (all features)
        attn_out, _ = self.cross_attn(query=merged, key=feat_kv, value=feat_kv, need_weights=False)
        combined = self.attn_norm(merged + attn_out)

        # Dwell-only cross-attention
        dwell_attn_out, _ = self.dwell_attn(
            query=combined, key=dwell_kv, value=dwell_kv, need_weights=False
        )
        combined = self.dwell_attn_norm(combined + dwell_attn_out)

        # Attention pooling
        pool_scores = self.pool_linear(combined)
        pool_weights = torch.softmax(pool_scores, dim=1)
        context_vector = (pool_weights * combined).sum(dim=1)

        logits: torch.Tensor = self.classifier(context_vector)
        return logits


class TCNDwellResidualLNDwellAttn(TCNDwellResidualDwellAttn):
    """TCNDwellResidualDwellAttn with LayerNorm."""

    def __init__(self, **kwargs):
        kwargs.setdefault("norm_type", "layernorm")
        super().__init__(**kwargs)
