"""
TCNDwellSplitResidual: Separate TCN branches for raw signal and kmer residual.

Unlike TCNDwellResidual which mixes raw signal and residual in the first Conv1d
(in_channels=2), this model processes them through independent TCN branches and
concatenates their outputs late. This prevents the model from trivially
extracting kmer-context information by mixing the two channels early.

Architecture:
    raw signal  → signal_tcn (1ch) → pool → ┐
    kmer resid  → residual_tcn (1ch) → pool → ├ cat → cross-attn → classifier
    sequence    → seq_tcn → pool →            ┘
    features    → feature_branch → (K/V for cross-attn)
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


class TCNDwellSplitResidual(BaseModel):
    """
    TCN model with separate branches for raw signal and kmer residual.

    The raw signal and residual are processed through independent TCN branches
    and only merged after feature extraction. This prevents early mixing that
    would let the model extract kmer-context information from the raw signal
    channel.

    Requires 2-channel signal input (signal_mode="both"). Falls back to
    duplicating the single channel if only 1 channel is provided.
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

        # Separate signal branches: each processes 1 channel independently
        self.signal_tcn = TCN(
            in_channels=1,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            norm_type=norm_type,
        )
        self.signal_pool = nn.AdaptiveAvgPool1d(kmer_len)

        self.residual_tcn = TCN(
            in_channels=1,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            norm_type=norm_type,
        )
        self.residual_pool = nn.AdaptiveAvgPool1d(kmer_len)

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

        # Merged signal + residual + sequence dimension
        merged_dim = hidden_channels * 3

        # Cross-attention: merged (Q) attends to wide dwell features (K/V)
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
            signal: (batch, 2, signal_len) for 2-channel, or (batch, signal_len) for 1-channel
            sequence: Encoded sequence (batch, 4, kmer_len) or (batch, 36, signal_len)
            features: Dwell + signal level features with full margin

        Returns:
            Logits (batch, num_out)
        """
        # Split 2-channel input into separate branches
        if signal.dim() == 3 and signal.shape[1] >= 2:
            raw_signal = signal[:, 0:1, :]  # (batch, 1, signal_len)
            residual = signal[:, 1:2, :]  # (batch, 1, signal_len)
        else:
            # 1-channel fallback: use same input for both branches
            if signal.dim() == 2:
                raw_signal = signal.unsqueeze(1)
            else:
                raw_signal = signal
            residual = raw_signal

        # Process each signal branch independently
        signal_feat = self.signal_tcn(raw_signal)  # (batch, hidden, signal_len)
        signal_feat = self.signal_pool(signal_feat)  # (batch, hidden, kmer_len)

        resid_feat = self.residual_tcn(residual)  # (batch, hidden, signal_len)
        resid_feat = self.residual_pool(resid_feat)  # (batch, hidden, kmer_len)

        # Sequence branch
        seq_feat = self.seq_tcn(sequence)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)

        # Feature branch on full-width features
        feat_out = self.feature_branch(features)  # (batch, 256, feat_len)
        feat_out = feat_out.transpose(1, 2)  # (batch, feat_len, 256)

        # Merge signal + residual + sequence
        merged = torch.cat([signal_feat, resid_feat, seq_feat], dim=1)
        merged = merged.transpose(1, 2)  # (batch, kmer_len, hidden*3)

        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=merged,
            key=feat_out,
            value=feat_out,
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


class TCNDwellSplitResidualLN(TCNDwellSplitResidual):
    """TCNDwellSplitResidual with LayerNorm instead of BatchNorm.

    Inherits all architecture from TCNDwellSplitResidual, overriding only the
    default norm_type to "layernorm".
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("norm_type", "layernorm")
        super().__init__(**kwargs)
