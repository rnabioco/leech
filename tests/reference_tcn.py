"""Frozen reference implementations of the TCN family.

These are the hand-written classes that ``leech/models/configs/tcn_dwell.toml``,
``tcn_dwell_residual.toml``, ``tcn_dwell_split_residual.toml``,
``tcn_dwell_residual_motor.toml`` and ``tcn_dwell_residual_dwell_attn.toml``
replaced.  They live here — deliberately outside ``src/`` — as a golden
reference: ``tests/test_models.py`` asserts that the config-built models have
identical ``state_dict()`` keys, identical same-seed initialisation, and
bit-identical outputs given the same weights.  Do not modernise this file; its
whole value is that it does not change.

``TemporalBlock`` and ``TCN`` are copied here too (they now live in
``leech.models.components``) so the reference stands on its own.

Architectures:
    signal   -> TCN -> pool -> |
    sequence -> TCN -> pool -> +- cat -> cross-attn (Q) -> attn pool -> MLP
    features -> FeatureBranch -----------^ (K/V, full width)

TCNDwellSplitResidual splits the two signal channels into independent TCNs;
TCNDwellResidualMotor adds a motor-region pooling pathway that bypasses
cross-attention; TCNDwellResidualDwellAttn adds a second, dwell-only
cross-attention head.
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

# Default: first 5 channels are dwell features
NUM_DWELL_FEATURES = 5
DWELL_CONV_CHANNELS = [4, 16, 64]


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
        # Signal branch: handle both 1-channel and multi-channel input
        if signal.dim() == 2:
            signal_feat = signal.unsqueeze(1)  # (batch, 1, signal_len)
        else:
            signal_feat = signal  # (batch, signal_in_channels, signal_len)
        signal_feat = self.signal_tcn(signal_feat)
        signal_feat = self.signal_pool(signal_feat)

        # Sequence branch
        seq_feat = self.seq_tcn(sequence)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)

        # Feature branch on full-width features
        feat_out = self.feature_branch(features)  # (batch, 256, feat_len)
        feat_out = feat_out.transpose(1, 2)  # (batch, feat_len, 256)

        # Merge signal + sequence
        merged = torch.cat([signal_feat, seq_feat], dim=1)
        merged = merged.transpose(1, 2)  # (batch, kmer_len, hidden_channels*2)

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

        logits: torch.Tensor = self.classifier(context_vector)
        return logits


class TCNDwellResidual(BaseModel):
    """
    TCN model with 2-channel signal input and cross-attention for dwell features.

    Same architecture as TCNDwell, but the signal branch accepts multiple input
    channels (raw signal + kmer residual).
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

        merged_dim = hidden_channels * 2

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=merged_dim,
            num_heads=num_attn_heads,
            kdim=conv_channels[-1],
            vdim=conv_channels[-1],
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(merged_dim)

        self.pool_linear = nn.Linear(merged_dim, 1)

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
        if signal.dim() == 2:
            signal_feat = signal.unsqueeze(1)
        else:
            signal_feat = signal
        signal_feat = self.signal_tcn(signal_feat)
        signal_feat = self.signal_pool(signal_feat)

        seq_feat = self.seq_tcn(sequence)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)

        feat_out = self.feature_branch(features)
        feat_out = feat_out.transpose(1, 2)

        merged = torch.cat([signal_feat, seq_feat], dim=1)
        merged = merged.transpose(1, 2)

        attn_out, _ = self.cross_attn(
            query=merged,
            key=feat_out,
            value=feat_out,
            need_weights=False,
        )

        combined = self.attn_norm(merged + attn_out)

        pool_scores = self.pool_linear(combined)
        pool_weights = torch.softmax(pool_scores, dim=1)
        context_vector = (pool_weights * combined).sum(dim=1)

        logits: torch.Tensor = self.classifier(context_vector)
        return logits


class TCNDwellSplitResidual(BaseModel):
    """
    TCN model with separate branches for raw signal and kmer residual.

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

        self.feature_branch = FeatureBranch(
            num_features=num_features, conv_channels=conv_channels, norm_type=norm_type
        )

        # Merged signal + residual + sequence dimension
        merged_dim = hidden_channels * 3

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=merged_dim,
            num_heads=num_attn_heads,
            kdim=conv_channels[-1],
            vdim=conv_channels[-1],
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(merged_dim)

        self.pool_linear = nn.Linear(merged_dim, 1)

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
        # Split 2-channel input into separate branches
        if signal.dim() == 3 and signal.shape[1] >= 2:
            raw_signal = signal[:, 0:1, :]
            residual = signal[:, 1:2, :]
        else:
            # 1-channel fallback: use same input for both branches
            if signal.dim() == 2:
                raw_signal = signal.unsqueeze(1)
            else:
                raw_signal = signal
            residual = raw_signal

        signal_feat = self.signal_tcn(raw_signal)
        signal_feat = self.signal_pool(signal_feat)

        resid_feat = self.residual_tcn(residual)
        resid_feat = self.residual_pool(resid_feat)

        seq_feat = self.seq_tcn(sequence)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)

        feat_out = self.feature_branch(features)
        feat_out = feat_out.transpose(1, 2)

        merged = torch.cat([signal_feat, resid_feat, seq_feat], dim=1)
        merged = merged.transpose(1, 2)

        attn_out, _ = self.cross_attn(
            query=merged,
            key=feat_out,
            value=feat_out,
            need_weights=False,
        )

        combined = self.attn_norm(merged + attn_out)

        pool_scores = self.pool_linear(combined)
        pool_weights = torch.softmax(pool_scores, dim=1)
        context_vector = (pool_weights * combined).sum(dim=1)

        logits: torch.Tensor = self.classifier(context_vector)
        return logits


class TCNDwellResidualMotor(TCNDwellResidual):
    """TCNDwellResidual with explicit motor-region pooling.

    Adds a parallel pathway that mean-pools feature branch output at motor
    positions and concatenates with the attention-pooled context vector.

    NOTE: ``super().__init__()`` builds a ``classifier`` that this constructor
    immediately replaces with a wider one.  The discarded head still consumes
    random numbers, which is why the config-built model cannot reproduce this
    class's same-seed initialisation (it does match key names, key order, and
    forward outputs).
    """

    def __init__(
        self,
        motor_pool_start: int = 8,
        motor_pool_end: int = 13,
        motor_proj_dim: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.motor_pool_start = motor_pool_start
        self.motor_pool_end = motor_pool_end

        # Feature branch output channels (find last Conv1d in sequential)
        feat_out_dim = [
            m.out_channels for m in self.feature_branch.conv_layers if isinstance(m, nn.Conv1d)
        ][-1]  # 256

        # Project pooled motor features to a compact representation
        norm_type = kwargs.get("norm_type", "batchnorm")
        self.motor_proj = nn.Sequential(
            nn.Linear(feat_out_dim, motor_proj_dim),
            nn.ReLU(),
            make_norm(norm_type, motor_proj_dim),
        )

        # Rebuild classifier with wider input (merged_dim + motor_proj_dim)
        merged_dim = kwargs.get("hidden_channels", 64) * 2
        dropout = kwargs.get("dropout", 0.1)
        num_out = kwargs.get("num_out", 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(merged_dim + motor_proj_dim, 256),
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
        if signal.dim() == 2:
            signal_feat = signal.unsqueeze(1)
        else:
            signal_feat = signal
        signal_feat = self.signal_tcn(signal_feat)
        signal_feat = self.signal_pool(signal_feat)

        seq_feat = self.seq_tcn(sequence)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)

        # Feature branch (full width for cross-attention AND motor pooling)
        feat_out = self.feature_branch(features)  # (batch, 256, feat_len)

        # Motor-region pooling: bypass cross-attention
        motor_start = min(self.motor_pool_start, feat_out.shape[2])
        motor_end = min(self.motor_pool_end, feat_out.shape[2])
        if motor_end > motor_start:
            motor_pooled = feat_out[:, :, motor_start:motor_end].mean(dim=2)
        else:
            motor_pooled = feat_out.mean(dim=2)
        motor_vec = self.motor_proj(motor_pooled)

        feat_kv = feat_out.transpose(1, 2)
        merged = torch.cat([signal_feat, seq_feat], dim=1)
        merged = merged.transpose(1, 2)

        attn_out, _ = self.cross_attn(query=merged, key=feat_kv, value=feat_kv, need_weights=False)
        combined = self.attn_norm(merged + attn_out)

        pool_scores = self.pool_linear(combined)
        pool_weights = torch.softmax(pool_scores, dim=1)
        context_vector = (pool_weights * combined).sum(dim=1)

        # Concatenate motor pathway with attention pathway
        context_vector = torch.cat([context_vector, motor_vec], dim=1)

        logits: torch.Tensor = self.classifier(context_vector)
        return logits


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
        if signal.dim() == 2:
            signal_feat = signal.unsqueeze(1)
        else:
            signal_feat = signal
        signal_feat = self.signal_tcn(signal_feat)
        signal_feat = self.signal_pool(signal_feat)

        seq_feat = self.seq_tcn(sequence)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)

        # Full feature branch (all channels) for main cross-attention
        feat_out = self.feature_branch(features)
        feat_kv = feat_out.transpose(1, 2)

        # Dwell-only feature branch (first 5 channels)
        dwell_feat = self.dwell_branch(features[:, : self.num_dwell_features, :])
        dwell_kv = dwell_feat.transpose(1, 2)

        merged = torch.cat([signal_feat, seq_feat], dim=1)
        merged = merged.transpose(1, 2)

        # Main cross-attention (all features)
        attn_out, _ = self.cross_attn(query=merged, key=feat_kv, value=feat_kv, need_weights=False)
        combined = self.attn_norm(merged + attn_out)

        # Dwell-only cross-attention
        dwell_attn_out, _ = self.dwell_attn(
            query=combined, key=dwell_kv, value=dwell_kv, need_weights=False
        )
        combined = self.dwell_attn_norm(combined + dwell_attn_out)

        pool_scores = self.pool_linear(combined)
        pool_weights = torch.softmax(pool_scores, dim=1)
        context_vector = (pool_weights * combined).sum(dim=1)

        logits: torch.Tensor = self.classifier(context_vector)
        return logits
