"""
ConvLSTM models with attention pooling.

Six variants parameterized by has_features (Base/Dwell) and norm_type:

Base variants (no dwell features, simple attention pooling over LSTM):
- ConvLSTMBaseAttn: no normalization
- ConvLSTMBaseBNAttn: BatchNorm

Dwell variants (dwell features + cross-attention + attention pooling):
- ConvLSTMDwellAttn: no normalization
- ConvLSTMDwellBNAttn: BatchNorm
- ConvLSTMDwellGNAttn: GroupNorm
- ConvLSTMDwellLNAttn: LayerNorm

Base architecture:
    signal  → SignalBranch → pool → ┐
    sequence → SequenceBranch →     ├ cat → BiLSTM → attn pool → FC
                                    ┘

Dwell architecture:
    signal  → SignalBranch → pool → ┐
    sequence → SequenceBranch →     ├ cat → BiLSTM → cross-attn → attn pool → FC
    features → FeatureBranch →      ┘ (K/V for cross-attention)
"""

import torch
import torch.nn as nn

from leech.constants import (
    DEFAULT_CONV_CHANNELS,
    DEFAULT_DROPOUT,
    DEFAULT_DWELL_MARGIN,
    DEFAULT_FC_HIDDEN,
    DEFAULT_KMER_LEN,
    DEFAULT_LSTM_HIDDEN,
    DEFAULT_LSTM_LAYERS,
    DEFAULT_NUM_FEATURES,
    DEFAULT_SIGNAL_KMER_CONTEXT,
    DEFAULT_SIGNAL_LEN,
)
from leech.models.components import BaseModel, FeatureBranch, SequenceBranch, SignalBranch


class _ConvLSTMAttn(BaseModel):
    """Internal base for attention-based ConvLSTM variants.

    When _has_features=True: builds FeatureBranch + cross-attention (LSTM Q → feature K/V)
        + attention pooling via pool_linear.
    When _has_features=False: simple attention pooling over LSTM output via attn_linear.

    The different attribute names (pool_linear vs attn_linear) preserve state_dict
    compatibility with the original standalone classes.
    """

    def __init__(
        self,
        *,
        _has_features: bool,
        _norm_type: str = "none",
        signal_len: int = DEFAULT_SIGNAL_LEN,
        kmer_len: int = DEFAULT_KMER_LEN,
        num_features: int = DEFAULT_NUM_FEATURES,
        dwell_margin: int = DEFAULT_DWELL_MARGIN,
        conv_channels: list[int] | None = None,
        lstm_hidden: int = DEFAULT_LSTM_HIDDEN,
        num_attn_heads: int = 4,
        dropout: float = DEFAULT_DROPOUT,
        seq_encoding: str = "base_onehot",
        signal_kmer_context: tuple[int, int] = DEFAULT_SIGNAL_KMER_CONTEXT,
        num_out: int = 1,
        signal_in_channels: int = 1,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = DEFAULT_CONV_CHANNELS

        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.lstm_hidden = lstm_hidden
        self.seq_encoding = seq_encoding
        if _has_features:
            self.num_features = num_features
            self.dwell_margin = dwell_margin
            self.num_out = num_out

        # Compute sequence input channels
        if seq_encoding == "signal_kmer":
            seq_in_channels = 4 * (signal_kmer_context[0] + signal_kmer_context[1] + 1)
        else:
            seq_in_channels = 4

        norm_kw = {"norm_type": _norm_type} if _norm_type != "none" else {}

        self.signal_branch = SignalBranch(
            conv_channels=conv_channels, in_channels=signal_in_channels, **norm_kw
        )
        self.sequence_branch = SequenceBranch(
            in_channels=seq_in_channels, conv_channels=conv_channels, **norm_kw
        )

        if _has_features:
            self.feature_branch = FeatureBranch(
                num_features=num_features, conv_channels=conv_channels, **norm_kw
            )

        self.signal_pool = nn.AdaptiveAvgPool1d(kmer_len)
        if seq_encoding == "signal_kmer":
            self.seq_pool = nn.AdaptiveAvgPool1d(kmer_len)

        # BiLSTM on merged signal + sequence
        self.lstm = nn.LSTM(
            input_size=conv_channels[2] * 2,
            hidden_size=lstm_hidden,
            num_layers=DEFAULT_LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        if _has_features:
            # Cross-attention: LSTM output (Q) → dwell features (K/V)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=lstm_hidden * 2,
                num_heads=num_attn_heads,
                kdim=conv_channels[2],
                vdim=conv_channels[2],
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(lstm_hidden * 2)
            self.pool_linear = nn.Linear(lstm_hidden * 2, 1)
        else:
            # Simple attention pooling over LSTM positions
            self.attn_linear = nn.Linear(lstm_hidden * 2, 1)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, DEFAULT_FC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(DEFAULT_FC_HIDDEN, num_out),
        )

        self._has_features = _has_features

    def forward(
        self,
        signal: torch.Tensor,
        sequence: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        signal_feat = self.signal_branch(signal)
        signal_feat = self.signal_pool(signal_feat)

        seq_feat = self.sequence_branch(sequence)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)

        merged = torch.cat([signal_feat, seq_feat], dim=1)
        merged = merged.transpose(1, 2)
        lstm_out, _ = self.lstm(merged)

        if self._has_features:
            # Cross-attention with full-width dwell features
            feat_out = self.feature_branch(features)
            feat_out = feat_out.transpose(1, 2)

            attn_out, _ = self.cross_attn(
                query=lstm_out,
                key=feat_out,
                value=feat_out,
            )
            combined = self.attn_norm(lstm_out + attn_out)

            # Attention pooling
            pool_scores = self.pool_linear(combined)
            pool_weights = torch.softmax(pool_scores, dim=1)
            context_vector = (pool_weights * combined).sum(dim=1)
        else:
            # Simple attention pooling over LSTM
            attn_scores = self.attn_linear(lstm_out)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context_vector = (attn_weights * lstm_out).sum(dim=1)

        logits: torch.Tensor = self.fc(context_vector)
        return logits


# --- Base variants (no dwell features) ---


class ConvLSTMBaseAttn(_ConvLSTMAttn):
    """Baseline model with learned attention pooling over LSTM positions."""

    def __init__(self, **kwargs):
        super().__init__(_has_features=False, _norm_type="none", **kwargs)

    def forward(self, signal: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        return super().forward(signal, sequence)


class ConvLSTMBaseBNAttn(_ConvLSTMAttn):
    """Baseline model with BatchNorm and attention pooling."""

    def __init__(self, **kwargs):
        super().__init__(_has_features=False, _norm_type="batchnorm", **kwargs)

    def forward(self, signal: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        return super().forward(signal, sequence)


# --- Dwell variants (features + cross-attention) ---


class ConvLSTMDwellAttn(_ConvLSTMAttn):
    """Dwell model with cross-attention for learning motor-sensor offset."""

    def __init__(self, **kwargs):
        super().__init__(_has_features=True, _norm_type="none", **kwargs)


class ConvLSTMDwellBNAttn(_ConvLSTMAttn):
    """Dwell model with BatchNorm and cross-attention."""

    def __init__(self, **kwargs):
        super().__init__(_has_features=True, _norm_type="batchnorm", **kwargs)


class ConvLSTMDwellGNAttn(_ConvLSTMAttn):
    """Dwell model with GroupNorm and cross-attention."""

    def __init__(self, **kwargs):
        super().__init__(_has_features=True, _norm_type="groupnorm", **kwargs)


class ConvLSTMDwellLNAttn(_ConvLSTMAttn):
    """Dwell model with LayerNorm and cross-attention."""

    def __init__(self, **kwargs):
        super().__init__(_has_features=True, _norm_type="layernorm", **kwargs)
