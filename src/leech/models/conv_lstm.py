"""
Unified ConvLSTM model architecture.

Consolidates 8 ConvLSTM variants (Base/Dwell x BN x Attn) into a single
parameterized ``ConvLSTMCore`` class with boolean flags:

- ``use_features``: enables feature branch (Dwell vs Base)
- ``use_batchnorm``: enables BatchNorm in branch convolutions
- ``use_attention``: enables cross-attention (Dwell) or attention pooling (Base)

Named subclasses preserve backward compatibility for model registry and
isinstance checks.
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


class ConvLSTMCore(BaseModel):
    """
    Unified ConvLSTM model with configurable feature, batchnorm, and attention.

    Four forward paths depending on ``use_features`` and ``use_attention``:

    - features=True,  attention=False: signal+seq+features -> LSTM(3x) -> center select
    - features=True,  attention=True:  signal+seq -> LSTM(2x), features -> cross_attn -> attn pool
    - features=False, attention=False: signal+seq -> LSTM(2x) -> center select
    - features=False, attention=True:  signal+seq -> LSTM(2x) -> attn pool

    Args:
        use_features: Enable feature branch (dwell + signal level features).
        use_batchnorm: Enable BatchNorm in branch convolutions.
        use_attention: Enable cross-attention (features) or attention pooling (base).
        signal_len: Length of input signal.
        kmer_len: Length of k-mer sequence.
        num_features: Number of feature channels.
        dwell_margin: Extra bases on each side for attention models.
        conv_channels: Channel sizes for conv layers.
        lstm_hidden: Hidden size for BiLSTM.
        num_attn_heads: Number of attention heads for cross-attention.
        dropout: Dropout probability.
        seq_encoding: Sequence encoding type.
        signal_kmer_context: Kmer context for signal_kmer encoding.
        num_out: Number of output units.
    """

    def __init__(
        self,
        *,
        use_features: bool = True,
        use_batchnorm: bool = False,
        use_attention: bool = False,
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
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = DEFAULT_CONV_CHANNELS

        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.num_features = num_features
        self.lstm_hidden = lstm_hidden
        self.seq_encoding = seq_encoding
        self.num_out = num_out
        self.use_features = use_features
        self.use_attention = use_attention

        # Compute sequence input channels
        if seq_encoding == "signal_kmer":
            seq_in_channels = 4 * (signal_kmer_context[0] + signal_kmer_context[1] + 1)
        else:
            seq_in_channels = 4

        # Signal branch
        self.signal_branch = SignalBranch(
            conv_channels=conv_channels, use_batchnorm=use_batchnorm
        )

        # Sequence branch
        self.sequence_branch = SequenceBranch(
            in_channels=seq_in_channels,
            conv_channels=conv_channels,
            use_batchnorm=use_batchnorm,
        )

        # Feature branch (only for Dwell variants)
        if use_features:
            self.feature_branch = FeatureBranch(
                num_features=num_features,
                conv_channels=conv_channels,
                use_batchnorm=use_batchnorm,
            )
            if use_attention:
                self.dwell_margin = dwell_margin

        # Adaptive pooling to match dimensions
        self.signal_pool = nn.AdaptiveAvgPool1d(kmer_len)
        if seq_encoding == "signal_kmer":
            self.seq_pool = nn.AdaptiveAvgPool1d(kmer_len)

        # LSTM input size depends on whether features are concatenated before LSTM
        if use_features and not use_attention:
            lstm_input = conv_channels[2] * 3  # signal + seq + features
        else:
            lstm_input = conv_channels[2] * 2  # signal + seq

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden,
            num_layers=DEFAULT_LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        # Attention layers (only for Attn variants)
        if use_attention:
            if use_features:
                # Cross-attention: LSTM output (Q) attends to dwell features (K, V)
                self.cross_attn = nn.MultiheadAttention(
                    embed_dim=lstm_hidden * 2,
                    num_heads=num_attn_heads,
                    kdim=conv_channels[2],
                    vdim=conv_channels[2],
                    dropout=dropout,
                    batch_first=True,
                )
                self.attn_norm = nn.LayerNorm(lstm_hidden * 2)

            # Attention pooling over positions
            self.pool_linear = nn.Linear(lstm_hidden * 2, 1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, DEFAULT_FC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(DEFAULT_FC_HIDDEN, num_out),
        )

    def forward(
        self,
        signal: torch.Tensor,
        sequence: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            signal: Raw signal (batch, signal_len)
            sequence: Encoded sequence (batch, C, seq_len)
            features: Dwell + signal level features (batch, num_features, feat_len).
                      Required for Dwell variants, ignored for Base variants.

        Returns:
            Logits (batch, num_out)
        """
        # Signal branch
        signal_feat = self.signal_branch(signal)
        signal_feat = self.signal_pool(signal_feat)

        # Sequence branch
        seq_feat = self.sequence_branch(sequence)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)

        if self.use_features and not self.use_attention:
            # Dwell (no attention): concat all three -> LSTM -> center select
            feat_feat = self.feature_branch(features)
            merged = torch.cat([signal_feat, seq_feat, feat_feat], dim=1)
            merged = merged.transpose(1, 2)
            lstm_out, _ = self.lstm(merged)
            center_idx = self.kmer_len // 2
            out = lstm_out[:, center_idx, :]

        elif self.use_features and self.use_attention:
            # Dwell + Attention: signal+seq -> LSTM, features -> cross_attn -> pool
            feat_out = self.feature_branch(features)
            feat_out = feat_out.transpose(1, 2)

            merged = torch.cat([signal_feat, seq_feat], dim=1)
            merged = merged.transpose(1, 2)
            lstm_out, _ = self.lstm(merged)

            attn_out, _ = self.cross_attn(
                query=lstm_out, key=feat_out, value=feat_out
            )
            combined = self.attn_norm(lstm_out + attn_out)

            pool_scores = self.pool_linear(combined)
            pool_weights = torch.softmax(pool_scores, dim=1)
            out = (pool_weights * combined).sum(dim=1)

        elif not self.use_features and self.use_attention:
            # Base + Attention: signal+seq -> LSTM -> attn pool
            merged = torch.cat([signal_feat, seq_feat], dim=1)
            merged = merged.transpose(1, 2)
            lstm_out, _ = self.lstm(merged)

            attn_scores = self.pool_linear(lstm_out)
            attn_weights = torch.softmax(attn_scores, dim=1)
            out = (attn_weights * lstm_out).sum(dim=1)

        else:
            # Base (no attention): signal+seq -> LSTM -> center select
            merged = torch.cat([signal_feat, seq_feat], dim=1)
            merged = merged.transpose(1, 2)
            lstm_out, _ = self.lstm(merged)
            center_idx = self.kmer_len // 2
            out = lstm_out[:, center_idx, :]

        logits: torch.Tensor = self.fc(out)
        return logits


# ---------------------------------------------------------------------------
# Named subclasses for backward compatibility
# ---------------------------------------------------------------------------


class ConvLSTMDwell(ConvLSTMCore):
    """Full model with dwell time features."""

    def __init__(self, **kwargs):
        super().__init__(use_features=True, use_batchnorm=False, use_attention=False, **kwargs)


class ConvLSTMDwellBN(ConvLSTMCore):
    """Full model with dwell features and BatchNorm."""

    def __init__(self, **kwargs):
        super().__init__(use_features=True, use_batchnorm=True, use_attention=False, **kwargs)


class ConvLSTMDwellAttn(ConvLSTMCore):
    """Dwell model with cross-attention for learned offset."""

    def __init__(self, **kwargs):
        super().__init__(use_features=True, use_batchnorm=False, use_attention=True, **kwargs)


class ConvLSTMDwellBNAttn(ConvLSTMCore):
    """Dwell model with BatchNorm and cross-attention."""

    def __init__(self, **kwargs):
        super().__init__(use_features=True, use_batchnorm=True, use_attention=True, **kwargs)


class ConvLSTMBase(ConvLSTMCore):
    """Baseline model without dwell features."""

    def __init__(self, **kwargs):
        super().__init__(use_features=False, use_batchnorm=False, use_attention=False, **kwargs)


class ConvLSTMBaseBN(ConvLSTMCore):
    """Baseline model with BatchNorm."""

    def __init__(self, **kwargs):
        super().__init__(use_features=False, use_batchnorm=True, use_attention=False, **kwargs)


class ConvLSTMBaseAttn(ConvLSTMCore):
    """Baseline model with attention pooling."""

    def __init__(self, **kwargs):
        super().__init__(use_features=False, use_batchnorm=False, use_attention=True, **kwargs)


class ConvLSTMBaseBNAttn(ConvLSTMCore):
    """Baseline model with BatchNorm and attention pooling."""

    def __init__(self, **kwargs):
        super().__init__(use_features=False, use_batchnorm=True, use_attention=True, **kwargs)
