"""
ConvLSTMDwellGNAttn: Dwell model with GroupNorm + cross-attention.

Same architecture as ConvLSTMDwellBNAttn but uses GroupNorm (batch-independent)
instead of BatchNorm. GroupNorm provides normalization benefits without
sensitivity to batch composition, which may reduce task-level variance
in imbalanced OVA classification.
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


class ConvLSTMDwellGNAttn(BaseModel):
    """
    Dwell model with GroupNorm and cross-attention for learning motor-sensor offset.

    Same architecture as ConvLSTMDwellBNAttn but with GroupNorm instead of BatchNorm.

    Args:
        signal_len: Length of input signal
        kmer_len: Length of k-mer sequence (e.g., 2*context+1)
        num_features: Number of feature channels (dwell + signal levels)
        dwell_margin: Extra bases on each side of dwell window (default: 15)
        conv_channels: List of channel sizes for conv layers (default: [4, 16, 256])
        lstm_hidden: Hidden size for BiLSTM (default: 96)
        num_attn_heads: Number of attention heads for cross-attention (default: 4)
        dropout: Dropout probability (default: 0.1)
        seq_encoding: Sequence encoding type ("base_onehot" or "signal_kmer")
        signal_kmer_context: Kmer context for signal_kmer encoding (default: (4, 4))
    """

    def __init__(
        self,
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
        self.num_features = num_features
        self.dwell_margin = dwell_margin
        self.lstm_hidden = lstm_hidden
        self.seq_encoding = seq_encoding
        self.num_out = num_out

        # Compute sequence input channels
        if seq_encoding == "signal_kmer":
            seq_in_channels = 4 * (signal_kmer_context[0] + signal_kmer_context[1] + 1)
        else:
            seq_in_channels = 4

        # Signal branch (with GroupNorm)
        self.signal_branch = SignalBranch(
            conv_channels=conv_channels, norm_type="groupnorm", in_channels=signal_in_channels
        )

        # Sequence branch (with GroupNorm)
        self.sequence_branch = SequenceBranch(
            in_channels=seq_in_channels, conv_channels=conv_channels, norm_type="groupnorm"
        )

        # Feature branch (dwell + signal level features, with GroupNorm)
        self.feature_branch = FeatureBranch(
            num_features=num_features, conv_channels=conv_channels, norm_type="groupnorm"
        )

        # Adaptive pooling to match dimensions
        self.signal_pool = nn.AdaptiveAvgPool1d(kmer_len)
        if seq_encoding == "signal_kmer":
            self.seq_pool = nn.AdaptiveAvgPool1d(kmer_len)

        # BiLSTM on merged signal + sequence features
        self.lstm = nn.LSTM(
            input_size=conv_channels[2] * 2,  # signal + sequence
            hidden_size=lstm_hidden,
            num_layers=DEFAULT_LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

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
        self, signal: torch.Tensor, sequence: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            signal: Raw signal (batch, signal_len)
            sequence: Encoded sequence — (batch, 4, kmer_len) for base_onehot
                or (batch, 36, signal_len) for signal_kmer
            features: Dwell + signal level features with full margin
                (batch, num_features, kmer_len + 2*dwell_margin)

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

        # Feature branch on full-width features
        feat_out = self.feature_branch(features)
        feat_out = feat_out.transpose(1, 2)

        # Merge signal + sequence → BiLSTM
        merged = torch.cat([signal_feat, seq_feat], dim=1)
        merged = merged.transpose(1, 2)
        lstm_out, _ = self.lstm(merged)

        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=lstm_out,
            key=feat_out,
            value=feat_out,
        )

        # Residual connection + layer norm
        combined = self.attn_norm(lstm_out + attn_out)

        # Attention pooling over all positions
        pool_scores = self.pool_linear(combined)
        pool_weights = torch.softmax(pool_scores, dim=1)
        context_vector = (pool_weights * combined).sum(dim=1)

        # FC layers
        logits: torch.Tensor = self.fc(context_vector)
        return logits
