"""
TransformerDwell: Transformer-based model with dwell time features.

Architecture:
- Signal branch: Conv1d + positional encoding + transformer encoder
- Sequence branch: Embedding + positional encoding + transformer encoder
- Feature branch: Conv1d (full-width) for cross-attention K/V
- Cross-attention: transformer output (Q) attends to wide dwell features (K/V)
- Attention pooling → MLP classifier

The dwell feature branch receives the full-width feature array
(kmer_len + margin_left + margin_right positions) so cross-attention
can learn the physical motor-sensor offset.
"""

import math

import torch
import torch.nn as nn

from leech.constants import (
    DEFAULT_DROPOUT,
    DEFAULT_DWELL_MARGIN,
    DEFAULT_KMER_LEN,
    DEFAULT_NUM_FEATURES,
    DEFAULT_SIGNAL_KERNEL,
    DEFAULT_SIGNAL_KMER_CONTEXT,
    DEFAULT_SIGNAL_LEN,
)
from leech.models.components import BaseModel


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.

    Args:
        d_model: Dimension of model embeddings
        max_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added (batch, seq_len, d_model)
        """
        assert isinstance(self.pe, torch.Tensor)
        x = x + self.pe[:, : x.size(1), :]
        out: torch.Tensor = self.dropout(x)
        return out


class TransformerDwell(BaseModel):
    """
    Transformer-based model with cross-attention for learning motor-sensor offset.

    Signal and sequence branches each get their own transformer encoder.
    Their outputs are concatenated and serve as Q for cross-attention
    against full-width dwell features (K/V), allowing each position to
    attend to dwell features at any offset.

    Args:
        signal_len: Length of input signal
        kmer_len: Length of k-mer sequence (e.g., 2*context+1)
        num_features: Number of feature channels (dwell + signal levels)
        dwell_margin: Extra bases on each side of dwell window (default: 15)
        d_model: Dimension of transformer model (default: 256)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer encoder layers (default: 4)
        dim_feedforward: Dimension of feedforward network (default: 1024)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        signal_len: int = DEFAULT_SIGNAL_LEN,
        kmer_len: int = DEFAULT_KMER_LEN,
        num_features: int = DEFAULT_NUM_FEATURES,
        dwell_margin: int = DEFAULT_DWELL_MARGIN,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
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
        self.d_model = d_model
        self.seq_encoding = seq_encoding

        # Compute sequence input channels
        if seq_encoding == "signal_kmer":
            seq_in_channels = 4 * (signal_kmer_context[0] + signal_kmer_context[1] + 1)
        else:
            seq_in_channels = 4

        # Signal branch: Conv1d to project signal to d_model dimensions
        self.signal_conv = nn.Sequential(
            nn.Conv1d(
                signal_in_channels,
                64,
                kernel_size=DEFAULT_SIGNAL_KERNEL,
                padding=DEFAULT_SIGNAL_KERNEL // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                64, d_model, kernel_size=DEFAULT_SIGNAL_KERNEL, padding=DEFAULT_SIGNAL_KERNEL // 2
            ),
            nn.ReLU(),
        )
        self.signal_pool = nn.AdaptiveAvgPool1d(kmer_len)

        # Sequence branch: Project encoding to d_model dimensions
        self.seq_conv = nn.Sequential(
            nn.Conv1d(seq_in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        if seq_encoding == "signal_kmer":
            self.seq_pool = nn.AdaptiveAvgPool1d(kmer_len)

        # Feature branch: Conv1d on full-width features for cross-attention K/V
        # Input: (batch, num_features, kmer_len + margin_left + margin_right)
        self.feature_conv = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )

        # Positional encoding for signal/sequence branches
        self.pos_encoding = PositionalEncoding(
            d_model, max_len=max(kmer_len, signal_len), dropout=dropout
        )

        # Transformer encoder for signal and sequence branches
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.signal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.seq_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention: merged signal+seq (Q) attends to wide dwell features (K/V)
        # Q dim = d_model*2 (signal + sequence concatenated)
        # K/V dim = d_model (from feature conv)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model * 2,
            num_heads=nhead,
            kdim=d_model,
            vdim=d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model * 2)

        # Attention pooling over positions
        self.pool_linear = nn.Linear(d_model * 2, 1)

        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
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
            signal_in = signal.unsqueeze(1)  # (batch, 1, signal_len)
        else:
            signal_in = signal  # (batch, signal_in_channels, signal_len)
        signal_feat = self.signal_conv(signal_in)  # (batch, d_model, signal_len)
        signal_feat = self.signal_pool(signal_feat)  # (batch, d_model, kmer_len)
        signal_feat = signal_feat.transpose(1, 2)  # (batch, kmer_len, d_model)
        signal_feat = self.pos_encoding(signal_feat)
        signal_feat = self.signal_transformer(signal_feat)  # (batch, kmer_len, d_model)

        # Sequence branch
        seq_feat = self.seq_conv(sequence)  # (batch, d_model, seq_len)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)  # (batch, d_model, kmer_len)
        seq_feat = seq_feat.transpose(1, 2)  # (batch, kmer_len, d_model)
        seq_feat = self.pos_encoding(seq_feat)
        seq_feat = self.seq_transformer(seq_feat)  # (batch, kmer_len, d_model)

        # Feature branch on full-width features
        feat_out = self.feature_conv(features)  # (batch, d_model, feat_len)
        feat_out = feat_out.transpose(1, 2)  # (batch, feat_len, d_model)

        # Merge signal + sequence
        merged = torch.cat([signal_feat, seq_feat], dim=2)  # (batch, kmer_len, d_model*2)

        # Cross-attention: merged (kmer_len positions) queries
        # dwell features (kmer_len + margin positions)
        attn_out, _ = self.cross_attn(
            query=merged,  # (batch, kmer_len, d_model*2)
            key=feat_out,  # (batch, feat_len, d_model)
            value=feat_out,  # (batch, feat_len, d_model)
        )  # attn_out: (batch, kmer_len, d_model*2)

        # Residual connection + layer norm
        combined = self.attn_norm(merged + attn_out)  # (batch, kmer_len, d_model*2)

        # Attention pooling over all positions
        pool_scores = self.pool_linear(combined)  # (batch, kmer_len, 1)
        pool_weights = torch.softmax(pool_scores, dim=1)  # (batch, kmer_len, 1)
        context_vector = (pool_weights * combined).sum(dim=1)  # (batch, d_model*2)

        # Classifier
        logits: torch.Tensor = self.classifier(context_vector)  # (batch, 1)

        return logits

    # predict_proba() is inherited from BaseModel
