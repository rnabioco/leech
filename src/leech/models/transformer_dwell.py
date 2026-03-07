"""
TransformerDwell: Transformer-based model with dwell time features.

Architecture:
- Signal branch: Conv1d + positional encoding + transformer encoder
- Sequence branch: Embedding + positional encoding + transformer encoder
- Feature branch: Conv1d + positional encoding + transformer encoder
- Cross-attention fusion → MLP classifier

Rationale:
- Self-attention captures long-range dependencies better than LSTM (important for 9500 context)
- Multi-head attention naturally handles multi-modal fusion
- Attention weights provide interpretability
"""

import math

import torch
import torch.nn as nn

from leech.constants import (
    DEFAULT_DROPOUT,
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
        pe: torch.Tensor = self.pe  # type: ignore[assignment]
        x = x + pe[:, : x.size(1), :]
        out: torch.Tensor = self.dropout(x)  # type: ignore[assignment]
        return out


class TransformerDwell(BaseModel):
    """
    Transformer-based model with signal, sequence, and dwell feature branches.

    Args:
        signal_len: Length of input signal
        kmer_len: Length of k-mer sequence (e.g., 2*context+1)
        num_features: Number of feature channels (dwell + signal levels)
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
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = DEFAULT_DROPOUT,
        seq_encoding: str = "base_onehot",
        signal_kmer_context: tuple[int, int] = DEFAULT_SIGNAL_KMER_CONTEXT,
    ):
        super().__init__()

        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.num_features = num_features
        self.d_model = d_model
        self.seq_encoding = seq_encoding

        # Compute sequence input channels
        if seq_encoding == "signal_kmer":
            seq_in_channels = 4 * (signal_kmer_context[0] + signal_kmer_context[1] + 1)
        else:
            seq_in_channels = 4

        # Signal branch: Conv1d to project signal to d_model dimensions
        # Input: (batch, 1, signal_len)
        self.signal_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=DEFAULT_SIGNAL_KERNEL, padding=DEFAULT_SIGNAL_KERNEL // 2),
            nn.ReLU(),
            nn.Conv1d(
                64, d_model, kernel_size=DEFAULT_SIGNAL_KERNEL, padding=DEFAULT_SIGNAL_KERNEL // 2
            ),
            nn.ReLU(),
        )
        self.signal_pool = nn.AdaptiveAvgPool1d(kmer_len)  # Match sequence length

        # Sequence branch: Project encoding to d_model dimensions
        self.seq_conv = nn.Sequential(
            nn.Conv1d(seq_in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        if seq_encoding == "signal_kmer":
            self.seq_pool = nn.AdaptiveAvgPool1d(kmer_len)

        # Feature branch: Project features to d_model dimensions
        # Input: (batch, num_features, kmer_len)
        self.feature_conv = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Positional encoding for each branch
        self.pos_encoding = PositionalEncoding(d_model, max_len=kmer_len, dropout=dropout)

        # Transformer encoder for each branch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.signal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.seq_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.feature_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention for multi-modal fusion
        # Query: concatenated features, Key/Value: concatenated features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model * 3,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # MLP classifier
        # Takes center position from fused features
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),  # Binary classification
        )

    def forward(
        self, signal: torch.Tensor, sequence: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            signal: Raw signal (batch, signal_len)
            sequence: One-hot encoded sequence (batch, 4, kmer_len)
            features: Dwell + signal level features (batch, num_features, kmer_len)

        Returns:
            Logits for binary classification (batch, 1)
        """
        # Signal branch
        # (batch, signal_len) -> (batch, 1, signal_len)
        signal_in = signal.unsqueeze(1)
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

        # Feature branch
        feat_feat = self.feature_conv(features)  # (batch, d_model, kmer_len)
        feat_feat = feat_feat.transpose(1, 2)  # (batch, kmer_len, d_model)
        feat_feat = self.pos_encoding(feat_feat)
        feat_feat = self.feature_transformer(feat_feat)  # (batch, kmer_len, d_model)

        # Concatenate all three branches
        # (batch, kmer_len, d_model) x 3 -> (batch, kmer_len, d_model*3)
        merged = torch.cat([signal_feat, seq_feat, feat_feat], dim=2)

        # Cross-attention (self-attention on merged features)
        attended, _ = self.cross_attention(merged, merged, merged)  # (batch, kmer_len, d_model*3)

        # Take center position
        center_idx = self.kmer_len // 2
        center_out = attended[:, center_idx, :]  # (batch, d_model*3)

        # Classifier
        logits: torch.Tensor = self.classifier(center_out)  # type: ignore[assignment]  # (batch, 1)

        return logits

    # predict_proba() is inherited from BaseModel
