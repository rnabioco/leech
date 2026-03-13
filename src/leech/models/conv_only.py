"""
ConvOnly: Pure CNN model without recurrent layers, with dwell cross-attention.

Architecture:
- Signal branch: Multi-scale convolutional layers (Inception-style)
- Sequence branch: Multi-scale convolutional layers
- Feature branch: Conv1d (full-width) for cross-attention K/V
- Cross-attention: CNN output (Q) attends to wide dwell features (K/V)
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
from leech.models.components import BaseModel, FeatureBranch


class InceptionBlock(nn.Module):
    """
    Inception-style block with parallel convolutions of different kernel sizes.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels per branch
        kernel_sizes: List of kernel sizes for parallel convolutions
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int] | None = None,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5, 7]

        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2  # Keep same length
            branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
            self.branches.append(branch)

        # Max pooling branch
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, length)

        Returns:
            Concatenated outputs from all branches (batch, out_channels*5, length)
        """
        branch_outputs = [branch(x) for branch in self.branches]
        pool_output = self.pool_branch(x)
        return torch.cat(branch_outputs + [pool_output], dim=1)


class ConvOnly(BaseModel):
    """
    Pure CNN model with cross-attention for learning motor-sensor offset.

    Signal and sequence branches use Inception-style multi-scale convolutions.
    Their outputs are merged and serve as Q for cross-attention against
    full-width dwell features (K/V), allowing each position to attend
    to dwell features at any offset.

    Args:
        signal_len: Length of input signal
        kmer_len: Length of k-mer sequence (e.g., 2*context+1)
        num_features: Number of feature channels (dwell + signal levels)
        dwell_margin: Extra bases on each side of dwell window (default: 15)
        base_channels: Base number of channels for inception blocks (default: 16)
        num_blocks: Number of inception blocks per branch (default: 3)
        num_attn_heads: Number of attention heads for cross-attention (default: 4)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        signal_len: int = DEFAULT_SIGNAL_LEN,
        kmer_len: int = DEFAULT_KMER_LEN,
        num_features: int = DEFAULT_NUM_FEATURES,
        dwell_margin: int = DEFAULT_DWELL_MARGIN,
        base_channels: int = 16,
        num_blocks: int = 3,
        num_attn_heads: int = 4,
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
        self.seq_encoding = seq_encoding

        conv_channels = DEFAULT_CONV_CHANNELS

        # Compute sequence input channels
        if seq_encoding == "signal_kmer":
            seq_in_channels = 4 * (signal_kmer_context[0] + signal_kmer_context[1] + 1)
        else:
            seq_in_channels = 4

        # Signal branch: Stack of inception blocks
        self.signal_conv = nn.ModuleList()
        in_ch = signal_in_channels
        for i in range(num_blocks):
            out_ch = base_channels * (2**i)
            self.signal_conv.append(InceptionBlock(in_ch, out_ch))
            in_ch = out_ch * 5  # 4 kernel sizes + 1 pool branch

        # Pool to match kmer length
        self.signal_pool = nn.AdaptiveMaxPool1d(kmer_len)

        # Sequence branch: Stack of inception blocks
        self.seq_conv = nn.ModuleList()
        in_ch = seq_in_channels
        for i in range(num_blocks):
            out_ch = base_channels * (2**i)
            self.seq_conv.append(InceptionBlock(in_ch, out_ch))
            in_ch = out_ch * 5

        if seq_encoding == "signal_kmer":
            self.seq_pool_sk = nn.AdaptiveMaxPool1d(kmer_len)

        # Calculate final channel size after all inception blocks
        final_inception_ch = base_channels * (2 ** (num_blocks - 1)) * 5

        # Feature branch: Conv1d on full-width features for cross-attention K/V
        self.feature_branch = FeatureBranch(
            num_features=num_features, conv_channels=conv_channels, use_batchnorm=True
        )

        # Merged signal+seq dimension
        merged_dim = final_inception_ch * 2

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

        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(merged_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
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
            signal_feat = signal.unsqueeze(1)  # (batch, 1, signal_len)
        else:
            signal_feat = signal  # (batch, signal_in_channels, signal_len)
        for conv_block in self.signal_conv:
            signal_feat = conv_block(signal_feat)
        signal_feat = self.signal_pool(signal_feat)  # (batch, channels, kmer_len)

        # Sequence branch
        seq_feat = sequence
        for conv_block in self.seq_conv:
            seq_feat = conv_block(seq_feat)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool_sk(seq_feat)  # Pool to kmer_len

        # Feature branch on full-width features
        feat_out = self.feature_branch(features)  # (batch, 256, feat_len)
        feat_out = feat_out.transpose(1, 2)  # (batch, feat_len, 256)

        # Merge signal + sequence → (batch, kmer_len, merged_dim)
        merged = torch.cat([signal_feat, seq_feat], dim=1)  # (batch, merged_dim, kmer_len)
        merged = merged.transpose(1, 2)  # (batch, kmer_len, merged_dim)

        # Cross-attention: merged (kmer_len positions) queries
        # dwell features (kmer_len + margin positions)
        attn_out, _ = self.cross_attn(
            query=merged,    # (batch, kmer_len, merged_dim)
            key=feat_out,    # (batch, feat_len, 256)
            value=feat_out,  # (batch, feat_len, 256)
        )  # attn_out: (batch, kmer_len, merged_dim)

        # Residual connection + layer norm
        combined = self.attn_norm(merged + attn_out)  # (batch, kmer_len, merged_dim)

        # Attention pooling over all positions
        pool_scores = self.pool_linear(combined)  # (batch, kmer_len, 1)
        pool_weights = torch.softmax(pool_scores, dim=1)  # (batch, kmer_len, 1)
        context_vector = (pool_weights * combined).sum(dim=1)  # (batch, merged_dim)

        # Classifier
        logits: torch.Tensor = self.classifier(context_vector)  # (batch, 1)

        return logits

    # predict_proba() is inherited from BaseModel
