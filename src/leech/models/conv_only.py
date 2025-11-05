"""
ConvOnly: Pure CNN model without recurrent or attention layers.

Architecture:
- Signal branch: Multi-scale convolutional layers (Inception-style)
- Sequence branch: Multi-scale convolutional layers
- Feature branch: Multi-scale convolutional layers
- Concatenate → global pooling → MLP classifier

Rationale:
- Tests whether temporal dynamics (LSTM/Transformer) are necessary
- Faster inference and training than RNN/Transformer models
- Easier to interpret via filter visualization
- Multi-scale convolutions capture both local and broader patterns
"""

import torch
import torch.nn as nn


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


class ConvOnly(nn.Module):
    """
    Pure CNN model with multi-scale convolutions and no recurrent layers.

    Args:
        signal_len: Length of input signal
        kmer_len: Length of k-mer sequence (e.g., 2*context+1)
        num_features: Number of feature channels (dwell + signal levels)
        base_channels: Base number of channels for inception blocks (default: 16)
        num_blocks: Number of inception blocks per branch (default: 3)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        signal_len: int = 400,
        kmer_len: int = 11,
        num_features: int = 5,  # e.g., dwell + mean + median + std + range
        base_channels: int = 16,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.num_features = num_features

        # Signal branch: Stack of inception blocks
        # Input: (batch, 1, signal_len)
        self.signal_conv = nn.ModuleList()
        in_ch = 1
        for i in range(num_blocks):
            out_ch = base_channels * (2**i)
            self.signal_conv.append(InceptionBlock(in_ch, out_ch))
            in_ch = out_ch * 5  # 4 kernel sizes + 1 pool branch

        # Max pooling to reduce dimensionality
        self.signal_pool = nn.AdaptiveMaxPool1d(kmer_len)

        # Sequence branch: Stack of inception blocks
        # Input: (batch, 4, kmer_len) - 4 nucleotides (A, C, G, T)
        self.seq_conv = nn.ModuleList()
        in_ch = 4
        for i in range(num_blocks):
            out_ch = base_channels * (2**i)
            self.seq_conv.append(InceptionBlock(in_ch, out_ch))
            in_ch = out_ch * 5

        # Feature branch: Stack of inception blocks
        # Input: (batch, num_features, kmer_len)
        self.feature_conv = nn.ModuleList()
        in_ch = num_features
        for i in range(num_blocks):
            out_ch = base_channels * (2**i)
            self.feature_conv.append(InceptionBlock(in_ch, out_ch))
            in_ch = out_ch * 5

        # Calculate final channel size after all inception blocks
        # Each inception block outputs: out_ch * 5 (4 kernel sizes + 1 pool)
        # After num_blocks: base_channels * (2^(num_blocks-1)) * 5
        final_signal_ch = base_channels * (2 ** (num_blocks - 1)) * 5
        final_seq_ch = base_channels * (2 ** (num_blocks - 1)) * 5
        final_feat_ch = base_channels * (2 ** (num_blocks - 1)) * 5

        # Global pooling (mean and max)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Total features: (avg + max) * 3 branches
        total_features = (final_signal_ch + final_seq_ch + final_feat_ch) * 2

        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
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
        signal_feat = signal.unsqueeze(1)
        for conv_block in self.signal_conv:
            signal_feat = conv_block(signal_feat)
        signal_feat = self.signal_pool(signal_feat)  # Match kmer length

        # Sequence branch
        seq_feat = sequence
        for conv_block in self.seq_conv:
            seq_feat = conv_block(seq_feat)

        # Feature branch
        feat_feat = features
        for conv_block in self.feature_conv:
            feat_feat = conv_block(feat_feat)

        # Global pooling for each branch
        # Average pooling
        signal_avg = self.global_avg_pool(signal_feat).squeeze(-1)  # (batch, channels)
        seq_avg = self.global_avg_pool(seq_feat).squeeze(-1)
        feat_avg = self.global_avg_pool(feat_feat).squeeze(-1)

        # Max pooling
        signal_max = self.global_max_pool(signal_feat).squeeze(-1)
        seq_max = self.global_max_pool(seq_feat).squeeze(-1)
        feat_max = self.global_max_pool(feat_feat).squeeze(-1)

        # Concatenate all features
        merged = torch.cat([signal_avg, signal_max, seq_avg, seq_max, feat_avg, feat_max], dim=1)

        # Classifier
        logits = self.classifier(merged)  # (batch, 1)

        return logits

    def predict_proba(
        self, signal: torch.Tensor, sequence: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get probability predictions.

        Args:
            signal: Raw signal (batch, signal_len)
            sequence: One-hot encoded sequence (batch, 4, kmer_len)
            features: Dwell + signal level features (batch, num_features, kmer_len)

        Returns:
            Probabilities (batch, 1)
        """
        logits = self.forward(signal, sequence, features)
        return torch.sigmoid(logits)
