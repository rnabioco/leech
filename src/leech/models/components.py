"""
Reusable neural network components for leech models.

This module provides shared building blocks used across all model architectures,
eliminating code duplication and making it easier to create new models.
"""

import torch
import torch.nn as nn

from leech.constants import (
    DEFAULT_CONV_CHANNELS,
    DEFAULT_FEATURE_KERNEL,
    DEFAULT_SEQ_KERNEL,
    DEFAULT_SIGNAL_KERNEL,
)


class SignalBranch(nn.Module):
    """
    Reusable 1D convolutional branch for raw signal processing.

    Applies a series of 1D convolutions to extract features from nanopore signal data.

    Args:
        conv_channels: List of channel sizes for conv layers (default: [4, 16, 256])
        kernel_size: Kernel size for convolutions (default: 5)

    Input shape:
        (batch_size, signal_len)

    Output shape:
        (batch_size, conv_channels[-1], signal_len)
    """

    def __init__(
        self,
        conv_channels: list[int] | None = None,
        kernel_size: int = DEFAULT_SIGNAL_KERNEL,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = DEFAULT_CONV_CHANNELS

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, conv_channels[0], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through signal branch.

        Args:
            signal: Raw signal tensor (batch_size, signal_len)

        Returns:
            Extracted features (batch_size, conv_channels[-1], signal_len)
        """
        # Add channel dimension: (batch, signal_len) -> (batch, 1, signal_len)
        signal = signal.unsqueeze(1)
        return self.conv_layers(signal)


class SequenceBranch(nn.Module):
    """
    Reusable 1D convolutional branch for one-hot encoded sequence processing.

    Applies a series of 1D convolutions to extract features from sequence k-mers.

    Args:
        conv_channels: List of channel sizes for conv layers (default: [4, 16, 256])
        kernel_size: Kernel size for convolutions (default: 3)

    Input shape:
        (batch_size, 4, kmer_len) - 4 channels for A, C, G, T

    Output shape:
        (batch_size, conv_channels[-1], kmer_len)
    """

    def __init__(
        self,
        conv_channels: list[int] | None = None,
        kernel_size: int = DEFAULT_SEQ_KERNEL,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = DEFAULT_CONV_CHANNELS

        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, conv_channels[0], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through sequence branch.

        Args:
            sequence: One-hot encoded sequence (batch_size, 4, kmer_len)

        Returns:
            Extracted features (batch_size, conv_channels[-1], kmer_len)
        """
        return self.conv_layers(sequence)


class FeatureBranch(nn.Module):
    """
    Reusable 1D convolutional branch for engineered features (dwell + signal levels).

    Applies a series of 1D convolutions to extract patterns from feature channels.

    Args:
        num_features: Number of input feature channels (e.g., 5 for dwell + 4 signal stats)
        conv_channels: List of channel sizes for conv layers (default: [4, 16, 256])
        kernel_size: Kernel size for convolutions (default: 3)

    Input shape:
        (batch_size, num_features, kmer_len)

    Output shape:
        (batch_size, conv_channels[-1], kmer_len)
    """

    def __init__(
        self,
        num_features: int,
        conv_channels: list[int] | None = None,
        kernel_size: int = DEFAULT_FEATURE_KERNEL,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = DEFAULT_CONV_CHANNELS

        self.conv_layers = nn.Sequential(
            nn.Conv1d(num_features, conv_channels[0], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature branch.

        Args:
            features: Engineered features (batch_size, num_features, kmer_len)

        Returns:
            Extracted features (batch_size, conv_channels[-1], kmer_len)
        """
        return self.conv_layers(features)


class BaseModel(nn.Module):
    """
    Base class for all leech models with shared predict_proba() method.

    All leech models should inherit from this class to get the standard
    predict_proba() implementation and ensure consistent interfaces.
    """

    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        """
        Predict probability of positive class (charged tRNA).

        This method wraps the forward() pass with evaluation mode and
        sigmoid activation to produce probabilities in [0, 1].

        Args:
            *args: Arguments passed to forward()
            **kwargs: Keyword arguments passed to forward()

        Returns:
            Probabilities for positive class (batch_size, 1)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(*args, **kwargs)
            probs = torch.sigmoid(logits)
        return probs
