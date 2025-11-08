"""
PyTorch Dataset classes for leech training data.

Provides efficient loading and batching of training chunks for PyTorch models.
Handles both baseline models (signal + sequence) and full models (signal + sequence + features).

Classes:
    LeechDataset: Main dataset class that loads chunks from .npz files

Functions:
    collate_fn(): Custom collate function for batching variable-length data

Data Format:
    Each training chunk is a dictionary with:
    - signal: Raw signal array (signal_len,)
    - sequence: DNA sequence string (kmer_len bases)
    - features: Engineered features (num_features, kmer_len) - optional
    - label: Binary label (0=uncharged, 1=charged)
    - read_id: Read identifier
    - base_idx: Position within read

Example:
    >>> from leech.dataset import LeechDataset, collate_fn
    >>> from torch.utils.data import DataLoader
    >>>
    >>> # Create dataset
    >>> dataset = LeechDataset("chunks.npz", model_type="ConvLSTMDwell")
    >>> print(f"Dataset size: {len(dataset)}")
    >>>
    >>> # Create DataLoader
    >>> loader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn)
    >>> batch = next(iter(loader))
    >>> print(batch.keys())  # ['signal', 'sequence', 'features', 'label']
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from leech.data_prep import encode_kmer, load_chunks


class LeechDataset(Dataset):
    """
    PyTorch Dataset for leech training chunks.

    Handles loading and preprocessing of training data.
    """

    def __init__(
        self,
        chunk_path: Path,
        signal_len: int = 400,
        kmer_len: int = 11,
        model_type: str = "ConvLSTMDwell",
    ):
        """
        Initialize dataset.

        Args:
            chunk_path: Path to .npz file with training chunks
            signal_len: Expected signal length (will pad/truncate)
            kmer_len: Expected k-mer length
            model_type: "ConvLSTMDwell" or "ConvLSTMBase"
        """
        self.chunk_path = chunk_path
        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.model_type = model_type

        # Load chunks
        self.chunks = load_chunks(chunk_path)

        # Filter chunks with valid labels
        self.chunks = [c for c in self.chunks if c["label"] is not None]

        if len(self.chunks) == 0:
            raise ValueError(f"No valid chunks found in {chunk_path}")

    def __len__(self) -> int:
        """Return number of chunks."""
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single training example.

        Returns:
            Dictionary with:
            - signal: (signal_len,) tensor
            - sequence: (4, kmer_len) one-hot encoded tensor
            - features: (num_features, kmer_len) tensor (if ConvLSTMDwell)
            - label: (1,) tensor
        """
        chunk = self.chunks[idx]

        # Process signal - pad or truncate to signal_len
        signal = chunk["signal"].astype(np.float32)
        if len(signal) < self.signal_len:
            # Pad with zeros
            signal = np.pad(signal, (0, self.signal_len - len(signal)), mode="constant")
        elif len(signal) > self.signal_len:
            # Truncate from center
            start = (len(signal) - self.signal_len) // 2
            signal = signal[start : start + self.signal_len]

        signal_tensor = torch.from_numpy(signal)

        # Process sequence - one-hot encode
        sequence = chunk["sequence"]
        if len(sequence) != self.kmer_len:
            raise ValueError(f"Expected k-mer length {self.kmer_len}, got {len(sequence)}")
        sequence_tensor = encode_kmer(sequence)

        # Process features (for ConvLSTMDwell)
        features = chunk["features"]
        if features.size > 0:
            features_tensor = torch.from_numpy(features.astype(np.float32))
        else:
            # If no features, create dummy features
            features_tensor = torch.zeros(1, self.kmer_len, dtype=torch.float32)

        # Label
        label = torch.tensor([chunk["label"]], dtype=torch.float32)

        result = {
            "signal": signal_tensor,
            "sequence": sequence_tensor,
            "label": label,
        }

        if self.model_type == "ConvLSTMDwell":
            result["features"] = features_tensor

        return result


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched tensors
    """
    # Stack all tensors
    signals = torch.stack([item["signal"] for item in batch])
    sequences = torch.stack([item["sequence"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    result = {
        "signal": signals,
        "sequence": sequences,
        "label": labels,
    }

    # Add features if present
    if "features" in batch[0]:
        features = torch.stack([item["features"] for item in batch])
        result["features"] = features

    return result
