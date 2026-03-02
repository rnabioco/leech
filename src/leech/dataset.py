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
    >>> dataset = LeechDataset("chunks.npz", model_type="TransformerDwell")
    >>> print(f"Dataset size: {len(dataset)}")
    >>>
    >>> # Create DataLoader
    >>> loader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn)
    >>> batch = next(iter(loader))
    >>> print(batch.keys())  # ['signal', 'sequence', 'features', 'label'] for feature models
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from leech.chunking import load_chunks

logger = logging.getLogger("leech.dataset")

# ASCII lookup table for vectorized sequence encoding (A=0, C=1, G=2, T=3, else=255)
_BASE_MAP = np.full(256, 255, dtype=np.uint8)
_BASE_MAP[ord("A")] = 0
_BASE_MAP[ord("C")] = 1
_BASE_MAP[ord("G")] = 2
_BASE_MAP[ord("T")] = 3
_BASE_MAP[ord("a")] = 0
_BASE_MAP[ord("c")] = 1
_BASE_MAP[ord("g")] = 2
_BASE_MAP[ord("t")] = 3

# Models that require dwell/signal features as third input
# Must match ModelInferenceWrapper.FEATURE_MODELS
FEATURE_MODELS = {
    "ConvLSTMDwell",
    "TransformerDwell",
    "ConvOnly",
    "TCNDwell",
    "ResNetDwell",
}


class LeechDataset(Dataset):
    """
    PyTorch Dataset for leech training chunks.

    Handles loading and preprocessing of training data.
    """

    def __init__(
        self,
        chunk_path: Path | None = None,
        signal_len: int = 400,
        kmer_len: int = 11,
        model_type: str = "ConvLSTMDwell",
        dwell_offset: int = 0,
        chunks: list[dict] | None = None,
    ):
        """
        Initialize dataset.

        Args:
            chunk_path: Path to .npz file with training chunks
            signal_len: Expected signal length (will pad/truncate)
            kmer_len: Expected k-mer length
            model_type: Model architecture name (e.g., "ConvLSTMDwell", "TransformerDwell")
            dwell_offset: Shift dwell/feature window toward 3' end (bases).
                Compensates for physical offset between motor protein and
                sensing region. Requires chunks extracted with dwell_margin >= offset.
            chunks: Pre-loaded list of chunk dicts. When provided, chunk_path is
                ignored and no disk I/O occurs (useful for grid search caching).
        """
        self.chunk_path = chunk_path
        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.model_type = model_type
        self.dwell_offset = dwell_offset

        # Use pre-loaded chunks or load from file
        if chunks is not None:
            logger.info(f"Using {len(chunks)} pre-loaded chunks (skipping disk I/O)")
            self.chunks = chunks
        elif chunk_path is not None:
            self.chunks = load_chunks(chunk_path)
        else:
            raise ValueError("Either chunk_path or chunks must be provided")

        # Filter chunks with valid numeric labels (label_int)
        self.chunks = [c for c in self.chunks if c["label_int"] is not None]

        if len(self.chunks) == 0:
            raise ValueError(
                f"No valid chunks found"
                f"{f' in {chunk_path}' if chunk_path else ''}"
            )

        # Pre-tensorize: encode sequences, labels, and convert dtypes once
        self._encoded_seqs: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

        for chunk in self.chunks:
            # Pre-encode sequence (vectorized, no Python loop)
            self._encoded_seqs.append(self._encode_sequence(chunk["sequence"]))

            # Pre-create label tensor
            self._labels.append(
                torch.tensor([chunk["label_int"]], dtype=torch.float32)
            )

            # Convert signal and features to float32 in-place (avoid per-epoch astype)
            if chunk["signal"].dtype != np.float32:
                chunk["signal"] = chunk["signal"].astype(np.float32)
            if chunk["features"].dtype != np.float32:
                chunk["features"] = chunk["features"].astype(np.float32)

        logger.debug(
            f"Pre-tensorized {len(self.chunks)} chunks "
            f"({len(self._encoded_seqs)} sequences encoded)"
        )

    @staticmethod
    def _encode_sequence(sequence: str) -> torch.Tensor:
        """Vectorized one-hot encoding of a DNA sequence.

        Uses a pre-built ASCII lookup table instead of a Python for-loop.

        Args:
            sequence: DNA sequence string (A, C, G, T, N)

        Returns:
            One-hot encoded tensor of shape (4, len(sequence))
        """
        indices = _BASE_MAP[np.frombuffer(sequence.encode(), dtype=np.uint8)]
        encoded = np.zeros((4, len(sequence)), dtype=np.float32)
        valid = indices < 4
        encoded[indices[valid], np.where(valid)[0]] = 1.0
        return torch.from_numpy(encoded)

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
            - features: (num_features, kmer_len) tensor (if model requires features)
            - label: (1,) tensor
        """
        chunk = self.chunks[idx]

        # Process signal - pad or truncate to signal_len (already float32)
        signal = chunk["signal"]
        if len(signal) < self.signal_len:
            signal = np.pad(signal, (0, self.signal_len - len(signal)), mode="constant")
        elif len(signal) > self.signal_len:
            start = (len(signal) - self.signal_len) // 2
            signal = signal[start : start + self.signal_len]

        signal_tensor = torch.from_numpy(signal)

        # Pre-encoded sequence lookup
        sequence = chunk["sequence"]
        if len(sequence) != self.kmer_len:
            raise ValueError(f"Expected k-mer length {self.kmer_len}, got {len(sequence)}")
        sequence_tensor = self._encoded_seqs[idx]

        # Process dwell/features — apply dwell_offset slicing if margin exists
        dwell = chunk["dwell"]
        features = chunk["features"]
        if len(dwell) > self.kmer_len:
            margin = (len(dwell) - self.kmer_len) // 2
            if self.dwell_offset > margin:
                raise ValueError(
                    f"dwell_offset ({self.dwell_offset}) exceeds available "
                    f"dwell_margin ({margin})"
                )
            start = margin + self.dwell_offset
            dwell = dwell[start : start + self.kmer_len]
            if features.size > 0:
                features = features[:, start : start + self.kmer_len]

        if features.size > 0:
            features_tensor = torch.from_numpy(features)
        else:
            features_tensor = torch.zeros(1, self.kmer_len, dtype=torch.float32)

        # Pre-computed label lookup
        label = self._labels[idx]

        result = {
            "signal": signal_tensor,
            "sequence": sequence_tensor,
            "label": label,
        }

        # Include features for models that require them
        if self.model_type in FEATURE_MODELS:
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
