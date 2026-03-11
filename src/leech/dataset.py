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
from leech.features import encode_signal_kmer, sequence_to_int
from leech.models.inference_wrapper import ModelInferenceWrapper

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
FEATURE_MODELS = ModelInferenceWrapper.FEATURE_MODELS

# Models that receive the full dwell margin (no dwell_offset slicing)
WIDE_FEATURE_MODELS = ModelInferenceWrapper.WIDE_FEATURE_MODELS


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
        augmentation: dict | None = None,
        seq_encoding: str = "signal_kmer",
        signal_kmer_context: tuple[int, int] = (4, 4),
        left_context: int | None = None,
        right_context: int | None = None,
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
            augmentation: Signal augmentation config dict. Keys:
                - jitter_std (float): Gaussian noise std dev (0 = disabled)
                - scale_range (tuple[float, float]): Random scale range (1.0, 1.0 = disabled)
            left_context: Left signal context (samples before focus base).
                When both left_context and right_context are provided, crop
                asymmetrically around the focus base instead of center-cropping.
            right_context: Right signal context (samples after focus base).
        """
        self.chunk_path = chunk_path
        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.model_type = model_type
        self.dwell_offset = dwell_offset
        self.augmentation = augmentation
        self.seq_encoding = seq_encoding
        self.signal_kmer_context = signal_kmer_context
        self.left_context = left_context
        self.right_context = right_context

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
            raise ValueError(f"No valid chunks found{f' in {chunk_path}' if chunk_path else ''}")

        # Pre-tensorize: encode sequences, labels, and convert dtypes once
        self._encoded_seqs: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

        # Determine effective encoding: fall back to base_onehot if chunks lack signal_kmer data
        self._effective_seq_encoding = seq_encoding
        if seq_encoding == "signal_kmer":
            first = self.chunks[0]
            if not first.get("seq_to_sig_map") is not None or not first.get(
                "sequence_with_kmer_context"
            ):
                logger.warning(
                    "Chunks lack seq_to_sig_map/sequence_with_kmer_context; "
                    "falling back to base_onehot encoding"
                )
                self._effective_seq_encoding = "base_onehot"

        # Detect multi-class: if any label_int > 1, use long dtype for CrossEntropyLoss
        max_label = max(c["label_int"] for c in self.chunks)
        self._multiclass = max_label > 1

        for chunk in self.chunks:
            if self._effective_seq_encoding == "signal_kmer":
                self._encoded_seqs.append(
                    self._encode_signal_kmer(chunk, signal_len, signal_kmer_context)
                )
            else:
                # Pre-encode sequence (vectorized, no Python loop)
                self._encoded_seqs.append(self._encode_sequence(chunk["sequence"]))

            # Pre-create label tensor: long for multi-class, float for binary
            if self._multiclass:
                self._labels.append(torch.tensor(chunk["label_int"], dtype=torch.long))
            else:
                self._labels.append(torch.tensor([chunk["label_int"]], dtype=torch.float32))

            # Convert signal and features to float32 in-place (avoid per-epoch astype)
            if chunk["signal"].dtype != np.float32:
                chunk["signal"] = chunk["signal"].astype(np.float32)
            if chunk["features"].dtype != np.float32:
                chunk["features"] = chunk["features"].astype(np.float32)

        logger.debug(
            f"Pre-tensorized {len(self.chunks)} chunks "
            f"({len(self._encoded_seqs)} sequences encoded, encoding={self._effective_seq_encoding})"
        )

    def _apply_augmentation(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply signal augmentation (jitter and/or scaling)."""
        jitter_std = self.augmentation.get("jitter_std", 0.0)
        if jitter_std > 0:
            signal = signal + torch.randn_like(signal) * jitter_std
        scale_range = self.augmentation.get("scale_range", (1.0, 1.0))
        if scale_range != (1.0, 1.0):
            scale = torch.empty(1).uniform_(scale_range[0], scale_range[1]).item()
            signal = signal * scale
        return signal

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

    @staticmethod
    def _encode_signal_kmer(
        chunk: dict,
        signal_len: int,
        kmer_context: tuple[int, int],
    ) -> torch.Tensor:
        """Encode chunk using signal-level kmer encoding.

        Args:
            chunk: Chunk dict with seq_to_sig_map and sequence_with_kmer_context
            signal_len: Target signal length (pad/truncate)
            kmer_context: (kmer_before, kmer_after) for encoding

        Returns:
            Encoded tensor of shape (4 * kmer_len, signal_len)
        """
        seq_ctx = chunk["sequence_with_kmer_context"]
        seq_to_sig = chunk["seq_to_sig_map"]
        seq_ints = sequence_to_int(seq_ctx)
        enc = encode_signal_kmer(seq_ints, seq_to_sig, signal_len, kmer_context)
        return torch.from_numpy(enc)

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
        if self.left_context is not None and self.right_context is not None:
            # Asymmetric crop around focus base (at center of symmetrically-prepared chunk)
            focus_pos = len(signal) // 2
            start = focus_pos - self.left_context
            end = focus_pos + self.right_context
            if start < 0 or end > len(signal):
                cropped = np.zeros(self.signal_len, dtype=np.float32)
                src_start, src_end = max(0, start), min(len(signal), end)
                dst_start = max(0, -start)
                cropped[dst_start : dst_start + (src_end - src_start)] = signal[src_start:src_end]
                signal = cropped
            else:
                signal = signal[start:end]
        elif len(signal) < self.signal_len:
            signal = np.pad(signal, (0, self.signal_len - len(signal)), mode="constant")
        elif len(signal) > self.signal_len:
            start = (len(signal) - self.signal_len) // 2
            signal = signal[start : start + self.signal_len]

        signal_tensor = torch.from_numpy(signal)

        # Apply augmentation if configured
        if self.augmentation is not None:
            signal_tensor = self._apply_augmentation(signal_tensor)

        # Pre-encoded sequence lookup
        sequence_tensor = self._encoded_seqs[idx]
        if self._effective_seq_encoding == "base_onehot":
            sequence = chunk["sequence"]
            if len(sequence) != self.kmer_len:
                raise ValueError(f"Expected k-mer length {self.kmer_len}, got {len(sequence)}")

        # Process dwell/features — apply dwell_offset slicing if margin exists
        # Wide feature models (e.g., ConvLSTMDwellAttn) receive the full margin
        # so cross-attention can learn the offset
        dwell = chunk["dwell"]
        features = chunk["features"]
        if self.model_type in WIDE_FEATURE_MODELS:
            # Pass full-width features (kmer_len + margin_left + margin_right)
            pass
        elif len(dwell) > self.kmer_len:
            # Use stored dwell_margin_left for correct slicing with asymmetric margins
            margin_left = chunk.get("dwell_margin_left", (len(dwell) - self.kmer_len) // 2)
            if self.dwell_offset + margin_left > len(dwell) - self.kmer_len:
                raise ValueError(f"dwell_offset ({self.dwell_offset}) exceeds dwell_margin")
            start = margin_left + self.dwell_offset
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
