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
        confound_map: dict[int, int] | None = None,
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
                sensing region. Requires feature_left >= kmer_context + offset.
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
        self.confound_map = confound_map

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

        # Pre-tensorize: encode sequences, labels, signals, and features once
        self._encoded_seqs: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._signals: list[torch.Tensor] = []
        self._features: list[torch.Tensor] = []
        self._needs_features = model_type in FEATURE_MODELS

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

        # Detect signal_residual channel
        self._has_signal_residual = self.chunks[0].get("signal_residual") is not None
        self.signal_channels = 2 if self._has_signal_residual else 1

        # Detect multi-class: if any label_int > 1, use long dtype for CrossEntropyLoss
        max_label = max(c["label_int"] for c in self.chunks)
        self._multiclass = max_label > 1

        for chunk in self.chunks:
            if self._effective_seq_encoding == "signal_kmer":
                self._encoded_seqs.append(
                    self._encode_signal_kmer(
                        chunk,
                        signal_len,
                        signal_kmer_context,
                        left_context=self.left_context,
                        right_context=self.right_context,
                    )
                )
            else:
                # Pre-encode sequence (vectorized, no Python loop)
                self._encoded_seqs.append(self._encode_sequence(chunk["sequence"]))

            # Pre-create label tensor: long for multi-class, float for binary
            if self._multiclass:
                self._labels.append(torch.tensor(chunk["label_int"], dtype=torch.long))
            else:
                self._labels.append(torch.tensor([chunk["label_int"]], dtype=torch.float32))

            # Pre-tensorize signal: pad/crop once instead of every __getitem__ call
            signal = chunk["signal"]
            if signal.dtype != np.float32:
                signal = signal.astype(np.float32)
            signal_residual = chunk.get("signal_residual")
            if signal_residual is not None and signal_residual.dtype != np.float32:
                signal_residual = signal_residual.astype(np.float32)
            self._signals.append(self._prepare_signal(signal, signal_residual))

            # Pre-tensorize features: apply dwell_offset slicing once
            if self._needs_features:
                self._features.append(self._prepare_features(chunk))
            else:
                self._features.append(torch.empty(0))

        # Stack into contiguous tensors for cache-friendly access
        try:
            self._signals_tensor = torch.stack(self._signals)
            self._signals = []  # free the list
        except RuntimeError as e:
            logger.warning("Signal shapes differ, falling back to list access: %s", e)
            self._signals_tensor = None

        # Pre-compute confound labels (for adversarial training)
        self._confound_labels: list[torch.Tensor] | None = None
        if confound_map is not None:
            self._confound_labels = []
            for chunk in self.chunks:
                label_int = chunk["label_int"]
                cl = confound_map.get(label_int, -1)  # -1 → ignored by CE
                self._confound_labels.append(torch.tensor(cl, dtype=torch.long))

        logger.debug(
            f"Pre-tensorized {len(self.chunks)} chunks "
            f"({len(self._encoded_seqs)} sequences encoded, encoding={self._effective_seq_encoding})"
        )

    def _prepare_signal(
        self, signal: np.ndarray, signal_residual: np.ndarray | None = None
    ) -> torch.Tensor:
        """Pad/crop signal (and optional residual) to target length. Called once during __init__."""
        if self.left_context is not None and self.right_context is not None:
            focus_pos = len(signal) // 2
            start = focus_pos - self.left_context
            end = focus_pos + self.right_context
            if start < 0 or end > len(signal):
                cropped = np.zeros(self.signal_len, dtype=np.float32)
                src_start, src_end = max(0, start), min(len(signal), end)
                dst_start = max(0, -start)
                cropped[dst_start : dst_start + (src_end - src_start)] = signal[src_start:src_end]
                signal = cropped
                if signal_residual is not None:
                    cropped_r = np.zeros(self.signal_len, dtype=np.float32)
                    cropped_r[dst_start : dst_start + (src_end - src_start)] = signal_residual[
                        src_start:src_end
                    ]
                    signal_residual = cropped_r
            else:
                signal = signal[start:end]
                if signal_residual is not None:
                    signal_residual = signal_residual[start:end]
        elif len(signal) < self.signal_len:
            signal = np.pad(signal, (0, self.signal_len - len(signal)), mode="constant")
            if signal_residual is not None:
                signal_residual = np.pad(
                    signal_residual, (0, self.signal_len - len(signal_residual)), mode="constant"
                )
        elif len(signal) > self.signal_len:
            start = (len(signal) - self.signal_len) // 2
            signal = signal[start : start + self.signal_len]
            if signal_residual is not None:
                signal_residual = signal_residual[start : start + self.signal_len]

        if signal_residual is not None:
            # Stack into (2, signal_len) for 2-channel signal input
            stacked = np.stack([signal, signal_residual], axis=0)
            return torch.from_numpy(np.ascontiguousarray(stacked))
        return torch.from_numpy(np.ascontiguousarray(signal))

    def _prepare_features(self, chunk: dict) -> torch.Tensor:
        """Apply dwell_offset slicing and tensorize features. Called once during __init__."""
        dwell = chunk["dwell"]
        features = chunk["features"]
        if features.dtype != np.float32:
            features = features.astype(np.float32)

        kmer_context = self.kmer_len // 2

        if self.model_type in WIDE_FEATURE_MODELS:
            pass  # full-width features
        elif len(dwell) > self.kmer_len:
            # Determine feature_start (signed offset from focus).
            # New chunks have it directly; old chunks need conversion.
            if "feature_start" in chunk:
                feat_start = int(chunk["feature_start"])
            elif "feature_left" in chunk:
                feat_start = -int(chunk["feature_left"])
            elif "dwell_margin_left" in chunk:
                feat_start = -(kmer_context + int(chunk["dwell_margin_left"]))
            else:
                feat_start = -(len(dwell) - 1) // 2  # symmetric fallback
            # kmer-aligned start within the feature array
            # Feature array starts at focus + feat_start, kmer starts at focus - kmer_context
            kmer_start = (-kmer_context) - feat_start
            start = kmer_start + self.dwell_offset
            if start < 0 or start + self.kmer_len > features.shape[1]:
                raise ValueError(
                    f"dwell_offset ({self.dwell_offset}) with feature_start={feat_start} "
                    f"exceeds feature width ({features.shape[1]})"
                )
            if features.size > 0:
                features = features[:, start : start + self.kmer_len]

        if features.size > 0:
            return torch.from_numpy(np.ascontiguousarray(features))
        else:
            return torch.zeros(1, self.kmer_len, dtype=torch.float32)

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
        left_context: int | None = None,
        right_context: int | None = None,
    ) -> torch.Tensor:
        """Encode chunk using signal-level kmer encoding.

        Args:
            chunk: Chunk dict with seq_to_sig_map and sequence_with_kmer_context
            signal_len: Target signal length (pad/truncate)
            kmer_context: (kmer_before, kmer_after) for encoding
            left_context: Left signal context for asymmetric crop adjustment
            right_context: Right signal context for asymmetric crop adjustment

        Returns:
            Encoded tensor of shape (4 * kmer_len, signal_len)
        """
        seq_ctx = chunk["sequence_with_kmer_context"]
        seq_to_sig = chunk["seq_to_sig_map"].copy()

        # Adjust seq_to_sig_map when signal will be asymmetrically cropped.
        # Stored chunks have coordinates in [0, stored_signal_len] but after
        # _prepare_signal crops to [focus - left_context, focus + right_context],
        # the signal_kmer encoder needs coordinates relative to the cropped window.
        if left_context is not None and right_context is not None:
            focus_pos = seq_to_sig[-1] // 2  # focus is at center of stored chunk
            crop_start = focus_pos - left_context
            seq_to_sig = seq_to_sig - crop_start
            seq_to_sig = np.clip(seq_to_sig, 0, signal_len)

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
        # Pre-computed signal lookup (padded/cropped once in __init__)
        if self._signals_tensor is not None:
            signal_tensor = self._signals_tensor[idx]
        else:
            signal_tensor = self._signals[idx]

        # Apply augmentation if configured (creates new tensor, doesn't modify stored data)
        if self.augmentation is not None:
            signal_tensor = self._apply_augmentation(signal_tensor.clone())

        # Pre-encoded sequence lookup
        sequence_tensor = self._encoded_seqs[idx]
        if self._effective_seq_encoding == "base_onehot":
            sequence = self.chunks[idx]["sequence"]
            if len(sequence) != self.kmer_len:
                raise ValueError(f"Expected k-mer length {self.kmer_len}, got {len(sequence)}")

        # Pre-computed label lookup
        label = self._labels[idx]

        result: dict[str, torch.Tensor] = {
            "signal": signal_tensor,
            "sequence": sequence_tensor,
            "label": label,
        }

        # Include pre-computed features for models that require them
        if self._needs_features:
            result["features"] = self._features[idx]

        # Include confound label for adversarial training
        if self._confound_labels is not None:
            result["confound_label"] = self._confound_labels[idx]

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

    # Add confound labels if present (adversarial training)
    if "confound_label" in batch[0]:
        result["confound_label"] = torch.stack([item["confound_label"] for item in batch])

    return result
