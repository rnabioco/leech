"""
Sequence encoding utilities for neural network input.

This module provides functions for converting DNA/RNA sequences into
numeric representations suitable for model training and inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

#: A=0, C=1, G=2, T=3, with U folded onto T.
#:
#: U matters: RNA references and some basecaller outputs carry it, and every
#: other encoder in the tree maps it to 3 -- ``features.sequence_to_int`` and
#: both Rust encoders (``sequence_to_int`` / ``encode_base_onehot`` in
#: ``rust/src/inference_pipeline/features.rs``). This table did not, so a U
#: encoded as an all-zero column here and as a T everywhere else -- the same
#: base, two different model inputs, depending on which encoder ran.
_BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def encode_kmer(sequence: str) -> torch.Tensor:
    """
    One-hot encode a DNA sequence for model input.

    This is the canonical sequence encoding function used throughout leech.
    Returns a PyTorch tensor suitable for direct model input.

    Args:
        sequence: DNA sequence string (A, C, G, T, N)

    Returns:
        One-hot encoded tensor of shape (4, len(sequence))
        Bases are encoded as: A=0, C=1, G=2, T=3 (U folds onto T)
        Unknown bases (e.g., N) are encoded as all zeros

    Examples:
        >>> seq = "ACGT"
        >>> encoded = encode_kmer(seq)
        >>> encoded.shape
        torch.Size([4, 4])
        >>> encoded[:, 0]  # First base 'A'
        tensor([1., 0., 0., 0.])
    """
    import torch

    seq_len = len(sequence)
    encoded = torch.zeros(4, seq_len, dtype=torch.float32)

    for i, base in enumerate(sequence.upper()):
        idx = _BASE_TO_IDX.get(base)
        if idx is not None:
            encoded[idx, i] = 1.0

    return encoded
