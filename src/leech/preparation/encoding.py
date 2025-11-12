"""
Sequence encoding utilities for neural network input.

This module provides functions for converting DNA/RNA sequences into
numeric representations suitable for model training and inference.
"""

from typing import Any

import numpy as np


def seq_to_int(seq: str) -> np.ndarray:
    """
    Convert DNA sequence to integer encoding.

    A=0, C=1, G=2, T=3, N=4, U=3 (treat as T)

    Args:
        seq: DNA/RNA sequence string

    Returns:
        Integer array

    Examples:
        >>> seq_to_int("ACGT")
        array([0, 1, 2, 3])
        >>> seq_to_int("ACGTN")
        array([0, 1, 2, 3, 4])
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3, "N": 4}
    return np.array([mapping.get(b.upper(), 4) for b in seq], dtype=np.int64)


def int_to_seq(int_seq: np.ndarray) -> str:
    """
    Convert integer encoding back to sequence.

    Args:
        int_seq: Integer array with values 0-4

    Returns:
        DNA sequence string

    Examples:
        >>> int_to_seq(np.array([0, 1, 2, 3]))
        'ACGT'
        >>> int_to_seq(np.array([0, 1, 2, 3, 4]))
        'ACGTN'
    """
    bases = ["A", "C", "G", "T", "N"]
    return "".join(bases[i] if i < 5 else "N" for i in int_seq)


def encode_kmer(sequence: str) -> Any:  # Returns torch.Tensor
    """
    One-hot encode a DNA sequence for model input.

    This is the canonical sequence encoding function used throughout leech.
    Returns a PyTorch tensor suitable for direct model input.

    Args:
        sequence: DNA sequence string (A, C, G, T, N)

    Returns:
        One-hot encoded tensor of shape (4, len(sequence))
        Bases are encoded as: A=0, C=1, G=2, T=3
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

    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq_len = len(sequence)
    encoded = torch.zeros(4, seq_len, dtype=torch.float32)

    for i, base in enumerate(sequence.upper()):
        if base in base_to_idx:
            encoded[base_to_idx[base], i] = 1.0
        # If base not in dict (e.g., N), leave as zeros

    return encoded


def one_hot_encode_sequence(seq: str, kmer_len: int = 1) -> np.ndarray:
    """
    One-hot encode a sequence with k-mer context (advanced version).

    NOTE: For standard sequence encoding, use encode_kmer() instead.
    This function is for specialized k-mer context encoding where each
    position includes information from neighboring bases.

    Args:
        seq: DNA sequence
        kmer_len: K-mer length for encoding (context window)

    Returns:
        Array of shape (kmer_len * 4, seq_len) for model input
        Each position encodes kmer_len neighboring bases

    Examples:
        >>> seq = "ACGT"
        >>> encoded = one_hot_encode_sequence(seq, kmer_len=1)
        >>> encoded.shape
        (4, 4)

    See Also:
        encode_kmer: Standard one-hot encoding function (recommended)
    """
    int_seq = seq_to_int(seq)
    seq_len = len(int_seq)

    # Create one-hot encoding for each k-mer position
    encoding = np.zeros((kmer_len * 4, seq_len), dtype=np.float32)

    for pos in range(seq_len):
        for k in range(kmer_len):
            offset = pos - kmer_len // 2 + k
            if 0 <= offset < seq_len:
                base = int_seq[offset]
                if base < 4:  # Valid base (not N)
                    encoding[k * 4 + base, pos] = 1.0

    return encoding
