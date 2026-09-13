"""Sequence-encoding helpers moved out of ``leech.preparation.encoding`` in #277.

``seq_to_int``, ``int_to_seq`` and ``one_hot_encode_sequence`` had no
production callers -- ``encode_kmer`` (the canonical encoder, kept in
``leech.preparation.encoding``) uses its own ``_BASE_TO_IDX`` table, and
``leech.features.sequence_to_int`` is the encoder the rest of the pipeline
actually calls. Only ``tests/test_data_prep.py`` and
``tests/test_base_encoding_parity.py`` exercised these three, so they live
here now rather than in ``src/``.
"""

from __future__ import annotations

import numpy as np

# A=0, C=1, G=2, T=3, N=4, U=3 (treat as T)
_SEQ_LOOKUP = np.full(256, 4, dtype=np.int64)
_SEQ_LOOKUP[ord("A")] = 0
_SEQ_LOOKUP[ord("C")] = 1
_SEQ_LOOKUP[ord("G")] = 2
_SEQ_LOOKUP[ord("T")] = 3
_SEQ_LOOKUP[ord("U")] = 3
_SEQ_LOOKUP[ord("N")] = 4
_SEQ_LOOKUP[ord("a")] = 0
_SEQ_LOOKUP[ord("c")] = 1
_SEQ_LOOKUP[ord("g")] = 2
_SEQ_LOOKUP[ord("t")] = 3
_SEQ_LOOKUP[ord("u")] = 3
_SEQ_LOOKUP[ord("n")] = 4


def seq_to_int(seq: str) -> np.ndarray:
    """Convert DNA sequence to integer encoding.

    A=0, C=1, G=2, T=3, N=4, U=3 (treat as T)

    Examples:
        >>> seq_to_int("ACGT")
        array([0, 1, 2, 3])
        >>> seq_to_int("ACGTN")
        array([0, 1, 2, 3, 4])
    """
    return _SEQ_LOOKUP[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def int_to_seq(int_seq: np.ndarray) -> str:
    """Convert integer encoding back to sequence.

    Examples:
        >>> int_to_seq(np.array([0, 1, 2, 3]))
        'ACGT'
        >>> int_to_seq(np.array([0, 1, 2, 3, 4]))
        'ACGTN'
    """
    bases = ["A", "C", "G", "T", "N"]
    return "".join(bases[i] if i < 5 else "N" for i in int_seq)


def one_hot_encode_sequence(seq: str, kmer_len: int = 1) -> np.ndarray:
    """One-hot encode a sequence with k-mer context (advanced version).

    NOTE: For standard sequence encoding, use
    ``leech.preparation.encoding.encode_kmer`` instead. This function is for
    specialized k-mer context encoding where each position includes
    information from neighboring bases.

    Returns:
        Array of shape (kmer_len * 4, seq_len) for model input.
        Each position encodes kmer_len neighboring bases.

    Examples:
        >>> encoded = one_hot_encode_sequence("ACGT", kmer_len=1)
        >>> encoded.shape
        (4, 4)
    """
    int_seq = seq_to_int(seq)
    seq_len = len(int_seq)

    encoding = np.zeros((kmer_len * 4, seq_len), dtype=np.float32)

    for pos in range(seq_len):
        for k in range(kmer_len):
            offset = pos - kmer_len // 2 + k
            if 0 <= offset < seq_len:
                base = int_seq[offset]
                if base < 4:  # Valid base (not N)
                    encoding[k * 4 + base, pos] = 1.0

    return encoding
