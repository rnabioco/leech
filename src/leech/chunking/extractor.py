"""
Chunk extraction from LeechReads.

Provides functionality for extracting training chunks from processed reads,
with support for motif-based filtering.
"""

import logging

import numpy as np

from leech.constants import DEFAULT_KMER_CONTEXT, DEFAULT_SIGNAL_CONTEXT
from leech.io.motif_search import MotifSearcher

logger = logging.getLogger("leech.chunking.extractor")


class LeechRead:
    """
    Container for a single read's data with all features.

    Attributes:
        read_id: Unique read identifier
        sequence: Basecalled sequence
        signal: Normalized signal array
        seq_to_sig_map: Mapping from base indices to signal indices
        dwells: Per-base dwell times
        dwell_features: Dict of dwell-derived features
        signal_features: Dict of signal-level features
        labels: Optional labels for training (e.g., 0=uncharged, 1=charged)
        metadata: Additional metadata (alignment info, etc.)
    """

    def __init__(
        self,
        read_id: str,
        sequence: str,
        signal: np.ndarray,
        seq_to_sig_map: np.ndarray,
        dwells: np.ndarray,
        dwell_features: dict[str, np.ndarray],
        signal_features: dict[str, np.ndarray],
        labels: np.ndarray | None = None,
        metadata: dict | None = None,
    ):
        """Initialize LeechRead."""
        self.read_id = read_id
        self.sequence = sequence
        self.signal = signal
        self.seq_to_sig_map = seq_to_sig_map
        self.dwells = dwells
        self.dwell_features = dwell_features
        self.signal_features = signal_features
        self.labels = labels
        self.metadata = metadata if metadata is not None else {}

    @property
    def num_bases(self) -> int:
        """Number of bases in the read."""
        return len(self.sequence)

    @property
    def num_samples(self) -> int:
        """Number of signal samples."""
        return len(self.signal)

    def get_chunk(
        self,
        base_idx: int,
        signal_context: tuple[int, int] = DEFAULT_SIGNAL_CONTEXT,
        kmer_context: int = DEFAULT_KMER_CONTEXT,
        base_justify: str = "center",
    ) -> dict[str, np.ndarray | str | int | None] | None:
        """
        Extract a training chunk centered on a specific base.

        Args:
            base_idx: Index of the focus base
            signal_context: (left, right) signal padding around focus base
            kmer_context: Number of bases on each side for k-mer encoding
            base_justify: Where to center the chunk within the focus base's
                signal span. One of:
                - "center" (default): midpoint of signal span (Remora default)
                - "start": first signal sample of the base
                - "end": last signal sample of the base (first sample of next
                  base). Useful for tRNA aa-charging where the amino acid is
                  attached to the 3' hydroxyl of the terminal A.

        Returns:
            Dictionary with 'signal', 'kmer', 'dwell', 'features' arrays,
            or None if chunk cannot be extracted

        Example:
            >>> chunk = read.get_chunk(base_idx=100)
            >>> if chunk:
            ...     print(f"Signal length: {len(chunk['signal'])}")
            ...     print(f"Sequence: {chunk['sequence']}")
        """
        # Check boundaries
        if base_idx < kmer_context or base_idx >= self.num_bases - kmer_context:
            return None

        # Extract signal chunk
        if base_justify == "start":
            focus_sig_pos = self.seq_to_sig_map[base_idx]
        elif base_justify == "end":
            focus_sig_pos = self.seq_to_sig_map[base_idx + 1]
        else:
            # Center on midpoint of focus base's signal span (Remora default)
            focus_sig_pos = (
                self.seq_to_sig_map[base_idx] + self.seq_to_sig_map[base_idx + 1]
            ) // 2
        sig_start = max(0, focus_sig_pos - signal_context[0])
        sig_end = min(self.num_samples, focus_sig_pos + signal_context[1])

        if sig_end - sig_start < signal_context[0] + signal_context[1]:
            return None  # Not enough signal context

        signal_chunk = self.signal[sig_start:sig_end]

        # Extract k-mer sequence context
        kmer_start = base_idx - kmer_context
        kmer_end = base_idx + kmer_context + 1
        kmer_seq = self.sequence[kmer_start:kmer_end]

        # Extract dwell features
        dwell_start = base_idx - kmer_context
        dwell_end = base_idx + kmer_context + 1
        dwell_chunk = self.dwells[dwell_start:dwell_end]

        # Compile additional features
        features = []
        for _feat_name, feat_array in {**self.dwell_features, **self.signal_features}.items():
            features.append(feat_array[dwell_start:dwell_end])

        return {
            "signal": signal_chunk,
            "sequence": kmer_seq,
            "dwell": dwell_chunk,
            "features": np.stack(features, axis=0) if features else np.array([]),
            "base_idx": base_idx,
            "label": self.labels[base_idx] if self.labels is not None else None,
        }


def extract_training_chunks(
    leech_read: LeechRead,
    motif: str | None = None,
    motif_offset: int = 0,
    label: str | None = None,
    label_int: int | None = None,
    motif_searcher: MotifSearcher | None = None,
    base_justify: str = "center",
) -> list[dict[str, np.ndarray | str | int | None]]:
    """
    Extract all training chunks from a read, optionally filtered by motif.

    Args:
        leech_read: LeechRead object
        motif: Optional sequence motif to filter (e.g., "CCAGGC")
        motif_offset: Offset within motif for focus base
        label: String label identifier (e.g., "Ala", "Gly", "charged", "uncharged")
        label_int: Optional numeric label (0, 1) - assigned during merge for pairwise comparisons
        motif_searcher: MotifSearcher instance (required if motif is provided)

    Returns:
        List of chunk dictionaries

    Example:
        >>> from leech.io.motif_search import get_motif_searcher
        >>> searcher = get_motif_searcher("bam")
        >>> chunks = extract_training_chunks(
        ...     read,
        ...     motif="CCAGGC",
        ...     motif_offset=0,
        ...     label="charged",
        ...     motif_searcher=searcher
        ... )
        >>> print(f"Extracted {len(chunks)} chunks")
    """
    chunks: list[dict] = []

    # Set numeric labels for all bases if provided
    if label_int is not None:
        leech_read.labels = np.full(leech_read.num_bases, label_int, dtype=np.int64)

    # Find focus bases (either all or motif matches)
    if motif is None:
        # No motif - use all bases (avoiding edges)
        focus_bases = list(range(5, leech_read.num_bases - 5))
    else:
        # Motif-based search
        if motif_searcher is None:
            raise ValueError("motif_searcher required when motif is provided")

        # Get alignment from metadata (may be None for basecalled search)
        alignment = leech_read.metadata.get("alignment")

        # Find motif positions using the searcher strategy
        motif_positions = motif_searcher.find_motif_positions(
            read_id=leech_read.read_id,
            sequence=leech_read.sequence,
            alignment=alignment,
            motif=motif,
        )

        # Apply offset to get focus bases
        focus_bases = [pos + motif_offset for pos in motif_positions]

    # Extract chunks
    for base_idx in focus_bases:
        chunk = leech_read.get_chunk(base_idx, base_justify=base_justify)
        if chunk is not None:
            chunk["read_id"] = leech_read.read_id
            # Rename numeric "label" from get_chunk() to "label_int"
            chunk["label_int"] = chunk.pop("label", None)
            # Add string label
            chunk["label"] = label
            chunks.append(chunk)

    return chunks
