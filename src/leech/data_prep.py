"""
Data preparation: reading POD5 and BAM files, extracting chunks for training.

Adapted from Remora but modernized with NumPy arrays and type hints.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pysam
import torch
from pod5 import DatasetReader

from leech.constants import DEFAULT_KMER_CONTEXT, DEFAULT_SIGNAL_CONTEXT
from leech.features import (
    compute_dwell_features,
    compute_dwell_times,
    compute_signal_features,
    extract_move_table,
    normalize_signal,
)


@dataclass
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

    read_id: str
    sequence: str
    signal: np.ndarray
    seq_to_sig_map: np.ndarray
    dwells: np.ndarray
    dwell_features: dict[str, np.ndarray] = field(default_factory=dict)
    signal_features: dict[str, np.ndarray] = field(default_factory=dict)
    labels: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

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
    ) -> dict[str, np.ndarray | str | int | None] | None:
        """
        Extract a training chunk centered on a specific base.

        Args:
            base_idx: Index of the focus base
            signal_context: (left, right) signal padding around focus base
            kmer_context: Number of bases on each side for k-mer encoding

        Returns:
            Dictionary with 'signal', 'kmer', 'dwell', 'features' arrays,
            or None if chunk cannot be extracted
        """
        # Check boundaries
        if base_idx < kmer_context or base_idx >= self.num_bases - kmer_context:
            return None

        # Extract signal chunk
        focus_sig_pos = self.seq_to_sig_map[base_idx]
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


def read_pod5_signal(pod5_path: Path, read_id: str) -> tuple[np.ndarray, dict]:
    """
    Read raw signal from POD5 file for a specific read.

    Args:
        pod5_path: Path to POD5 file
        read_id: Read identifier

    Returns:
        Tuple of (signal_array, metadata_dict)
    """
    with DatasetReader(pod5_path) as reader:
        for read in reader.reads([read_id]):
            signal = read.signal
            metadata = {
                "read_id": str(read.read_id),
                "channel": read.channel,
                "well": read.well,
                "pore_type": read.pore_type,
                "calibration_offset": read.calibration.offset,
                "calibration_scale": read.calibration.scale,
                "sample_rate": read.sample_rate,
            }
            return signal, metadata

    raise ValueError(f"Read {read_id} not found in {pod5_path}")


def iter_bam_with_pod5(
    bam_path: Path,
    pod5_path: Path,
    reference_fasta: Path | None = None,
    min_mapq: int = 0,
    require_tags: list[str] | None = None,
) -> Iterator[LeechRead]:
    """
    Iterate over aligned reads, loading signal from POD5.

    Args:
        bam_path: Path to BAM file with mv tags
        pod5_path: Path to POD5 file
        reference_fasta: Optional reference for MD tag parsing
        min_mapq: Minimum mapping quality
        require_tags: BAM tags that must be present

    Yields:
        LeechRead objects with full feature extraction
    """
    if require_tags is None:
        require_tags = ["mv", "ns"]
    bam = pysam.AlignmentFile(str(bam_path), "rb")

    for aln in bam:
        # Filter alignments
        if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
            continue
        if aln.mapping_quality < min_mapq:
            continue

        # Check required tags
        if not all(aln.has_tag(tag) for tag in require_tags):
            continue

        try:
            # Extract move table
            move_table = extract_move_table(aln)

            # Check for required query fields
            read_id = aln.query_name
            read_seq = aln.query_sequence
            if read_id is None or read_seq is None:
                continue

            # Read signal from POD5
            raw_signal, pod5_metadata = read_pod5_signal(pod5_path, read_id)

            # Normalize signal
            norm_signal, norm_params = normalize_signal(raw_signal, method="median_mad")

            # Compute seq-to-signal mapping
            seq_to_sig_map = move_table.to_seq_to_sig_map()

            # Compute features
            dwells = compute_dwell_times(move_table)
            dwell_feats = compute_dwell_features(dwells)
            signal_feats = compute_signal_features(norm_signal, seq_to_sig_map)

            # Build metadata
            metadata = {
                **pod5_metadata,
                "normalization": norm_params,
                "mapping_quality": aln.mapping_quality,
                "reference_name": aln.reference_name,
                "reference_start": aln.reference_start,
                "reference_end": aln.reference_end,
                "is_reverse": aln.is_reverse,
            }

            yield LeechRead(
                read_id=read_id,
                sequence=read_seq,
                signal=norm_signal,
                seq_to_sig_map=seq_to_sig_map,
                dwells=dwells,
                dwell_features=dwell_feats,
                signal_features=signal_feats,
                metadata=metadata,
            )

        except Exception as e:
            print(f"Warning: Skipping read {aln.query_name}: {e}")
            continue

    bam.close()


def extract_training_chunks(
    leech_read: LeechRead,
    motif: str | None = None,
    motif_offset: int = 0,
    label: int = 0,
) -> list[dict[str, np.ndarray | str | int | None]]:
    """
    Extract all training chunks from a read, optionally filtered by motif.

    Args:
        leech_read: LeechRead object
        motif: Optional sequence motif to filter (e.g., "CCA")
        motif_offset: Offset within motif for focus base
        label: Label for all chunks from this read

    Returns:
        List of chunk dictionaries
    """
    chunks = []

    # Set labels for all bases
    leech_read.labels = np.full(leech_read.num_bases, label, dtype=np.int64)

    # Find focus bases (either all or motif matches)
    if motif is None:
        focus_bases = list(range(5, leech_read.num_bases - 5))  # Avoid edges
    else:
        # Find motif occurrences
        focus_bases = []
        motif_len = len(motif)
        for i in range(len(leech_read.sequence) - motif_len + 1):
            if leech_read.sequence[i : i + motif_len] == motif:
                focus_bases.append(i + motif_offset)

    # Extract chunks
    for base_idx in focus_bases:
        chunk = leech_read.get_chunk(base_idx)
        if chunk is not None:
            chunk["read_id"] = leech_read.read_id
            chunks.append(chunk)

    return chunks


def seq_to_int(seq: str) -> np.ndarray:
    """
    Convert DNA sequence to integer encoding.

    A=0, C=1, G=2, T=3, N=4, U=3 (treat as T)

    Args:
        seq: DNA/RNA sequence string

    Returns:
        Integer array
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3, "N": 4}
    return np.array([mapping.get(b.upper(), 4) for b in seq], dtype=np.int64)


def int_to_seq(int_seq: np.ndarray) -> str:
    """Convert integer encoding back to sequence."""
    bases = ["A", "C", "G", "T", "N"]
    return "".join(bases[i] if i < 5 else "N" for i in int_seq)


def encode_kmer(sequence: str) -> torch.Tensor:
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

    Example:
        >>> seq = "ACGT"
        >>> encoded = encode_kmer(seq)
        >>> encoded.shape
        torch.Size([4, 4])
    """
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


def save_chunks(chunks: list[dict], output_path: Path) -> None:
    """
    Save training chunks to compressed numpy format.

    Args:
        chunks: List of chunk dictionaries from extract_training_chunks
        output_path: Output file path (.npz)

    Format:
        Saves as .npz with arrays:
        - signals: (N, signal_len) raw signal chunks
        - sequences: (N,) string array of k-mer sequences
        - dwells: (N, kmer_len) dwell times
        - features: (N, num_features, kmer_len) feature arrays
        - labels: (N,) integer labels
        - read_ids: (N,) string array of read IDs
        - base_indices: (N,) base indices
    """
    if not chunks:
        raise ValueError("No chunks to save")

    # Collect arrays
    signals = []
    sequences = []
    dwells = []
    features = []
    labels = []
    read_ids = []
    base_indices = []

    for chunk in chunks:
        signals.append(chunk["signal"])
        sequences.append(chunk["sequence"])
        dwells.append(chunk["dwell"])
        features.append(chunk["features"])
        labels.append(chunk["label"] if chunk["label"] is not None else -1)
        read_ids.append(chunk["read_id"])
        base_indices.append(chunk["base_idx"])

    # Convert to arrays
    # Signals may have variable length, so we'll save them as object array
    signals_arr = np.array(signals, dtype=object)
    sequences_arr = np.array(sequences, dtype=str)
    dwells_arr = np.array(dwells, dtype=object)
    features_arr = np.array(features, dtype=object)
    labels_arr = np.array(labels, dtype=np.int64)
    read_ids_arr = np.array(read_ids, dtype=str)
    base_indices_arr = np.array(base_indices, dtype=np.int64)

    # Save
    np.savez_compressed(
        output_path,
        signals=signals_arr,
        sequences=sequences_arr,
        dwells=dwells_arr,
        features=features_arr,
        labels=labels_arr,
        read_ids=read_ids_arr,
        base_indices=base_indices_arr,
    )

    print(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(input_path: Path) -> list[dict]:
    """
    Load training chunks from compressed numpy format.

    Args:
        input_path: Path to .npz file

    Returns:
        List of chunk dictionaries compatible with extract_training_chunks output
    """
    data = np.load(input_path, allow_pickle=True)

    chunks = []
    n_chunks = len(data["labels"])

    for i in range(n_chunks):
        chunk = {
            "signal": data["signals"][i],
            "sequence": str(data["sequences"][i]),
            "dwell": data["dwells"][i],
            "features": data["features"][i],
            "label": int(data["labels"][i]) if data["labels"][i] >= 0 else None,
            "read_id": str(data["read_ids"][i]),
            "base_idx": int(data["base_indices"][i]),
        }
        chunks.append(chunk)

    return chunks
