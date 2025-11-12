"""
Chunk serialization utilities.

Provides functions for saving and loading training chunks to/from compressed
numpy format (.npz files).
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("leech.chunking.serialization")


def save_chunks(chunks: list[dict], output_path: Path) -> None:
    """
    Save training chunks to compressed numpy format.

    Args:
        chunks: List of chunk dictionaries from extract_training_chunks
        output_path: Output file path (.npz)

    Raises:
        ValueError: If chunks list is empty

    Format:
        Saves as .npz with arrays:
        - signals: (N, signal_len) raw signal chunks (object array for variable length)
        - sequences: (N,) string array of k-mer sequences
        - dwells: (N, kmer_len) dwell times (object array for variable length)
        - features: (N, num_features, kmer_len) feature arrays (object array)
        - labels: (N,) string labels (e.g., "Ala", "Gly")
        - labels_int: (N,) integer labels (0, 1, or -1 if unset)
        - read_ids: (N,) string array of read IDs
        - base_indices: (N,) base indices

    Example:
        >>> chunks = extract_training_chunks(read, motif="CCAGGC")
        >>> save_chunks(chunks, Path("output/chunks.npz"))
    """
    if not chunks:
        raise ValueError("No chunks to save")

    # Collect arrays
    signals = []
    sequences = []
    dwells = []
    features = []
    labels = []
    labels_int = []
    read_ids = []
    base_indices = []

    for chunk in chunks:
        signals.append(chunk["signal"])
        sequences.append(chunk["sequence"])
        dwells.append(chunk["dwell"])
        features.append(chunk["features"])
        labels.append(chunk.get("label", ""))  # String label (e.g., "Ala", "Gly")
        labels_int.append(
            chunk.get("label_int", -1) if chunk.get("label_int") is not None else -1
        )  # Numeric label or -1
        read_ids.append(chunk["read_id"])
        base_indices.append(chunk["base_idx"])

    # Convert to arrays
    # Signals may have variable length, so we'll save them as object array
    signals_arr = np.array(signals, dtype=object)
    sequences_arr = np.array(sequences, dtype=str)
    dwells_arr = np.array(dwells, dtype=object)
    features_arr = np.array(features, dtype=object)
    labels_arr = np.array(labels, dtype=str)  # String labels
    labels_int_arr = np.array(labels_int, dtype=np.int64)  # Numeric labels
    read_ids_arr = np.array(read_ids, dtype=str)
    base_indices_arr = np.array(base_indices, dtype=np.int64)

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    np.savez_compressed(
        output_path,
        signals=signals_arr,
        sequences=sequences_arr,
        dwells=dwells_arr,
        features=features_arr,
        labels=labels_arr,
        labels_int=labels_int_arr,
        read_ids=read_ids_arr,
        base_indices=base_indices_arr,
    )

    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(input_path: Path) -> list[dict]:
    """
    Load training chunks from compressed numpy format.

    Args:
        input_path: Path to .npz file

    Returns:
        List of chunk dictionaries compatible with extract_training_chunks output

    Note:
        This function loads all arrays into memory at once. The arrays are stored
        as numpy object arrays (dtype=object) to handle variable-length signals.
        The loaded data is kept in memory-mapped form when possible, but converting
        to individual dictionaries will create copies in memory.

    Example:
        >>> chunks = load_chunks(Path("output/chunks.npz"))
        >>> print(f"Loaded {len(chunks)} chunks")
        >>> for chunk in chunks[:5]:
        ...     print(f"{chunk['read_id']}: {chunk['label']}")
    """
    # Load all arrays at once (keeps data memory-mapped when possible)
    with np.load(input_path, allow_pickle=True) as data:
        # Extract all arrays first (this creates copies but only once)
        signals = data["signals"]
        sequences = data["sequences"]
        dwells = data["dwells"]
        features = data["features"]
        labels_arr = data["labels"]  # String labels
        labels_int_arr = data["labels_int"]  # Numeric labels
        read_ids = data["read_ids"]
        base_indices = data["base_indices"]

        n_chunks = len(labels_arr)
        chunks = []

        # Create dictionaries with references to array elements
        # This is more memory efficient than accessing data[key][i] each time
        for i in range(n_chunks):
            chunk = {
                "signal": signals[i],
                "sequence": str(sequences[i]),
                "dwell": dwells[i],
                "features": features[i],
                "read_id": str(read_ids[i]),
                "base_idx": int(base_indices[i]),
                "label": str(labels_arr[i]) if labels_arr[i] != "" else None,
                "label_int": int(labels_int_arr[i]) if labels_int_arr[i] >= 0 else None,
            }

            chunks.append(chunk)

    logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
    return chunks


def get_chunk_statistics(chunks: list[dict]) -> dict:
    """
    Compute statistics about a list of chunks.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Dictionary with statistics:
        - n_chunks: Number of chunks
        - n_reads: Number of unique reads
        - labels: Distribution of string labels
        - label_ints: Distribution of numeric labels
        - signal_lengths: Mean/std/min/max signal lengths
        - sequence_lengths: Mean/std/min/max sequence lengths

    Example:
        >>> chunks = load_chunks(Path("chunks.npz"))
        >>> stats = get_chunk_statistics(chunks)
        >>> print(f"Chunks: {stats['n_chunks']}")
        >>> print(f"Reads: {stats['n_reads']}")
        >>> print(f"Labels: {stats['labels']}")
    """
    if not chunks:
        return {
            "n_chunks": 0,
            "n_reads": 0,
            "labels": {},
            "label_ints": {},
            "signal_lengths": {"mean": 0, "std": 0, "min": 0, "max": 0},
            "sequence_lengths": {"mean": 0, "std": 0, "min": 0, "max": 0},
        }

    # Count unique reads
    unique_reads = {chunk["read_id"] for chunk in chunks}

    # Label distribution
    label_counts: dict[str, int] = {}
    for chunk in chunks:
        label = chunk.get("label")
        if label is not None:
            label_counts[label] = label_counts.get(label, 0) + 1

    # Numeric label distribution
    label_int_counts: dict[int, int] = {}
    for chunk in chunks:
        label_int = chunk.get("label_int")
        if label_int is not None and label_int >= 0:
            label_int_counts[label_int] = label_int_counts.get(label_int, 0) + 1

    # Signal length statistics
    signal_lengths = [len(chunk["signal"]) for chunk in chunks]
    signal_stats = {
        "mean": float(np.mean(signal_lengths)),
        "std": float(np.std(signal_lengths)),
        "min": int(np.min(signal_lengths)),
        "max": int(np.max(signal_lengths)),
    }

    # Sequence length statistics
    seq_lengths = [len(chunk["sequence"]) for chunk in chunks]
    seq_stats = {
        "mean": float(np.mean(seq_lengths)),
        "std": float(np.std(seq_lengths)),
        "min": int(np.min(seq_lengths)),
        "max": int(np.max(seq_lengths)),
    }

    return {
        "n_chunks": len(chunks),
        "n_reads": len(unique_reads),
        "labels": label_counts,
        "label_ints": label_int_counts,
        "signal_lengths": signal_stats,
        "sequence_lengths": seq_stats,
    }
