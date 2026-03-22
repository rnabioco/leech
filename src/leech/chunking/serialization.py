"""
Chunk serialization utilities.

Provides functions for saving and loading training chunks to/from compressed
numpy format (.npz files).
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("leech.chunking.serialization")


def save_chunks(chunks: list[dict], output_path: Path, *, compressed: bool = True) -> None:
    """
    Save training chunks to numpy format.

    Args:
        chunks: List of chunk dictionaries from extract_training_chunks
        output_path: Output file path (.npz)
        compressed: If True (default), use np.savez_compressed (zlib);
            if False, use np.savez for faster writes at larger file size.

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

    Examples:
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
    seq_to_sig_maps = []
    sequences_with_kmer_context = []

    feature_starts = []
    feature_ends = []
    source_groups = []
    reference_names = []
    signal_residuals = []
    cl_values = []
    focus_signal_pos_list = []
    has_signal_residual = "signal_residual" in chunks[0]
    has_focus_signal_pos = "focus_signal_pos" in chunks[0]

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
        feature_starts.append(chunk.get("feature_start", -5))
        feature_ends.append(chunk.get("feature_end", 5))
        source_groups.append(chunk.get("source_group", ""))
        reference_names.append(chunk.get("reference_name", ""))
        # Charging level: sentinel -1 for missing
        cl_val = chunk.get("cl_value")
        cl_values.append(cl_val if cl_val is not None else -1)
        if has_signal_residual:
            signal_residuals.append(chunk["signal_residual"])
        if has_focus_signal_pos:
            focus_signal_pos_list.append(chunk["focus_signal_pos"])
        # Signal-level kmer encoding fields (may be None for old chunks)
        s2s = chunk.get("seq_to_sig_map")
        seq_ctx = chunk.get("sequence_with_kmer_context")
        seq_to_sig_maps.append(s2s if s2s is not None else np.array([], dtype=np.int64))
        sequences_with_kmer_context.append(seq_ctx if seq_ctx is not None else "")

    # Convert to arrays — use flat (non-object) arrays when shapes are uniform
    # for faster serialization (avoids pickle overhead on object arrays)
    sequences_arr = np.array(sequences, dtype=str)
    labels_arr = np.array(labels, dtype=str)  # String labels
    labels_int_arr = np.array(labels_int, dtype=np.int64)  # Numeric labels
    read_ids_arr = np.array(read_ids, dtype=str)
    base_indices_arr = np.array(base_indices, dtype=np.int64)
    feature_starts_arr = np.array(feature_starts, dtype=np.int64)
    feature_ends_arr = np.array(feature_ends, dtype=np.int64)
    source_groups_arr = np.array(source_groups, dtype=str)
    reference_names_arr = np.array(reference_names, dtype=str)
    sequences_with_kmer_context_arr = np.array(sequences_with_kmer_context, dtype=str)
    # seq_to_sig_maps are variable length (depend on read dwell times), keep as object
    seq_to_sig_maps_arr = np.array(seq_to_sig_maps, dtype=object)

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cl_values_arr = np.array(cl_values, dtype=np.int16)

    save_kwargs: dict[str, np.ndarray] = {
        "sequences": sequences_arr,
        "labels": labels_arr,
        "labels_int": labels_int_arr,
        "read_ids": read_ids_arr,
        "base_indices": base_indices_arr,
        "feature_starts": feature_starts_arr,
        "feature_ends": feature_ends_arr,
        "source_groups": source_groups_arr,
        "reference_names": reference_names_arr,
        "seq_to_sig_maps": seq_to_sig_maps_arr,
        "sequences_with_kmer_context": sequences_with_kmer_context_arr,
        "cl_values": cl_values_arr,
    }

    if has_focus_signal_pos:
        save_kwargs["focus_signal_pos"] = np.array(focus_signal_pos_list, dtype=np.int64)

    # Signals: try stacking into 2D float32 (all chunks should be same length)
    sig_shapes = {s.shape for s in signals}
    if len(sig_shapes) == 1:
        save_kwargs["signals_flat"] = np.stack(signals).astype(np.float32)
    else:
        save_kwargs["signals"] = np.array(signals, dtype=object)

    # Signal residuals (optional, same shape as signals)
    if has_signal_residual and signal_residuals:
        sr_shapes = {s.shape for s in signal_residuals}
        if len(sr_shapes) == 1:
            save_kwargs["signal_residuals_flat"] = np.stack(signal_residuals).astype(np.float32)
        else:
            save_kwargs["signal_residuals"] = np.array(signal_residuals, dtype=object)

    # Dwells: try stacking into 2D
    dwell_shapes = {d.shape for d in dwells}
    if len(dwell_shapes) == 1:
        save_kwargs["dwells_flat"] = np.stack(dwells).astype(np.float32)
    else:
        save_kwargs["dwells"] = np.array(dwells, dtype=object)

    # Features: try stacking into 3D
    feat_shapes = {f.shape for f in features}
    if len(feat_shapes) == 1 and features[0].size > 0:
        save_kwargs["features_flat"] = np.stack(features).astype(np.float32)
    else:
        save_kwargs["features"] = np.array(features, dtype=object)

    # Save
    if compressed:
        np.savez_compressed(output_path, **save_kwargs)
    else:
        np.savez(output_path, **save_kwargs)

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

    Examples:
        >>> chunks = load_chunks(Path("output/chunks.npz"))
        >>> print(f"Loaded {len(chunks)} chunks")
        >>> for chunk in chunks[:5]:
        ...     print(f"{chunk['read_id']}: {chunk['label']}")
    """
    # Load all arrays at once (keeps data memory-mapped when possible)
    with np.load(input_path, allow_pickle=True) as data:
        # Detect format: flat arrays (new, fast) vs object arrays (old, backward compat)
        has_flat_signals = "signals_flat" in data
        has_flat_dwells = "dwells_flat" in data
        has_flat_features = "features_flat" in data

        signals = data["signals_flat"] if has_flat_signals else data["signals"]
        sequences = data["sequences"]
        dwells = data["dwells_flat"] if has_flat_dwells else data["dwells"]
        features = data["features_flat"] if has_flat_features else data["features"]
        labels_arr = data["labels"]  # String labels
        labels_int_arr = data["labels_int"]  # Numeric labels
        read_ids = data["read_ids"]
        base_indices = data["base_indices"]
        # New fields for signal_kmer encoding (backward compatible)
        has_sig_kmer = "seq_to_sig_maps" in data
        if has_sig_kmer:
            seq_to_sig_maps = data["seq_to_sig_maps"]
            sequences_with_kmer_context = data["sequences_with_kmer_context"]

        # Signal residual channel (backward compatible)
        has_signal_residual_data = "signal_residuals_flat" in data or "signal_residuals" in data
        if has_signal_residual_data:
            signal_residuals_loaded = (
                data["signal_residuals_flat"]
                if "signal_residuals_flat" in data
                else data["signal_residuals"]
            )

        # Feature window params (new format: feature_starts/feature_ends)
        has_feature_se = "feature_starts" in data
        if has_feature_se:
            feature_starts_loaded = data["feature_starts"]
            feature_ends_loaded = data["feature_ends"]

        # Backward compat: old format had dwell_margin_lefts
        has_dwell_margin = "dwell_margin_lefts" in data and not has_feature_se
        if has_dwell_margin:
            dwell_margin_lefts = data["dwell_margin_lefts"]

        # Source group for balanced sampling (backward compatible)
        has_source_groups = "source_groups" in data
        if has_source_groups:
            source_groups = data["source_groups"]

        # Reference name (e.g., tRNA isodecoder identity; backward compatible)
        has_reference_names = "reference_names" in data
        if has_reference_names:
            reference_names_loaded = data["reference_names"]

        # Charging level (backward compatible)
        has_cl_values = "cl_values" in data
        if has_cl_values:
            cl_values_loaded = data["cl_values"]

        # Focus signal position (backward compatible — absent in old symmetric data)
        has_focus_signal_pos = "focus_signal_pos" in data
        if has_focus_signal_pos:
            focus_signal_pos_loaded = data["focus_signal_pos"]

        n_chunks = len(labels_arr)
        chunks = []

        # Create dictionaries with references to array elements
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
            if has_sig_kmer:
                s2s = seq_to_sig_maps[i]
                seq_ctx = str(sequences_with_kmer_context[i])
                chunk["seq_to_sig_map"] = s2s if len(s2s) > 0 else None
                chunk["sequence_with_kmer_context"] = seq_ctx if seq_ctx else None
            if has_signal_residual_data:
                chunk["signal_residual"] = signal_residuals_loaded[i]
            if has_feature_se:
                chunk["feature_start"] = int(feature_starts_loaded[i])
                chunk["feature_end"] = int(feature_ends_loaded[i])
            elif has_dwell_margin:
                # Old format: convert dwell_margin_left to feature_left
                chunk["dwell_margin_left"] = int(dwell_margin_lefts[i])
            if has_source_groups:
                sg = str(source_groups[i])
                chunk["source_group"] = sg if sg else None
            if has_reference_names:
                rn = str(reference_names_loaded[i])
                chunk["reference_name"] = rn if rn else ""
            if has_cl_values:
                cl_val = int(cl_values_loaded[i])
                chunk["cl_value"] = cl_val if cl_val >= 0 else None
            else:
                chunk["cl_value"] = None
            if has_focus_signal_pos:
                chunk["focus_signal_pos"] = int(focus_signal_pos_loaded[i])

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

    Examples:
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
