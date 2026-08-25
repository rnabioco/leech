"""
Chunk serialization utilities.

Provides functions for saving and loading training chunks to/from compressed
numpy format (.npz files).
"""

import contextlib
import logging
import zipfile
from collections.abc import Collection, Iterator
from pathlib import Path

import numpy as np

logger = logging.getLogger("leech.chunking.serialization")

# Chunk fields `load_chunks(..., defer=...)` can skip. These are the per-chunk
# arrays; everything else in the file is a scalar or a string and stays cheap.
DEFERRABLE_FIELDS = frozenset(
    {
        "signal",
        "signal_residual",
        "dwell",
        "features",
        "seq_to_sig_map",
        "sequence_with_kmer_context",
    }
)


def _read_npy_header(fp) -> tuple[tuple[int, ...], bool, np.dtype]:
    """Read the .npy magic + header from an open member stream.

    numpy 2.x moved the version-dispatching ``_read_array_header`` out of the
    public namespace, so dispatch on the magic ourselves.
    """
    major, _minor = np.lib.format.read_magic(fp)
    if major == 1:
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fp)
    else:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fp)
    return tuple(shape), fortran_order, dtype


# ZipExtFile has no native readinto: BufferedIOBase's default allocates a bytes
# object the size of the request. Fill the block in bounded slices so that
# transient stays small no matter how large a block is.
_READ_SLICE_BYTES = 1 << 20


def _readinto_exact(fp, buf: memoryview) -> int:
    """Fill ``buf`` from ``fp``, returning the bytes read (short only at EOF)."""
    total = 0
    while total < len(buf):
        end = min(total + _READ_SLICE_BYTES, len(buf))
        got = fp.readinto(buf[total:end])
        if not got:
            break
        total += got
    return total


def csr_offsets_from_lens(lens: np.ndarray) -> np.ndarray:
    """Row offsets for CSR storage: ``[0, lens[0], lens[0] + lens[1], ...]``."""
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    return offsets


def csr_gather_index(
    offsets: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index arrays for reading a subset of CSR rows as one concatenated run.

    Args:
        offsets: CSR row offsets.
        rows: Row indices to gather, ascending.

    Returns:
        ``(lens, col, src)``: per-row lengths, the within-row column of every
        gathered element, and the index into the values array of every gathered
        element. ``values[src]`` is the concatenation of the selected rows.
    """
    lens = (offsets[rows + 1] - offsets[rows]).astype(np.int64)
    total = int(lens.sum())
    col = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(lens) - lens, lens)
    src = np.repeat(offsets[rows], lens) + col
    return lens, col, src


def csr_from_object_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a legacy object array of variable-length rows into CSR form."""
    lens = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
    offsets = csr_offsets_from_lens(lens)
    values = (
        np.concatenate(list(rows)).astype(np.int32)
        if offsets[-1] > 0
        else np.empty(0, dtype=np.int32)
    )
    return values, offsets


def npz_member_names(input_path: Path) -> set[str]:
    """Names of every member of an .npz, without reading any data.

    Unlike :func:`npz_array_members` this includes object (pickled) members, so
    it answers "was this field written at all" for formats that
    :func:`iter_npz_row_blocks` cannot stream.
    """
    with zipfile.ZipFile(input_path) as zf:
        return {name[:-4] for name in zf.namelist() if name.endswith(".npy")}


def npz_array_members(input_path: Path) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Shapes and dtypes of the row-streamable members of an .npz, without reading data.

    Only the zip central directory and each member's .npy header are read, so
    this is O(number of members) regardless of file size. Members that
    :func:`iter_npz_row_blocks` cannot stream — object dtypes (pickled),
    Fortran order, 0-d — are omitted, so ``name in npz_array_members(path)``
    is the test for "can I stream this".

    Args:
        input_path: Path to .npz file

    Returns:
        Mapping of member name (without the .npy suffix) to ``(shape, dtype)``.

    Examples:
        >>> members = npz_array_members(Path("chunks.npz"))
        >>> members["signals_flat"][0]  # doctest: +SKIP
        (6668328, 540)
    """
    members: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    with zipfile.ZipFile(input_path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with zf.open(info) as fp:
                try:
                    shape, fortran_order, dtype = _read_npy_header(fp)
                except ValueError:  # not a .npy stream we understand
                    continue
            if fortran_order or dtype.hasobject or not shape:
                continue
            members[info.filename[:-4]] = (shape, dtype)
    return members


def iter_npz_row_blocks(
    input_path: Path,
    names: Collection[str],
    block_rows: int | None = None,
    *,
    block_bytes: int = 8 << 20,
) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    """Yield ``(row_start, {name: block})`` over the rows of fixed-shape npz members.

    ``np.load`` reads a whole member at once — it never memory-maps a zip
    member, compressed or not — so converting a large corpus means holding the
    numpy source and the converted output at the same time (#211). This walks
    the members as sequential row blocks instead, so only one block per member
    is resident.

    **Blocks are reused between iterations.** The yielded arrays are views into
    buffers that the next iteration overwrites; copy anything you need to keep.

    Args:
        input_path: Path to .npz file
        names: Member names to stream, without the .npy suffix. All must have
            the same number of rows (see :func:`npz_array_members`).
        block_rows: Rows per block. Default sizes the block from ``block_bytes``
            so a wide signal does not silently allocate a huge buffer, and
            never exceeds the member's row count.
        block_bytes: Target resident bytes across all streamed members, used
            when ``block_rows`` is not given.

    Yields:
        ``(row_start, blocks)`` where ``blocks[name]`` has ``min(block_rows,
        n_rows - row_start)`` rows.

    Raises:
        ValueError: If a member is not streamable, row counts disagree, or the
            member is truncated.
    """
    names = list(names)
    if not names or (block_rows is not None and block_rows < 1):
        return

    with contextlib.ExitStack() as stack:
        opened: dict[str, tuple] = {}
        n_rows: int | None = None
        for name in names:
            # One handle per member: each is then a single sequential read
            # rather than interleaved seeks through one shared file object.
            zf = stack.enter_context(zipfile.ZipFile(input_path))
            fp = stack.enter_context(zf.open(name + ".npy"))
            shape, fortran_order, dtype = _read_npy_header(fp)
            if fortran_order or dtype.hasobject or not shape:
                raise ValueError(f"npz member '{name}' is not row-streamable")
            if n_rows is None:
                n_rows = shape[0]
            elif shape[0] != n_rows:
                raise ValueError(f"npz member '{name}' has {shape[0]} rows, expected {n_rows}")
            row_shape = shape[1:]
            row_bytes = int(np.prod(row_shape, dtype=np.int64)) * dtype.itemsize
            opened[name] = (fp, row_shape, dtype, row_bytes)

        assert n_rows is not None
        if block_rows is None:
            per_row = sum(entry[3] for entry in opened.values())
            block_rows = max(1, block_bytes // max(per_row, 1))
        block_rows = min(block_rows, max(n_rows, 1))

        streams: dict[str, tuple] = {}
        for name, (fp, row_shape, dtype, row_bytes) in opened.items():
            block = np.empty((block_rows, *row_shape), dtype=dtype)
            streams[name] = (fp, block, block.reshape(-1).view(np.uint8), row_bytes)

        start = 0
        while start < n_rows:
            rows = min(block_rows, n_rows - start)
            blocks = {}
            for name, (fp, block, raw, row_bytes) in streams.items():
                want = rows * row_bytes
                got = _readinto_exact(fp, memoryview(raw)[:want])
                if got != want:
                    raise ValueError(
                        f"npz member '{name}' truncated at row {start}: read {got} of {want} bytes"
                    )
                blocks[name] = block[:rows]
            yield start, blocks
            start += rows


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
        - seq_to_sig_values / seq_to_sig_offsets: base-to-signal maps in CSR
          form; row i is ``values[offsets[i]:offsets[i + 1]]``. Files written
          before v0.6.8 carry a pickled object array named ``seq_to_sig_maps``
          instead, which :func:`load_chunks` still reads.

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
    # seq_to_sig_maps are variable length (they depend on the read's dwell
    # times), so store them CSR-style: one flat values array plus row offsets.
    # An object array would be pickled, which costs a Python ndarray per chunk
    # on load and makes the member unstreamable (#211).
    seq_to_sig_values_arr, seq_to_sig_offsets_arr = csr_from_object_rows(seq_to_sig_maps)

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
        "seq_to_sig_values": seq_to_sig_values_arr,
        "seq_to_sig_offsets": seq_to_sig_offsets_arr,
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


def load_seq_to_sig_csr(input_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load the base-to-signal maps in CSR form: ``(values, offsets)``.

    Row ``i`` is ``values[offsets[i]:offsets[i + 1]]``. Files written before
    v0.6.8 store these as a pickled object array (``seq_to_sig_maps``); those
    are converted here so callers see one representation.

    Args:
        input_path: Path to .npz file

    Returns:
        ``(values, offsets)``, or None if the file has no base-to-signal maps.
    """
    with np.load(input_path, allow_pickle=True) as data:
        if "seq_to_sig_values" in data:
            return data["seq_to_sig_values"], data["seq_to_sig_offsets"]
        if "seq_to_sig_maps" not in data:
            return None
        return csr_from_object_rows(data["seq_to_sig_maps"])


def load_chunks(input_path: Path, *, defer: Collection[str] = ()) -> list[dict]:
    """
    Load training chunks from compressed numpy format.

    Args:
        input_path: Path to .npz file
        defer: Chunk array fields to leave unread (see :data:`DEFERRABLE_FIELDS`).
            The key is still present on each chunk, set to None. Callers that
            convert the arrays themselves — :class:`~leech.dataset.LeechDataset`
            streams them with :func:`iter_npz_row_blocks` — use this to avoid
            holding a second full copy.

    Returns:
        List of chunk dictionaries compatible with extract_training_chunks output

    Note:
        Every member requested is read in full: ``np.load`` does not memory-map
        zip members, compressed or not, so there is no lazy path here. On a
        large corpus this dominates peak memory, which is what ``defer`` and
        :func:`iter_npz_row_blocks` exist to avoid (#211).

    Examples:
        >>> chunks = load_chunks(Path("output/chunks.npz"))
        >>> print(f"Loaded {len(chunks)} chunks")
        >>> for chunk in chunks[:5]:
        ...     print(f"{chunk['read_id']}: {chunk['label']}")
    """
    deferred = frozenset(defer)
    unknown = deferred - DEFERRABLE_FIELDS
    if unknown:
        raise ValueError(
            f"Cannot defer unknown chunk field(s): {sorted(unknown)}. "
            f"Deferrable: {sorted(DEFERRABLE_FIELDS)}"
        )

    # Repeated string fields: a few hundred distinct values across millions of
    # chunks, and str(arr[i]) mints a fresh object every time.
    _interned: dict[str, str] = {}

    def _intern(value) -> str:
        text = str(value)
        return _interned.setdefault(text, text)

    with np.load(input_path, allow_pickle=True) as data:
        # Detect format: flat arrays (new, fast) vs object arrays (old, backward compat)
        has_flat_signals = "signals_flat" in data
        has_flat_dwells = "dwells_flat" in data
        has_flat_features = "features_flat" in data

        signals = (
            None
            if "signal" in deferred
            else data["signals_flat" if has_flat_signals else "signals"]
        )
        sequences = data["sequences"]
        dwells = (
            None if "dwell" in deferred else data["dwells_flat" if has_flat_dwells else "dwells"]
        )
        features = (
            None
            if "features" in deferred
            else data["features_flat" if has_flat_features else "features"]
        )
        labels_arr = data["labels"]  # String labels
        labels_int_arr = data["labels_int"]  # Numeric labels
        read_ids = data["read_ids"]
        base_indices = data["base_indices"]
        # Base-to-signal maps: CSR pair (new) or pickled object array (old)
        has_sig_kmer = "seq_to_sig_values" in data or "seq_to_sig_maps" in data
        seq_to_sig_values = seq_to_sig_offsets = seq_to_sig_maps = None
        sequences_with_kmer_context = None
        if has_sig_kmer:
            if "seq_to_sig_map" not in deferred:
                if "seq_to_sig_values" in data:
                    seq_to_sig_values = data["seq_to_sig_values"]
                    seq_to_sig_offsets = data["seq_to_sig_offsets"]
                else:
                    seq_to_sig_maps = data["seq_to_sig_maps"]
            if "sequence_with_kmer_context" not in deferred:
                sequences_with_kmer_context = data["sequences_with_kmer_context"]

        # Signal residual channel (backward compatible)
        has_signal_residual_data = "signal_residuals_flat" in data or "signal_residuals" in data
        if has_signal_residual_data and "signal_residual" not in deferred:
            signal_residuals_loaded = (
                data["signal_residuals_flat"]
                if "signal_residuals_flat" in data
                else data["signal_residuals"]
            )
        else:
            signal_residuals_loaded = None

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
                "signal": None if signals is None else signals[i],
                "sequence": _intern(sequences[i]),
                "dwell": None if dwells is None else dwells[i],
                "features": None if features is None else features[i],
                "read_id": str(read_ids[i]),
                "base_idx": int(base_indices[i]),
                "label": _intern(labels_arr[i]) if labels_arr[i] != "" else None,
                "label_int": int(labels_int_arr[i]) if labels_int_arr[i] >= 0 else None,
            }
            if has_sig_kmer:
                if seq_to_sig_offsets is not None:
                    s2s = seq_to_sig_values[seq_to_sig_offsets[i] : seq_to_sig_offsets[i + 1]]
                elif seq_to_sig_maps is not None:
                    s2s = seq_to_sig_maps[i]
                else:
                    s2s = None
                chunk["seq_to_sig_map"] = s2s if s2s is not None and len(s2s) > 0 else None
                if sequences_with_kmer_context is None:
                    chunk["sequence_with_kmer_context"] = None
                else:
                    seq_ctx = str(sequences_with_kmer_context[i])
                    chunk["sequence_with_kmer_context"] = seq_ctx if seq_ctx else None
            if has_signal_residual_data:
                chunk["signal_residual"] = (
                    None if signal_residuals_loaded is None else signal_residuals_loaded[i]
                )
            if has_feature_se:
                chunk["feature_start"] = int(feature_starts_loaded[i])
                chunk["feature_end"] = int(feature_ends_loaded[i])
            elif has_dwell_margin:
                # Old format: convert dwell_margin_left to feature_left
                chunk["dwell_margin_left"] = int(dwell_margin_lefts[i])
            if has_source_groups:
                sg = _intern(source_groups[i])
                chunk["source_group"] = sg if sg else None
            if has_reference_names:
                rn = _intern(reference_names_loaded[i])
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
