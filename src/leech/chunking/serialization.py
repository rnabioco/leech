"""
Chunk serialization utilities.

Provides functions for saving and loading training chunks to/from compressed
numpy format (.npz files).
"""

import contextlib
import logging
import tempfile
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


def npz_path(output_path: Path) -> Path:
    """The file ``np.savez`` would write for ``output_path`` (it appends .npz)."""
    text = str(output_path)
    return output_path if text.endswith(".npz") else Path(text + ".npz")


def open_npz_zip(output_path: Path, compressed: bool) -> zipfile.ZipFile:
    """Open an .npz for writing with the same zip settings ``np.savez`` uses."""
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    return zipfile.ZipFile(
        npz_path(output_path), mode="w", compression=compression, allowZip64=True
    )


def write_npz_member(zf: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    """Write one array into an open .npz exactly as ``np.savez`` would."""
    with zf.open(name + ".npy", "w", force_zip64=True) as fp:
        np.lib.format.write_array(fp, np.asanyarray(array), allow_pickle=True)


def write_npy_header(fp, shape: tuple[int, ...], dtype: np.dtype) -> None:
    """Write the .npy magic + header for ``shape``/``dtype``, payload to follow.

    :func:`np.lib.format.write_array` needs the whole array; this is the split
    version, for members whose rows are streamed in afterwards.
    """
    header = np.lib.format.header_data_from_array_1_0(np.empty((0,), dtype=dtype))
    header["shape"] = tuple(int(n) for n in shape)
    try:
        np.lib.format.write_array_header_1_0(fp, header)
    except ValueError:  # header too long for the 1.0 format
        np.lib.format.write_array_header_2_0(fp, header)


#: ``np.strings`` since numpy 2.0, ``np.char`` before it.
_str_len = getattr(np, "strings", np.char).str_len


def _text_dtype(values: np.ndarray) -> np.dtype:
    """The ``<U`` dtype ``np.array(list_of_str, dtype=str)`` would give ``values``.

    Selecting rows out of a member does not narrow its width, but
    :func:`save_chunks` on the same subset would size it to that subset's
    longest string. Recompute it so a split file is dtype-identical to one
    written from a list.
    """
    width = int(_str_len(values).max()) if len(values) else 0
    return np.dtype(f"<U{max(width, 1)}")


#: Transient bytes allowed when copying array payloads around. Deliberately
#: small: this is what keeps the writer's peak a constant rather than a
#: fraction of the corpus, and a memcpy this size costs nothing.
_COPY_SLICE_BYTES = 1 << 20


def _row_step(dtype: np.dtype, row_shape: tuple[int, ...]) -> int:
    """Rows per payload slice, so a copy never exceeds ``_COPY_SLICE_BYTES``."""
    row_bytes = int(np.prod(row_shape, dtype=np.int64)) * dtype.itemsize
    return max(1, _COPY_SLICE_BYTES // max(int(row_bytes), 1))


def _write_rows(fp, array: np.ndarray) -> None:
    """Write an array's payload in bounded slices (never one big ``tobytes``)."""
    step = _row_step(array.dtype, array.shape[1:])
    for start in range(0, len(array), step):
        fp.write(array[start : start + step].tobytes())


def _text(value) -> str:
    """Coerce an optional text field to the empty-string convention.

    `None` means "absent" throughout the chunk format, and the readers already
    honour that on the way in: `load_chunks` and `ChunkTable` both map `""` back
    to `None`. The writer did not honour it on the way out. `chunk.get("label",
    "")` returns the default only when the key is *missing*, so a key present
    with the value `None` — which is what `--label` produces when it is not
    passed, and what `load_chunks` produces for an absent group — reached
    `np.array(..., dtype=str)` and was stringified to the literal `"None"`.

    That made a save/load round trip non-idempotent: `"" -> None -> "None"`, so
    merging a corpus renamed its empty source groups to a group *called*
    "None", which `--balance-groups` then weighted as a real one.
    """
    return "" if value is None else str(value)


def iter_chunk_columns(chunks: list[dict]) -> Iterator[tuple[str, np.ndarray | list]]:
    """Yield ``(member_name, value)`` for every npz member of ``chunks``, in write order.

    This is the single definition of the on-disk chunk format: both
    :func:`save_chunks` and :class:`ChunkSpool` consume it, so a field can only
    be added in one place.

    ``value`` is an ndarray for the members stored as fixed-shape arrays, or a
    plain list of per-chunk rows for the ragged (object-array) fallback —
    ``signals``/``dwells``/``features`` when the chunks do not all share a
    shape. Members are yielded one at a time and the caller is expected to
    write each one before asking for the next, so the stacked copies never
    coexist (#211).

    Args:
        chunks: List of chunk dictionaries from extract_training_chunks

    Yields:
        ``(name, value)`` pairs. Names never repeat.
    """
    if not chunks:
        return

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
        labels.append(_text(chunk.get("label")))  # String label (e.g., "Ala", "Gly")
        labels_int.append(
            chunk.get("label_int", -1) if chunk.get("label_int") is not None else -1
        )  # Numeric label or -1
        read_ids.append(chunk["read_id"])
        base_indices.append(chunk["base_idx"])
        feature_starts.append(chunk.get("feature_start", -5))
        feature_ends.append(chunk.get("feature_end", 5))
        source_groups.append(_text(chunk.get("source_group")))
        reference_names.append(_text(chunk.get("reference_name")))
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
        sequences_with_kmer_context.append(_text(seq_ctx))

    # Convert to arrays — use flat (non-object) arrays when shapes are uniform
    # for faster serialization (avoids pickle overhead on object arrays).
    # Each member is built at the point it is yielded and its source list is
    # dropped straight after, so at most one stacked array is alive at a time.
    yield "sequences", np.array(sequences, dtype=str)
    del sequences
    yield "labels", np.array(labels, dtype=str)  # String labels
    del labels
    yield "labels_int", np.array(labels_int, dtype=np.int64)  # Numeric labels
    del labels_int
    yield "read_ids", np.array(read_ids, dtype=str)
    del read_ids
    yield "base_indices", np.array(base_indices, dtype=np.int64)
    del base_indices
    yield "feature_starts", np.array(feature_starts, dtype=np.int64)
    del feature_starts
    yield "feature_ends", np.array(feature_ends, dtype=np.int64)
    del feature_ends
    yield "source_groups", np.array(source_groups, dtype=str)
    del source_groups
    yield "reference_names", np.array(reference_names, dtype=str)
    del reference_names

    # seq_to_sig_maps are variable length (they depend on the read's dwell
    # times), so store them CSR-style: one flat values array plus row offsets.
    # An object array would be pickled, which costs a Python ndarray per chunk
    # on load and makes the member unstreamable (#211).
    seq_to_sig_values_arr, seq_to_sig_offsets_arr = csr_from_object_rows(seq_to_sig_maps)
    del seq_to_sig_maps
    yield "seq_to_sig_values", seq_to_sig_values_arr
    del seq_to_sig_values_arr
    yield "seq_to_sig_offsets", seq_to_sig_offsets_arr
    del seq_to_sig_offsets_arr

    yield "sequences_with_kmer_context", np.array(sequences_with_kmer_context, dtype=str)
    del sequences_with_kmer_context
    yield "cl_values", np.array(cl_values, dtype=np.int16)
    del cl_values

    if has_focus_signal_pos:
        yield "focus_signal_pos", np.array(focus_signal_pos_list, dtype=np.int64)
    del focus_signal_pos_list

    # Signals: try stacking into 2D float32 (all chunks should be same length).
    # `copy=False` matters: chunk signals are already float32, so the default
    # `astype` duplicates the largest array in the process for nothing (#211).
    sig_shapes = {s.shape for s in signals}
    if len(sig_shapes) == 1:
        yield "signals_flat", np.stack(signals).astype(np.float32, copy=False)
    else:
        yield "signals", signals
    del signals

    # Signal residuals (optional, same shape as signals)
    if has_signal_residual and signal_residuals:
        sr_shapes = {s.shape for s in signal_residuals}
        if len(sr_shapes) == 1:
            yield (
                "signal_residuals_flat",
                np.stack(signal_residuals).astype(np.float32, copy=False),
            )
        else:
            yield "signal_residuals", signal_residuals
    del signal_residuals

    # Dwells: try stacking into 2D
    dwell_shapes = {d.shape for d in dwells}
    if len(dwell_shapes) == 1:
        yield "dwells_flat", np.stack(dwells).astype(np.float32, copy=False)
    else:
        yield "dwells", dwells
    del dwells

    # Features: try stacking into 3D
    feat_shapes = {f.shape for f in features}
    if len(feat_shapes) == 1 and features[0].size > 0:
        yield "features_flat", np.stack(features).astype(np.float32, copy=False)
    else:
        yield "features", features
    del features


def save_chunks(chunks: list[dict], output_path: Path, *, compressed: bool = True) -> None:
    """
    Save training chunks to numpy format.

    Args:
        chunks: List of chunk dictionaries from extract_training_chunks
        output_path: Output file path (.npz)
        compressed: If True (default), compress members with zlib
            (``np.savez_compressed``); if False, store them (``np.savez``) for
            faster writes at larger file size.

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

    Note:
        Members are stacked and written one at a time rather than collected
        into a ``np.savez`` call, so only one of them is resident at a time.
        Callers that do not already have the whole corpus as a list should use
        :class:`ChunkNpzWriter`, which never builds one.

    Examples:
        >>> chunks = extract_training_chunks(read, motif="CCAGGC")
        >>> save_chunks(chunks, Path("output/chunks.npz"))
    """
    if not chunks:
        raise ValueError("No chunks to save")

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open_npz_zip(output_path, compressed) as zf:
        for name, value in iter_chunk_columns(chunks):
            array = np.array(value, dtype=object) if isinstance(value, list) else value
            write_npz_member(zf, name, array)
            del array, value

    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


class _SpilledColumn:
    """One fixed-shape npz member, accumulated row by row in a temp file."""

    __slots__ = ("name", "dtype", "row_shape", "row_bytes", "n_rows", "_fp", "_mm", "_sealed")

    def __init__(self, name: str, dtype: np.dtype, row_shape: tuple[int, ...], spill_dir: Path):
        self.name = name
        self.dtype = dtype
        self.row_shape = row_shape
        self.row_bytes = int(np.prod(row_shape, dtype=np.int64)) * dtype.itemsize
        self.n_rows = 0
        self._fp = tempfile.TemporaryFile(dir=spill_dir)
        self._mm: np.ndarray | None = None
        self._sealed = False

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n_rows, *self.row_shape)

    def append(self, array: np.ndarray) -> None:
        if array.dtype != self.dtype or array.shape[1:] != self.row_shape:
            raise ValueError(
                f"npz member '{self.name}' changed layout between batches: "
                f"had {self.dtype}{self.row_shape}, got {array.dtype}{array.shape[1:]}"
            )
        if self._sealed:
            # Reading rewinds the spill; appending after that would overwrite it.
            raise ValueError(f"npz member '{self.name}' cannot be appended to after a read")
        array = np.ascontiguousarray(array)
        try:
            # Writes straight out of the array's buffer: no `tobytes` copy of a
            # member that may be the largest array in the process.
            array.tofile(self._fp)
        except (OSError, ValueError):  # not a real file (mocked, in-memory, ...)
            _write_rows(self._fp, array)
        self.n_rows += len(array)

    def stream_to(self, fp) -> None:
        """Copy the whole payload into ``fp`` through one reused buffer."""
        self._sealed = True
        self._fp.flush()
        self._fp.seek(0)
        remaining = self.n_rows * self.row_bytes
        buffer = bytearray(min(_COPY_SLICE_BYTES, max(remaining, 1)))
        view = memoryview(buffer)
        while remaining > 0:
            want = min(len(buffer), remaining)
            got = self._fp.readinto(view[:want])
            if not got:
                raise ValueError(f"spill for npz member '{self.name}' is truncated")
            fp.write(view[:got])
            remaining -= got

    def view(self) -> np.ndarray:
        """A read-only view of every row, without reading the file into memory.

        A memory map where the filesystem allows one, otherwise the member read
        back into an array.
        """
        self._sealed = True
        if self.n_rows == 0:
            return np.empty(self.shape, dtype=self.dtype)
        if self._mm is None:
            self._fp.flush()
            try:
                self._mm = np.memmap(self._fp, dtype=self.dtype, mode="r", shape=self.shape)
            except (OSError, ValueError) as exc:
                # Some filesystems refuse to mmap. Reading the member back is
                # the whole point of the spill, so fall back rather than fail —
                # at the cost of holding this one member.
                logger.warning(
                    f"Cannot memory-map the spill for npz member '{self.name}' ({exc}); "
                    f"reading it into memory instead"
                )
                self._fp.seek(0)
                count = self.n_rows * max(int(np.prod(self.row_shape, dtype=np.int64)), 1)
                self._mm = np.fromfile(self._fp, dtype=self.dtype, count=count).reshape(self.shape)
        return self._mm

    def close(self) -> None:
        self._mm = None
        with contextlib.suppress(Exception):
            self._fp.close()


class ChunkSpool:
    """Accumulate chunk batches on disk, then write one or more .npz corpora.

    ``save_chunks`` needs the whole corpus as a list before it writes anything,
    which is what makes ``data prepare`` peak at the corpus plus its stacked
    copy (#211). A spool takes the same chunks a batch at a time, spills each
    npz member to its own temp file as it goes, and assembles the .npz at the
    end — so no batch outlives the ``append`` call that delivered it.

    Output is byte-compatible with :func:`save_chunks`: same member names,
    order, dtypes, shapes and values, including the CSR
    ``seq_to_sig_values``/``seq_to_sig_offsets`` pair and the object-array
    fallbacks for ragged chunks. ``tests/test_chunk_writer.py`` holds the two
    writers to that.

    Trade-off: the corpus is written twice (once to the spill, once into the
    .npz) and the spill needs corpus-sized scratch space in ``spill_dir``,
    which defaults to the output directory. That buys back the corpus-sized
    peak in RAM.

    What is *not* spilled: string members are held in memory (a few hundred
    bytes per chunk — they are what the read-level split is computed from), and
    the ragged object-array fallback is buffered like ``save_chunks`` does,
    since a pickled member cannot be appended to.

    Args:
        spill_dir: Directory for the temp files. Must have room for the corpus.
        compressed: Default for :meth:`write_npz`.
        batch_rows: Chunks buffered before a spill write. Callers that append
            one read at a time (the sequential prepare path) would otherwise
            pay a per-member array build per read.

    Examples:
        >>> with ChunkSpool(Path("out")) as spool:  # doctest: +SKIP
        ...     for batch in batches:
        ...         spool.append(batch)
        ...     spool.write_npz(Path("out/all.npz"))
    """

    def __init__(self, spill_dir: Path, *, compressed: bool = True, batch_rows: int = 4096):
        self.spill_dir = Path(spill_dir)
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        self.compressed = compressed
        self.batch_rows = max(1, batch_rows)
        self._order: list[str] = []
        self._flat: dict[str, _SpilledColumn] = {}
        self._text: dict[str, list[np.ndarray]] = {}
        self._object: dict[str, list] = {}
        self._pending: list[dict] = []
        self._n_chunks = 0
        self._csr_total = 0
        self._csr_started = False
        self._closed = False

    # -- accumulation ------------------------------------------------------

    def __len__(self) -> int:
        return self._n_chunks

    @property
    def n_chunks(self) -> int:
        """Chunks appended so far."""
        return self._n_chunks

    def append(self, chunks: list[dict]) -> None:
        """Add a batch of chunks. The caller may drop them immediately after."""
        if self._closed:
            raise ValueError("ChunkSpool is closed")
        if not chunks:
            return
        self._pending.extend(chunks)
        self._n_chunks += len(chunks)
        if len(self._pending) >= self.batch_rows:
            self._flush()

    def _flush(self) -> None:
        """Turn the buffered chunks into npz members and spill them."""
        if not self._pending:
            return
        chunks = self._pending
        self._pending = []
        names = []
        for name, value in iter_chunk_columns(chunks):
            names.append(name)
            self._append_member(name, value)
            del value
        del chunks
        if not self._order:
            self._order = names
        elif names != self._order:
            raise ValueError(
                "chunk batches disagree on the npz members they produce: "
                f"{self._order} then {names}. A spool cannot mix batches whose "
                "chunks carry different fields or different array shapes."
            )

    def _append_member(self, name: str, value: np.ndarray | list) -> None:
        if isinstance(value, list):
            # Ragged fallback: the member is pickled, so it cannot be streamed.
            self._object.setdefault(name, []).extend(value)
            return
        if value.dtype.kind in "US":
            # Fixed-width text: the width is per batch, so these are promoted
            # to the widest at write time rather than spilled at a width that
            # a later batch might outgrow.
            self._text.setdefault(name, []).append(value)
            return
        if name == "seq_to_sig_offsets":
            value = self._continue_csr_offsets(value)
        column = self._flat.get(name)
        if column is None:
            column = _SpilledColumn(name, value.dtype, value.shape[1:], self.spill_dir)
            self._flat[name] = column
        column.append(value)

    def _continue_csr_offsets(self, local: np.ndarray) -> np.ndarray:
        """Rebase one batch's CSR row offsets onto the values already spilled."""
        base = self._csr_total
        if self._csr_started:
            rebased = local[1:] + base
        else:
            # The first batch keeps its leading 0, which is row 0's start.
            self._csr_started = True
            rebased = local
        self._csr_total = base + int(local[-1])
        return rebased

    # -- reading back ------------------------------------------------------

    def text_column(self, name: str) -> np.ndarray:
        """One text member as a single array, at the width the .npz will use."""
        self._flush()
        parts = self._text[name]
        if len(parts) > 1:
            dtype = parts[0].dtype
            for part in parts[1:]:
                dtype = np.promote_types(dtype, part.dtype)
            joined = np.concatenate([p.astype(dtype, copy=False) for p in parts])
            # Collapse in place: the per-batch arrays are a second copy.
            parts.clear()
            parts.append(joined)
        return parts[0]

    def read_ids(self) -> np.ndarray:
        """The ``read_ids`` column, for computing a read-level split."""
        return self.text_column("read_ids")

    # -- writing -----------------------------------------------------------

    def write_npz(
        self,
        output_path: Path,
        *,
        rows: np.ndarray | None = None,
        compressed: bool | None = None,
    ) -> int:
        """Write the spooled chunks to ``output_path`` as an .npz.

        Args:
            output_path: Output file path (.npz appended if absent).
            rows: Row indices to write, in output order. ``None`` writes every
                row in arrival order. Used to split a spooled corpus into
                train/val/test without ever materialising a split as a list.
            compressed: Overrides the spool's default.

        Returns:
            Number of chunks written.

        Raises:
            ValueError: If the spool is empty or a member has the wrong length.
        """
        if self._closed:
            raise ValueError("ChunkSpool is closed")
        if self._n_chunks == 0:
            raise ValueError("No chunks to save")
        self._flush()
        self._check_lengths()
        if compressed is None:
            compressed = self.compressed
        if rows is not None:
            rows = np.asarray(rows, dtype=np.int64)
        n_written = self._n_chunks if rows is None else len(rows)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        csr = self._csr_gather(rows) if rows is not None else None
        with open_npz_zip(output_path, compressed) as zf:
            for name in self._order:
                if csr is not None and name in ("seq_to_sig_values", "seq_to_sig_offsets"):
                    self._write_csr_member(zf, name, csr)
                elif name in self._object:
                    self._write_object_member(zf, name, rows)
                elif name in self._text:
                    self._write_text_member(zf, name, rows)
                else:
                    self._write_flat_member(zf, name, rows)

        logger.info(f"Saved {n_written} chunks to {output_path}")
        return n_written

    def _check_lengths(self) -> None:
        """Every per-chunk member must have one row per chunk before we write."""
        expected = self._n_chunks
        for name, column in self._flat.items():
            if name == "seq_to_sig_values":
                continue  # flat CSR values, one row per map entry
            want = expected + 1 if name == "seq_to_sig_offsets" else expected
            if column.n_rows != want:
                raise ValueError(f"npz member '{name}' has {column.n_rows} rows, expected {want}")
        for name in self._text:
            got = sum(len(p) for p in self._text[name])
            if got != expected:
                raise ValueError(f"npz member '{name}' has {got} rows, expected {expected}")
        for name, values in self._object.items():
            if len(values) != expected:
                raise ValueError(f"npz member '{name}' has {len(values)} rows, expected {expected}")

    def _write_flat_member(self, zf: zipfile.ZipFile, name: str, rows: np.ndarray | None) -> None:
        column = self._flat[name]
        if rows is None:
            with zf.open(name + ".npy", "w", force_zip64=True) as fp:
                write_npy_header(fp, column.shape, column.dtype)
                column.stream_to(fp)
            return
        view = column.view()
        with zf.open(name + ".npy", "w", force_zip64=True) as fp:
            write_npy_header(fp, (len(rows), *column.row_shape), column.dtype)
            step = _row_step(column.dtype, column.row_shape)
            for start in range(0, len(rows), step):
                fp.write(np.asarray(view[rows[start : start + step]]).tobytes())

    def _write_text_member(self, zf: zipfile.ZipFile, name: str, rows: np.ndarray | None) -> None:
        parts = self._text[name]
        dtype = parts[0].dtype
        for part in parts[1:]:
            dtype = np.promote_types(dtype, part.dtype)
        if rows is None:
            with zf.open(name + ".npy", "w", force_zip64=True) as fp:
                write_npy_header(fp, (sum(len(p) for p in parts),), dtype)
                for part in parts:
                    _write_rows(fp, np.ascontiguousarray(part.astype(dtype, copy=False)))
            return
        selected = self.text_column(name)[rows]
        dtype = _text_dtype(selected)
        with zf.open(name + ".npy", "w", force_zip64=True) as fp:
            write_npy_header(fp, (len(rows),), dtype)
            _write_rows(fp, np.ascontiguousarray(selected.astype(dtype, copy=False)))

    def _write_object_member(self, zf: zipfile.ZipFile, name: str, rows: np.ndarray | None) -> None:
        values = self._object[name]
        if rows is not None:
            values = [values[i] for i in rows.tolist()]
        write_npz_member(zf, name, np.array(values, dtype=object))

    def _csr_gather(self, rows: np.ndarray) -> dict:
        """Row lengths and output offsets for a gathered subset of the CSR pair."""
        offsets = np.asarray(self._flat["seq_to_sig_offsets"].view())
        return {
            "offsets": offsets,
            "out_offsets": csr_offsets_from_lens(offsets[rows + 1] - offsets[rows]),
            "rows": rows,
        }

    def _write_csr_member(self, zf: zipfile.ZipFile, name: str, csr: dict) -> None:
        if name == "seq_to_sig_offsets":
            write_npz_member(zf, name, csr["out_offsets"])
            return
        values = self._flat["seq_to_sig_values"].view()
        rows = csr["rows"]
        offsets = csr["offsets"]
        total = int(csr["out_offsets"][-1])
        with zf.open(name + ".npy", "w", force_zip64=True) as fp:
            write_npy_header(fp, (total,), values.dtype)
            mean_row_bytes = max(1, (total // max(len(rows), 1)) * values.dtype.itemsize)
            step = max(1, _COPY_SLICE_BYTES // mean_row_bytes)
            for start in range(0, len(rows), step):
                _lens, _col, src = csr_gather_index(offsets, rows[start : start + step])
                fp.write(np.asarray(values[src]).tobytes())

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Drop the spill files. The spool cannot be used afterwards."""
        if self._closed:
            return
        self._closed = True
        self._pending.clear()
        for column in self._flat.values():
            column.close()
        self._flat.clear()
        self._text.clear()
        self._object.clear()

    def __enter__(self) -> "ChunkSpool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ChunkNpzWriter:
    """Write one .npz from chunk batches, without ever holding the corpus.

    The streaming counterpart of :func:`save_chunks`, for callers that produce
    chunks a batch at a time (``data prepare``). Output is byte-compatible; see
    :class:`ChunkSpool` for the mechanics and the disk-space trade-off.

    Args:
        output_path: Output file path (.npz appended if absent).
        compressed: If True (default), compress members with zlib.
        spill_dir: Where the temp files go. Defaults to the output directory.
        batch_rows: Chunks buffered before a spill write.

    Examples:
        >>> with ChunkNpzWriter(Path("out/all.npz")) as writer:  # doctest: +SKIP
        ...     for batch in batches:
        ...         writer.append(batch)
    """

    def __init__(
        self,
        output_path: Path,
        *,
        compressed: bool = True,
        spill_dir: Path | None = None,
        batch_rows: int = 4096,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._spool = ChunkSpool(
            spill_dir if spill_dir is not None else self.output_path.parent,
            compressed=compressed,
            batch_rows=batch_rows,
        )

    def append(self, chunks: list[dict]) -> None:
        """Add a batch of chunks. The caller may drop them immediately after."""
        self._spool.append(chunks)

    @property
    def n_chunks(self) -> int:
        return self._spool.n_chunks

    def close(self) -> None:
        """Write the .npz and drop the spill files."""
        try:
            self._spool.write_npz(self.output_path)
        finally:
            self._spool.close()

    def __enter__(self) -> "ChunkNpzWriter":
        return self

    def __exit__(self, exc_type, *_exc) -> None:
        if exc_type is None:
            self.close()
        else:
            self._spool.close()


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
