"""Columnar chunk metadata.

:func:`~leech.chunking.serialization.load_chunks` builds one dict per chunk.
Measured at 780 bytes each — with the repeated strings already interned — that
is 5.2 GB for a 6.7M-chunk corpus, spent on dicts whose values are a handful of
small integers and a few hundred distinct strings (#211).

:class:`ChunkTable` keeps the npz's own arrays as columns and materialises a row
view only when something asks for a chunk, which costs roughly 130 bytes per
chunk for the same access patterns. Text is held as fixed-width bytes rather
than as ``<U`` (numpy stores those as UTF-32, four bytes per character) and
integers are narrowed to the smallest dtype that holds their range.

A row is a read-only ``Mapping``, so consumers that already do
``chunk["label_int"]``, ``chunk.get("source_group")`` or
``"feature_start" in chunk`` work unchanged. Fields the file does not carry are
absent rather than None, which is what those ``in`` tests are asking about.
"""

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import numpy as np

from leech.chunking.serialization import iter_npz_row_blocks, npz_array_members

#: npz member -> (chunk field, is the negative value a missing marker)
_INT_FIELDS: tuple[tuple[str, str, bool], ...] = (
    ("base_indices", "base_idx", False),
    ("labels_int", "label_int", True),
    ("feature_starts", "feature_start", False),
    ("feature_ends", "feature_end", False),
    ("cl_values", "cl_value", True),
    ("focus_signal_pos", "focus_signal_pos", False),
)

#: npz member -> (chunk field, does the empty string mean None)
_TEXT_FIELDS: tuple[tuple[str, str, bool], ...] = (
    ("sequences", "sequence", False),
    ("read_ids", "read_id", False),
    ("labels", "label", True),
    ("source_groups", "source_group", True),
    # An absent reference name reads back as "" rather than None, matching
    # load_chunks — callers concatenate it into report keys.
    ("reference_names", "reference_name", False),
    ("sequences_with_kmer_context", "sequence_with_kmer_context", True),
)


class _Column:
    """One metadata field across every chunk."""

    __slots__ = ()

    def value(self, index: int):
        raise NotImplementedError

    def take(self, rows: np.ndarray) -> "_Column":
        raise NotImplementedError

    @property
    def raw(self) -> np.ndarray | None:
        """The underlying array, without the missing-value translation."""
        return None


class _IntColumn(_Column):
    __slots__ = ("values", "none_if_negative")

    def __init__(self, values: np.ndarray, none_if_negative: bool):
        self.values = values
        self.none_if_negative = none_if_negative

    def value(self, index: int):
        number = int(self.values[index])
        if self.none_if_negative and number < 0:
            return None
        return number

    def take(self, rows: np.ndarray) -> "_IntColumn":
        return _IntColumn(self.values[rows], self.none_if_negative)

    @property
    def raw(self) -> np.ndarray:
        return self.values


class _TextColumn(_Column):
    """Text as fixed-width bytes, decoded on access.

    Decoding allocates a str per read, but a transient one: the dicts this
    replaces held every one of them for the life of the dataset.
    """

    __slots__ = ("values", "none_if_empty", "_encoded")

    def __init__(self, values: np.ndarray, none_if_empty: bool):
        self.values = values
        self.none_if_empty = none_if_empty
        self._encoded = values.dtype.kind == "S"

    def value(self, index: int):
        text = self.values[index]
        text = text.decode() if self._encoded else str(text)
        if self.none_if_empty and not text:
            return None
        return text

    def take(self, rows: np.ndarray) -> "_TextColumn":
        return _TextColumn(self.values[rows], self.none_if_empty)

    @property
    def raw(self) -> np.ndarray:
        return self.values


class _ConstColumn(_Column):
    """A field every chunk shares — currently only ``cl_value = None``."""

    __slots__ = ("constant",)

    def __init__(self, constant):
        self.constant = constant

    def value(self, index: int):
        return self.constant

    def take(self, rows: np.ndarray) -> "_ConstColumn":
        return self


def _narrow_ints(values: np.ndarray) -> np.ndarray:
    """Cast to the smallest signed dtype that holds the column's range."""
    if values.size == 0:
        return values.astype(np.int8)
    low, high = int(values.min()), int(values.max())
    for dtype in (np.int8, np.int16, np.int32):
        info = np.iinfo(dtype)
        if info.min <= low and high <= info.max:
            return values.astype(dtype)
    return values.astype(np.int64)


def _read_text_member(input_path: Path, member: str, shape, dtype) -> np.ndarray:
    """Read a text member as fixed-width bytes, four times smaller than ``<U``.

    Chunk text is ASCII — read ids, base sequences, amino-acid and reference
    names — and numpy stores ``<U`` as UTF-32. Reading the member whole and
    casting would hold both forms at once, which for a 6.7M-chunk corpus is a
    gigabyte of transient per column, so convert a block at a time instead.
    Text that is not ASCII stays unicode rather than failing the load.
    """
    if dtype.kind != "U":
        with np.load(input_path, allow_pickle=False) as data:
            return data[member]

    packed = np.empty(shape[0], dtype=f"S{dtype.itemsize // 4}")
    try:
        for start, blocks in iter_npz_row_blocks(input_path, [member]):
            block = blocks[member]
            packed[start : start + len(block)] = block
    except UnicodeEncodeError:
        with np.load(input_path, allow_pickle=False) as data:
            return data[member]
    return packed


class ChunkRow(Mapping):
    """A single chunk's metadata, read through its table's columns."""

    __slots__ = ("_table", "_index")

    def __init__(self, table: "ChunkTable", index: int):
        self._table = table
        self._index = index

    def __getitem__(self, key: str):
        try:
            column = self._table.columns[key]
        except KeyError:
            raise KeyError(key) from None
        return column.value(self._index)

    def __contains__(self, key: object) -> bool:
        # Mapping's default answers this by reading the value, which for a
        # column means an index and a conversion. Presence is a property of the
        # table, not of the row.
        return key in self._table.columns

    def __iter__(self):
        return iter(self._table.columns)

    def __len__(self) -> int:
        return len(self._table.columns)

    def __repr__(self) -> str:
        return f"ChunkRow({dict(self)!r})"


class ChunkTable(Sequence):
    """Chunk metadata as columns, presented as a sequence of read-only mappings.

    Indexing yields a :class:`ChunkRow`; iterating yields one per chunk. Rows
    are views, so they cost nothing to keep out of and nothing is shared with
    the caller to mutate.

    Examples:
        >>> table = ChunkTable.from_npz(Path("chunks.npz"))  # doctest: +SKIP
        >>> table[0]["label_int"]                            # doctest: +SKIP
        1
        >>> table.values("label_int")  # raw column, for vectorized tallies
        ... # doctest: +SKIP
        array([1, 0, 1, ...], dtype=int8)
    """

    __slots__ = ("columns", "_n")

    def __init__(self, columns: dict[str, _Column], n_chunks: int):
        self.columns = columns
        self._n = n_chunks

    @classmethod
    def from_npz(
        cls,
        input_path: Path,
        *,
        skip: Collection[str] = (),
    ) -> "ChunkTable":
        """Read a corpus's metadata members — never its per-chunk arrays.

        Args:
            input_path: Path to .npz file.
            skip: Chunk field names to leave out. Text the run will not read is
                worth skipping: ``sequence_with_kmer_context`` is 56 bytes a
                chunk that only ``signal_kmer`` encoding touches.

        Returns:
            A table with one row per chunk in file order.
        """
        skip = set(skip)
        columns: dict[str, _Column] = {}
        text_members = npz_array_members(input_path)

        # allow_pickle stays off: every member read here is a plain array, and
        # the one pickled member (legacy seq_to_sig_maps) is never metadata.
        with np.load(input_path, allow_pickle=False) as data:
            n_chunks = len(data["labels_int"])
            has_feature_window = "feature_starts" in data
            present = set(data.files)

            for member, field, none_if_negative in _INT_FIELDS:
                if field in skip or member not in present:
                    continue
                columns[field] = _IntColumn(_narrow_ints(data[member]), none_if_negative)

            # Old corpora carry dwell_margin_lefts instead of the signed window.
            if not has_feature_window and "dwell_margin_lefts" in present:
                columns["dwell_margin_left"] = _IntColumn(
                    _narrow_ints(data["dwell_margin_lefts"]), False
                )

        for member, field, none_if_empty in _TEXT_FIELDS:
            if field in skip or member not in present:
                continue
            shape, dtype = text_members[member]
            columns[field] = _TextColumn(
                _read_text_member(input_path, member, shape, dtype), none_if_empty
            )

        # load_chunks reports cl_value as None when the corpus predates it, and
        # callers read it unguarded.
        if "cl_value" not in columns and "cl_value" not in skip:
            columns["cl_value"] = _ConstColumn(None)

        return cls(columns, n_chunks)

    def select(self, mask: np.ndarray) -> "ChunkTable":
        """Return a table holding only the rows where ``mask`` is True."""
        rows = np.nonzero(mask)[0]
        return ChunkTable({name: col.take(rows) for name, col in self.columns.items()}, len(rows))

    def values(self, field: str) -> np.ndarray | None:
        """The raw column for ``field``, or None if the table lacks it.

        Raw means as stored: text comes back as bytes, and missing integers as
        their negative sentinel rather than as None. Use it to tally a field
        across a whole corpus without building a row per chunk.
        """
        column = self.columns.get(field)
        return None if column is None else column.raw

    def require_values(self, field: str) -> np.ndarray:
        """The raw column for ``field``, raising if the corpus lacks it."""
        column = self.columns.get(field)
        if column is None or column.raw is None:
            raise KeyError(f"chunk metadata has no column '{field}'")
        return column.raw

    def nbytes(self) -> int:
        """Total bytes held by the columns."""
        return sum(col.raw.nbytes for col in self.columns.values() if col.raw is not None)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> ChunkRow:
        if isinstance(index, slice):
            raise TypeError("ChunkTable indexes one chunk at a time; use select() for subsets")
        if index < 0:
            index += self._n
        if not 0 <= index < self._n:
            raise IndexError(f"chunk index out of range: {index}")
        return ChunkRow(self, index)

    def __iter__(self):
        for index in range(self._n):
            yield ChunkRow(self, index)

    def __repr__(self) -> str:
        return f"ChunkTable({self._n} chunks, fields={sorted(self.columns)})"
