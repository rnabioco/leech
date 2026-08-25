"""
PyTorch Dataset classes for leech training data.

Provides efficient loading and batching of training chunks for PyTorch models.
Handles both baseline models (signal + sequence) and full models (signal + sequence + features).

Classes:
    LeechDataset: Main dataset class that loads chunks from .npz files

Functions:
    collate_fn(): Custom collate function for batching variable-length data

Data Format:
    Each training chunk is a dictionary with:
    - signal: Raw signal array (signal_len,)
    - sequence: DNA sequence string (kmer_len bases)
    - features: Engineered features (num_features, kmer_len) - optional
    - label: Binary label (0=uncharged, 1=charged)
    - read_id: Read identifier
    - base_idx: Position within read

Example:
    >>> from leech.dataset import LeechDataset, collate_fn
    >>> from torch.utils.data import DataLoader
    >>>
    >>> # Create dataset
    >>> dataset = LeechDataset("chunks.npz", model_type="TransformerDwell")
    >>> print(f"Dataset size: {len(dataset)}")
    >>>
    >>> # Create DataLoader
    >>> loader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn)
    >>> batch = next(iter(loader))
    >>> print(batch.keys())  # ['signal', 'sequence', 'features', 'label'] for feature models
"""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from leech.chunking import (
    ChunkTable,
    iter_npz_row_blocks,
    load_chunks,
    load_seq_to_sig_csr,
    npz_array_members,
    npz_member_names,
)
from leech.constants import AUTO_DATALOADER_WORKERS
from leech.features import encode_signal_kmer, sequence_to_int
from leech.models.inference_wrapper import ModelInferenceWrapper

if TYPE_CHECKING:
    from leech.confounds import ConfoundEncoder

logger = logging.getLogger("leech.dataset")

# ASCII lookup table for vectorized sequence encoding (A=0, C=1, G=2, T=3, else=255)
_BASE_MAP = np.full(256, 255, dtype=np.uint8)
_BASE_MAP[ord("A")] = 0
_BASE_MAP[ord("C")] = 1
_BASE_MAP[ord("G")] = 2
_BASE_MAP[ord("T")] = 3
_BASE_MAP[ord("a")] = 0
_BASE_MAP[ord("c")] = 1
_BASE_MAP[ord("g")] = 2
_BASE_MAP[ord("t")] = 3

# Byte -> base int for the block-wise signal_kmer path, derived from
# `sequence_to_int` rather than restated so the two cannot drift. Bytes a
# fixed-width bytes column cannot hold (>= 128, which never survive the ASCII
# packing in `_read_text_member`) map to -1, the same "not a base" value.
_BASE_INT_MAP = np.full(256, -1, dtype=np.int8)
_BASE_INT_MAP[:128] = sequence_to_int("".join(map(chr, range(128))))


def _gather_rows(tensor: torch.Tensor, rows: np.ndarray) -> torch.Tensor:
    """Take ``rows`` out of ``tensor`` along dim 0, off the ATen thread pool.

    ``tensor[torch.as_tensor(rows)]`` dispatches to ``index_select``, which
    splits the copy across ``at::parallel_for``. Measured on a 200k-chunk
    corpus, batch 256: 0.12 ms per batch on an idle node against 0.25 ms for
    the numpy gather — and **115 ms** on the same node under load, because
    every batch pays to hand a memcpy to as many threads as the machine has
    cores, against a run queue that already has work on it. A training loop
    contends with its own forward and backward for that pool, so take the
    version whose cost does not depend on who else is running.
    """
    return torch.from_numpy(tensor.numpy()[rows])


def _byte_matrix(column: np.ndarray | None) -> np.ndarray | None:
    """A ``(rows, width)`` uint8 view of a fixed-width bytes column, or None.

    None means the column is absent or held as unicode — the fallback
    `ChunkTable` takes for text that is not ASCII — and the caller has to read
    it one row at a time.
    """
    if column is None or column.dtype.kind != "S":
        return None
    width = column.dtype.itemsize
    # `view(np.uint8)` resolves to an overload whose dtype parameter numpy types
    # as `type[uint8] | property`, which is not an `np.dtype` — passing the
    # dtype instance picks the overload that is.
    flat = np.ascontiguousarray(column).view(np.dtype(np.uint8))
    return flat.reshape(len(column), width)


# Models that require dwell/signal features as third input
FEATURE_MODELS = ModelInferenceWrapper.FEATURE_MODELS

# Models that receive the full dwell margin (no dwell_offset slicing)
WIDE_FEATURE_MODELS = ModelInferenceWrapper.WIDE_FEATURE_MODELS


# =============================================================================
# Dwell template helpers — shared between training (LeechDataset.__getitem__)
# and inference (run_bundle_inference / run_inference chunk consumption).
#
# The template append is LABEL-INDEPENDENT: it computes `dwell / expected_dwell
# [AA_i, pos]` for all 20 AAs per feature position without ever looking at the
# ground-truth label, so the same routine works verbatim at inference time on
# models that were trained with templates.
# =============================================================================


def load_dwell_template_table(
    table_path: Path,
) -> tuple[np.ndarray, int, list[str]]:
    """Load per-AA per-position expected dwell from a TSV.

    Args:
        table_path: Path to TSV with columns ``aa``, ``position``, ``dwell_mean``.

    Returns:
        (template, template_min_pos, aa_order) where:
          - template: ndarray of shape (n_aa, n_positions), indexed by
            ``[aa_idx, pos - template_min_pos]``.
          - template_min_pos: integer offset such that column 0 corresponds to
            position ``template_min_pos``.
          - aa_order: list of amino-acid three-letter codes, sorted
            alphabetically. Index into this list to interpret the first axis of
            ``template``.
    """
    df = pl.read_csv(table_path, separator="\t")
    aa_list = sorted(df["aa"].unique().to_list())
    n_aa = len(aa_list)
    aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}

    min_pos = int(df["position"].min())
    max_pos = int(df["position"].max())

    # Grand mean dwell at each position (fallback for missing entries)
    grand_mean = dict(df.group_by("position").agg(pl.col("dwell_mean").mean()).iter_rows())

    n_positions = max_pos - min_pos + 1
    template = np.ones((n_aa, n_positions), dtype=np.float32)
    for row in df.iter_rows(named=True):
        ai = aa_to_idx[row["aa"]]
        pi = int(row["position"]) - min_pos
        template[ai, pi] = row["dwell_mean"]

    # Fill missing entries with grand mean
    for pos in range(min_pos, max_pos + 1):
        pi = pos - min_pos
        gm = grand_mean.get(pos, 1.0)
        for ai in range(n_aa):
            if template[ai, pi] <= 0:
                template[ai, pi] = gm

    logger.info(
        f"Loaded dwell template table: {n_aa} AAs, "
        f"positions {min_pos} to {max_pos} from {table_path}"
    )
    return template, min_pos, aa_list


def append_dwell_template_channels(
    features: np.ndarray,
    *,
    feat_start: int,
    dwell_templates: np.ndarray,
    template_min_pos: int,
) -> np.ndarray:
    """Append N_AA dwell ratio channels to a feature array.

    For each of the AAs in ``dwell_templates`` computes
    ``dwell / expected_dwell[AA_i, pos]`` at each feature position, where
    raw dwell is expected to be channel 0 of ``features``. Positions not
    covered by the template use 1.0 (unit ratio, no information).

    Args:
        features: Base features of shape ``(num_features, feat_width)``.
        feat_start: Position of column 0 of ``features`` in the same coordinate
            frame used to build the template (typically a signed offset from
            the focus base — matches ``chunk["feature_start"]``).
        dwell_templates: (n_aa, n_template_positions) array from
            ``load_dwell_template_table``.
        template_min_pos: ``template_min_pos`` from ``load_dwell_template_table``.

    Returns:
        Expanded feature array of shape ``(num_features + n_aa, feat_width)``.
    """
    raw_dwell = features[0]  # (feat_width,)
    feat_width = features.shape[1]
    n_aa = dwell_templates.shape[0]
    template_channels = np.ones((n_aa, feat_width), dtype=np.float32)

    for fi in range(feat_width):
        pos = feat_start + fi
        ti = pos - template_min_pos
        dwell_val = raw_dwell[fi]
        if dwell_val <= 0:
            continue  # no dwell data at this position, leave as 1.0
        if 0 <= ti < dwell_templates.shape[1]:
            expected = dwell_templates[:, ti]
            safe_expected = np.where(expected > 0, expected, 1.0)
            template_channels[:, fi] = dwell_val / safe_expected
        # else: outside transit table range, leave as 1.0

    return np.concatenate([features, template_channels], axis=0)


# Rows per block when expanding the CSR base-to-signal maps. Keeps the gather
# index arrays to tens of MB regardless of corpus size.
_S2S_BLOCK_ROWS = 65536


class _TensorFill:
    """Accumulate per-chunk tensors into one preallocated contiguous tensor.

    ``torch.stack`` allocates the whole output *while the input list is still
    alive*, so stacking N chunk tensors peaks at twice the output — 33 GB of
    transient on a large corpus (#211). The output shape is known before the
    loop, so fill a buffer allocated up front instead and peak at once the
    output.

    Chunk tensors whose shape disagrees with the first one fall back to the old
    list, which ``__getitem__`` still handles.
    """

    #: Rows staged before one bulk write. Assigning row by row costs a few
    #: microseconds of dispatch each — seconds over a multi-million-chunk
    #: corpus — while ``torch.stack(..., out=)`` writes a run at C speed with
    #: no transient. Small enough that what it pins is irrelevant.
    _BATCH_ROWS = 256

    def __init__(self, name: str, capacity: int):
        self.name = name
        self.capacity = capacity
        self._tensor: torch.Tensor | None = None
        self._items: list[torch.Tensor] = []
        self._pending: list[torch.Tensor] = []
        self._n = 0

    def append(self, tensor: torch.Tensor) -> None:
        if self._items:  # already degraded to list access
            self._items.append(tensor)
            return
        if self._tensor is None:
            self._tensor = torch.empty((self.capacity, *tensor.shape), dtype=tensor.dtype)
        elif (
            tuple(tensor.shape) != tuple(self._tensor.shape[1:])
            or tensor.dtype != self._tensor.dtype
        ):
            self._degrade(tensor)
            return
        self._pending.append(tensor)
        if len(self._pending) >= self._BATCH_ROWS:
            self._flush()

    def extend(self, batch: torch.Tensor) -> None:
        """Write a whole ``(rows, *row_shape)`` batch into the buffer at once.

        The block-wise filler prepares a row block in one vectorized go, so it
        already holds the rows contiguously — there is nothing to stage and
        nothing to stack. Interleaves with :meth:`append`: whatever is pending
        is flushed first, so rows land in call order either way.

        The copy goes through numpy rather than ``Tensor.copy_`` for the same
        reason :func:`_gather_rows` does: the ATen copy splits a memcpy across
        ``at::parallel_for``, and when the thread pool is contended — grid
        search runs one of these processes per worker — that turns a block
        copy into a scheduling problem. Measured on a loaded node, it was the
        difference between the block-wise filler winning 2x and losing.
        """
        rows = int(batch.shape[0])
        if self._items:  # already degraded to list access
            self._items.extend(batch[i].clone() for i in range(rows))
            return
        if self._tensor is None:
            self._tensor = torch.empty((self.capacity, *batch.shape[1:]), dtype=batch.dtype)
        elif (
            tuple(batch.shape[1:]) != tuple(self._tensor.shape[1:])
            or batch.dtype != self._tensor.dtype
        ):
            self._degrade_batch(batch)
            return
        self._flush()
        np.copyto(self._tensor[self._n : self._n + rows].numpy(), batch.numpy())
        self._n += rows

    def _flush(self) -> None:
        if not self._pending:
            return
        rows = len(self._pending)
        torch.stack(self._pending, out=self._tensor[self._n : self._n + rows])
        self._n += rows
        self._pending.clear()

    def _degrade(self, tensor: torch.Tensor) -> None:
        """Fall back to a list of per-chunk tensors, keeping what was filled."""
        self._start_list(tuple(tensor.shape))
        self._items.append(tensor)

    def _degrade_batch(self, batch: torch.Tensor) -> None:
        """:meth:`_degrade` for a whole block."""
        self._start_list(tuple(batch.shape[1:]))
        self._items.extend(batch[i].clone() for i in range(int(batch.shape[0])))

    def _start_list(self, shape: tuple[int, ...]) -> None:
        logger.warning(
            "%s shapes differ (%s vs %s), falling back to list access",
            self.name,
            shape,
            tuple(self._tensor.shape[1:]),
        )
        self._flush()
        self._items = [self._tensor[i].clone() for i in range(self._n)]
        self._tensor = None

    def finish(self) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
        """Return ``(stacked_tensor, fallback_list)``; exactly one is populated."""
        if self._tensor is None:
            # Nothing appended (the caller uses a different tensor) or shapes
            # differed — both cases the old _try_stack signalled with None.
            return None, self._items
        self._flush()
        return self._tensor[: self._n], []


@dataclass
class _ArrayStream:
    """Row-block source for the per-chunk arrays of an npz corpus.

    Reading the arrays this way — instead of through ``load_chunks``, which
    materialises every member — keeps the numpy source out of the peak: only
    one block per member is resident while the output tensors are filled.

    Attributes:
        path: The npz being streamed.
        members: Chunk field name -> npz member name, for the fields this run
            actually consumes. Members nothing reads are never decompressed.
        keep: One bool per npz row: False for rows dropped by label filtering,
            so the stream stays aligned with ``LeechDataset.chunks``.
        dwell_width: ``len(chunk["dwell"])``, constant in the flat format and
            read from the member header — the only thing the tensorize loop
            needs from the dwells, so that member is never read.
        row_shapes: Chunk field name -> the shape of one row of that member,
            also from the header. Lets the block-wise filler decide up front
            whether a whole corpus can be vectorized without reading a byte
            of it.
    """

    path: Path
    members: dict[str, str]
    dwell_width: int | None
    row_shapes: dict[str, tuple[int, ...]] = dataclass_field(default_factory=dict)
    keep: np.ndarray | None = None

    @classmethod
    def build(
        cls,
        path: Path,
        *,
        needs_features: bool,
        wants_residual: bool,
    ) -> "_ArrayStream | None":
        """Return a stream for ``path``, or None if it cannot be streamed.

        Corpora written before the flat format store the arrays as pickled
        object members; those still take the eager path.
        """
        members = npz_array_members(path)
        if "signals_flat" not in members:
            return None
        wanted = {"signal": "signals_flat"}
        if wants_residual:
            if "signal_residuals_flat" in members:
                wanted["signal_residual"] = "signal_residuals_flat"
            elif "signal_residuals" in npz_member_names(path):
                return None  # residuals present but not streamable
        if needs_features:
            # dwells_flat carries no values the loop needs, only its width —
            # but without it there is no way to know that width without reading
            # the per-chunk dwells, so fall back.
            if "features_flat" not in members or "dwells_flat" not in members:
                return None
            wanted["features"] = "features_flat"
        dwell_width = members["dwells_flat"][0][1] if "dwells_flat" in members else None
        return cls(
            path=path,
            members=wanted,
            dwell_width=dwell_width,
            row_shapes={field: tuple(members[member][0][1:]) for field, member in wanted.items()},
        )

    def __iter__(self):
        """Yield one dict of arrays per kept row, in ``LeechDataset.chunks`` order.

        Rows are copied out of the block buffer, which the next block read
        overwrites. Without the copy the caller would have to consume each row
        before the next one arrives — and it does not: preparing a chunk can
        return a tensor that aliases its input (a contiguous crop is a view),
        and those are staged in batches. Copying a few KB per row is cheaper
        than the alternatives and removes the lifetime question entirely.
        """
        if self.keep is None:
            raise RuntimeError("_ArrayStream.keep must be set before iterating")
        for start, blocks in iter_npz_row_blocks(self.path, list(self.members.values())):
            rows = len(next(iter(blocks.values())))
            for j in np.nonzero(self.keep[start : start + rows])[0]:
                yield {field: blocks[member][j].copy() for field, member in self.members.items()}

    def blocks(self):
        """Yield ``(out_start, arrays)`` a row block at a time, kept rows only.

        The same rows :meth:`__iter__` yields, in the same order, handed over
        as ``(rows, ...)`` arrays instead of one row at a time. Consuming them
        block-wise is what keeps the tensorize loop off the 72 µs/chunk path:
        the per-row copy, stack, encode and ``torch.tensor`` calls all become
        one vectorized call per block.

        ``out_start`` is the row's index in ``LeechDataset.chunks``, which the
        ``keep`` mask makes different from its npz row. The arrays are fresh
        (the fancy index copies), so unlike :meth:`__iter__` there is no
        recycled buffer to worry about within a block — but the *next* block
        still invalidates nothing the caller kept, because nothing is shared.
        """
        if self.keep is None:
            raise RuntimeError("_ArrayStream.keep must be set before iterating")
        out_start = 0
        for start, blocks in iter_npz_row_blocks(self.path, list(self.members.values())):
            rows = len(next(iter(blocks.values())))
            selected = np.nonzero(self.keep[start : start + rows])[0]
            if len(selected) == 0:
                continue
            yield (
                out_start,
                {field: blocks[member][selected] for field, member in self.members.items()},
            )
            out_start += len(selected)


def _expand_seq_to_sig_csr(
    values: np.ndarray,
    offsets: np.ndarray,
    rows: np.ndarray,
    *,
    signal_len: int,
    crop_starts: np.ndarray | None,
) -> np.ndarray:
    """Expand CSR base-to-signal maps into one padded ``(len(rows), max_len)`` array.

    Args:
        values: Flat concatenated map values.
        offsets: Row offsets, one per row plus a final total.
        rows: npz row indices to expand, ascending.
        signal_len: Padding value — the encoder's ``sig_start < signal_len``
            test fails at padded positions, so they contribute nothing.
        crop_starts: Per-row signal offset to subtract (asymmetric crop), or
            None to keep the stored coordinates.

    Returns:
        int64 array of shape ``(len(rows), max_len)``.
    """
    lens = (offsets[rows + 1] - offsets[rows]).astype(np.int64)
    n = len(rows)
    max_len = int(lens.max()) if n else 0
    padded = np.full((n, max_len), signal_len, dtype=np.int64)
    starts = offsets[rows]

    # Blocked so the gather indices stay small on a multi-million-chunk corpus.
    for block_start in range(0, n, _S2S_BLOCK_ROWS):
        block_end = min(block_start + _S2S_BLOCK_ROWS, n)
        block_lens = lens[block_start:block_end]
        total = int(block_lens.sum())
        if total == 0:
            continue
        row_idx = np.repeat(np.arange(block_start, block_end), block_lens)
        col_idx = np.arange(total) - np.repeat(np.cumsum(block_lens) - block_lens, block_lens)
        gathered = values[np.repeat(starts[block_start:block_end], block_lens) + col_idx].astype(
            np.int64
        )
        if crop_starts is not None:
            gathered -= np.repeat(crop_starts[block_start:block_end], block_lens)
            np.clip(gathered, 0, signal_len, out=gathered)
        padded[row_idx, col_idx] = gathered
    return padded


class LeechDataset(Dataset):
    """
    PyTorch Dataset for leech training chunks.

    Handles loading and preprocessing of training data.
    """

    def __init__(
        self,
        chunk_path: Path | None = None,
        signal_len: int = 400,
        kmer_len: int = 11,
        model_type: str = "ConvLSTMDwell",
        dwell_offset: int = 0,
        chunks: list[dict] | None = None,
        augmentation: dict | None = None,
        seq_encoding: str = "signal_kmer",
        signal_kmer_context: tuple[int, int] = (4, 4),
        left_context: int | None = None,
        right_context: int | None = None,
        confound_encoder: "ConfoundEncoder | None" = None,
        cl_regression: bool = False,
        signal_mode: str = "both",
        time_mask_bases: int = 0,
        time_mask_count: int = 1,
        shift_max_bases: float = 0.0,
        feature_noise_scale: float = 0.0,
        dwell_template_table: str | Path | None = None,
    ):
        """
        Initialize dataset.

        Args:
            chunk_path: Path to .npz file with training chunks
            signal_len: Expected signal length (will pad/truncate)
            kmer_len: Expected k-mer length
            model_type: Model architecture name (e.g., "ConvLSTMDwell", "TransformerDwell")
            dwell_offset: Shift dwell/feature window toward 3' end (bases).
                Compensates for physical offset between motor protein and
                sensing region. Requires feature_left >= kmer_context + offset.
            chunks: Pre-loaded list of chunk dicts. When provided, chunk_path is
                ignored and no disk I/O occurs (useful for grid search caching).
            augmentation: Signal augmentation config dict. Keys:
                - jitter_std (float): Gaussian noise std dev (0 = disabled)
                - scale_range (tuple[float, float]): Random scale range (1.0, 1.0 = disabled)
            left_context: Left signal context (samples before focus base).
                When both left_context and right_context are provided, crop
                asymmetrically around the focus base instead of center-cropping.
            right_context: Right signal context (samples after focus base).
            confound_encoder: Optional :class:`~leech.confounds.ConfoundEncoder`
                that maps each chunk to an integer confound class. When provided,
                each batch includes a ``confound_label`` tensor for adversarial
                training. Chunks whose confound value is unknown get ``-1``
                (ignored by the CE loss via ``ignore_index=-1``).
            cl_regression: When True, include ``cl_target`` in each batch
                (cl_value / 255.0 in [0,1]; sentinel -1.0 for missing).
            time_mask_bases: Max width in bases for time masking (0 = disabled).
                Zeros out contiguous blocks across signal, features, and sequence.
            time_mask_count: Number of time masks to apply per sample.
            shift_max_bases: Max cross-layer shift in bases (0 = disabled).
                Simulates motif anchor offset, applied consistently to all branches.
            feature_noise_scale: Per-channel Gaussian noise multiplier (0 = disabled).
                Noise std = feature_noise_scale * per-channel empirical std.
            dwell_template_table: Path to TSV with per-AA per-position expected
                dwell times. When provided, 20 dwell ratio channels are appended
                to features: ``dwell / expected_dwell[AA_i, pos]`` for each of
                20 AAs. The correct AA's channel has ratio closest to 1.0.
                TSV columns: aa, position, dwell_mean, ...
        """
        self.chunk_path = chunk_path
        self.signal_len = signal_len
        self.kmer_len = kmer_len
        self.model_type = model_type
        self.dwell_offset = dwell_offset
        self.augmentation = augmentation
        self.seq_encoding = seq_encoding
        self.signal_kmer_context = signal_kmer_context
        self.left_context = left_context
        self.right_context = right_context
        self._time_mask_bases = time_mask_bases
        self._time_mask_count = time_mask_count
        self._shift_max_bases = shift_max_bases
        self._feature_noise_scale = feature_noise_scale

        # Load dwell template table for 20-channel AA template features
        self._dwell_templates: np.ndarray | None = None
        self._template_aa_order: list[str] = []
        if dwell_template_table is not None:
            self._load_dwell_templates(Path(dwell_template_table))

        self._needs_features = model_type in FEATURE_MODELS

        # Use pre-loaded chunks or load from file. Loading from a path streams
        # the per-chunk arrays out of the npz a row block at a time instead of
        # holding a full numpy copy alongside the tensors built from it (#211);
        # pre-loaded chunks have already paid that cost.
        self._array_stream: _ArrayStream | None = None
        self._npz_members: set[str] = set()
        self._s2s_csr: tuple[np.ndarray, np.ndarray] | None = None
        self._s2s_rows: np.ndarray | None = None
        if chunks is not None:
            logger.info(f"Using {len(chunks)} pre-loaded chunks (skipping disk I/O)")
            self.chunks = chunks
        elif chunk_path is not None:
            self._load_from_path(
                Path(chunk_path), signal_mode=signal_mode, seq_encoding=seq_encoding
            )
        else:
            raise ValueError("Either chunk_path or chunks must be provided")

        # Filter chunks with valid numeric labels (label_int). The columnar
        # path applied this at load, recording the same mask so npz rows stay
        # aligned with self.chunks — a mismatch here would pair signals with
        # the wrong labels silently.
        if not isinstance(self.chunks, ChunkTable):
            self.chunks = [c for c in self.chunks if c["label_int"] is not None]

        if len(self.chunks) == 0:
            raise ValueError(f"No valid chunks found{f' in {chunk_path}' if chunk_path else ''}")

        # Pre-tensorize: encode sequences, labels, signals, and features once.
        # Each accumulator fills one preallocated contiguous tensor — see
        # _TensorFill for why stacking a list instead doubles the peak.
        n_chunks = len(self.chunks)
        fill_encoded_seqs = _TensorFill("Encoded-sequence", n_chunks)
        fill_labels = _TensorFill("Label", n_chunks)
        fill_signals = _TensorFill("Signal", n_chunks)
        fill_features = _TensorFill("Feature", n_chunks)
        fill_confounds = _TensorFill("Confound", n_chunks)
        fill_cl_targets = _TensorFill("CL target", n_chunks)
        self._encoded_seqs: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._signals: list[torch.Tensor] = []
        self._features: list[torch.Tensor] = []
        self._confound_encoder = confound_encoder
        self._has_confound = confound_encoder is not None
        self._confound_labels: list[torch.Tensor] = []
        self._cl_regression = cl_regression
        self._cl_targets: list[torch.Tensor] = []

        # Determine effective encoding: fall back to base_onehot if chunks lack signal_kmer data
        self._effective_seq_encoding = seq_encoding
        if seq_encoding == "signal_kmer":
            first = self.chunks[0]
            if self._s2s_csr is not None:
                # Streaming path: the maps were deferred out of the chunk dicts,
                # so ask the CSR arrays whether the first chunk has one.
                _offsets = self._s2s_csr[1]
                _row = self._s2s_rows[0]
                has_seq_to_sig = bool(_offsets[_row + 1] > _offsets[_row])
            else:
                has_seq_to_sig = first.get("seq_to_sig_map") is not None
            if not has_seq_to_sig or not first.get("sequence_with_kmer_context"):
                logger.warning(
                    "Chunks lack seq_to_sig_map/sequence_with_kmer_context; "
                    "falling back to base_onehot encoding"
                )
                self._effective_seq_encoding = "base_onehot"
                self._s2s_csr = None

        # Detect signal_residual channel and apply signal_mode
        self._signal_mode = signal_mode
        if self._array_stream is not None:
            # Arrays were deferred, so presence is a property of the file.
            self._has_signal_residual = bool(
                {"signal_residuals_flat", "signal_residuals"} & self._npz_members
            )
        else:
            self._has_signal_residual = self.chunks[0].get("signal_residual") is not None
        if signal_mode == "both" and self._has_signal_residual:
            self.signal_channels = 2
        else:
            self.signal_channels = 1

        # Detect multi-class: if any label_int > 1, use long dtype for
        # CrossEntropyLoss. Off the column when there is one — the generator
        # form builds a row view per chunk, 150 ms per 200k chunks for a max.
        if isinstance(self.chunks, ChunkTable):
            max_label = int(self.chunks.require_values("label_int").max())
        else:
            max_label = max(c["label_int"] for c in self.chunks)
        self._multiclass = max_label > 1

        # For signal_kmer encoding, the on-the-fly inputs are ~30 B/chunk
        # (seq_ints + seq_to_sig_map) vs ~77 KB/chunk for the encoded output
        # — a >2000x reduction. We stash the compact inputs here and run
        # `encode_signal_kmer` lazily in __getitem__, where DataLoader workers
        # parallelize it behind prefetch. base_onehot stays eagerly tensorized
        # (88 floats/chunk; not worth deferring).
        self._seq_ints: list[np.ndarray] = []
        self._seq_to_sig: list[np.ndarray] = []
        self._seq_ints_tensor: torch.Tensor | None = None

        # Arrays come either from the chunk dicts (pre-loaded / legacy corpus)
        # or a row-block stream over the npz. Both fill the same accumulators.
        # A streamed corpus is handled a whole block at a time: the arrays
        # arrive in blocks of ~1,500 rows, and taking them apart to prepare one
        # row at a time cost 58-67 us per chunk — six to seven minutes of
        # single-threaded startup on a 6.7M-chunk corpus, none of it I/O.
        fills = {
            "signals": fill_signals,
            "encoded_seqs": fill_encoded_seqs,
            "labels": fill_labels,
            "features": fill_features,
            "confounds": fill_confounds,
            "cl_targets": fill_cl_targets,
        }
        if self._block_fill_supported():
            self._fill_from_blocks(fills)
        else:
            self._fill_from_rows(fills)

        # Every per-chunk tensor now lives in one contiguous buffer. That isn't
        # just for cache friendliness — it's required for fork-safety. A
        # DataLoader with num_workers > 0 forks worker processes that COW-inherit
        # the parent's address space. CPython refcounts live inside each
        # PyObject header, so a worker iterating a list of N tensors writes to N
        # separate page-resident headers and faults every page into a private
        # copy, multiplying peak RSS by (1 + num_workers). A single contiguous
        # tensor keeps its data buffer outside Python's GC, so the buffer
        # pages are genuinely shared across the fork.
        self._signals_tensor, self._signals = fill_signals.finish()
        self._encoded_seqs_tensor, self._encoded_seqs = fill_encoded_seqs.finish()

        # Compact signal_kmer inputs — only populated when encoding == signal_kmer.
        # Stacked into fork-safe int tensors. encode_signal_kmer is then called
        # lazily per-sample in __getitem__ (workers parallelize behind prefetch).
        # Chunks have variable basecalled sequence lengths near the motif, so we
        # pad to the per-array max. Padding values are chosen so the encoder
        # gracefully ignores them: seq_ints=-1 hits the encoder's `base < 0`
        # skip; seq_to_sig=signal_len makes its `sig_start < signal_len` check
        # fail, writing nothing for the padded positions.
        #
        # The block-wise filler builds the padded int8 matrix straight off the
        # text column and leaves ``_seq_ints_tensor`` set; the per-chunk path
        # leaves a list to pad here. Either way the CSR expansion below runs.
        self._seq_to_sig_tensor: torch.Tensor | None = None
        if self._effective_seq_encoding == "signal_kmer" and (
            self._seq_ints or self._seq_ints_tensor is not None
        ):
            if self._seq_ints:
                max_seq_ints_len = max(s.shape[0] for s in self._seq_ints)
                n = len(self._seq_ints)
                padded_seq_ints = np.full((n, max_seq_ints_len), -1, dtype=np.int8)
                for i, si in enumerate(self._seq_ints):
                    padded_seq_ints[i, : si.shape[0]] = si
                self._seq_ints_tensor = torch.from_numpy(padded_seq_ints)
                self._seq_ints = []
            n = int(self._seq_ints_tensor.shape[0])

            if self._s2s_csr is not None:
                # Streaming path: expand every map at once from the CSR pair,
                # instead of one astype/clip per chunk.
                values, offsets = self._s2s_csr
                crop_starts = None
                if self.left_context is not None and self.right_context is not None:
                    crop_starts = self._crop_starts(values, offsets) - self.left_context
                padded_s2s = _expand_seq_to_sig_csr(
                    values,
                    offsets,
                    self._s2s_rows,
                    signal_len=signal_len,
                    crop_starts=crop_starts,
                )
                self._s2s_csr = None
            else:
                max_s2s_len = max(s.shape[0] for s in self._seq_to_sig)
                padded_s2s = np.full((n, max_s2s_len), signal_len, dtype=np.int64)
                for i, s2s in enumerate(self._seq_to_sig):
                    padded_s2s[i, : s2s.shape[0]] = s2s
            self._seq_to_sig_tensor = torch.from_numpy(padded_s2s)
            self._seq_to_sig = []

        self._labels_tensor, self._labels = fill_labels.finish()

        self._features_tensor: torch.Tensor | None = None
        if self._needs_features:
            self._features_tensor, self._features = fill_features.finish()

        self._confound_labels_tensor: torch.Tensor | None = None
        if self._has_confound:
            self._confound_labels_tensor, self._confound_labels = fill_confounds.finish()

        self._cl_targets_tensor: torch.Tensor | None = None
        if self._cl_regression:
            self._cl_targets_tensor, self._cl_targets = fill_cl_targets.finish()

        # Drop the raw numpy arrays from self.chunks now that everything has
        # been pre-tensorized. External code (samplers, label tally, feature
        # window introspection) still reads the small scalar/string fields,
        # so we keep self.chunks alive but null out the per-chunk arrays.
        # Without this, each chunk dict keeps a ~50 KB numpy view alive and
        # the same COW blowup hits during DataLoader fork.
        if not isinstance(self.chunks, ChunkTable):
            for chunk in self.chunks:
                for key in (
                    "signal",
                    "signal_residual",
                    "dwell",
                    "features",
                    "seq_to_sig_map",
                    "sequence_with_kmer_context",
                ):
                    if key in chunk:
                        chunk[key] = None

        # Same reasoning for the streaming bookkeeping: one row per chunk each,
        # and a DataLoader that spawns workers pickles whatever is still here.
        self._s2s_csr = None
        self._s2s_rows = None
        if self._array_stream is not None:
            self._array_stream.keep = None

        # Precompute per-channel feature stds for feature noise augmentation.
        # Reuse the already-stacked features tensor when available.
        self._feature_stds: torch.Tensor | None = None
        if self._feature_noise_scale > 0 and self._needs_features:
            if self._features_tensor is not None:
                self._feature_stds = self._features_tensor.std(dim=0)  # (C, K)
            else:
                logger.warning("Feature shapes differ, feature noise disabled")
                self._feature_noise_scale = 0.0

        # Approx samples per base for cross-layer shift/mask
        self._samples_per_base = signal_len / max(kmer_len, 1)

        # Whether __getitems__ can gather a batch straight out of the
        # contiguous tensors. Every field has to be one: a field that degraded
        # to a list of per-chunk tensors has no batch to gather.
        self._batched_fetch = (
            self._signals_tensor is not None
            and self._labels_tensor is not None
            and (not self._needs_features or self._features_tensor is not None)
            and (not self._has_confound or self._confound_labels_tensor is not None)
            and (not self._cl_regression or self._cl_targets_tensor is not None)
            and (
                (self._seq_ints_tensor is not None and self._seq_to_sig_tensor is not None)
                if self._effective_seq_encoding == "signal_kmer"
                else self._encoded_seqs_tensor is not None
            )
        )

        _n_encoded = (
            self._encoded_seqs_tensor.shape[0]
            if self._encoded_seqs_tensor is not None
            else len(self._encoded_seqs)
        )
        logger.debug(
            f"Pre-tensorized {len(self.chunks)} chunks "
            f"({_n_encoded} sequences encoded, encoding={self._effective_seq_encoding})"
        )

    # =========================================================================
    # Tensorizing: one row at a time, or a whole row block at a time
    # =========================================================================

    def _fill_from_rows(self, fills: dict[str, _TensorFill]) -> None:
        """Prepare one chunk at a time — the original, and still the reference.

        Used for pre-loaded chunk lists, corpora too old to stream, and every
        option combination :meth:`_block_fill_supported` declines. The
        block-wise filler must agree with this one bit for bit;
        ``tests/test_dataset_streaming.py`` holds it to that.
        """
        array_iter = iter(self._array_stream) if self._array_stream is not None else None
        # The one metadata field read for every chunk on the default encoding.
        # Reading the column directly hands `_encode_sequence` the bytes it
        # wants and skips a row view per chunk; dicts have nothing to hoist.
        sequence_column = (
            self.chunks.values("sequence") if isinstance(self.chunks, ChunkTable) else None
        )
        stream_dwell_width = (
            self._array_stream.dwell_width if self._array_stream is not None else None
        )
        signal_len = self.signal_len

        for row, chunk in enumerate(self.chunks):
            arrays = chunk if array_iter is None else next(array_iter)

            if self._effective_seq_encoding == "signal_kmer":
                seq_ctx = chunk["sequence_with_kmer_context"]
                seq_ints = sequence_to_int(seq_ctx).astype(np.int8)
                self._seq_ints.append(seq_ints)

                if self._s2s_csr is None:
                    # Non-streaming path: one map per chunk dict. The streaming
                    # path expands all of them at once, after this loop.
                    s2s = chunk["seq_to_sig_map"].astype(np.int64, copy=True)
                    if self.left_context is not None and self.right_context is not None:
                        stored_focus = chunk.get("focus_signal_pos")
                        focus_pos = stored_focus if stored_focus is not None else int(s2s[-1]) // 2
                        crop_start = focus_pos - self.left_context
                        s2s -= crop_start
                        np.clip(s2s, 0, signal_len, out=s2s)
                    self._seq_to_sig.append(s2s)
            else:
                # Pre-encode sequence (vectorized, no Python loop)
                sequence = chunk["sequence"] if sequence_column is None else sequence_column[row]
                fills["encoded_seqs"].append(self._encode_sequence(sequence))

            # Pre-create label tensor: long for multi-class, float for binary
            if self._multiclass:
                fills["labels"].append(torch.tensor(chunk["label_int"], dtype=torch.long))
            else:
                fills["labels"].append(torch.tensor([chunk["label_int"]], dtype=torch.float32))

            # Pre-tensorize signal: pad/crop once instead of every __getitem__ call
            signal = arrays["signal"]
            if signal.dtype != np.float32:
                signal = signal.astype(np.float32)
            signal_residual = arrays.get("signal_residual")
            if signal_residual is not None and signal_residual.dtype != np.float32:
                signal_residual = signal_residual.astype(np.float32)
            fills["signals"].append(
                self._prepare_signal(
                    signal, signal_residual, focus_signal_pos=chunk.get("focus_signal_pos")
                )
            )

            # Pre-tensorize features: apply dwell_offset slicing once
            if self._needs_features:
                dwell_width = (
                    stream_dwell_width if stream_dwell_width is not None else len(chunk["dwell"])
                )
                fills["features"].append(
                    self._prepare_features(arrays["features"], dwell_width, chunk)
                )

            # Confound label for adversarial training. The encoder reads the
            # configured chunk field and maps it to a class int (-1 = ignore).
            if self._confound_encoder is not None:
                confound_class = self._confound_encoder.encode(chunk)
                fills["confounds"].append(torch.tensor(confound_class, dtype=torch.long))

            # CL regression target (cl_value / 255.0; sentinel -1.0 for missing)
            if self._cl_regression:
                cl_val = chunk.get("cl_value")
                if cl_val is not None and cl_val >= 0:
                    fills["cl_targets"].append(torch.tensor(cl_val / 255.0, dtype=torch.float32))
                else:
                    fills["cl_targets"].append(torch.tensor(-1.0, dtype=torch.float32))

    def _block_fill_supported(self) -> bool:
        """Can this corpus and option set be tensorized a block at a time?

        One decision for the whole dataset rather than one per block: every
        case the vectorized filler cannot express is a property of the corpus
        or of the options, and all of them are readable before the first block
        arrives — a text column whose rows are not all the same length, a dwell
        template (whose append is inherently per chunk), an asymmetric crop
        whose in-bounds width differs from ``signal_len``, a feature window
        that runs off the array. When any holds, everything takes
        :meth:`_fill_from_rows`, which is the definition of the right answer.
        """
        table = self.chunks
        stream = self._array_stream
        if stream is None or not isinstance(table, ChunkTable):
            return False
        if self._dwell_templates is not None:
            return False  # the per-AA template append reads one chunk at a time

        if self._effective_seq_encoding == "signal_kmer":
            if _byte_matrix(table.values("sequence_with_kmer_context")) is None:
                return False
        else:
            flat = _byte_matrix(table.values("sequence"))
            if flat is None:
                return False
            # numpy strips trailing NULs from a fixed-width row, so a padded
            # row is a short sequence: ragged, and the row path degrades it to
            # a list of per-chunk tensors.
            if flat.shape[1] and bool(np.any(flat[:, -1] == 0)):
                return False

        stored_len = stream.row_shapes["signal"][0]
        if self.left_context is not None and self.right_context is not None:
            width = self.left_context + self.right_context
            if stored_len == 0:
                return False
            if width != self.signal_len:
                # The overhang branch pads to signal_len while the in-bounds
                # branch is `width` wide; when they disagree the output is
                # ragged and only the row path can produce it.
                starts = self._focus_positions(stored_len) - self.left_context
                if np.any(starts < 0) or np.any(starts + width > stored_len):
                    return False

        if self._needs_features:
            shape = stream.row_shapes["features"]
            if len(shape) != 2 or shape[0] * shape[1] == 0:
                return False
            dwell_width = stream.dwell_width
            if (
                dwell_width is not None
                and dwell_width > self.kmer_len
                and self.model_type not in WIDE_FEATURE_MODELS
            ):
                starts = self._feature_slice_starts(dwell_width)
                if np.any(starts < 0) or np.any(starts + self.kmer_len > shape[1]):
                    return False  # let the row path raise the documented ValueError
        return True

    def _fill_from_blocks(self, fills: dict[str, _TensorFill]) -> None:
        """Prepare a whole row block at a time.

        Same outputs as :meth:`_fill_from_rows`, reached with one vectorized
        call per block instead of one Python call per chunk. Everything the
        row loop reads out of the metadata one chunk at a time — labels,
        sequences, focus positions, feature windows, confound classes, CL
        targets — is a column, so it is read once for the corpus and sliced
        per block; only the signal and feature arrays are genuinely per block.
        """
        table = self.chunks
        stream = self._array_stream
        assert stream is not None

        if self._effective_seq_encoding == "signal_kmer":
            self._seq_ints_tensor = self._seq_ints_from_column()
            sequence_bytes = None
        else:
            # A view of the column, not a copy: the one-hot expansion is 4
            # floats per base and only ever exists for one block at a time.
            sequence_bytes = _byte_matrix(table.values("sequence"))

        labels = self._label_column()
        confounds = self._confound_column() if self._has_confound else None
        cl_targets = self._cl_target_column() if self._cl_regression else None

        stored_len = stream.row_shapes["signal"][0]
        asymmetric = self.left_context is not None and self.right_context is not None
        focus = self._focus_positions(stored_len) if asymmetric else None
        feature_starts = (
            self._feature_slice_starts(stream.dwell_width)
            if self._needs_features and stream.dwell_width is not None
            else None
        )

        for start, arrays in stream.blocks():
            stop = start + len(arrays["signal"])

            signal = arrays["signal"]
            if signal.dtype != np.float32:
                signal = signal.astype(np.float32)
            residual = arrays.get("signal_residual")
            if residual is not None and residual.dtype != np.float32:
                residual = residual.astype(np.float32)
            fills["signals"].extend(
                self._prepare_signals_block(
                    signal, residual, None if focus is None else focus[start:stop]
                )
            )

            if sequence_bytes is not None:
                fills["encoded_seqs"].extend(
                    self._encode_sequences_block(sequence_bytes[start:stop])
                )
            fills["labels"].extend(labels[start:stop])
            if self._needs_features:
                fills["features"].extend(
                    self._prepare_features_block(
                        arrays["features"],
                        stream.dwell_width,
                        None if feature_starts is None else feature_starts[start:stop],
                    )
                )
            if confounds is not None:
                fills["confounds"].extend(confounds[start:stop])
            if cl_targets is not None:
                fills["cl_targets"].extend(cl_targets[start:stop])

    # -- column readers: the metadata half of the block-wise filler ----------

    def _label_column(self) -> torch.Tensor:
        """Every chunk's label, in the dtype and shape the model expects.

        Long ``(N,)`` for multi-class (CrossEntropyLoss), float ``(N, 1)`` for
        binary — the stacked form of the per-chunk ``torch.tensor`` calls.
        """
        raw = self.chunks.require_values("label_int")
        if self._multiclass:
            return torch.from_numpy(raw.astype(np.int64))
        return torch.from_numpy(raw.astype(np.float32)).unsqueeze(1)

    def _cl_target_column(self) -> torch.Tensor:
        """``cl_value / 255.0`` per chunk, with -1.0 where the value is missing."""
        raw = self.chunks.values("cl_value")
        n = len(self.chunks)
        if raw is None:
            return torch.full((n,), -1.0, dtype=torch.float32)
        return torch.from_numpy(np.where(raw >= 0, raw / 255.0, -1.0).astype(np.float32))

    def _confound_column(self) -> torch.Tensor:
        """Confound class per chunk (-1 = ignore), mapped through the encoder.

        The encoder's table is keyed by the chunk's *translated* value (a str,
        or an int with the missing-value sentinel resolved), so the mapping is
        applied to the column's distinct values — a few hundred at most — and
        broadcast back rather than looked up per chunk.
        """
        encoder = self._confound_encoder
        assert encoder is not None
        table = self.chunks
        n = len(table)
        raw = table.values(encoder.source)
        if raw is None:
            # The corpus has no such field; every chunk reads as None.
            return torch.full((n,), encoder.value_to_class.get(None, -1), dtype=torch.long)
        uniques, first_row, inverse = np.unique(raw, return_index=True, return_inverse=True)
        lookup = np.array(
            [
                encoder.value_to_class.get(table.value(encoder.source, int(row)), -1)
                for row in first_row
            ],
            dtype=np.int64,
        )
        return torch.from_numpy(lookup[inverse.reshape(-1)])

    def _focus_positions(self, stored_len: int) -> np.ndarray:
        """Focus signal position per chunk, or the centre when none is stored."""
        column = self.chunks.values("focus_signal_pos")
        if column is None:
            return np.full(len(self.chunks), stored_len // 2, dtype=np.int64)
        return column.astype(np.int64)

    def _feature_slice_starts(self, dwell_width: int) -> np.ndarray:
        """Per-chunk k-mer-aligned start into the feature array.

        The column form of the ``feature_start`` resolution in
        :meth:`_prepare_features`. ``feature_left`` is not represented here
        because it is never a column — only a legacy chunk-dict field, which
        goes down the row path anyway.
        """
        table = self.chunks
        kmer_context = self.kmer_len // 2
        column = table.values("feature_start")
        if column is not None:
            feat_start = column.astype(np.int64)
        else:
            margin = table.values("dwell_margin_left")
            if margin is not None:
                feat_start = -(kmer_context + margin.astype(np.int64))
            else:
                feat_start = np.full(len(table), -(dwell_width - 1) // 2, dtype=np.int64)
        return (-kmer_context) - feat_start + self.dwell_offset

    def _seq_ints_from_column(self) -> torch.Tensor | None:
        """The padded int8 ``seq_ints`` matrix, straight off the text column.

        ``sequence_to_int`` maps anything that is not a base to -1, which is
        also the padding value the encoder skips — so the NUL padding numpy
        already holds in a fixed-width bytes column encodes to exactly what
        the per-chunk path pads with, and no mask is needed.
        """
        flat = _byte_matrix(self.chunks.values("sequence_with_kmer_context"))
        if flat is None:
            return None
        width = flat.shape[1]
        if width:
            # Trailing NULs are padding; the widest real sequence sets the
            # matrix width, exactly as max(len) does on the row path.
            trailing = (flat[:, ::-1] != 0).argmax(axis=1)
            empty = ~flat.any(axis=1)
            lengths = np.where(empty, 0, width - trailing)
            width = int(lengths.max()) if len(lengths) else 0
        return torch.from_numpy(np.ascontiguousarray(_BASE_INT_MAP[flat[:, :width]]))

    # -- array preparers: the block twins of _prepare_signal/_prepare_features

    @staticmethod
    def _encode_sequences_block(flat: np.ndarray) -> torch.Tensor:
        """One-hot a block of equal-length sequences at once.

        ``flat`` is the ``(rows, kmer_len)`` uint8 view of the bytes column.
        Same output as stacking :meth:`_encode_sequence` over the rows.
        """
        rows, width = flat.shape
        indices = _BASE_MAP[flat]
        encoded = np.zeros((rows, 4, width), dtype=np.float32)
        valid = indices < 4
        row_idx, col_idx = np.nonzero(valid)
        encoded[row_idx, indices[valid], col_idx] = 1.0
        return torch.from_numpy(encoded)

    def _prepare_signals_block(
        self,
        signal: np.ndarray,
        signal_residual: np.ndarray | None,
        focus: np.ndarray | None,
    ) -> torch.Tensor:
        """Pad/crop a ``(rows, stored_len)`` signal block. Twin of :meth:`_prepare_signal`.

        The asymmetric crop is the only per-row part: each chunk's window
        starts at its own focus position, so it is one gather with the
        out-of-range positions zeroed — which is what the per-chunk zero-pad
        branch computes, one row at a time.
        """
        rows, stored_len = signal.shape
        if focus is not None:
            left, right = self.left_context, self.right_context
            assert left is not None and right is not None  # focus implies both
            width = left + right
            starts = focus - left
            columns = starts[:, None] + np.arange(width, dtype=np.int64)
            inside = (columns >= 0) & (columns < stored_len)
            np.clip(columns, 0, stored_len - 1, out=columns)
            gather = np.arange(rows)[:, None]
            signal = signal[gather, columns]
            signal[~inside] = 0.0
            if signal_residual is not None:
                signal_residual = signal_residual[gather, columns]
                signal_residual[~inside] = 0.0
        elif stored_len < self.signal_len:
            signal = np.pad(signal, ((0, 0), (0, self.signal_len - stored_len)), mode="constant")
            if signal_residual is not None:
                signal_residual = np.pad(
                    signal_residual, ((0, 0), (0, self.signal_len - stored_len)), mode="constant"
                )
        elif stored_len > self.signal_len:
            start = (stored_len - self.signal_len) // 2
            signal = signal[:, start : start + self.signal_len]
            if signal_residual is not None:
                signal_residual = signal_residual[:, start : start + self.signal_len]

        if signal_residual is not None:
            if self._signal_mode == "both":
                return torch.from_numpy(np.stack([signal, signal_residual], axis=1))
            elif self._signal_mode == "residual":
                return torch.from_numpy(np.ascontiguousarray(signal_residual))
        return torch.from_numpy(np.ascontiguousarray(signal))

    def _prepare_features_block(
        self,
        features: np.ndarray,
        dwell_width: int | None,
        starts: np.ndarray | None,
    ) -> torch.Tensor:
        """Slice a ``(rows, channels, width)`` feature block. Twin of :meth:`_prepare_features`.

        ``starts`` comes from :meth:`_feature_slice_starts` and is already
        known to be in range — :meth:`_block_fill_supported` checked it, and
        sends the corpus down the row path when it is not, so the documented
        ValueError still comes from the one place that raises it.
        """
        if features.dtype != np.float32:
            features = features.astype(np.float32)
        if (
            starts is not None
            and dwell_width is not None
            and dwell_width > self.kmer_len
            and self.model_type not in WIDE_FEATURE_MODELS
        ):
            first = int(starts[0])
            if bool(np.all(starts == first)):
                features = features[:, :, first : first + self.kmer_len]
            else:
                columns = starts[:, None] + np.arange(self.kmer_len, dtype=np.int64)
                features = np.take_along_axis(features, columns[:, None, :], axis=2)
        return torch.from_numpy(np.ascontiguousarray(features))

    def _load_from_path(self, chunk_path: Path, *, signal_mode: str, seq_encoding: str) -> None:
        """Load chunk metadata from an npz, deferring the arrays a stream can supply.

        Sets ``self.chunks`` and, when the corpus is row-streamable,
        ``self._array_stream`` (plus the CSR base-to-signal maps when the run
        needs them). Falls back to loading everything eagerly for corpora
        written before the flat array format.
        """
        self._npz_members = npz_member_names(chunk_path)
        stream = _ArrayStream.build(
            chunk_path,
            needs_features=self._needs_features,
            wants_residual=signal_mode in ("both", "residual"),
        )
        if stream is None:
            logger.debug("%s is not row-streamable; loading arrays eagerly", chunk_path)
            self.chunks = load_chunks(chunk_path)
            return

        # Metadata goes into columns rather than a dict per chunk: the stream
        # supplies signal/residual/features, the dwell width comes from the
        # member header, and the base-to-signal maps are expanded from CSR only
        # when signal_kmer needs them, so no per-chunk array is read at all.
        skip = () if seq_encoding == "signal_kmer" else ("sequence_with_kmer_context",)
        table = ChunkTable.from_npz(chunk_path, skip=skip)

        keep = table.require_values("label_int") >= 0
        stream.keep = keep
        self._array_stream = stream
        self.chunks = table if keep.all() else table.select(keep)
        if seq_encoding == "signal_kmer":
            self._s2s_csr = load_seq_to_sig_csr(chunk_path)
            self._s2s_rows = np.nonzero(keep)[0]

    def _crop_starts(self, values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """Per-chunk focus signal position for the streamed base-to-signal maps.

        Mirrors the per-chunk rule: the stored ``focus_signal_pos`` when the
        corpus has one, else half the map's last value (old symmetric data).
        """
        rows = self._s2s_rows
        assert rows is not None
        # A column is all-or-nothing: either every chunk carries a focus
        # position or none does, so reading it whole answers both questions
        # without a row view per chunk (150 ms per 200k chunks for the loop).
        if isinstance(self.chunks, ChunkTable):
            column = self.chunks.values("focus_signal_pos")
            if column is not None:
                return column.astype(np.int64)
            focus = np.zeros(len(rows), dtype=np.int64)
            missing: list[int] = list(range(len(rows)))
        else:
            focus = np.zeros(len(rows), dtype=np.int64)
            missing = []
            for i, chunk in enumerate(self.chunks):
                stored = chunk.get("focus_signal_pos")
                if stored is None:
                    missing.append(i)
                else:
                    focus[i] = stored
        if missing:
            idx = np.asarray(missing)
            starts = offsets[rows[idx]]
            ends = offsets[rows[idx] + 1]
            if np.any(ends <= starts):
                raise ValueError(
                    "Chunk without focus_signal_pos has an empty seq_to_sig_map; "
                    "cannot place the asymmetric crop"
                )
            focus[idx] = values[ends - 1].astype(np.int64) // 2
        return focus

    def _prepare_signal(
        self,
        signal: np.ndarray,
        signal_residual: np.ndarray | None = None,
        focus_signal_pos: int | None = None,
    ) -> torch.Tensor:
        """Pad/crop signal (and optional residual) to target length. Called once during __init__."""
        if self.left_context is not None and self.right_context is not None:
            # Use stored focus position when available (asymmetric prepare);
            # fall back to center for old symmetric data.
            focus_pos = focus_signal_pos if focus_signal_pos is not None else len(signal) // 2
            start = focus_pos - self.left_context
            end = focus_pos + self.right_context
            if start < 0 or end > len(signal):
                cropped = np.zeros(self.signal_len, dtype=np.float32)
                src_start, src_end = max(0, start), min(len(signal), end)
                dst_start = max(0, -start)
                cropped[dst_start : dst_start + (src_end - src_start)] = signal[src_start:src_end]
                signal = cropped
                if signal_residual is not None:
                    cropped_r = np.zeros(self.signal_len, dtype=np.float32)
                    cropped_r[dst_start : dst_start + (src_end - src_start)] = signal_residual[
                        src_start:src_end
                    ]
                    signal_residual = cropped_r
            else:
                signal = signal[start:end]
                if signal_residual is not None:
                    signal_residual = signal_residual[start:end]
        elif len(signal) < self.signal_len:
            signal = np.pad(signal, (0, self.signal_len - len(signal)), mode="constant")
            if signal_residual is not None:
                signal_residual = np.pad(
                    signal_residual, (0, self.signal_len - len(signal_residual)), mode="constant"
                )
        elif len(signal) > self.signal_len:
            start = (len(signal) - self.signal_len) // 2
            signal = signal[start : start + self.signal_len]
            if signal_residual is not None:
                signal_residual = signal_residual[start : start + self.signal_len]

        if signal_residual is not None:
            if self._signal_mode == "both":
                stacked = np.stack([signal, signal_residual], axis=0)
                return torch.from_numpy(np.ascontiguousarray(stacked))
            elif self._signal_mode == "residual":
                return torch.from_numpy(np.ascontiguousarray(signal_residual))
        return torch.from_numpy(np.ascontiguousarray(signal))

    def _load_dwell_templates(self, table_path: Path) -> None:
        """Instance wrapper: load templates via the module-level helper."""
        template, template_min_pos, aa_order = load_dwell_template_table(table_path)
        self._dwell_templates = template
        self._template_min_pos = template_min_pos
        self._template_aa_order = aa_order

    def _append_template_channels(self, features: np.ndarray, chunk: dict) -> np.ndarray:
        """Instance wrapper: delegate to the module-level helper.

        The label-independent nature of ``append_dwell_template_channels`` means
        it works at inference time too — it computes ``dwell /
        expected_dwell[AA_i, pos]`` for all 20 AAs without ever looking at the
        ground-truth label.
        """
        if self._dwell_templates is None:
            return features
        feat_start = int(chunk.get("feature_start", 0))
        return append_dwell_template_channels(
            features,
            feat_start=feat_start,
            dwell_templates=self._dwell_templates,
            template_min_pos=self._template_min_pos,
        )

    def _prepare_features(
        self, features: np.ndarray, dwell_width: int, chunk: dict
    ) -> torch.Tensor:
        """Apply dwell_offset slicing and tensorize features. Called once during __init__.

        Args:
            features: The chunk's feature array, ``(num_features, feat_width)``.
            dwell_width: ``len(chunk["dwell"])``. Only its width matters here,
                so the streaming path takes it from the member header rather
                than reading the dwells at all.
            chunk: The chunk dict, for the feature-window metadata.
        """
        if features.dtype != np.float32:
            features = features.astype(np.float32)

        # Append 20 dwell template ratio channels before slicing
        features = self._append_template_channels(features, chunk)

        kmer_context = self.kmer_len // 2

        if self.model_type in WIDE_FEATURE_MODELS:
            pass  # full-width features
        elif dwell_width > self.kmer_len:
            # Determine feature_start (signed offset from focus).
            # New chunks have it directly; old chunks need conversion.
            if "feature_start" in chunk:
                feat_start = int(chunk["feature_start"])
            elif "feature_left" in chunk:
                feat_start = -int(chunk["feature_left"])
            elif "dwell_margin_left" in chunk:
                feat_start = -(kmer_context + int(chunk["dwell_margin_left"]))
            else:
                feat_start = -(dwell_width - 1) // 2  # symmetric fallback
            # kmer-aligned start within the feature array
            # Feature array starts at focus + feat_start, kmer starts at focus - kmer_context
            kmer_start = (-kmer_context) - feat_start
            start = kmer_start + self.dwell_offset
            if start < 0 or start + self.kmer_len > features.shape[1]:
                raise ValueError(
                    f"dwell_offset ({self.dwell_offset}) with feature_start={feat_start} "
                    f"exceeds feature width ({features.shape[1]})"
                )
            if features.size > 0:
                features = features[:, start : start + self.kmer_len]

        if features.size > 0:
            return torch.from_numpy(np.ascontiguousarray(features))
        else:
            return torch.zeros(1, self.kmer_len, dtype=torch.float32)

    def _apply_augmentation(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply signal augmentation (jitter and/or scaling).

        Both jitter_std and scale_range can be either scalar (uniform across
        channels) or dict for per-channel control::

            {"signal": 0.02, "signal_residual": 0.001}

        Per-channel dicts only apply to 2-channel (dim=2) input; for 1-channel
        input the scalar path is used regardless.
        """
        jitter_std = self.augmentation.get("jitter_std", 0.0)
        if jitter_std:
            if isinstance(jitter_std, dict) and signal.dim() == 2:
                noise = torch.zeros_like(signal)
                if jitter_std.get("signal", 0.0) > 0:
                    noise[0] = torch.randn(signal.shape[1]) * jitter_std["signal"]
                if jitter_std.get("signal_residual", 0.0) > 0:
                    noise[1] = torch.randn(signal.shape[1]) * jitter_std["signal_residual"]
                signal = signal + noise
            elif isinstance(jitter_std, (int, float)) and jitter_std > 0:
                signal = signal + torch.randn_like(signal) * jitter_std

        scale_range = self.augmentation.get("scale_range", (1.0, 1.0))
        if scale_range and scale_range != (1.0, 1.0):
            if isinstance(scale_range, dict) and signal.dim() == 2:
                for ch_idx, ch_name in enumerate(["signal", "signal_residual"]):
                    ch_range = scale_range.get(ch_name, (1.0, 1.0))
                    if ch_range != (1.0, 1.0):
                        s = torch.empty(1).uniform_(ch_range[0], ch_range[1]).item()
                        signal[ch_idx] = signal[ch_idx] * s
            else:
                scale = torch.empty(1).uniform_(scale_range[0], scale_range[1]).item()
                signal = signal * scale
        return signal

    def _apply_shift(
        self,
        signal: torch.Tensor,
        features: torch.Tensor,
        sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply cross-layer shift (simulates motif anchor offset).

        Shifts all branches consistently. The shift is drawn as a continuous
        float in [-shift_max_bases, +shift_max_bases], allowing sub-base
        resolution. Signal is shifted in sample space; features and sequence
        are shifted by the nearest integer base.
        """
        base_shift = torch.empty(1).uniform_(-self._shift_max_bases, self._shift_max_bases).item()
        if abs(base_shift) < 1e-6:
            return signal, features, sequence

        # Shift signal (sample-level, fractional)
        sample_shift = int(round(base_shift * self._samples_per_base))
        if sample_shift != 0 and signal.numel() > 0:
            signal = torch.roll(signal, shifts=sample_shift, dims=-1)
            if sample_shift > 0:
                signal[..., :sample_shift] = 0.0
            else:
                signal[..., sample_shift:] = 0.0

        # Integer base shift for features and sequence
        int_base_shift = int(round(base_shift))
        if int_base_shift != 0:
            # Shift features (base-level)
            if features.numel() > 0 and features.shape[-1] > 0:
                features = torch.roll(features, shifts=int_base_shift, dims=-1)
                if int_base_shift > 0:
                    features[..., :int_base_shift] = 0.0
                else:
                    features[..., int_base_shift:] = 0.0

            # Shift sequence (base-level)
            if sequence.numel() > 0 and sequence.shape[-1] > 0:
                sequence = torch.roll(sequence, shifts=int_base_shift, dims=-1)
                if int_base_shift > 0:
                    sequence[..., :int_base_shift] = 0.0
                else:
                    sequence[..., int_base_shift:] = 0.0

        return signal, features, sequence

    def _apply_time_mask(
        self,
        signal: torch.Tensor,
        features: torch.Tensor,
        sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply time masking: zero contiguous blocks across all branches.

        Mask widths are in base units. Signal is masked at the corresponding
        sample-level range using the approximate samples_per_base ratio.
        """
        kmer_len = sequence.shape[-1] if sequence.numel() > 0 else self.kmer_len

        for _ in range(self._time_mask_count):
            width = torch.randint(1, self._time_mask_bases + 1, (1,)).item()
            start = torch.randint(0, max(1, kmer_len - width + 1), (1,)).item()
            end = min(start + width, kmer_len)

            # Mask features (base-level)
            if features.numel() > 0 and features.shape[-1] > 0:
                features[..., start:end] = 0.0

            # Mask sequence (base-level)
            if sequence.numel() > 0 and sequence.shape[-1] > 0:
                sequence[..., start:end] = 0.0

            # Mask signal (sample-level)
            if signal.numel() > 0:
                sig_start = int(round(start * self._samples_per_base))
                sig_end = int(round(end * self._samples_per_base))
                sig_end = min(sig_end, signal.shape[-1])
                signal[..., sig_start:sig_end] = 0.0

        return signal, features, sequence

    def _apply_feature_noise(self, features: torch.Tensor) -> torch.Tensor:
        """Apply per-channel Gaussian noise scaled by empirical channel std."""
        if self._feature_stds is not None:
            noise = torch.randn_like(features) * self._feature_stds * self._feature_noise_scale
            features = features + noise
        return features

    @staticmethod
    def _encode_sequence(sequence: str | bytes) -> torch.Tensor:
        """Vectorized one-hot encoding of a DNA sequence.

        Uses a pre-built ASCII lookup table instead of a Python for-loop.

        Args:
            sequence: DNA sequence (A, C, G, T, N), str or ASCII bytes. The
                columnar metadata store holds sequences as bytes, and the
                lookup wants bytes, so accept them and skip the round trip.

        Returns:
            One-hot encoded tensor of shape (4, len(sequence))
        """
        raw = sequence if isinstance(sequence, bytes) else sequence.encode()
        indices = _BASE_MAP[np.frombuffer(raw, dtype=np.uint8)]
        encoded = np.zeros((4, len(raw)), dtype=np.float32)
        valid = indices < 4
        encoded[indices[valid], np.where(valid)[0]] = 1.0
        return torch.from_numpy(encoded)

    @staticmethod
    def _encode_signal_kmer(
        chunk: dict,
        signal_len: int,
        kmer_context: tuple[int, int],
        left_context: int | None = None,
        right_context: int | None = None,
    ) -> torch.Tensor:
        """Encode chunk using signal-level kmer encoding.

        Args:
            chunk: Chunk dict with seq_to_sig_map and sequence_with_kmer_context
            signal_len: Target signal length (pad/truncate)
            kmer_context: (kmer_before, kmer_after) for encoding
            left_context: Left signal context for asymmetric crop adjustment
            right_context: Right signal context for asymmetric crop adjustment

        Returns:
            Encoded tensor of shape (4 * kmer_len, signal_len)
        """
        seq_ctx = chunk["sequence_with_kmer_context"]
        seq_to_sig = chunk["seq_to_sig_map"].copy()

        # Adjust seq_to_sig_map when signal will be asymmetrically cropped.
        # Stored chunks have coordinates in [0, stored_signal_len] but after
        # _prepare_signal crops to [focus - left_context, focus + right_context],
        # the signal_kmer encoder needs coordinates relative to the cropped window.
        if left_context is not None and right_context is not None:
            # Use stored focus position when available (asymmetric prepare);
            # fall back to center for old symmetric data.
            stored_focus = chunk.get("focus_signal_pos")
            focus_pos = stored_focus if stored_focus is not None else seq_to_sig[-1] // 2
            crop_start = focus_pos - left_context
            seq_to_sig = seq_to_sig - crop_start
            seq_to_sig = np.clip(seq_to_sig, 0, signal_len)

        seq_ints = sequence_to_int(seq_ctx)
        enc = encode_signal_kmer(seq_ints, seq_to_sig, signal_len, kmer_context)
        return torch.from_numpy(enc)

    def __len__(self) -> int:
        """Return number of chunks."""
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single training example.

        Returns:
            Dictionary with:
            - signal: (signal_len,) tensor
            - sequence: (4, kmer_len) one-hot encoded tensor
            - features: (num_features, kmer_len) tensor (if model requires features)
            - label: (1,) tensor
        """
        # Pre-computed signal lookup (padded/cropped once in __init__).
        # Tensor path is the fork-safe fast path; list path is a fallback when
        # per-chunk shapes were inconsistent and torch.stack failed in __init__.
        if self._signals_tensor is not None:
            signal_tensor = self._signals_tensor[idx]
        else:
            signal_tensor = self._signals[idx]

        if self._effective_seq_encoding == "signal_kmer":
            # Compute one-hot kmer encoding on the fly from the compact
            # inputs. The encoded output is (4*kmer_len, signal_len) float32
            # — ~77 KB per chunk. Materializing 789K of these for a 20-class
            # run is ~60 GB; we keep memory bounded by deferring.
            if self._seq_ints_tensor is not None:
                seq_ints = self._seq_ints_tensor[idx].numpy()
                seq_to_sig = self._seq_to_sig_tensor[idx].numpy()
            else:
                seq_ints = self._seq_ints[idx]
                seq_to_sig = self._seq_to_sig[idx]
            enc = encode_signal_kmer(
                seq_ints, seq_to_sig, self.signal_len, self.signal_kmer_context
            )
            sequence_tensor = torch.from_numpy(enc)
        elif self._encoded_seqs_tensor is not None:
            sequence_tensor = self._encoded_seqs_tensor[idx]
        else:
            sequence_tensor = self._encoded_seqs[idx]

        if self._needs_features:
            if self._features_tensor is not None:
                features_tensor = self._features_tensor[idx]
            else:
                features_tensor = self._features[idx]
        else:
            features_tensor = torch.empty(0)

        # Augmentation pipeline (training only — augmentation dict is None for val/test)
        needs_cross_layer = self._shift_max_bases > 0 or self._time_mask_bases > 0
        needs_any_aug = (
            self.augmentation is not None or needs_cross_layer or self._feature_noise_scale > 0
        )

        if needs_any_aug:
            # Clone to avoid mutating stored tensors
            signal_tensor = signal_tensor.clone()
            if needs_cross_layer or self._feature_noise_scale > 0:
                if self._needs_features:
                    features_tensor = features_tensor.clone()
                sequence_tensor = sequence_tensor.clone()

            # 1. Cross-layer shift
            if self._shift_max_bases > 0:
                signal_tensor, features_tensor, sequence_tensor = self._apply_shift(
                    signal_tensor, features_tensor, sequence_tensor
                )

            # 2. Cross-layer time mask
            if self._time_mask_bases > 0:
                signal_tensor, features_tensor, sequence_tensor = self._apply_time_mask(
                    signal_tensor, features_tensor, sequence_tensor
                )

            # 3. Signal jitter + scale (existing y-axis augmentation)
            if self.augmentation is not None:
                signal_tensor = self._apply_augmentation(signal_tensor)

            # 4. Feature noise
            if self._feature_noise_scale > 0 and self._needs_features:
                features_tensor = self._apply_feature_noise(features_tensor)

        if self._labels_tensor is not None:
            label = self._labels_tensor[idx]
        else:
            label = self._labels[idx]

        result: dict[str, torch.Tensor] = {
            "signal": signal_tensor,
            "sequence": sequence_tensor,
            "label": label,
        }

        # Include features for models that require them
        if self._needs_features:
            result["features"] = features_tensor

        # Include confound label for adversarial training
        if self._has_confound:
            if self._confound_labels_tensor is not None:
                result["confound_label"] = self._confound_labels_tensor[idx]
            else:
                result["confound_label"] = self._confound_labels[idx]

        if self._cl_regression:
            if self._cl_targets_tensor is not None:
                result["cl_target"] = self._cl_targets_tensor[idx]
            else:
                result["cl_target"] = self._cl_targets[idx]

        return result

    def __getitems__(self, indices: Sequence[int]) -> dict[str, torch.Tensor] | list[dict]:
        """Fetch a whole batch at once, already collated.

        ``DataLoader``'s fetcher calls this instead of ``__getitem__`` once per
        index when a dataset defines it (torch >= 2.0), and hands whatever it
        returns to ``collate_fn`` — which passes an already-collated batch
        through untouched. Everything the per-sample path does is a gather out
        of a contiguous tensor, and one gather of B rows is far cheaper than B
        gathers plus a ``torch.stack`` per field: measured on a 200k-chunk
        corpus at batch 256, 925,425 chunks/s against 101,042 — 9.2x.

        Falls back to the per-sample path — returning a list for ``collate_fn``
        to stack — whenever a field is held as a list of per-chunk tensors, or
        when cross-layer shift/masking is on, since those draw one offset per
        sample and roll by it.
        """
        if not self._batched_fetch or self._shift_max_bases > 0 or self._time_mask_bases > 0:
            return [self[int(i)] for i in indices]

        rows = np.asarray(indices, dtype=np.int64)
        # A gather copies, so nothing below aliases the stored tensors and
        # augmentation needs no clone.
        signal = _gather_rows(self._signals_tensor, rows)

        if self._effective_seq_encoding == "signal_kmer":
            # encode_signal_kmer is per sample on both paths — batching it
            # needs a Rust batch entry point, which is out of scope here.
            seq_ints = self._seq_ints_tensor.numpy()[rows]
            seq_to_sig = self._seq_to_sig_tensor.numpy()[rows]
            sequence = torch.from_numpy(
                np.stack(
                    [
                        encode_signal_kmer(
                            seq_ints[i], seq_to_sig[i], self.signal_len, self.signal_kmer_context
                        )
                        for i in range(len(rows))
                    ]
                )
            )
        else:
            sequence = _gather_rows(self._encoded_seqs_tensor, rows)

        features = (
            _gather_rows(self._features_tensor, rows) if self._needs_features else torch.empty(0)
        )

        if self.augmentation is not None:
            signal = self._apply_augmentation_batch(signal)
        if self._feature_noise_scale > 0 and self._needs_features:
            features = self._apply_feature_noise(features)

        result: dict[str, torch.Tensor] = {
            "signal": signal,
            "sequence": sequence,
            "label": _gather_rows(self._labels_tensor, rows),
        }
        if self._needs_features:
            result["features"] = features
        if self._has_confound:
            result["confound_label"] = _gather_rows(self._confound_labels_tensor, rows)
        if self._cl_regression:
            result["cl_target"] = _gather_rows(self._cl_targets_tensor, rows)
        return result

    def _apply_augmentation_batch(self, signal: torch.Tensor) -> torch.Tensor:
        """:meth:`_apply_augmentation` over a leading batch dimension.

        One draw per *sample*, not one per batch: the jitter is elementwise
        anyway, and the scale is drawn as a ``(B, 1, ...)`` tensor so each row
        gets its own factor exactly as the per-sample path does.
        """
        batch = signal.shape[0]
        # Per-channel dicts describe a 2-channel sample, which is dim 3 here.
        channelwise = signal.dim() == 3

        jitter_std = self.augmentation.get("jitter_std", 0.0)
        if jitter_std:
            if isinstance(jitter_std, dict) and channelwise:
                noise = torch.zeros_like(signal)
                for ch_idx, ch_name in enumerate(["signal", "signal_residual"]):
                    if jitter_std.get(ch_name, 0.0) > 0:
                        noise[:, ch_idx] = (
                            torch.randn(batch, signal.shape[-1]) * jitter_std[ch_name]
                        )
                signal = signal + noise
            elif isinstance(jitter_std, (int, float)) and jitter_std > 0:
                signal = signal + torch.randn_like(signal) * jitter_std

        scale_range = self.augmentation.get("scale_range", (1.0, 1.0))
        if scale_range and scale_range != (1.0, 1.0):
            if isinstance(scale_range, dict) and channelwise:
                for ch_idx, ch_name in enumerate(["signal", "signal_residual"]):
                    ch_range = scale_range.get(ch_name, (1.0, 1.0))
                    if ch_range != (1.0, 1.0):
                        scale = torch.empty(batch, 1).uniform_(ch_range[0], ch_range[1])
                        signal[:, ch_idx] = signal[:, ch_idx] * scale
            else:
                shape = (batch, *([1] * (signal.dim() - 1)))
                scale = torch.empty(shape).uniform_(scale_range[0], scale_range[1])
                signal = signal * scale
        return signal


def collate_fn(
    batch: list[dict[str, torch.Tensor]] | dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of samples from ``__getitem__``, or the already-collated
            batch ``LeechDataset.__getitems__`` returns.

    Returns:
        Batched tensors
    """
    if isinstance(batch, dict):
        # __getitems__ gathered the batch out of the contiguous tensors in one
        # go; there is nothing left to stack.
        return batch

    # Stack all tensors
    signals = torch.stack([item["signal"] for item in batch])
    sequences = torch.stack([item["sequence"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    result = {
        "signal": signals,
        "sequence": sequences,
        "label": labels,
    }

    # Add features if present
    if "features" in batch[0]:
        features = torch.stack([item["features"] for item in batch])
        result["features"] = features

    # Add confound labels if present (adversarial training)
    if "confound_label" in batch[0]:
        result["confound_label"] = torch.stack([item["confound_label"] for item in batch])

    # Add CL regression targets if present
    if "cl_target" in batch[0]:
        result["cl_target"] = torch.stack([item["cl_target"] for item in batch])

    return result


def _usable_cpus() -> int:
    """CPUs this process is allowed to run on, not CPUs the machine has."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return os.cpu_count() or 1


def resolve_dataloader_workers(num_workers: int, device: str) -> int:
    """Resolve how many DataLoader workers to actually use.

    ``num_workers=0`` means AUTO here, not "no workers": on CUDA it becomes
    ``AUTO_DATALOADER_WORKERS``, on CPU it stays 0. Feeding a GPU from the main
    process serializes collate, host-to-device copy and forward pass onto one
    core, which is how ``eval test`` sat at 8% GPU on an A5000 (issue #205).
    On CPU the workers would compete with the compute for the same cores, and
    ``__getitem__`` is trivially fast against pre-tensorized data, so they only
    add overhead.

    The daemon check is not an optimization: daemonic processes (a
    ``multiprocessing.Pool`` worker, as in grid search) cannot spawn children,
    so a DataLoader with workers raises there. Every caller that builds a
    loader goes through this function, so that guard lives in one place.

    The auto count is capped by the CPUs this process may actually run on --
    ``sched_getaffinity``, which respects the Slurm cpuset -- because a GPU job
    allocated 2 cores would otherwise fork 8 workers onto them and thrash. An
    explicit request is honoured as given; only "auto" is capped.
    """
    import multiprocessing

    is_daemon = multiprocessing.current_process().daemon
    if is_daemon:
        effective = 0
    elif num_workers > 0:
        effective = num_workers
    elif device == "cpu":
        effective = 0
    else:
        effective = min(AUTO_DATALOADER_WORKERS, max(1, _usable_cpus() - 1))

    logger.info(
        f"DataLoader workers: {effective} "
        f"(requested={num_workers}, daemon={is_daemon}, device={device})"
    )
    return effective


def resolve_val_dataloader_workers(val_dataset, num_workers: int, device: str) -> int:
    """Workers for the VALIDATION loader.

    Same rule as [`resolve_dataloader_workers`], with one exception: a dataset
    that fell back to per-chunk lists gets 0.

    Validation used to be hardcoded to 0 on the grounds that its `__getitem__`
    is trivially fast. That does not follow -- collate, pin, host-to-device and
    the forward pass still serialize onto one core, which cost ~5 minutes of
    near-idle GPU at every epoch boundary on a 1.18M-chunk val set (issue #207,
    the same failure as #205 for `eval test`).

    The memory half of the old rationale is real but narrow. `LeechDataset`
    stacks per-chunk tensors into contiguous buffers precisely so a fork
    COW-shares them; only the `_try_stack` list fallback makes each worker
    fault N PyObject headers into private copies and multiply peak RSS. So the
    exception is scoped to exactly that case rather than applied to every run.
    """
    workers = resolve_dataloader_workers(num_workers, device)
    if workers and getattr(val_dataset, "_signals_tensor", None) is None:
        logger.info(
            "Validation dataset is not contiguously stacked; using 0 DataLoader "
            "workers to avoid multiplying peak RSS across forks"
        )
        return 0
    return workers


class SignalDataset(Dataset):
    """Minimal signal-only dataset for ``SignalCNN``.

    Yields the leech batch contract (``signal``, a dummy ``sequence``, ``label``)
    so ``collate_fn`` and ``Trainer`` work unchanged for signal-only
    classification (e.g. barcode demux from the adapter signal). ``X`` is
    ``(N, L)`` or ``(N, 1, L)``; ``y`` is integer class labels.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, None, :]
        self.X = torch.from_numpy(X)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        # SignalCNN ignores sequence; a tiny placeholder keeps collate_fn happy.
        self._dummy_seq = torch.zeros(1, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"signal": self.X[idx], "sequence": self._dummy_seq, "label": self.y[idx]}
