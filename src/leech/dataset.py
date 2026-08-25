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
from dataclasses import dataclass
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

    def _flush(self) -> None:
        if not self._pending:
            return
        rows = len(self._pending)
        torch.stack(self._pending, out=self._tensor[self._n : self._n + rows])
        self._n += rows
        self._pending.clear()

    def _degrade(self, tensor: torch.Tensor) -> None:
        """Fall back to a list of per-chunk tensors, keeping what was filled."""
        logger.warning(
            "%s shapes differ (%s vs %s), falling back to list access",
            self.name,
            tuple(tensor.shape),
            tuple(self._tensor.shape[1:]),
        )
        self._flush()
        self._items = [self._tensor[i].clone() for i in range(self._n)]
        self._tensor = None
        self._items.append(tensor)

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
    """

    path: Path
    members: dict[str, str]
    dwell_width: int | None
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
        return cls(path=path, members=wanted, dwell_width=dwell_width)

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

        # Detect multi-class: if any label_int > 1, use long dtype for CrossEntropyLoss
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

        # Arrays come either from the chunk dicts (pre-loaded / legacy corpus)
        # or a row-block stream over the npz. Both yield the same field names,
        # so the loop below does not care which.
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
                fill_encoded_seqs.append(self._encode_sequence(sequence))

            # Pre-create label tensor: long for multi-class, float for binary
            if self._multiclass:
                fill_labels.append(torch.tensor(chunk["label_int"], dtype=torch.long))
            else:
                fill_labels.append(torch.tensor([chunk["label_int"]], dtype=torch.float32))

            # Pre-tensorize signal: pad/crop once instead of every __getitem__ call
            signal = arrays["signal"]
            if signal.dtype != np.float32:
                signal = signal.astype(np.float32)
            signal_residual = arrays.get("signal_residual")
            if signal_residual is not None and signal_residual.dtype != np.float32:
                signal_residual = signal_residual.astype(np.float32)
            fill_signals.append(
                self._prepare_signal(
                    signal, signal_residual, focus_signal_pos=chunk.get("focus_signal_pos")
                )
            )

            # Pre-tensorize features: apply dwell_offset slicing once
            if self._needs_features:
                dwell_width = (
                    stream_dwell_width if stream_dwell_width is not None else len(chunk["dwell"])
                )
                fill_features.append(self._prepare_features(arrays["features"], dwell_width, chunk))

            # Confound label for adversarial training. The encoder reads the
            # configured chunk field and maps it to a class int (-1 = ignore).
            if self._confound_encoder is not None:
                confound_class = self._confound_encoder.encode(chunk)
                fill_confounds.append(torch.tensor(confound_class, dtype=torch.long))

            # CL regression target (cl_value / 255.0; sentinel -1.0 for missing)
            if self._cl_regression:
                cl_val = chunk.get("cl_value")
                if cl_val is not None and cl_val >= 0:
                    fill_cl_targets.append(torch.tensor(cl_val / 255.0, dtype=torch.float32))
                else:
                    fill_cl_targets.append(torch.tensor(-1.0, dtype=torch.float32))

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
        self._seq_ints_tensor: torch.Tensor | None = None
        self._seq_to_sig_tensor: torch.Tensor | None = None
        if self._effective_seq_encoding == "signal_kmer" and self._seq_ints:
            max_seq_ints_len = max(s.shape[0] for s in self._seq_ints)
            n = len(self._seq_ints)
            padded_seq_ints = np.full((n, max_seq_ints_len), -1, dtype=np.int8)
            for i, si in enumerate(self._seq_ints):
                padded_seq_ints[i, : si.shape[0]] = si
            self._seq_ints_tensor = torch.from_numpy(padded_seq_ints)
            self._seq_ints = []

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

        _n_encoded = (
            self._encoded_seqs_tensor.shape[0]
            if self._encoded_seqs_tensor is not None
            else len(self._encoded_seqs)
        )
        logger.debug(
            f"Pre-tensorized {len(self.chunks)} chunks "
            f"({_n_encoded} sequences encoded, encoding={self._effective_seq_encoding})"
        )

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
        focus = np.zeros(len(rows), dtype=np.int64)
        missing: list[int] = []
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


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched tensors
    """
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
