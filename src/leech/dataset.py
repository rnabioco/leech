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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from leech.chunking import load_chunks
from leech.features import encode_signal_kmer, sequence_to_int
from leech.models.inference_wrapper import ModelInferenceWrapper

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
    df = pd.read_csv(table_path, sep="\t")
    aa_list = sorted(df["aa"].unique())
    n_aa = len(aa_list)
    aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}

    min_pos = int(df["position"].min())
    max_pos = int(df["position"].max())

    # Grand mean dwell at each position (fallback for missing entries)
    grand_mean = df.groupby("position")["dwell_mean"].mean()

    n_positions = max_pos - min_pos + 1
    template = np.ones((n_aa, n_positions), dtype=np.float32)
    for _, row in df.iterrows():
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
        confound_map: dict[int, int] | None = None,
        ref_confound_map: dict[str, int] | None = None,
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
            confound_map: Optional mapping from ``label_int`` to confound class
                (e.g. discriminator base identity).  When provided, each batch
                includes a ``confound_label`` tensor for adversarial training.
                Labels not found in the map get ``-1`` (ignored by CE loss).
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

        # Use pre-loaded chunks or load from file
        if chunks is not None:
            logger.info(f"Using {len(chunks)} pre-loaded chunks (skipping disk I/O)")
            self.chunks = chunks
        elif chunk_path is not None:
            self.chunks = load_chunks(chunk_path)
        else:
            raise ValueError("Either chunk_path or chunks must be provided")

        # Filter chunks with valid numeric labels (label_int)
        self.chunks = [c for c in self.chunks if c["label_int"] is not None]

        if len(self.chunks) == 0:
            raise ValueError(f"No valid chunks found{f' in {chunk_path}' if chunk_path else ''}")

        # Pre-tensorize: encode sequences, labels, signals, and features once
        self._encoded_seqs: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._signals: list[torch.Tensor] = []
        self._features: list[torch.Tensor] = []
        self._needs_features = model_type in FEATURE_MODELS
        self._confound_map = confound_map
        self._ref_confound_map = ref_confound_map
        self._has_confound = confound_map is not None or ref_confound_map is not None
        self._confound_labels: list[torch.Tensor] = []
        self._cl_regression = cl_regression
        self._cl_targets: list[torch.Tensor] = []

        # Determine effective encoding: fall back to base_onehot if chunks lack signal_kmer data
        self._effective_seq_encoding = seq_encoding
        if seq_encoding == "signal_kmer":
            first = self.chunks[0]
            if not first.get("seq_to_sig_map") is not None or not first.get(
                "sequence_with_kmer_context"
            ):
                logger.warning(
                    "Chunks lack seq_to_sig_map/sequence_with_kmer_context; "
                    "falling back to base_onehot encoding"
                )
                self._effective_seq_encoding = "base_onehot"

        # Detect signal_residual channel and apply signal_mode
        self._signal_mode = signal_mode
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

        for chunk in self.chunks:
            if self._effective_seq_encoding == "signal_kmer":
                seq_ctx = chunk["sequence_with_kmer_context"]
                seq_ints = sequence_to_int(seq_ctx).astype(np.int8)
                self._seq_ints.append(seq_ints)

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
                self._encoded_seqs.append(self._encode_sequence(chunk["sequence"]))

            # Pre-create label tensor: long for multi-class, float for binary
            if self._multiclass:
                self._labels.append(torch.tensor(chunk["label_int"], dtype=torch.long))
            else:
                self._labels.append(torch.tensor([chunk["label_int"]], dtype=torch.float32))

            # Pre-tensorize signal: pad/crop once instead of every __getitem__ call
            signal = chunk["signal"]
            if signal.dtype != np.float32:
                signal = signal.astype(np.float32)
            signal_residual = chunk.get("signal_residual")
            if signal_residual is not None and signal_residual.dtype != np.float32:
                signal_residual = signal_residual.astype(np.float32)
            self._signals.append(
                self._prepare_signal(
                    signal, signal_residual, focus_signal_pos=chunk.get("focus_signal_pos")
                )
            )

            # Pre-tensorize features: apply dwell_offset slicing once
            if self._needs_features:
                self._features.append(self._prepare_features(chunk))
            else:
                self._features.append(torch.empty(0))

            # Confound label for adversarial training. Two modes:
            # 1. label-level (disc_base): confound_map maps label_int → class
            # 2. chunk-level (trna_id): ref_confound_map maps reference_name → class
            if self._has_confound:
                if self._ref_confound_map is not None:
                    ref_name = chunk.get("reference_name", "")
                    confound_class = self._ref_confound_map.get(ref_name, -1)
                else:
                    label_int = chunk["label_int"]
                    confound_class = confound_map.get(label_int, -1)
                self._confound_labels.append(torch.tensor(confound_class, dtype=torch.long))

            # CL regression target (cl_value / 255.0; sentinel -1.0 for missing)
            if self._cl_regression:
                cl_val = chunk.get("cl_value")
                if cl_val is not None and cl_val >= 0:
                    self._cl_targets.append(torch.tensor(cl_val / 255.0, dtype=torch.float32))
                else:
                    self._cl_targets.append(torch.tensor(-1.0, dtype=torch.float32))

        # Stack every per-chunk list into one contiguous tensor. This isn't just
        # for cache friendliness — it's required for fork-safety. A DataLoader
        # with num_workers > 0 forks worker processes that COW-inherit the
        # parent's address space. CPython refcounts live inside each PyObject
        # header, so a worker iterating a list of N tensors writes to N
        # separate page-resident headers and faults every page into a private
        # copy, multiplying peak RSS by (1 + num_workers). A single contiguous
        # tensor keeps its data buffer outside Python's GC, so the buffer
        # pages are genuinely shared across the fork.
        def _try_stack(name: str, items: list[torch.Tensor]) -> torch.Tensor | None:
            if not items:
                # Expected when a feature mode isn't active (e.g. signal_kmer
                # leaves _encoded_seqs empty); the consumer will use a different
                # tensor instead.
                return None
            try:
                return torch.stack(items)
            except RuntimeError as e:
                logger.warning("%s shapes differ, falling back to list access: %s", name, e)
                return None

        self._signals_tensor = _try_stack("Signal", self._signals)
        if self._signals_tensor is not None:
            self._signals = []

        self._encoded_seqs_tensor = _try_stack("Encoded-sequence", self._encoded_seqs)
        if self._encoded_seqs_tensor is not None:
            self._encoded_seqs = []

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
            max_s2s_len = max(s.shape[0] for s in self._seq_to_sig)
            n = len(self._seq_ints)
            padded_seq_ints = np.full((n, max_seq_ints_len), -1, dtype=np.int8)
            padded_s2s = np.full((n, max_s2s_len), signal_len, dtype=np.int64)
            for i, (si, s2s) in enumerate(zip(self._seq_ints, self._seq_to_sig, strict=True)):
                padded_seq_ints[i, : si.shape[0]] = si
                padded_s2s[i, : s2s.shape[0]] = s2s
            self._seq_ints_tensor = torch.from_numpy(padded_seq_ints)
            self._seq_to_sig_tensor = torch.from_numpy(padded_s2s)
            self._seq_ints = []
            self._seq_to_sig = []

        self._labels_tensor = _try_stack("Label", self._labels)
        if self._labels_tensor is not None:
            self._labels = []

        self._features_tensor: torch.Tensor | None = None
        if self._needs_features:
            self._features_tensor = _try_stack("Feature", self._features)
            if self._features_tensor is not None:
                self._features = []

        self._confound_labels_tensor: torch.Tensor | None = None
        if self._has_confound:
            self._confound_labels_tensor = _try_stack("Confound", self._confound_labels)
            if self._confound_labels_tensor is not None:
                self._confound_labels = []

        self._cl_targets_tensor: torch.Tensor | None = None
        if self._cl_regression:
            self._cl_targets_tensor = _try_stack("CL target", self._cl_targets)
            if self._cl_targets_tensor is not None:
                self._cl_targets = []

        # Drop the raw numpy arrays from self.chunks now that everything has
        # been pre-tensorized. External code (samplers, label tally, feature
        # window introspection) still reads the small scalar/string fields,
        # so we keep self.chunks alive but null out the per-chunk arrays.
        # Without this, each chunk dict keeps a ~50 KB numpy view alive and
        # the same COW blowup hits during DataLoader fork.
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

    def _prepare_features(self, chunk: dict) -> torch.Tensor:
        """Apply dwell_offset slicing and tensorize features. Called once during __init__."""
        dwell = chunk["dwell"]
        features = chunk["features"]
        if features.dtype != np.float32:
            features = features.astype(np.float32)

        # Append 20 dwell template ratio channels before slicing
        features = self._append_template_channels(features, chunk)

        kmer_context = self.kmer_len // 2

        if self.model_type in WIDE_FEATURE_MODELS:
            pass  # full-width features
        elif len(dwell) > self.kmer_len:
            # Determine feature_start (signed offset from focus).
            # New chunks have it directly; old chunks need conversion.
            if "feature_start" in chunk:
                feat_start = int(chunk["feature_start"])
            elif "feature_left" in chunk:
                feat_start = -int(chunk["feature_left"])
            elif "dwell_margin_left" in chunk:
                feat_start = -(kmer_context + int(chunk["dwell_margin_left"]))
            else:
                feat_start = -(len(dwell) - 1) // 2  # symmetric fallback
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
    def _encode_sequence(sequence: str) -> torch.Tensor:
        """Vectorized one-hot encoding of a DNA sequence.

        Uses a pre-built ASCII lookup table instead of a Python for-loop.

        Args:
            sequence: DNA sequence string (A, C, G, T, N)

        Returns:
            One-hot encoded tensor of shape (4, len(sequence))
        """
        indices = _BASE_MAP[np.frombuffer(sequence.encode(), dtype=np.uint8)]
        encoded = np.zeros((4, len(sequence)), dtype=np.float32)
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
