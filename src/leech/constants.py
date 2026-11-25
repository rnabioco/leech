"""
Constants for the leech package.

This module provides a single source of truth for default values,
magic numbers, and configuration constants used throughout the codebase.
"""

import json
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path

# Required BAM tags for nanopore signal analysis
REQUIRED_BAM_TAGS = ["mv", "ns"]

# How many reads `iter_read_info_batches` checks before it warns that a BAM
# carries neither the uppercase `CL` nor lowercase `cl` charging-level tag
# (issue #254). One warning per BAM, not per read -- a corpus prepared from a
# BAM missing the tag entirely gets `cl_value = None` (-> -1, the "missing"
# sentinel) for every chunk, which silently zeroes the `--cl-regression`
# gradient unless something says so.
CL_TAG_CHECK_READS = 1000

# Class value written to the prediction tag when a read falls below
# --min-confidence or --min-margin.
BELOW_THRESHOLD_LABEL = "unc"

# Signal processing defaults
DEFAULT_SIGNAL_CONTEXT = (225, 225)  # (left, right) signal samples around focus base
DEFAULT_KMER_CONTEXT = 5  # Number of bases on each side of focus base
DEFAULT_DWELL_MARGIN = 15  # Extra bases on each side for dwell_offset tuning (symmetric fallback)

# Conservative slow-read translocation rate (samples/base), used only to size
# the fixed `--signal-len` window `--signal-context-bases L,R` needs when the
# caller does not pass `--signal-len` explicitly (issue #278). A sample-based
# window can't reach the same *bases* of context on a fast read (~24
# samples/base) and a slow one (~36 samples/base) without either missing
# context on the slow read or overrunning it on the fast one -- sizing off the
# slow-read rate means a typical read's window is narrower than `signal_len`
# and gets zero-padded (the conservative default) rather than centre-cropped.
DEFAULT_MAX_SAMPLES_PER_BASE = 36

# Fast-read translocation rate floor (samples/base), the documented "fast"
# end (~24 samples/base). The label-divergence guard (`leech.leakage`) bounds
# how many bases a sample-defined signal window can reach by assuming a read
# no faster than this; a faster read reaches further. `--signal-context-bases`
# is exact and needs no floor.
DEFAULT_MIN_SAMPLES_PER_BASE = 24.0

# Model architecture defaults
DEFAULT_CONV_CHANNELS = [4, 16, 256]  # Channel sizes for conv layers
DEFAULT_SIGNAL_KERNEL = 5  # Kernel size for signal branch convolutions
DEFAULT_SEQ_KERNEL = 3  # Kernel size for sequence branch convolutions
DEFAULT_FEATURE_KERNEL = 3  # Kernel size for feature branch convolutions
DEFAULT_LSTM_HIDDEN = 96  # Hidden size for BiLSTM layers
DEFAULT_DROPOUT = 0.1  # Dropout probability
DEFAULT_LSTM_LAYERS = 2  # Number of LSTM layers
DEFAULT_FC_HIDDEN = 64  # Hidden size for fully connected layers

# Training defaults
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
# Workers a DataLoader gets when the caller asks for 0 ("auto") on a GPU.
# Resolved by ``dataset.resolve_dataloader_workers``; 0 on CPU.
AUTO_DATALOADER_WORKERS = 8

# Advanced training defaults
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MAX_GRAD_NORM = 0.0  # 0 = disabled
DEFAULT_QUANTILE_GRAD_CLIP = False  # Adaptive clipping at a quantile of recent grad norms
DEFAULT_GRAD_ACCUM_SPLIT = 1  # Sub-batches per optimizer step (1 = no accumulation)
DEFAULT_SAVE_OPTIM_EVERY = 1  # Write optimizer state every N epochs (1 = every checkpoint)
DEFAULT_SCHEDULER = "none"  # "none", "reduce_on_plateau", or "cosine"
DEFAULT_SCHEDULER_PATIENCE = 5
DEFAULT_SCHEDULER_FACTOR = 0.5
DEFAULT_WARMUP_EPOCHS = 0
DEFAULT_LOSS_TYPE = "bce"  # "bce", "focal", "cross_entropy", or "noise_corrected_bce"
DEFAULT_FOCAL_GAMMA = 2.0
# None = symmetric focal loss (neg_gamma == gamma), bit-for-bit identical to
# the pre-#280 loss. See FocalBCEWithLogitsLoss.
DEFAULT_FOCAL_NEG_GAMMA = None
DEFAULT_LABEL_SMOOTHING = 0.0  # 0 = disabled; e.g., 0.05 softens 0/1 targets
DEFAULT_MIXED_PRECISION = False
DEFAULT_AUGMENT_JITTER = 0.0
DEFAULT_AUGMENT_SCALE_MIN = 1.0
DEFAULT_AUGMENT_SCALE_MAX = 1.0
DEFAULT_AUGMENT_TIME_MASK_BASES = 0  # Max width in bases per mask (0 = disabled)
DEFAULT_AUGMENT_TIME_MASK_COUNT = 1  # Number of masks to apply
DEFAULT_AUGMENT_SHIFT_MAX_BASES = (
    0.0  # Max cross-layer shift in bases (0 = disabled, float for sub-base)
)
DEFAULT_AUGMENT_FEATURE_NOISE_SCALE = 0.0  # Per-channel Gaussian noise scale (0 = disabled)
DEFAULT_STANDARDIZE_FEATURES = False  # Per-channel feature standardization (frozen affine layer)
# Floor for a standardized channel's corpus std, so a near-constant channel
# (dwell_ratio is close to 1 everywhere in some corpora) divides by something
# other than ~0 instead of blowing up to +/-inf.
FEATURE_STANDARDIZE_EPS = 1e-6
DEFAULT_AUGMENT_TIME_STRETCH = (1.0, 1.0)  # (min, max) per-sample stretch factor (disabled)

# Model defaults
DEFAULT_SIGNAL_LEN = 400  # Default signal chunk length
DEFAULT_KMER_LEN = 11  # Default k-mer length (2*context+1)
DEFAULT_NUM_FEATURES = 5  # Default number of feature channels (dwell + signal levels)
DEFAULT_CAUSAL_TCN = True  # TemporalBlock padding: causal (left-only) by default

# Device defaults
DEFAULT_DEVICE = "cuda"
DEFAULT_SEED = None  # Generate random seed by default to avoid "seed=42" cargo-culting

# torch.compile gate shared by `predict` (inference/single.py) and
# `eval test` (evaluation.py), one policy for both (#264): below this many
# samples (reads for predict, test chunks for eval), compilation overhead
# (~15-30s) outweighs the speedup, so both auto-skip. Never used on CPU.
TORCH_COMPILE_MIN_SAMPLES = 5000


def generate_random_seed() -> int:
    """
    Generate a cryptographically random seed for reproducible randomness.

    Returns a 32-bit unsigned integer suitable for use with numpy, random, and torch.

    This function should be called when no explicit seed is provided, ensuring
    each run uses a different seed by default while still being reproducible
    via the logged seed value.

    Returns:
        Random integer in range [0, 2^32-1]
    """
    import secrets

    return secrets.randbits(32)


# Sequence encoding defaults
DEFAULT_SIGNAL_KMER_CONTEXT = (4, 4)  # Kmer context for signal-level kmer encoding

# What `seq_encoding` means for a checkpoint whose config lacks the key.
# Every consumer of an old/bare config must agree on this fallback (issue
# #269) -- `base_onehot`, because that is the model classes' OWN default
# (see e.g. `models/configs/conv_lstm.toml`'s `seq_encoding = "base_onehot"`
# [params] entry): a model built via `get_model(name, **kwargs)` with no
# seq_encoding kwarg is a base_onehot model, full stop. `--seq-encoding
# signal_kmer` is only the *CLI*'s default for `model train`, which always
# records the effective encoding it used into config.json -- so a config
# that lacks the key was never touched by that flag at all.
DEFAULT_SEQ_ENCODING_FALLBACK = "base_onehot"

# The feature-channel list every corpus/checkpoint used before it became a
# recorded, configurable setting (issue #358) -- `leech data prepare
# --feature-channels`'s default, verbatim, so nothing already prepared or
# trained changes. Row order here IS the model's input channel order: dwell
# family first (`compute_dwell_features`'s fixed output order), then the
# level-span stats, then the three k-mer residual rows (present only when a
# k-mer level table is configured -- see `merge_feature_channels`). Must match
# `rust/src/inference_pipeline/types.rs::DEFAULT_FEATURE_CHANNELS` exactly;
# `tests/test_data_prep.py::TestFeatureChannelOrder` pins the two together.
DEFAULT_FEATURE_CHANNELS: list[str] = [
    "dwell",
    "dwell_log",
    "dwell_mean",
    "dwell_std",
    "dwell_ratio",
    "level_mean",
    "level_median",
    "level_std",
    "level_range",
    "kmer_expected",
    "kmer_residual",
    "kmer_residual_abs",
]

# Every channel name either backend (Rust via escapepod's `FeatureChannel`,
# Python via `compute_dwell_features`/`compute_signal_features`) knows how to
# compute. `level_skew`/`level_kurtosis` (escapepod-rs#396) are known but not
# in the default list -- opt in with `--feature-channels`. `data prepare`
# refuses any name outside this set before touching a read (issue #358); a
# typo here would otherwise silently vanish at `merge_feature_channels`
# instead of raising.
KNOWN_FEATURE_CHANNELS: frozenset[str] = frozenset(
    [*DEFAULT_FEATURE_CHANNELS, "level_skew", "level_kurtosis"]
)

# `leech data prepare --feature-channels all` expands to this (issue #358
# follow-up: channel ablation) -- the historical twelve in their order, then
# the two opt-in channels. Recorded in `prepare_config.json` as this expanded
# list, never as the literal word "all", so every downstream reader
# (`model train`, `leech predict`, `data merge`) sees ordinary channel names
# and needs no special case for the token.
ALL_FEATURE_CHANNELS: list[str] = [*DEFAULT_FEATURE_CHANNELS, "level_skew", "level_kurtosis"]

# The only channels `merge_feature_channels` (chunking/extractor.py) ever
# drops from a recorded list -- silently, when no k-mer level table is
# configured. `resolve_stored_feature_channels` below applies the same rule
# in reverse, reconciling a corpus's recorded list against the actual row
# count of its stored feature tensor.
KMER_RESIDUAL_FEATURE_CHANNELS: frozenset[str] = frozenset(
    ["kmer_expected", "kmer_residual", "kmer_residual_abs"]
)


def validate_feature_channels(names: Iterable[str]) -> None:
    """Raise ``ValueError`` naming any channel neither backend can compute.

    Called at ``data prepare`` time (issue #358), before any read is
    touched, and defensively from
    :meth:`~leech.configs.SignalConfig.__post_init__` so a config built
    directly through the Python API (not the CLI) gets the same fail-fast
    check. Do not rely on ``merge_feature_channels`` to catch this instead --
    it silently drops a name that matches no computed column, which is
    exactly how a typo like ``"level_min"`` would vanish rather than raise.
    """
    unknown = [n for n in names if n not in KNOWN_FEATURE_CHANNELS]
    if unknown:
        raise ValueError(
            f"Unknown feature channel(s): {unknown}. Known channels: "
            f"{sorted(KNOWN_FEATURE_CHANNELS)}"
        )


def resolve_stored_feature_channels(recorded: Iterable[str], actual_row_count: int) -> list[str]:
    """Reconcile a corpus's *recorded* feature-channel list against its stored row count.

    ``prepare_config.json``'s ``feature_channels`` (issue #358) is the list
    requested at ``data prepare`` time, verbatim -- it is not filtered down to
    what actually landed in the npz. The only rows ``merge_feature_channels``
    (chunking/extractor.py) ever drops from that list are the three k-mer
    residual channels, and only when the run had no k-mer level table
    configured -- a property of the run this function doesn't have direct
    access to, so it infers presence from the row-count mismatch instead.

    Used when selecting a channel subset at train time (channel ablation):
    the row a channel name maps to in the stored tensor must be its position
    among the channels that actually survived prepare, not its position in
    the recorded list.

    Returns the subset of ``recorded`` that actually exists, in recorded
    order. Raises ``ValueError`` if dropping the k-mer residual names still
    doesn't reconcile the counts -- a mismatch this rule cannot explain.
    """
    recorded = list(recorded)
    if len(recorded) == actual_row_count:
        return recorded
    reduced = [n for n in recorded if n not in KMER_RESIDUAL_FEATURE_CHANNELS]
    if len(reduced) == actual_row_count:
        return reduced
    raise ValueError(
        f"Corpus's recorded feature_channels ({len(recorded)}: {recorded}) does not "
        f"match its stored feature row count ({actual_row_count}), even after "
        f"dropping the k-mer residual channels that merge_feature_channels can drop "
        f"({sorted(KMER_RESIDUAL_FEATURE_CHANNELS)}). This corpus may be corrupt or "
        "from an incompatible leech version."
    )


def select_feature_channels(available: Sequence[str], requested: Iterable[str]) -> list[int]:
    """Row indices into ``available`` for each name in ``requested``, in requested order.

    Used both to validate a ``--feature-channels`` subset against a corpus's
    channel list (the indices are then discarded, only the validation
    matters) and, in :class:`~leech.dataset.LeechDataset`, to actually select
    and reorder the stored feature tensor's rows.

    Raises ``ValueError`` naming the missing channels and ``available``
    itself when any requested name is not in ``available`` -- e.g. a typo, or
    a k-mer residual channel this corpus's actual data doesn't have (see
    :func:`resolve_stored_feature_channels`).
    """
    requested = list(requested)
    index = {name: i for i, name in enumerate(available)}
    missing = [n for n in requested if n not in index]
    if missing:
        raise ValueError(
            f"Unknown feature channel(s) for this corpus: {missing}. "
            f"Corpus feature channels ({len(available)}): {list(available)}"
        )
    return [index[n] for n in requested]


def corpus_recorded_feature_channels(corpus_path: Path) -> list[str]:
    """A corpus's recorded ``feature_channels`` list, from its
    ``prepare_config.json`` sidecar (falling back to the historical twelve
    when the sidecar is missing or predates issue #358) -- the read side of
    the same convention ``training.py``'s own ``_train_feature_channels``
    uses.

    Public (not just an implementation detail of
    :func:`resolve_corpus_feature_selection`): any caller that also passes
    ``selected_feature_channels`` to a fresh :class:`~leech.dataset.LeechDataset`
    (``leech eval test``, ``leech model calibrate``, ``leech model
    benchmark``) must pass this SAME list as ``feature_channel_names`` too --
    ``LeechDataset`` needs it to know the corpus's own nominal list, to
    reconcile against the actual stored row count and resolve
    ``selected_feature_channels`` by name. Passing neither (the default,
    ``None`` -- the historical twelve) is wrong for any corpus whose real
    recorded list differs, which is exactly the ablation case this whole
    feature is for.
    """
    sidecar = corpus_path.parent / "prepare_config.json"
    if sidecar.exists():
        with open(sidecar) as f:
            recorded = json.load(f).get("feature_channels")
        if recorded is not None:
            return list(recorded)
    return list(DEFAULT_FEATURE_CHANNELS)


def _corpus_actual_feature_row_count(corpus_path: Path) -> int | None:
    """The corpus's actual stored ``features`` channel count, or ``None`` if
    it cannot be determined cheaply (missing/corrupt file). Mirrors
    :meth:`~leech.dataset.LeechDataset._stored_feature_row_count`, but from a
    bare path rather than an already-open dataset: the fast path reads only
    the flat member's ``.npy`` header (no data), falling back to a full
    ``load_chunks`` for a corpus written before the flat array format.
    """
    from leech.chunking import load_chunks, npz_array_members

    try:
        members = npz_array_members(corpus_path)
    except (OSError, ValueError, zipfile.BadZipFile):
        return None
    shape_dtype = members.get("features_flat")
    if shape_dtype is not None and len(shape_dtype[0]) >= 2:
        return shape_dtype[0][1]
    try:
        chunks = load_chunks(corpus_path)
    except (OSError, ValueError, zipfile.BadZipFile):
        return None
    for chunk in chunks:
        feat = chunk.get("features") if isinstance(chunk, dict) else None
        if feat is not None:
            return int(feat.shape[0])
    return None


def resolve_corpus_feature_selection(
    wanted: Sequence[str] | None, corpus_path: Path | str
) -> list[str] | None:
    """The ``LeechDataset(selected_feature_channels=...)`` a consumer should
    pass to score/benchmark ``corpus_path`` against a ``wanted`` channel
    list -- the shared rule behind every place that builds a
    :class:`~leech.dataset.LeechDataset` from a corpus for something OTHER
    than ``model train``'s own corpus: ``leech eval test``
    (:func:`~leech.evaluation.evaluate_model`), ``leech model calibrate``
    (:func:`~leech.calibration.calibrate_model` and
    :func:`~leech.calibration.calibrate_model_multiclass`), and ``leech model
    benchmark`` (:func:`~leech.commands.benchmark.handle_benchmark`).

    Channel ablation (issue #358 follow-up) means a checkpoint's declared
    channel list can be a strict SUBSET of what a given corpus actually
    carries -- every arm of an ablation sweep is typically scored against
    the SAME all-channel corpus prepared once with ``--feature-channels
    all``. Without this, each of those four callers builds a dataset with
    every row the corpus has, which is the wrong width for a checkpoint
    trained on a subset (a channel-count crash at best; a silent misalign at
    worst, if two corpora happen to share a row count by coincidence).

    Args:
        wanted: The channel list to score against, in order -- typically a
            checkpoint's ``config.json["feature_channels"]`` (``None``
            passed in by the *caller* already means "the historical twelve",
            since that key is always populated by a config written after
            issue #358; a caller with no checkpoint at all, e.g. an ad hoc
            ``leech model benchmark --model-config`` snippet with no
            ``feature_channels`` key, should pass literal ``None`` here to
            mean "no selection was requested" instead). ``None`` returns
            ``None`` unconditionally, without even opening the corpus.
        corpus_path: The corpus file that will be handed to
            :class:`~leech.dataset.LeechDataset` (a ``train.npz``/
            ``val.npz`` or test corpus). Its ``prepare_config.json`` sidecar
            and its own stored ``features`` row count are both read here,
            cheaply (header/shape only, except for a legacy pre-flat-format
            corpus).

    Returns:
        ``None`` when ``wanted`` already equals the corpus's resolved list
        (:func:`resolve_stored_feature_channels`) -- pass this straight
        through as ``selected_feature_channels``, taking ``LeechDataset``'s
        identity/no-op path, so every EXISTING checkpoint/corpus pair (which
        never disagree) is unaffected byte for byte. Otherwise, ``wanted``
        itself, to pass as ``selected_feature_channels`` -- ``LeechDataset``
        then selects/reorders those rows, refusing clearly
        (:func:`select_feature_channels`) if the corpus's actual data lacks
        one of them. This is also what lets an OLD checkpoint (no recorded
        key, so the historical twelve) evaluate correctly against a NEW
        ``--feature-channels all`` corpus that carries more rows than that.
    """
    if wanted is None:
        return None
    wanted = list(wanted)
    corpus_path = Path(corpus_path)
    recorded = corpus_recorded_feature_channels(corpus_path)
    actual_count = _corpus_actual_feature_row_count(corpus_path)
    try:
        corpus_resolved = (
            resolve_stored_feature_channels(recorded, actual_count)
            if actual_count is not None
            else recorded
        )
    except ValueError:
        # The recorded list and the corpus's actual row count disagree in a
        # way this rule can't explain (a corpus older than any of this
        # feature, or one with a channel set this reconciliation wasn't
        # built to describe -- e.g. a hand-built test fixture). Identity is
        # the safe default: LeechDataset's un-selected path already handles
        # whatever the corpus actually has, exactly as it did before this
        # feature existed, so a caller here that can't confidently prove a
        # real mismatch must not force a NEW failure mode onto it.
        return None
    return None if wanted == corpus_resolved else wanted


# Signal map refinement defaults.
#
# The single source of truth for the refine-half-bandwidth default (5, not
# 300 -- a prior version of this constant was simply wrong and unread
# everywhere). `configs.SignalConfig.refine_half_bandwidth` and every
# `config.get("refine_half_bandwidth", ...)` fallback in the inference and
# training paths reference this constant so there is exactly one place that
# says "5".
DEFAULT_REFINE_HALF_BANDWIDTH = 5

# Remora model defaults
DEFAULT_REMORA_SIZE = 64
