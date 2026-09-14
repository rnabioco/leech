"""
Constants for the leech package.

This module provides a single source of truth for default values,
magic numbers, and configuration constants used throughout the codebase.
"""

# Required BAM tags for nanopore signal analysis
REQUIRED_BAM_TAGS = ["mv", "ns"]

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
