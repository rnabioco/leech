"""
Constants for the leech package.

This module provides a single source of truth for default values,
magic numbers, and configuration constants used throughout the codebase.
"""

# Required BAM tags for nanopore signal analysis
REQUIRED_BAM_TAGS = ["mv", "ns"]

# Signal processing defaults
DEFAULT_SIGNAL_CONTEXT = (225, 225)  # (left, right) signal samples around focus base
DEFAULT_KMER_CONTEXT = 5  # Number of bases on each side of focus base
DEFAULT_DWELL_MARGIN = 15  # Extra bases on each side for dwell_offset tuning (symmetric fallback)

# Model architecture defaults
DEFAULT_CONV_CHANNELS = [4, 16, 256]  # Channel sizes for conv layers
DEFAULT_SIGNAL_KERNEL = 5  # Kernel size for signal branch convolutions
DEFAULT_SEQ_KERNEL = 3  # Kernel size for sequence branch convolutions
DEFAULT_FEATURE_KERNEL = 3  # Kernel size for feature branch convolutions
DEFAULT_LSTM_HIDDEN = 96  # Hidden size for BiLSTM layers
DEFAULT_DROPOUT = 0.1  # Dropout probability
DEFAULT_LSTM_LAYERS = 2  # Number of LSTM layers
DEFAULT_FC_HIDDEN = 64  # Hidden size for fully connected layers

# Feature names - dwell time features
DWELL_FEATURES = [
    "dwell",  # Raw dwell time (number of signal samples per base)
    "dwell_log",  # Log-transformed dwell time
    "dwell_mean",  # Mean dwell time (sliding window)
    "dwell_std",  # Std dev of dwell time (sliding window)
    "dwell_ratio",  # Ratio to mean dwell time
]

# Feature names - signal level features
SIGNAL_FEATURES = [
    "level_mean",  # Mean signal level per base
    "level_median",  # Median signal level per base
    "level_std",  # Std dev of signal level per base
    "level_range",  # Range (max - min) of signal level per base
]

# Feature names - kmer residual features (requires kmer level table)
KMER_RESIDUAL_FEATURES = [
    "kmer_expected",  # Expected signal level per base from kmer table lookup
    "kmer_residual",  # level_mean - kmer_expected (signed deviation)
    "kmer_residual_abs",  # |kmer_residual| (unsigned magnitude)
]

# Training defaults
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50

# Advanced training defaults
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MAX_GRAD_NORM = 0.0  # 0 = disabled
DEFAULT_SCHEDULER = "none"  # "none", "reduce_on_plateau", or "cosine"
DEFAULT_SCHEDULER_PATIENCE = 5
DEFAULT_SCHEDULER_FACTOR = 0.5
DEFAULT_WARMUP_EPOCHS = 0
DEFAULT_LOSS_TYPE = "bce"  # "bce" or "focal"
DEFAULT_FOCAL_GAMMA = 2.0
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

# Model defaults
DEFAULT_SIGNAL_LEN = 400  # Default signal chunk length
DEFAULT_KMER_LEN = 11  # Default k-mer length (2*context+1)
DEFAULT_NUM_FEATURES = 5  # Default number of feature channels (dwell + signal levels)

# Device defaults
DEFAULT_DEVICE = "cuda"
DEFAULT_SEED = None  # Generate random seed by default to avoid "seed=42" cargo-culting


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

# Normalization methods
NORMALIZATION_METHODS = ["median_mad", "zscore", "quantile", "pa_scaling"]
DEFAULT_NORMALIZATION = "median_mad"

# Anchor modes for data preparation
ANCHOR_MODES = ["basecall", "reference"]
DEFAULT_ANCHOR = "basecall"

# Signal map refinement defaults
DEFAULT_REFINE_HALF_BANDWIDTH = 300
DEFAULT_REFINE_ROUGH_RESCALE = True

# Loss types
LOSS_TYPES = ["bce", "focal", "cross_entropy"]

# Remora model defaults
DEFAULT_REMORA_SIZE = 64
