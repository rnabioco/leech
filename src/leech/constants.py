"""
Constants for the leech package.

This module provides a single source of truth for default values,
magic numbers, and configuration constants used throughout the codebase.
"""

# Signal processing defaults
DEFAULT_SIGNAL_CONTEXT = (200, 200)  # (left, right) signal samples around focus base
DEFAULT_KMER_CONTEXT = 5  # Number of bases on each side of focus base
DEFAULT_DWELL_MARGIN = 15  # Extra bases on each side for dwell_offset tuning

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

# Training defaults
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_NUM_WORKERS = 2  # DataLoader num_workers

# Advanced training defaults
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MAX_GRAD_NORM = 0.0  # 0 = disabled
DEFAULT_SCHEDULER = "none"  # "none" or "reduce_on_plateau"
DEFAULT_SCHEDULER_PATIENCE = 5
DEFAULT_SCHEDULER_FACTOR = 0.5
DEFAULT_WARMUP_EPOCHS = 0
DEFAULT_LOSS_TYPE = "bce"  # "bce" or "focal"
DEFAULT_FOCAL_GAMMA = 2.0
DEFAULT_MIXED_PRECISION = False
DEFAULT_AUGMENT_JITTER = 0.0
DEFAULT_AUGMENT_SCALE_MIN = 1.0
DEFAULT_AUGMENT_SCALE_MAX = 1.0

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
DEFAULT_SEQ_ENCODING = "signal_kmer"  # "signal_kmer" or "base_onehot"
DEFAULT_SIGNAL_KMER_CONTEXT = (4, 4)  # Kmer context for signal-level kmer encoding

# Inference defaults
DEFAULT_MIN_MAPQ = 0  # Minimum mapping quality for inference
DEFAULT_MOTIF = "CCAGGC"  # Default motif for tRNA 3' end
DEFAULT_MOTIF_OFFSET = 2  # Default offset within motif (0-indexed, focuses on A in CCA)

# Normalization methods
NORMALIZATION_METHODS = ["median_mad", "zscore", "quantile"]
DEFAULT_NORMALIZATION = "median_mad"
