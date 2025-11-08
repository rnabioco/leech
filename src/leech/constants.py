"""
Constants for the leech package.

This module provides a single source of truth for default values,
magic numbers, and configuration constants used throughout the codebase.
"""

# Signal processing defaults
DEFAULT_SIGNAL_CONTEXT = (200, 200)  # (left, right) signal samples around focus base
DEFAULT_KMER_CONTEXT = 5  # Number of bases on each side of focus base

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

# Model defaults
DEFAULT_SIGNAL_LEN = 400  # Default signal chunk length
DEFAULT_KMER_LEN = 11  # Default k-mer length (2*context+1)
DEFAULT_NUM_FEATURES = 5  # Default number of feature channels (dwell + signal levels)

# Device defaults
DEFAULT_DEVICE = "cuda"
DEFAULT_SEED = 42

# Inference defaults
DEFAULT_MIN_MAPQ = 0  # Minimum mapping quality for inference
DEFAULT_MOTIF = "CCA"  # Default motif for tRNA 3' end
DEFAULT_MOTIF_OFFSET = 2  # Default offset within motif (0-indexed)

# Normalization methods
NORMALIZATION_METHODS = ["median_mad", "zscore", "quantile"]
DEFAULT_NORMALIZATION = "median_mad"
