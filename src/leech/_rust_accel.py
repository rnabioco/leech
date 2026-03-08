"""
Rust-accelerated functions with pure Python fallbacks.

Imports the Rust ``leech_core`` extension if available, otherwise falls back
to the pure Python/numpy implementations.
"""

import logging

logger = logging.getLogger("leech._rust_accel")

try:
    from leech_core import encode_signal_kmer as _rs_encode_signal_kmer
    from leech_core import extract_levels as _rs_extract_levels
    from leech_core import rough_rescale as _rs_rough_rescale
    from leech_core import seq_banded_dp as _rs_seq_banded_dp

    HAS_RUST = True
    logger.debug("Rust acceleration available (leech_core)")
except ImportError:
    HAS_RUST = False
    _rs_encode_signal_kmer = None
    _rs_extract_levels = None
    _rs_rough_rescale = None
    _rs_seq_banded_dp = None
    logger.debug("Rust acceleration not available, using pure Python fallbacks")
