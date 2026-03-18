"""
Rust-accelerated functions with pure Python fallbacks.

Imports the Rust ``leech_core`` extension if available, otherwise falls back
to the pure Python/numpy implementations.
"""

import logging

logger = logging.getLogger("leech._rust_accel")

try:
    from leech_core import _test_process_read as _rs_test_process_read
    from leech_core import compute_signal_stats as _rs_compute_signal_stats
    from leech_core import encode_signal_kmer as _rs_encode_signal_kmer
    from leech_core import extract_chunks_from_preloaded as _rs_extract_chunks_from_preloaded
    from leech_core import extract_inference_chunks as _rs_extract_inference_chunks
    from leech_core import extract_levels as _rs_extract_levels
    from leech_core import preload_pod5_signals as _rs_preload_pod5_signals
    from leech_core import read_pod5_batch as _rs_read_pod5_batch
    from leech_core import rough_rescale as _rs_rough_rescale
    from leech_core import seq_banded_dp as _rs_seq_banded_dp

    HAS_RUST = True
    logger.debug("Rust acceleration available (leech_core)")
except ImportError:
    HAS_RUST = False
    _rs_compute_signal_stats = None
    _rs_encode_signal_kmer = None
    _rs_extract_chunks_from_preloaded = None
    _rs_extract_inference_chunks = None
    _rs_extract_levels = None
    _rs_preload_pod5_signals = None
    _rs_read_pod5_batch = None
    _rs_test_process_read = None
    _rs_rough_rescale = None
    _rs_seq_banded_dp = None
    logger.debug("Rust acceleration not available, using pure Python fallbacks")


def check_rust() -> None:
    """Print Rust acceleration status."""
    if HAS_RUST:
        import leech_core

        version = getattr(leech_core, "__version__", None)
        label = f"leech_core {version}" if version else "leech_core"
        print(f"Rust acceleration: enabled ({label})")
        funcs = [
            "compute_signal_stats",
            "encode_signal_kmer",
            "extract_chunks_from_preloaded",
            "extract_inference_chunks",
            "extract_levels",
            "preload_pod5_signals",
            "read_pod5_batch",
            "rough_rescale",
            "seq_banded_dp",
        ]
        for f in funcs:
            status = "ok" if getattr(leech_core, f, None) is not None else "missing"
            print(f"  {f}: {status}")
    else:
        print("Rust acceleration: not available")
        print("Install with: uv sync --extra rust")
