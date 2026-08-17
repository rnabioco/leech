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
    from leech_core import extract_training_chunks as _rs_extract_training_chunks
    from leech_core import preload_pod5_signals as _rs_preload_pod5_signals
    from leech_core import read_pod5_batch as _rs_read_pod5_batch
    from leech_core import rough_rescale_quantile as _rs_rough_rescale_quantile
    from leech_core import seq_banded_dp as _rs_seq_banded_dp

    HAS_RUST = True
    logger.debug("Rust acceleration available (leech_core)")
except ImportError:
    HAS_RUST = False
    _rs_test_process_read = None
    _rs_compute_signal_stats = None
    _rs_encode_signal_kmer = None
    _rs_extract_chunks_from_preloaded = None
    _rs_extract_inference_chunks = None
    _rs_extract_training_chunks = None
    _rs_extract_levels = None
    _rs_preload_pod5_signals = None
    _rs_read_pod5_batch = None
    _rs_rough_rescale_quantile = None
    _rs_seq_banded_dp = None
    logger.debug("Rust acceleration not available, using pure Python fallbacks")


#: The only signal normalization the Rust pipeline implements.
#:
#: ``rust/src/inference_pipeline/processing.rs`` calls ``normalize_median_mad``
#: unconditionally — ``PipelineConfig`` carries no normalization field at all.
#: Callers must therefore check :func:`rust_supports_norm_method` before
#: dispatching to Rust, or a run configured for ``zscore`` / ``quantile`` /
#: ``pa_scaling`` would be silently normalized as ``median_mad`` instead.
RUST_NORM_METHOD = "median_mad"


def rust_supports_norm_method(norm_method: str | None) -> bool:
    """Whether the Rust extraction path can honor ``norm_method``.

    ``None`` means "caller did not configure one", which resolves to the
    :data:`RUST_NORM_METHOD` default and is therefore supported.
    """
    return norm_method is None or norm_method == RUST_NORM_METHOD


#: Whether the Rust pipeline implements ref-anchored soft-clip edge recovery.
#:
#: ``ChunkConfig.recover_softclip_signal`` fills chunk-window samples that fall
#: outside the aligned region with real soft-clipped signal instead of zeros.
#: Doing that requires keeping the full pre-crop signal plus its offset, which
#: the Python path stashes on ``LeechRead.full_signal`` / ``signal_offset``.
#: The Rust ``ProcessedRead`` has no such fields — ``process_read_signal``
#: overwrites ``norm_signal`` with the cropped slice and discards the rest —
#: so the flag cannot be honored there and callers must fall back to Python.
#: Flip this to ``True`` if that changes.
RUST_SUPPORTS_SOFTCLIP_RECOVERY = False


def rust_supports_softclip_recovery(recover_softclip_signal: bool) -> bool:
    """Whether the Rust extraction path can honor ``recover_softclip_signal``.

    Always ``True`` when the flag is off, since there is then nothing to honor.
    """
    return not recover_softclip_signal or RUST_SUPPORTS_SOFTCLIP_RECOVERY


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
            "extract_training_chunks",
            "preload_pod5_signals",
            "read_pod5_batch",
            "seq_banded_dp",
        ]
        for f in funcs:
            status = "ok" if getattr(leech_core, f, None) is not None else "missing"
            print(f"  {f}: {status}")
    else:
        print("Rust acceleration: not available")
        print("Install with: uv sync --extra rust")
