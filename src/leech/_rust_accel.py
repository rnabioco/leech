"""
Rust-accelerated functions with pure Python fallbacks.

Imports the Rust ``leech_core`` extension if available, otherwise falls back
to the pure Python/numpy implementations.
"""

import logging
import os
import re

logger = logging.getLogger("leech._rust_accel")

#: Set ``LEECH_DISABLE_RUST=1`` to take the pure-Python paths even when
#: ``leech_core`` is installed.
#:
#: "Rust is available" and "Rust is faster here" are different claims, and
#: without a switch there is no way to measure the second. They came apart on
#: reference-anchored ``data prepare`` over a 145 GB merged POD5 (#176):
#: 13 reads/s on the Rust path against 130 at 8 workers and 234 at 32 on the
#: Python one.
#:
#: That step is bound by random-read LATENCY into the POD5 -- a
#: coordinate-sorted BAM visits reads in an order unrelated to how they are
#: stored -- so throughput is set by how many reads are in flight. The Rust
#: path used to leave exactly one: it re-opened (and so re-scanned) the POD5
#: per batch, held the GIL across the I/O, and was driven from a serial batch
#: loop. #178 fixed all three, and ``prepare`` now logs achieved reads/s on
#: every progress line so the two paths can be compared without this switch.
#:
#: Still useful for measuring, and as an escape hatch on data where the
#: Python path happens to win. Leave it unset unless you have timed both on
#: your own data.
#:
#: If you are comparing the two paths, check that both halves of your install
#: are current: ``leech_core`` is a separate package from ``leech``, and a
#: freshly built extension paired with a stale ``leech`` gives the new Rust
#: with the old serial driver. The startup line names the dispatch, so a
#: build without "batches in flight" in it is the stale pairing.
DISABLE_RUST = os.environ.get("LEECH_DISABLE_RUST", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

try:
    if DISABLE_RUST:
        raise ImportError("leech_core disabled by LEECH_DISABLE_RUST")

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
    if DISABLE_RUST:
        logger.info("Rust acceleration disabled by LEECH_DISABLE_RUST")
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


#: Separators PEP 440 drops from a pre-release segment. Cargo keeps them.
_VERSION_SEP = re.compile(r"[-_.]")

#: Pre-release spellings PEP 440 folds together. Cargo passes them through.
_PRE_ALIASES = {"alpha": "a", "beta": "b", "c": "rc", "pre": "rc", "preview": "rc"}


def _normalize_version(version: str) -> str:
    """Reduce a version to a form comparable across Cargo and PEP 440.

    The two halves of the install report their versions in different dialects.
    ``leech``'s comes from ``importlib.metadata``, which gives the PEP 440
    normal form (``0.6.7rc1``). ``leech_core``'s comes from
    ``env!("CARGO_PKG_VERSION")`` -- the literal Cargo string, which must be
    semver (``0.6.7-rc.1``). A final release spells the same in both, so this
    only bites on pre-releases, where a raw ``==`` reports a mismatch on a
    correctly paired install and there is no way to release an rc at all.

    Comparing normal forms rather than parsing: ``packaging`` is not a runtime
    dependency (it happens to be present in dev environments, which is exactly
    how this would come back), and the comparison only needs the two spellings
    to agree, not a total order.
    """
    version = version.strip().lower()
    release, sep, suffix = version.partition("-")
    if not sep:
        # Already inline (PEP 440), or no pre-release at all.
        return version
    suffix = _VERSION_SEP.sub("", suffix)
    if match := re.match(r"([a-z]+)(.*)", suffix):
        word, rest = match.groups()
        suffix = _PRE_ALIASES.get(word, word) + rest
    return release + suffix


def rust_version_mismatch() -> tuple[str, str] | None:
    """``(leech_version, leech_core_version)`` when the two disagree.

    ``leech_core`` is a separate distribution from ``leech``, built from the
    same repository but installed independently, so an extension compiled at one
    revision can sit alongside a ``leech`` from another. That pairing produces
    wrong numbers rather than an error -- it is how issue #176 stayed hidden
    (new Rust, old serial driver), and how a stale ``uv`` cache entry silently
    reinstated pre-#188 chunk behaviour over a current build.

    Both versions move together on release, so a difference means one half of
    the install is stale. Returns ``None`` when they agree, or when either
    version cannot be determined (an old extension exports no ``__version__``,
    and there is nothing useful to say about that).
    """
    if not HAS_RUST:
        return None
    import leech_core

    import leech

    core_version = getattr(leech_core, "__version__", None)
    leech_version = getattr(leech, "__version__", None)
    if not core_version or not leech_version:
        return None
    if _normalize_version(core_version) == _normalize_version(leech_version):
        return None
    return (leech_version, core_version)


def check_rust() -> None:
    """Print Rust acceleration status."""
    if HAS_RUST:
        import leech_core

        version = getattr(leech_core, "__version__", None)
        label = f"leech_core {version}" if version else "leech_core (version unknown)"
        print(f"Rust acceleration: enabled ({label})")
        mismatch = rust_version_mismatch()
        if mismatch is not None:
            leech_version, core_version = mismatch
            print(
                f"  WARNING: leech {leech_version} paired with leech_core "
                f"{core_version}. They are built from one repository and "
                f"released together, so a mismatch means half the install is "
                f"stale. Rebuild the extension with `bash rust/build.sh`; if "
                f"that does not clear it, the stale half is leech's own "
                f"metadata -- reinstall it (`uv pip install -e .`), which an "
                f"editable install needs after a version bump."
            )
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
