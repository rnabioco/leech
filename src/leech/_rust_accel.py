"""
Rust-accelerated functions with pure Python fallbacks.

Imports the Rust ``leech_core`` extension if available, otherwise falls back
to the pure Python/numpy implementations.
"""

import logging
import re

logger = logging.getLogger("leech._rust_accel")

# `KmerLevels` is imported in the same all-or-nothing try/except as every
# other Rust symbol below, deliberately. A `leech_core` build stale enough to
# be missing it also predates the signature change that made
# `extract_training_chunks` / `extract_inference_chunks` /
# `extract_chunks_from_preloaded` require a `KmerLevels` handle instead of a
# raw dict for `kmer_table` (issue #259) -- so it is not a build this module
# could partially use anyway. Splitting `KmerLevels` into its own try/except
# so the rest of Rust acceleration stayed enabled would be actively worse: with
# `refine_signal_map=True` and no table, `make_kmer_levels()` below returns
# `None`, and Rust silently skips refinement rather than raising, producing
# *unrefined* chunks labeled as refined. Failing this import closed -- the
# whole module falls back to the (slower, but correct) pure-Python path --
# is what keeps that failure mode a performance regression instead of a
# silent correctness one.
try:
    from leech_core import KmerLevels as _RsKmerLevels
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
    _RsKmerLevels = None
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


def make_kmer_levels(kmer_to_level: dict[str, float] | None) -> "_RsKmerLevels | None":
    """Build a Rust ``KmerLevels`` handle once, for reuse across every batch call.

    ``KmerLevels`` owns the ``dict -> HashMap<String, f64>`` conversion that
    used to run inside every ``extract_training_chunks`` /
    ``extract_inference_chunks`` / ``extract_chunks_from_preloaded`` call, under
    the GIL and before the Rust side released it -- 51.8ms min / 71.7ms median
    for the 262,144-entry 9-mer table, serializing the ``ThreadPoolExecutor``
    workers batch dispatch exists to overlap (issue #259). Call this ONCE per
    prepare/predict run and pass the returned handle to every batch call in
    place of the raw ``dict``; the object is safe to share across concurrent
    calls (it is immutable after construction).

    Returns ``None`` when Rust acceleration is unavailable or there is no table
    to build (``kmer_to_level`` is ``None``/empty), in which case callers should
    pass the raw dict to the pure-Python fallback path as before.
    """
    if not HAS_RUST or _RsKmerLevels is None or not kmer_to_level:
        return None
    return _RsKmerLevels(kmer_to_level)


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
            "extract_levels",
            "extract_training_chunks",
            "preload_pod5_signals",
            "read_pod5_batch",
            "rough_rescale_quantile",
            "seq_banded_dp",
            "KmerLevels",
        ]
        for f in funcs:
            status = "ok" if getattr(leech_core, f, None) is not None else "missing"
            print(f"  {f}: {status}")
    else:
        print("Rust acceleration: not available")
        print("Install with: uv sync --extra rust")
