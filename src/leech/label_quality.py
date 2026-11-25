"""Per-chunk label quality: a number, gated at train time (issue #398).

Labels here are enrichment purities, barcode-caller calls and deacylation per
sample, so some are better than others. That is recorded as a numeric
``label_quality`` per chunk and **never as a ``keep`` boolean**: the threshold is
applied at training time (``--min-label-quality``) so it stays sweepable --
the same rule as ``leech.crf.manifest``'s ``quality_score``.

Estimating quality is out of scope. The numbers come from an external table
keyed by ``read_id`` (``data prepare --label-quality-table``).

An unscored chunk cannot pass a gate, so it would be dropped without a word and
the gated corpus would be a small, non-random subset. :func:`gate_mask` therefore
refuses partial coverage, and a corpus with no scores at all.

numpy only; no torch.
"""

from collections.abc import Iterable, MutableMapping
from pathlib import Path

import numpy as np


def label_quality_coverage(values: np.ndarray | None) -> float:
    """Fraction of chunks carrying a score. ``None`` (no column) is 0.0."""
    if values is None or len(values) == 0:
        return 0.0
    return float(1.0 - np.isnan(np.asarray(values, dtype=np.float64)).mean())


def gate_mask(values: np.ndarray | None, min_quality: float) -> np.ndarray:
    """Boolean keep-mask for ``label_quality >= min_quality``.

    Raises:
        ValueError: when the corpus has no scores at all, or only some chunks
            are scored. Both would otherwise shrink the corpus silently.
    """
    if values is None or label_quality_coverage(values) == 0.0:
        raise ValueError(
            f"--min-label-quality {min_quality} was requested but the corpus carries no "
            "label_quality scores. Prepare it with --label-quality-table, or drop the gate."
        )
    arr = np.asarray(values, dtype=np.float64)
    n_scored = int((~np.isnan(arr)).sum())
    if n_scored != len(arr):
        raise ValueError(
            f"label_quality coverage is partial: {n_scored} of {len(arr)} chunks "
            f"({100.0 * n_scored / len(arr):.1f}%) are scored. An unscored chunk cannot pass a "
            "gate and would be dropped silently, leaving a non-random subset; score every "
            "read or none."
        )
    return arr >= min_quality


def load_quality_table(path: str | Path) -> dict[str, float]:
    """Read a ``read_id`` -> ``label_quality`` table (TSV, CSV, optionally ``.gz``, or Parquet)."""
    import polars as pl

    path = Path(path)
    suffixes = [s.lower() for s in path.suffixes]
    if ".parquet" in suffixes:
        frame = pl.read_parquet(path)
    else:
        sep = "," if ".csv" in suffixes else "\t"
        frame = pl.read_csv(path, separator=sep, schema_overrides={"read_id": pl.Utf8})
    missing = {"read_id", "label_quality"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} must have read_id and label_quality columns; missing {sorted(missing)} "
            f"(found {frame.columns})"
        )
    frame = frame.select("read_id", "label_quality").drop_nulls()
    qualities = frame["label_quality"].cast(pl.Float64).to_list()
    return dict(zip(frame["read_id"].to_list(), qualities, strict=True))


def apply_label_quality(chunks: Iterable[MutableMapping], lookup: dict[str, float]) -> int:
    """Stamp ``label_quality`` on each chunk from its ``read_id``; return how many were scored.

    A read the table does not carry gets ``None`` (unscored), never a default.
    """
    scored = 0
    for chunk in chunks:
        quality = lookup.get(chunk["read_id"])
        chunk["label_quality"] = quality
        scored += quality is not None
    return scored
