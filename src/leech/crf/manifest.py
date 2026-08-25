"""The per-read manifest: the contract between a corpus's vocabulary and its signal.

One table, one row per read, naming everything leech needs to cut a CRF training
corpus — and nothing about where those facts came from. Which reads belong to
which barcode, which flowcell, how a label's trustworthiness was scored: those
are the producing project's business, and answering them is what
``escapepod_models``' panels and gates exist for. By the time a manifest reaches
here they are resolved into columns.

That seam is not a tidiness argument. ``extract_chunks.py`` in escapepod-models
used to reach into scratch for its labels and boundaries, so the Snakemake rule
running it declared **no inputs at all**: the DAG could not see them, could not
rebuild them, and would not fail when a purge removed them. The failure surfaced
as a corpus that silently got smaller. A declared manifest file is what makes
that a hard error, and keeping leech on this side of it is what stops the
vocabulary leaking back in — the extractor there still took ``--panel`` purely to
turn a class name into a target string, which is the one thread that kept it
tied to one assay.

Columns
-------
Required:

``read_id``      the read, as POD5 and BAM both name it
``pod5``         file or directory holding that read's signal
``anchor_end``   signal index the extraction window *ends* at (exclusive)
``target``       the CRF target sequence for this read

Optional:

``label``        class name, for evaluation and reporting
``group``        reporting/balancing bucket (defaults to ``label``)
``batch``        acquisition batch, for leave-one-batch-out holdout
``quality_score``   label-quality score
``quality_margin``  label-quality margin
``split``        ``train``/``test``, when the producer carved one

Deliberately absent
-------------------
**No ``keep`` boolean.** Label quality travels as *numbers* and the threshold is
applied at training time, because the gate has to be sweepable: on the ldx panel
gating the labels moved accuracy from 0.875 to 0.97, and a boolean decided here
would mean re-extracting an 8 GB corpus to try a different threshold. What the
two quality columns *mean* is the producer's definition and varies by project;
leech only ever compares them against a threshold it is given.

**No panel, no code, no oligo.** ``target`` is the resolved sequence. A class
name may ride along in ``label`` for reporting, but nothing here looks it up.

The geometry trap
-----------------
``anchor_end`` and ``target`` are **coupled**, and getting it wrong is silent.
A CRF with ``state_len`` emits ``len(target) - state_len`` bases at *any* window
width — the first ``state_len`` bases only fix the initial state — so a window
must hold every base of ``target`` even though the model will never emit the
first few. Size targets so those sacrificial bases fall in a constant prefix.
:func:`check_geometry` is the check; see ``leech.crf`` for the rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "CrfManifest",
    "check_geometry",
    "emitted_target",
    "load_manifest",
]

#: Every manifest must carry these. A missing one is an error, not a default:
#: each is a fact only the producer knows.
REQUIRED_COLUMNS: tuple[str, ...] = ("read_id", "pod5", "anchor_end", "target")

#: Column -> what its absence costs. Absent is fine; present-but-empty is what
#: :meth:`CrfManifest.quality_coverage` exists to catch.
OPTIONAL_COLUMNS: dict[str, str] = {
    "label": "no per-class reporting",
    "group": "balancing and reporting fall back to `label`, then to one bucket",
    "batch": "leave-one-batch-out holdout unavailable",
    "quality_score": "no label-quality gate on score",
    "quality_margin": "no label-quality gate on margin",
    "split": "the split is seeded at training time instead of carried",
}


def emitted_target(target: str, state_len: int) -> str:
    """The part of ``target`` the model can actually emit: ``target[state_len:]``.

    The first ``state_len`` bases fix the initial state and are never emitted,
    at any window width — widening the signal window does not lengthen the
    decode. Match decodes against this, never against the full-length target:
    the full-length comparison still picks the right reference but inflates
    every edit distance and compresses the confidence margin that ranking
    depends on.
    """
    if state_len < 0:
        raise ValueError(f"state_len must be >= 0, got {state_len}")
    return target[state_len:]


def check_geometry(
    window: int, target_len: int, samples_per_base: float, *, state_len: int = 4
) -> None:
    """Refuse a window that cannot hold ``target_len`` bases of signal.

    Raises rather than warns. A short window does not fail loudly at training
    time — it trains, converges, and quietly discriminates on fewer bases than
    the design intended, which is how a 27-nt barcode came to be classified on
    23 of them (rnabioco/escapepod-models#36).

    ``samples_per_base`` is the read population's own translocation rate.
    Measure it (leech has dwell times; take a median) rather than carrying a
    constant — it is chemistry- and speed-dependent, and a stale constant makes
    this check pass when it should not.
    """
    if target_len <= state_len:
        raise ValueError(
            f"target of {target_len} bases emits nothing at state_len={state_len}: "
            f"the first {state_len} bases only fix the initial state, so a target "
            f"must be longer than {state_len} to decode to anything."
        )
    needed = target_len * samples_per_base
    if window < needed:
        raise ValueError(
            f"window of {window} samples cannot hold {target_len} bases at "
            f"{samples_per_base:.1f} samples/base (needs ~{needed:.0f}). "
            f"The model would train on a truncated target and report nothing: "
            f"widen the window, or shorten the target and accept that it emits "
            f"{target_len - state_len} bases."
        )


@dataclass(frozen=True)
class CrfManifest:
    """A validated manifest table.

    ``frame`` is a ``polars.DataFrame``; the class is a thin wrapper that exists
    so validation happens in one place and callers can ask questions
    (``quality_coverage``, ``batches``) without re-deriving column conventions.
    """

    frame: Any
    path: Path | None = None

    def __len__(self) -> int:
        return self.frame.height

    @property
    def columns(self) -> list[str]:
        return list(self.frame.columns)

    def has(self, column: str) -> bool:
        return column in self.frame.columns

    def quality_coverage(self) -> float:
        """Fraction of rows carrying a non-null ``quality_score``.

        Worth checking before a gated run, and the reason it is a method rather
        than a caller's one-liner: an unscored read cannot pass a gate, so it is
        dropped *silently*, and a partially-scored manifest trains on a small
        non-random subset. This once cut a corpus from 56% to 13.5% without a
        word, because the score table covered only the reads of an earlier
        extraction. Returns 1.0 when no quality column exists at all — nothing
        is being gated, so nothing is being lost.
        """
        if not self.has("quality_score"):
            return 1.0
        if not len(self):
            return 0.0
        return 1.0 - (self.frame["quality_score"].null_count() / len(self))

    def batches(self) -> list[str]:
        """Distinct ``batch`` values, or ``[]`` when the column is absent."""
        if not self.has("batch"):
            return []
        return sorted(self.frame["batch"].drop_nulls().unique().to_list())

    def target_lengths(self) -> set[int]:
        """Distinct target lengths. More than one is legal but rarely intended."""
        return set(self.frame["target"].str.len_chars().unique().to_list())


def load_manifest(path: str | Path, *, require: tuple[str, ...] = ()) -> CrfManifest:
    """Read and validate a manifest from parquet, TSV or CSV.

    Args:
        path: manifest file; format is taken from the suffix.
        require: optional columns this caller additionally needs. Named up front
            so the failure is "your manifest has no `batch` column" rather than
            a `KeyError` an hour into a run.

    Raises:
        FileNotFoundError: the manifest is not there.
        ValueError: a required column is missing, the table is empty, or
            ``read_id`` is not unique.
    """
    import polars as pl  # lazy: `leech.crf` must import only torch and numpy

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pl.read_parquet(path)
    elif suffix in (".tsv", ".txt"):
        frame = pl.read_csv(path, separator="\t")
    elif suffix == ".csv":
        frame = pl.read_csv(path)
    else:
        raise ValueError(
            f"unrecognised manifest format {suffix!r} ({path}); "
            f"expected .parquet, .tsv, .txt or .csv"
        )

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{path}: manifest is missing required column(s) {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}; found: {list(frame.columns)}. "
            f"`target` is the resolved sequence — if you are porting a table that "
            f"carries a class name instead, the class -> sequence lookup belongs "
            f"in whatever produced it, not here."
        )

    unknown_required = [c for c in require if c not in frame.columns]
    if unknown_required:
        raise ValueError(
            f"{path}: this run needs optional column(s) {unknown_required}, which "
            f"the manifest does not carry. "
            + "; ".join(
                f"without `{c}`: {OPTIONAL_COLUMNS.get(c, 'unknown column')}"
                for c in unknown_required
            )
        )

    if frame.height == 0:
        raise ValueError(f"{path}: manifest is empty")

    duplicates = frame.height - frame["read_id"].n_unique()
    if duplicates:
        raise ValueError(
            f"{path}: {duplicates} duplicate read_id(s). One row per read — a "
            f"duplicate silently double-weights those reads in training and puts "
            f"the same read on both sides of a split."
        )

    return CrfManifest(frame=frame, path=path)
