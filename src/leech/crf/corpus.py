"""Cut a CRF training corpus from a manifest: which reads, which split, which signal.

Two stages, deliberately separated. :func:`plan_corpus` decides *which reads and
in which split*, touching no POD5 and returning a plain table; :func:`build_corpus`
then extracts their signal. Everything subtle is in the plan, so it is testable
without a gigabyte of fixture — and the alternative has already gone wrong: two
copies of this decision (a fused single-pass path and a two-pass one) would drift,
and the failure is silent, because a test set drawn from one batch still trains
and still reports a number.

Output is a streamed pair, not one array:

``<out>_X.npy``     ``(n, chunk)`` float32 raw pA, memory-mappable
``<out>_meta.npz``  targets, groups, read ids, split, batch, quality, geometry

The signal never exists in RAM as a whole. An 80-plex corpus is tens of
gigabytes; ``np.empty((N, chunk))`` is not an allocation that succeeds, and the
memmap is what makes the size a disk question instead of a RAM one.

Four rules that are not obvious and have each cost a rebuild
------------------------------------------------------------
**A cap is only a cap if every class can reach it.** Capping at a number above
the rarest class's depth de-balances the corpus it is meant to balance: one
class contributes everything it has while the others are held back. Reserve the
test fraction *first*, so the cap draws from the trainable pool, and default to
the rarest class's trainable depth.

**Carve the split before capping, and rank globally per class.** Ranking per
``(batch, class)`` instead silently multiplies the cap by the number of batches
for any panel whose classes are crossed with batch.

**Interleave batches; do not concatenate them.** Reads are shuffled *within*
each batch and then drawn round-robin *across* batches. Concatenating ranks
every row of the first batch ahead of the second, so the first ``test_frac`` of
each class — the test set — comes entirely from whichever batch sorts first, and
the headline number measures batch rather than signal. Sorting by ``read_id``
before shuffling keeps the draw independent of the manifest's row order.

**Shard after planning, never before.** The plan is deterministic in (manifest,
seed), so every shard computes the same global split and keeps its own share.
Filtering the manifest first would rank each shard's reads against only
themselves, giving each its own test set drawn from one batch.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import CrfManifest, load_manifest

__all__ = ["CorpusPlan", "build_corpus", "load_corpus", "load_corpus_meta", "plan_corpus"]

logger = logging.getLogger("leech.crf.corpus")

#: Samples of headroom required beyond the window itself. A read whose
#: ``anchor_end`` sits at exactly ``chunk`` has no margin for an off-by-a-few
#: boundary call, and this filter cannot live in the manifest because it depends
#: on ``chunk``: a read usable at one window is not usable at a wider one.
ANCHOR_MARGIN = 200

#: Rows extracted per POD5 call. Bounds peak RAM at ``BATCH * chunk * 4`` bytes
#: while letting the reader group a batch by owning file.
EXTRACT_BATCH = 2000


@dataclass(frozen=True)
class CorpusPlan:
    """Which reads go in, and in which split. No signal has been read yet."""

    frame: Any
    cap: int
    chunk: int
    dropped_short_anchor: int

    def __len__(self) -> int:
        return self.frame.height

    def counts_by_split(self) -> dict[str, int]:
        rows = self.frame.group_by("split").len().rows()
        return {str(k): int(v) for k, v in rows}

    def groups(self) -> list[str]:
        return sorted(self.frame["group"].unique().to_list())


def _group_column(frame):
    """`group`, falling back to `label`, then to a single bucket.

    Balancing and the per-class split need *some* notion of class. A manifest
    that carries neither is legal — it just means one bucket, which is the right
    answer for a corpus with one target.
    """
    import polars as pl

    if "group" in frame.columns:
        return frame
    if "label" in frame.columns:
        return frame.with_columns(pl.col("label").alias("group"))
    return frame.with_columns(pl.lit("all").alias("group"))


def plan_corpus(
    manifest: CrfManifest | str | Path,
    *,
    chunk: int,
    test_frac: float = 0.1,
    per_group: int | str = "auto",
    seed: int = 0,
    groups: list[str] | None = None,
    shard_batches: list[str] | None = None,
) -> CorpusPlan:
    """Decide the corpus. Reads no signal.

    Args:
        manifest: a :class:`CrfManifest` or a path to one.
        chunk: window width in samples; also sets the ``anchor_end`` filter.
        test_frac: fraction of each class held out, before capping.
        per_group: cap per class, or ``"auto"`` for the rarest class's trainable
            depth — the only value that actually balances.
        seed: fixes the shuffle, and therefore the split.
        groups: restrict to these classes (default: every class present).
        shard_batches: keep only these batches *after* planning globally.

    Returns:
        A :class:`CorpusPlan` whose frame carries a ``split`` column.
    """
    import polars as pl

    man = manifest if isinstance(manifest, CrfManifest) else load_manifest(manifest)
    frame = _group_column(man.frame)
    if "batch" not in frame.columns:
        frame = frame.with_columns(pl.lit("all").alias("batch"))

    if groups is not None:
        frame = frame.filter(pl.col("group").is_in(groups))
        if frame.height == 0:
            raise ValueError(f"no reads for groups {groups}")

    before = frame.height
    frame = frame.filter(pl.col("anchor_end") > chunk + ANCHOR_MARGIN)
    dropped = before - frame.height
    if frame.height == 0:
        raise ValueError(
            f"no read has anchor_end > {chunk + ANCHOR_MARGIN}; every one of "
            f"{before} was dropped. The window is wider than the signal available "
            f"ahead of the anchor."
        )

    # Shuffle within batch, then interleave round-robin across batches.
    parts = []
    for _batch, part in sorted(
        ((k[0], p) for k, p in frame.partition_by("batch", as_dict=True).items()),
        key=lambda kv: str(kv[0]),
    ):
        part = part.sort("read_id").sample(fraction=1.0, shuffle=True, seed=seed)
        parts.append(part.with_columns(pl.int_range(pl.len()).alias("_pos")))
    frame = pl.concat(parts).sort("_pos", maintain_order=True)

    avail = frame.group_by("group").len().sort("group")
    counts = dict(zip(avail["group"].to_list(), avail["len"].to_list(), strict=True))
    trainable = {g: int(n * (1 - test_frac)) for g, n in counts.items()}
    rarest_group = min(trainable, key=lambda g: trainable[g])
    rarest = trainable[rarest_group]

    if per_group == "auto":
        cap = rarest
        logger.info("per_group=auto -> %d (trainable depth of %s)", cap, rarest_group)
    else:
        cap = int(per_group)
        if cap > rarest:
            short = sorted(g for g, n in trainable.items() if n < cap)
            logger.warning(
                "per_group=%d exceeds the trainable depth of %d group(s); the corpus "
                "will NOT be balanced. Rarest is %s at %d — those groups contribute "
                "everything they have while the others are capped. Use per_group='auto' "
                "(%d) for a balanced corpus.",
                cap,
                len(short),
                rarest_group,
                rarest,
                rarest,
            )

    # Rank within class over the interleaved order, so rank is a random draw.
    # Per class and global across batches: per-(batch, class) would multiply the
    # cap by the number of batches whenever classes are crossed with batch.
    ranked = frame.with_columns(
        pl.int_range(pl.len()).over("group").alias("_rank"),
        pl.len().over("group").alias("_n"),
    ).with_columns(
        # At least one held-out read per class: a class with none is invisible to
        # evaluation rather than reported as missing.
        pl.max_horizontal(pl.lit(1), (pl.col("_n") * test_frac).cast(pl.Int64)).alias("_n_test")
    )
    test = ranked.filter(pl.col("_rank") < pl.col("_n_test")).with_columns(
        pl.lit("test").alias("split")
    )
    train = (
        ranked.filter(pl.col("_rank") >= pl.col("_n_test"))
        .with_columns((pl.col("_rank") - pl.col("_n_test")).alias("_train_rank"))
        .filter(pl.col("_train_rank") < cap)
        .with_columns(pl.lit("train").alias("split"))
        .drop("_train_rank")
    )
    planned = pl.concat([test, train]).drop("_pos", "_rank", "_n", "_n_test")

    if shard_batches is not None:
        want = set(shard_batches)
        missing = want - set(planned["batch"].unique().to_list())
        if missing:
            raise ValueError(
                f"shard_batches names {len(missing)} batch(es) with no reads in the "
                f"plan: {sorted(missing)}. A shard that quietly extracted nothing "
                f"would shrink the corpus without failing."
            )
        planned = planned.filter(pl.col("batch").is_in(list(want)))

    return CorpusPlan(frame=planned, cap=cap, chunk=chunk, dropped_short_anchor=dropped)


def build_corpus(
    plan: CorpusPlan,
    out: str | Path,
    *,
    state_len: int = 4,
    allow_shortfall: bool = False,
    extract_batch: int = EXTRACT_BATCH,
) -> Path:
    """Extract each planned read's window and stream it to ``<out>_X.npy``.

    Args:
        plan: from :func:`plan_corpus`.
        out: output stem; ``_X.npy`` and ``_meta.npz`` are appended.
        state_len: recorded in the metadata so consumers can derive what the
            model will emit without guessing.
        allow_shortfall: permit extracting less than half the plan.
        extract_batch: reads per POD5 call.

    Raises:
        RuntimeError: nothing was extracted, or more than half the plan is
            missing and ``allow_shortfall`` is false.
    """
    # Imported here, not at module scope: `leech.io.pod5_reader` pulls pysam and
    # escapepod, and `leech.crf` promises to cost only torch and numpy on import.
    from leech.io.pod5_reader import read_pod5_signals_batch_cached

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    x_path = out.with_name(out.name + "_X.npy")
    frame, chunk, total = plan.frame, plan.chunk, len(plan)

    signal = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(total, chunk))
    kept: dict[str, list] = {k: [] for k in ("target", "group", "read_id", "split", "batch")}
    quality: dict[str, list] = {"quality_score": [], "quality_margin": []}
    has_quality = {k: k in frame.columns for k in quality}
    n = 0

    # Grouped by POD5 source: that is the unit the reader caches and the unit
    # storage order is defined within. The split above is independent of it.
    for (source,), part in sorted(
        frame.partition_by("pod5", as_dict=True).items(), key=lambda kv: str(kv[0][0])
    ):
        rows = part.to_dicts()
        for start in range(0, len(rows), extract_batch):
            block = rows[start : start + extract_batch]
            found = read_pod5_signals_batch_cached(source, [r["read_id"] for r in block])
            for row in block:
                hit = found.get(row["read_id"])
                if hit is None:
                    continue
                raw = hit[0]
                end = int(row["anchor_end"])
                if end - chunk < 0 or end > len(raw):
                    continue
                signal[n] = np.asarray(raw[end - chunk : end], dtype=np.float32)
                for key in kept:
                    kept[key].append(row[key])
                for key, present in has_quality.items():
                    quality[key].append(float(row[key]) if present else np.nan)
                n += 1
        logger.info("%s: %d extracted", source, n)
    signal.flush()

    # An empty or badly short corpus is a BROKEN build, not a small one, and
    # everything upstream can be correct and still produce it — a manifest naming
    # a parent directory rather than the one holding the .pod5 files matches zero
    # reads, writes a 0-row array, and exits cleanly. The two failures are
    # distinguished because their causes differ.
    if n == 0:
        sources = sorted({str(s) for s in frame["pod5"].unique().to_list()})[:3]
        del signal
        raise RuntimeError(
            f"extracted 0 of {total} planned reads. Nothing was read from "
            f"{', '.join(sources)}. Check that these paths hold the .pod5 files "
            f"naming these reads."
        )
    if n < 0.5 * total and not allow_shortfall:
        del signal
        raise RuntimeError(
            f"extracted {n} of {total} planned reads ({n / total:.1%}). More than "
            f"half the corpus is missing, so the manifest and the POD5s disagree "
            f"about which reads exist. Pass allow_shortfall=True if expected."
        )

    if n < total:
        _trim(x_path, n, chunk)

    targets = np.array(kept["target"], dtype=str)
    meta = {
        "y": targets,
        "code": np.array(kept["group"], dtype=str),
        "group": np.array(kept["group"], dtype=str),
        "read_id": np.array(kept["read_id"], dtype=str),
        "split": np.array(kept["split"], dtype=str),
        "batch": np.array(kept["batch"], dtype=str),
        "run": np.array(kept["batch"], dtype=str),
        "gate_score": np.array(quality["quality_score"], dtype=np.float32),
        "gate_margin": np.array(quality["quality_margin"], dtype=np.float32),
        "chunk": chunk,
        "per_code": plan.cap,
        "target_len": int(len(targets[0])) if n else 0,
        "state_len": state_len,
    }
    np.savez(out.with_name(out.name + "_meta.npz"), **meta)
    logger.info("wrote %s (%d x %d) and %s_meta.npz", x_path, n, chunk, out)
    return x_path


def _trim(x_path: Path, n: int, chunk: int, step: int = 50_000) -> None:
    """Shrink the memmap to the rows actually written.

    ``j`` is clamped to ``n`` on BOTH sides. The source still has the planned
    row count, so an unclamped final block reads more rows than the destination
    holds and numpy refuses to broadcast — losing an already-extracted corpus at
    the very last step. It only fires when reads were dropped *and* the
    remainder straddles a block boundary, which is why it can survive many
    builds unnoticed.
    """
    full = np.load(x_path, mmap_mode="r")
    tmp = x_path.with_suffix(x_path.suffix + ".tmp")
    trimmed = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, chunk))
    for i in range(0, n, step):
        j = min(i + step, n)
        trimmed[i:j] = full[i:j]
    trimmed.flush()
    del trimmed, full
    os.replace(tmp, x_path)


def load_corpus(path: str | Path, *, mmap: bool = True):
    """Read a corpus written by :func:`build_corpus`, or the legacy single ``.npz``.

    Returns ``(signal, targets, groups, read_ids, split)``; ``split`` is ``None``
    for the legacy layout, which carries none. ``mmap`` keeps the signal on disk,
    which is the only way to train on a corpus larger than RAM — batches are
    indexed out of it per step.
    """
    base = Path(str(path)[:-4] if str(path).endswith(".npz") else path)
    streamed = base.with_name(base.name + "_X.npy")
    if streamed.exists():
        signal = np.load(streamed, mmap_mode="r" if mmap else None)
        meta = np.load(base.with_name(base.name + "_meta.npz"), allow_pickle=True)
        group_key = "group" if "group" in meta else "code"
        split = meta["split"].astype(str) if "split" in meta else None
        return (
            signal,
            meta["y"].astype(str),
            meta[group_key].astype(str),
            meta["read_id"].astype(str),
            split,
        )
    legacy = np.load(base.with_suffix(".npz"), allow_pickle=True)
    group_key = "group" if "group" in legacy else "code"
    return (
        legacy["X"],
        legacy["y"].astype(str),
        legacy[group_key].astype(str),
        legacy["read_id"].astype(str),
        None,
    )


def load_corpus_meta(path: str | Path) -> dict:
    """The corpus's ``_meta.npz`` as a dict, or ``{}`` for the legacy layout.

    Separate from :func:`load_corpus` because the optional columns really are
    optional: corpora written before ``batch`` and the quality pair existed must
    keep loading, so every consumer has to *ask* whether a column is there
    rather than assume it.
    """
    base = Path(str(path)[:-4] if str(path).endswith(".npz") else path)
    meta_path = base.with_name(base.name + "_meta.npz")
    if not meta_path.exists():
        return {}
    return dict(np.load(meta_path, allow_pickle=True))
