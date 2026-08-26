"""Score a CRF against a reference set: decode, match, report.

The generic half of CRF evaluation. What a *panel* is — which classes exist,
which of them share a flowcell, which grouping makes a number honest — stays
with whatever defines the panel. What arrives here is a reference set, a
grouping, and a corpus.

Three rules this module exists to hold
--------------------------------------
**Match against what the model EMITS.** A CRF with ``state_len`` cannot emit the
first ``state_len`` bases of its target, so scoring a decode against the
full-length target forces ``state_len`` leading deletions into every alignment.
That inflates every distance *and compresses the margin*, because an aligner
places those deletions wherever they help most — which discounts the wrong
references more than the right one. :func:`emitted_references` applies the rule
once, from the ``state_len`` the model declares.

**Report per group, never as one pooled table.** When classes are crossed with
batch — one class per flowcell is the usual way this happens — a pooled table
measures batch, not signal. The grouping is an argument because only the caller
knows whether their classes are confounded; the *refusal* to pool is here.

**Balanced, not raw, recall.** A pooled accuracy over unbalanced classes is
dominated by whichever class is deepest. Averaging per-class recall within a
group is what makes two arms comparable.

On the edit distance
--------------------
:func:`lev_vs_refs` scores one decode against a whole reference set at once,
which is the shape of every evaluation loop. Scoring R references one at a time
is R full DP tables per read in Python — affordable on a 16k sample, not on a
run. The vectorisation is exact, not approximate, and the argument is in that
function's docstring; :func:`_lev_py` stays as the named fallback so a test can
compare the two rather than comparing a fast path against itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Call",
    "balanced_recall",
    "call_references",
    "decode_corpus",
    "emitted_references",
    "encode_references",
    "lev",
    "lev_vs_refs",
]

logger = logging.getLogger("leech.crf.evaluate")


# ── edit distance ──────────────────────────────────────────────────────────


def _lev_py(a: str, b: str) -> int:
    """Pure-Python Levenshtein. The fallback, named so it stays testable.

    :func:`lev` is edlib where edlib is importable, so a test comparing ``lev``
    against edlib would compare edlib against itself on exactly the machines
    where edlib exists — and silently stop testing anything. The two
    implementations are compared through *this* name instead.

    The empty-``a`` case needs no guard: ``prev`` starts as ``range(len(b) + 1)``,
    so ``prev[-1]`` is already ``len(b)``.
    """
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


try:  # pragma: no cover - depends on the environment
    import edlib as _edlib

    def lev(a: str, b: str) -> int:
        """Levenshtein distance. edlib where available, :func:`_lev_py` otherwise."""
        return _edlib.align(a, b, task="distance")["editDistance"]

except ImportError:  # pragma: no cover - exercised only where edlib is absent
    lev = _lev_py


def lev_vs_refs(query: str, refs: np.ndarray) -> np.ndarray:
    """``lev(query, r)`` for every row of an ``(R, M)`` uint8 array, at once.

    The insertion term ``cur[j-1] + 1`` is a serial dependency along ``j``,
    which is what normally blocks vectorising this. It is recovered *exactly*:
    once the substitution and deletion terms give ``tmp``, the true row is
    ``cur[j] = min_{k<=j} (tmp[k] + (j - k))``, i.e.
    ``j + cummin(tmp[k] - k)``. That identity is the only thing making the
    algebra trustworthy, so it is asserted against :func:`_lev_py` on random
    strings rather than assumed.

    Returns an ``(R,)`` int array.
    """
    n_refs, m = refs.shape
    cols = np.arange(m + 1, dtype=np.int32)
    prev = np.broadcast_to(cols, (n_refs, m + 1)).copy()
    if not query:
        return prev[:, -1]
    for i, ch in enumerate(query.encode(), 1):
        sub = prev[:, :-1] + (refs != ch)
        dele = prev[:, 1:] + 1
        tmp = np.empty_like(prev)
        tmp[:, 0] = i
        np.minimum(sub, dele, out=tmp[:, 1:])
        prev = cols + np.minimum.accumulate(tmp - cols, axis=1)
    return prev[:, -1]


# ── references ─────────────────────────────────────────────────────────────


def emitted_references(targets: dict[str, str], state_len: int) -> dict[str, str]:
    """References as the model emits them: ``target[state_len:]``.

    Applied once, here, so no caller can hand a full-length target to a matcher
    and quietly inflate every distance.
    """
    from .manifest import emitted_target

    return {name: emitted_target(target, state_len) for name, target in targets.items()}


def encode_references(references: dict[str, str]) -> tuple[list[str], np.ndarray | None]:
    """``(names, (R, M) uint8 array)`` for :func:`lev_vs_refs`.

    The array is ``None`` when the references are ragged — the vectorised form
    needs one width, and padding them would change the distances it computes.
    :func:`call_references` falls back to the scalar path in that case rather
    than silently scoring something else.
    """
    names = sorted(references)
    if not names:
        raise ValueError("no references to match against")
    widths = {len(references[n]) for n in names}
    if len(widths) != 1:
        logger.info("references are ragged (%s widths); using the scalar path", len(widths))
        return names, None
    packed = np.frombuffer("".join(references[n] for n in names).encode(), dtype=np.uint8)
    return names, packed.reshape(len(names), widths.pop())


# ── calling ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Call:
    """One decode matched against the reference set."""

    name: str
    distance: int
    margin: int
    """Distance to the runner-up minus distance to the best. The quantity a
    recovery threshold ranks on — and it is a *reference separation* measure on
    a designed panel, not a decode-confidence one, so treat a large margin as
    saying the panel is well separated rather than the call is certain."""


def call_references(
    decodes: list[str], references: dict[str, str], *, candidates: list[str] | None = None
) -> list[Call]:
    """Match each decode to its nearest reference by edit distance.

    Args:
        decodes: one sequence per read.
        references: ``{name: emitted_reference}``. Pass
            :func:`emitted_references` output, not full-length targets.
        candidates: restrict matching to these names — the honest candidate set
            when a group cannot contain every class.

    Ties resolve to the lowest name with a margin of 0, rather than a silent
    coin flip.
    """
    if candidates is not None:
        references = {n: references[n] for n in candidates}
    names, packed = encode_references(references)

    calls: list[Call] = []
    for decode in decodes:
        if packed is not None:
            distances = lev_vs_refs(decode, packed)
        else:
            distances = np.array([lev(decode, references[n]) for n in names])
        order = np.argsort(distances, kind="stable")
        best = int(order[0])
        runner_up = int(distances[order[1]]) if len(order) > 1 else int(distances[best])
        calls.append(
            Call(
                name=names[best],
                distance=int(distances[best]),
                margin=runner_up - int(distances[best]),
            )
        )
    return calls


# ── reporting ──────────────────────────────────────────────────────────────


def balanced_recall(
    truth: list[str] | np.ndarray,
    calls: list[Call] | list[str],
    groups: list[str] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Per-group balanced recall, and the per-class recalls behind it.

    ``groups`` buckets reads by where they came from. **It is not optional in
    spirit**: when classes are crossed with batch, one pooled table measures
    batch rather than signal, and the number still looks reasonable. Passing
    ``None`` means the caller is asserting the classes are *not* confounded,
    and everything lands in one bucket named ``"all"``.

    Reads are bucketed by their own group, and each group's classes are the
    ones that occur in it. Filtering reads by "is this class in the group"
    instead is right only when classes partition the groups, and counts every
    read in every bucket when they do not.
    """
    called = [c.name if isinstance(c, Call) else c for c in calls]
    truth = [str(t) for t in truth]
    if groups is None:
        groups = ["all"] * len(truth)
    groups = [str(g) for g in groups]
    if not (len(truth) == len(called) == len(groups)):
        raise ValueError(
            f"lengths differ: truth={len(truth)}, calls={len(called)}, groups={len(groups)}"
        )

    per_group: dict[str, Any] = {}
    for group in sorted(set(groups)):
        tallies: dict[str, list[int]] = {}
        for t, c, g in zip(truth, called, groups, strict=True):
            if g != group:
                continue
            hit, total = tallies.setdefault(t, [0, 0])
            tallies[t] = [hit + (c == t), total + 1]
        if not tallies:
            continue
        recalls = {k: v[0] / v[1] for k, v in sorted(tallies.items())}
        per_group[group] = {
            "balanced_recall": float(np.mean(list(recalls.values()))),
            "n_classes": len(recalls),
            "n_reads": sum(v[1] for v in tallies.values()),
            "per_class": {k: {"recall": r, "n": tallies[k][1]} for k, r in recalls.items()},
        }

    if not per_group:
        raise ValueError(
            "no reporting group had any reads, so every metric would be null. That "
            "is the grouping/corpus mismatch, and it does not raise on its own — "
            "a null balanced_recall serializes fine and ships."
        )
    return {
        "groups": per_group,
        "balanced_recall": float(np.mean([g["balanced_recall"] for g in per_group.values()])),
    }


# ── decoding a corpus ──────────────────────────────────────────────────────


def decode_corpus(
    model,
    signal,
    indices: np.ndarray,
    *,
    mean: float,
    std: float,
    chunk: int,
    n_base: int = 4,
    state_len: int = 4,
    alphabet: str = "NACGT",
    batch_size: int = 256,
    device: str = "cpu",
) -> list[str]:
    """Decode the reads at ``indices``, in corpus order.

    Indices are sorted per batch before gathering, because the corpus is a
    memmap and a sorted gather reads it forwards instead of seeking per row.
    The returned list is in the order of ``indices`` as given, not sorted.
    """
    import torch

    from .decode import decode_batch

    model.eval()
    out: dict[int, str] = {}
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            rows = np.sort(indices[start : start + batch_size])
            window = np.asarray(signal[rows][:, -chunk:], dtype=np.float32)
            x = ((torch.from_numpy(window).to(device) - mean) / std).unsqueeze(1)
            for row, seq in zip(
                rows, decode_batch(model(x), n_base, state_len, alphabet), strict=True
            ):
                out[int(row)] = seq
    return [out[int(i)] for i in indices]
