"""Scoring a CRF against a reference set.

The weight is on two things: that the vectorised edit distance really is the
scalar one (its algebra recovers a serial dependency, which is worth an
assertion rather than a comment), and that the reporting refuses the shapes that
produce a plausible-looking wrong number.
"""

from __future__ import annotations

import numpy as np
import pytest

from leech.crf import (
    Call,
    balanced_recall,
    call_references,
    emitted_references,
    encode_references,
    lev,
    lev_vs_refs,
)
from leech.crf.evaluate import _lev_py

BASES = "ACGT"


def _random_strings(rng, n, width):
    return ["".join(rng.choice(list(BASES), size=width)) for _ in range(n)]


# ── edit distance ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("a", "b", "want"),
    [
        ("", "", 0),
        ("", "ACGT", 4),
        ("ACGT", "", 4),
        ("ACGT", "ACGT", 0),
        ("ACGT", "ACGA", 1),  # substitution
        ("ACGT", "ACT", 1),  # deletion
        ("ACT", "ACGT", 1),  # insertion
        ("ACGT", "TGCA", 4),
    ],
)
def test_lev_py_known_distances(a, b, want):
    assert _lev_py(a, b) == want


def test_lev_matches_the_python_fallback():
    """`lev` is edlib where edlib is importable. Comparing it against edlib
    would compare edlib with itself exactly where edlib exists; comparing it
    against the named fallback tests something on every machine."""
    rng = np.random.default_rng(0)
    for a, b in zip(_random_strings(rng, 60, 20), _random_strings(rng, 60, 22), strict=True):
        assert lev(a, b) == _lev_py(a, b)


def test_vectorised_distance_equals_the_scalar_one():
    """The identity the vectorisation rests on.

    `cur[j-1] + 1` is serial along j; it is recovered as
    `j + cummin(tmp[k] - k)`. That algebra is exact or it is silently wrong, so
    it gets asserted rather than commented.
    """
    rng = np.random.default_rng(1)
    refs = {f"r{i}": s for i, s in enumerate(_random_strings(rng, 24, 30))}
    names, packed = encode_references(refs)
    for query in _random_strings(rng, 40, 30) + _random_strings(rng, 10, 17):
        got = lev_vs_refs(query, packed)
        want = [_lev_py(query, refs[n]) for n in names]
        assert list(got) == want, f"diverged on {query!r}"


def test_vectorised_distance_handles_an_empty_query():
    """The `not query` early return: distance is just each reference's length."""
    _, packed = encode_references({"a": "ACGT", "b": "TTTT"})
    assert list(lev_vs_refs("", packed)) == [4, 4]


# ── references ─────────────────────────────────────────────────────────────


def test_emitted_references_drop_the_sacrificial_prefix():
    """Matching full-length targets forces `state_len` leading deletions into
    every alignment, inflating distances and compressing the margin."""
    out = emitted_references({"c1": "AAAACGT", "c2": "TTTTGCA"}, state_len=4)
    assert out == {"c1": "CGT", "c2": "GCA"}


def test_ragged_references_fall_back_instead_of_padding():
    """Padding would change the distances rather than compute them."""
    names, packed = encode_references({"a": "ACGT", "b": "ACG"})
    assert packed is None and names == ["a", "b"]


def test_encoding_refuses_an_empty_reference_set():
    with pytest.raises(ValueError, match="no references"):
        encode_references({})


# ── calling ────────────────────────────────────────────────────────────────


def test_calls_the_nearest_reference_with_its_margin():
    refs = {"a": "ACGTACGT", "b": "TTTTTTTT"}
    (call,) = call_references(["ACGTACGA"], refs)
    assert call.name == "a" and call.distance == 1
    assert call.margin == _lev_py("ACGTACGA", refs["b"]) - 1


def test_ties_resolve_to_the_lowest_name_with_zero_margin():
    """A silent coin flip would make the same input call differently run to run."""
    (call,) = call_references(["AAAA"], {"b": "CCCC", "a": "GGGG"})
    assert call.margin == 0
    assert call.name == "a"


def test_candidates_restrict_the_reference_set():
    """The honest candidate set when a group cannot contain every class."""
    refs = {"a": "ACGTACGT", "b": "ACGTACGA", "c": "TTTTTTTT"}
    (unrestricted,) = call_references(["ACGTACGA"], refs)
    (restricted,) = call_references(["ACGTACGA"], refs, candidates=["a", "c"])
    assert unrestricted.name == "b"
    assert restricted.name == "a"


def test_ragged_and_packed_paths_agree():
    packed_refs = {"a": "ACGTAC", "b": "TTTTTT"}
    ragged_refs = {"a": "ACGTAC", "b": "TTTTT"}  # forces the scalar path
    assert encode_references(packed_refs)[1] is not None
    assert encode_references(ragged_refs)[1] is None
    q = "ACGTAA"
    assert call_references([q], packed_refs)[0].name == "a"
    assert call_references([q], ragged_refs)[0].name == "a"


# ── reporting ──────────────────────────────────────────────────────────────


def test_balanced_recall_averages_classes_not_reads():
    """A pooled accuracy is dominated by whichever class is deepest."""
    truth = ["a"] * 100 + ["b"] * 4
    calls = ["a"] * 100 + ["b", "b", "x", "x"]
    out = balanced_recall(truth, calls)
    # pooled would be 102/104 = 0.98; balanced is (1.0 + 0.5) / 2
    assert out["balanced_recall"] == pytest.approx(0.75)


def test_reports_per_group_and_averages_across_them():
    truth = ["a", "a", "b", "b"]
    calls = ["a", "a", "b", "x"]
    groups = ["g1", "g1", "g2", "g2"]
    out = balanced_recall(truth, calls, groups)
    assert set(out["groups"]) == {"g1", "g2"}
    assert out["groups"]["g1"]["balanced_recall"] == 1.0
    assert out["groups"]["g2"]["balanced_recall"] == 0.5
    assert out["balanced_recall"] == pytest.approx(0.75)


def test_each_group_scores_only_the_classes_it_contains():
    """Reads bucket by their own group, and a group's classes are the ones that
    occur in it — filtering the other way counts every read in every bucket
    when classes do not partition groups."""
    truth = ["a", "a", "b", "b"]
    calls = ["a", "a", "b", "b"]
    groups = ["g1", "g1", "g2", "g2"]
    out = balanced_recall(truth, calls, groups)
    assert out["groups"]["g1"]["n_classes"] == 1
    assert set(out["groups"]["g1"]["per_class"]) == {"a"}


def test_accepts_call_objects_as_well_as_names():
    truth = ["a", "b"]
    calls = [Call("a", 0, 5), Call("b", 1, 3)]
    assert balanced_recall(truth, calls)["balanced_recall"] == 1.0


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="lengths differ"):
        balanced_recall(["a", "b"], ["a"], ["g", "g"])


def test_an_empty_grouping_is_refused_rather_than_reported_as_null():
    """The failure this catches is silent: with the wrong grouping every bucket
    comes out empty, the headline reduces to None, and a null balanced_recall
    serializes fine and ships."""
    with pytest.raises(ValueError, match="no reporting group"):
        balanced_recall([], [], [])


# ── decoding ───────────────────────────────────────────────────────────────


def test_decode_corpus_preserves_the_requested_order(monkeypatch):
    """Batches are gathered in sorted order for the memmap, but the result must
    come back in the caller's order or truth and calls silently misalign."""
    import leech.crf.decode
    import leech.crf.evaluate as ev

    signal = np.arange(40, dtype=np.float32).reshape(8, 5)
    # Patched at the source: `decode_corpus` imports it inside the function, so
    # there is no module attribute on `evaluate` to replace.
    monkeypatch.setattr(
        leech.crf.decode, "decode_batch", lambda x, *a, **k: [f"s{int(r[0][0])}" for r in x]
    )

    class Fake:
        def eval(self):
            return self

        def __call__(self, x):
            return x

    order = np.array([5, 1, 7, 0])
    out = ev.decode_corpus(
        Fake(), signal, order, mean=0.0, std=1.0, chunk=5, batch_size=2, device="cpu"
    )
    assert len(out) == len(order)
    # Whatever the batching did internally, position i must be index order[i].
    assert out == [f"s{int(signal[i, 0])}" for i in order]
