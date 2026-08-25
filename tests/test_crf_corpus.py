"""Corpus planning and extraction.

The planning tests carry the weight. Every rule they pin fails *silently* when
broken — a corpus built the wrong way still trains and still reports a number,
which is why the decision is factored out of the extraction and tested without
touching a POD5.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from leech.crf import CrfManifest, build_corpus, load_corpus, load_corpus_meta, plan_corpus

TARGET = "ACGT" * 12  # 48 nt


def _manifest(rows: list[dict]) -> CrfManifest:
    return CrfManifest(frame=pl.DataFrame(rows))


def _reads(n, *, group, batch="b1", anchor_end=3400, start=0, target=TARGET, **extra):
    return [
        {
            "read_id": f"{group}-{batch}-{i:05d}",
            "pod5": "reads.pod5",
            "anchor_end": anchor_end,
            "target": target,
            "group": group,
            "batch": batch,
            **extra,
        }
        for i in range(start, start + n)
    ]


# ── the anchor filter ──────────────────────────────────────────────────────


def test_reads_without_room_for_the_window_are_dropped():
    man = _manifest(_reads(10, group="g1", anchor_end=3000) + _reads(10, group="g2"))
    plan = plan_corpus(man, chunk=3000)
    assert plan.dropped_short_anchor == 10
    assert plan.groups() == ["g2"]


def test_all_reads_too_short_is_an_error_not_an_empty_plan():
    man = _manifest(_reads(10, group="g1", anchor_end=500))
    with pytest.raises(ValueError, match="anchor_end"):
        plan_corpus(man, chunk=3000)


def test_the_filter_depends_on_chunk_so_it_cannot_live_in_the_manifest():
    """The same read is usable at one window and not at a wider one."""
    man = _manifest(_reads(10, group="g1", anchor_end=2500))
    assert len(plan_corpus(man, chunk=2000)) == 10
    with pytest.raises(ValueError):
        plan_corpus(man, chunk=3000)


# ── capping ────────────────────────────────────────────────────────────────


def test_auto_cap_is_the_rarest_groups_trainable_depth():
    man = _manifest(_reads(1000, group="g1") + _reads(100, group="g2"))
    plan = plan_corpus(man, chunk=3000, test_frac=0.1, per_group="auto")
    assert plan.cap == 90  # 100 * (1 - 0.1)


def test_auto_cap_balances_the_corpus():
    man = _manifest(_reads(1000, group="g1") + _reads(100, group="g2"))
    plan = plan_corpus(man, chunk=3000, per_group="auto")
    train = plan.frame.filter(pl.col("split") == "train")
    per = train.group_by("group").len().sort("group")["len"].to_list()
    assert per[0] == per[1], "auto must balance the trainable pool"


def test_cap_above_the_rarest_depth_warns_and_does_not_balance(caplog):
    """A cap only caps if every class can reach it; above that it de-balances
    the corpus it is meant to balance."""
    man = _manifest(_reads(1000, group="g1") + _reads(100, group="g2"))
    with caplog.at_level("WARNING", logger="leech.crf.corpus"):
        plan = plan_corpus(man, chunk=3000, per_group=500)
    assert "will NOT be balanced" in caplog.text
    train = plan.frame.filter(pl.col("split") == "train")
    per = dict(train.group_by("group").len().sort("group").rows())
    assert per["g1"] != per["g2"]


def test_test_fraction_is_reserved_before_capping():
    """The cap draws from the trainable pool, not the whole class."""
    man = _manifest(_reads(100, group="g1") + _reads(100, group="g2"))
    plan = plan_corpus(man, chunk=3000, test_frac=0.2, per_group="auto")
    assert plan.cap == 80
    counts = plan.counts_by_split()
    assert counts["test"] == 40 and counts["train"] == 160


# ── the split ──────────────────────────────────────────────────────────────


def test_every_group_gets_at_least_one_held_out_read():
    """A class with no test reads is invisible to evaluation rather than
    reported as missing."""
    man = _manifest(_reads(3, group="tiny") + _reads(500, group="big"))
    plan = plan_corpus(man, chunk=3000, test_frac=0.01)
    test = plan.frame.filter(pl.col("split") == "test")
    assert set(test["group"].to_list()) == {"tiny", "big"}


def test_split_is_deterministic_in_seed():
    man = _manifest(_reads(200, group="g1") + _reads(200, group="g2"))
    a = plan_corpus(man, chunk=3000, seed=7).frame.sort("read_id")
    b = plan_corpus(man, chunk=3000, seed=7).frame.sort("read_id")
    assert a["split"].to_list() == b["split"].to_list()


def test_a_different_seed_gives_a_different_split():
    man = _manifest(_reads(200, group="g1"))
    a = plan_corpus(man, chunk=3000, seed=0).frame.sort("read_id")["split"].to_list()
    b = plan_corpus(man, chunk=3000, seed=1).frame.sort("read_id")["split"].to_list()
    assert a != b


def test_no_read_is_in_both_splits():
    man = _manifest(_reads(300, group="g1") + _reads(300, group="g2"))
    frame = plan_corpus(man, chunk=3000).frame
    test = set(frame.filter(pl.col("split") == "test")["read_id"].to_list())
    train = set(frame.filter(pl.col("split") == "train")["read_id"].to_list())
    assert not (test & train)


# ── the interleave ─────────────────────────────────────────────────────────


def test_held_out_reads_are_drawn_from_every_batch():
    """Concatenating batches instead of interleaving puts the whole test set in
    whichever batch sorts first, and the headline number then measures batch."""
    man = _manifest(
        _reads(500, group="g1", batch="b1") + _reads(500, group="g1", batch="b2", start=500)
    )
    test = plan_corpus(man, chunk=3000, test_frac=0.2).frame.filter(pl.col("split") == "test")
    per_batch = dict(test.group_by("batch").len().sort("batch").rows())
    assert set(per_batch) == {"b1", "b2"}
    # Round-robin, so the two batches contribute within a read or two.
    assert abs(per_batch["b1"] - per_batch["b2"]) <= 2


def test_the_draw_does_not_depend_on_manifest_row_order():
    """Reads are sorted by read_id before shuffling, so a reordered manifest
    plans identically."""
    rows = _reads(200, group="g1")
    a = plan_corpus(_manifest(rows), chunk=3000, seed=3).frame.sort("read_id")
    b = plan_corpus(_manifest(rows[::-1]), chunk=3000, seed=3).frame.sort("read_id")
    assert a["split"].to_list() == b["split"].to_list()


def test_cap_is_global_per_group_not_per_batch_and_group():
    """Ranking per (batch, group) silently multiplies the cap by the number of
    batches whenever classes are crossed with batch."""
    man = _manifest(
        _reads(300, group="g1", batch="b1") + _reads(300, group="g1", batch="b2", start=300)
    )
    plan = plan_corpus(man, chunk=3000, test_frac=0.1, per_group=100)
    train = plan.frame.filter(pl.col("split") == "train")
    assert train.height == 100, "the cap is per group, across batches"


# ── sharding ───────────────────────────────────────────────────────────────


def test_shard_keeps_the_global_split():
    """Planning globally then filtering must give the shard exactly the rows the
    unsharded plan assigned it — filtering first would give it its own test set
    drawn from one batch."""
    man = _manifest(
        _reads(400, group="g1", batch="b1") + _reads(400, group="g1", batch="b2", start=400)
    )
    full = plan_corpus(man, chunk=3000, seed=5).frame.filter(pl.col("batch") == "b1")
    shard = plan_corpus(man, chunk=3000, seed=5, shard_batches=["b1"]).frame
    assert shard.sort("read_id")["split"].to_list() == full.sort("read_id")["split"].to_list()


def test_shard_naming_an_absent_batch_is_an_error():
    """A shard that quietly extracted nothing would shrink the corpus without
    failing."""
    man = _manifest(_reads(50, group="g1", batch="b1"))
    with pytest.raises(ValueError, match="no reads in the plan"):
        plan_corpus(man, chunk=3000, shard_batches=["b9"])


# ── group fallback ─────────────────────────────────────────────────────────


def test_group_falls_back_to_label_then_to_one_bucket():
    rows = [
        {
            "read_id": f"r{i}",
            "pod5": "reads.pod5",
            "anchor_end": 3400,
            "target": TARGET,
            "label": "cls",
        }
        for i in range(20)
    ]
    assert plan_corpus(_manifest(rows), chunk=3000).groups() == ["cls"]
    bare = [{k: v for k, v in r.items() if k != "label"} for r in rows]
    assert plan_corpus(_manifest(bare), chunk=3000).groups() == ["all"]


# ── extraction ─────────────────────────────────────────────────────────────


@pytest.fixture
def fake_pod5(monkeypatch):
    """Stand in for the POD5 reader; extraction is what is under test, not I/O."""
    store: dict[str, np.ndarray] = {}

    def fake_read(source, read_ids):
        return {r: (store[r], {}) for r in read_ids if r in store}

    monkeypatch.setattr("leech.io.pod5_reader.read_pod5_signals_batch_cached", fake_read)
    return store


def test_extracts_the_window_ending_at_the_anchor(tmp_path, fake_pod5):
    rows = _reads(20, group="g1", anchor_end=3400)
    for i, r in enumerate(rows):
        fake_pod5[r["read_id"]] = np.arange(4000, dtype=np.float32) + i
    plan = plan_corpus(_manifest(rows), chunk=3000)

    build_corpus(plan, tmp_path / "corpus")
    signal, targets, groups, read_ids, split = load_corpus(tmp_path / "corpus")

    assert signal.shape == (20, 3000)
    assert set(targets) == {TARGET} and set(groups) == {"g1"}
    # window is [anchor_end - chunk, anchor_end)
    row = dict(zip(read_ids, signal, strict=True))[rows[0]["read_id"]]
    np.testing.assert_array_equal(row, np.arange(400, 3400, dtype=np.float32))
    assert set(split) <= {"train", "test"}


def test_reads_missing_from_the_pod5_are_dropped_and_the_array_is_trimmed(tmp_path, fake_pod5):
    rows = _reads(20, group="g1")
    for r in rows[:15]:
        fake_pod5[r["read_id"]] = np.arange(4000, dtype=np.float32)
    plan = plan_corpus(_manifest(rows), chunk=3000)

    build_corpus(plan, tmp_path / "corpus", allow_shortfall=True)
    signal, _, _, read_ids, _ = load_corpus(tmp_path / "corpus")
    assert signal.shape == (15, 3000) and len(read_ids) == 15


def test_trim_survives_a_remainder_straddling_a_block_boundary(tmp_path, fake_pod5):
    """The clamp that loses an extracted corpus at the very last step if it is
    applied on one side only."""
    from leech.crf.corpus import _trim

    path = tmp_path / "x.npy"
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(25, 4))
    arr[:] = np.arange(100, dtype=np.float32).reshape(25, 4)
    arr.flush()
    del arr
    _trim(path, 7, 4, step=5)  # 7 rows in blocks of 5 -> final block is short
    out = np.load(path)
    assert out.shape == (7, 4)
    np.testing.assert_array_equal(out, np.arange(28, dtype=np.float32).reshape(7, 4))


def test_extracting_nothing_is_a_hard_error(tmp_path, fake_pod5):
    """Zero means the path is wrong — and it exits cleanly if not checked."""
    plan = plan_corpus(_manifest(_reads(20, group="g1")), chunk=3000)
    with pytest.raises(RuntimeError, match="extracted 0"):
        build_corpus(plan, tmp_path / "corpus")


def test_a_large_shortfall_is_refused_by_default(tmp_path, fake_pod5):
    """A large shortfall means the manifest and the POD5s disagree — a different
    failure from a wrong path, so it gets its own message."""
    rows = _reads(20, group="g1")
    for r in rows[:4]:
        fake_pod5[r["read_id"]] = np.arange(4000, dtype=np.float32)
    plan = plan_corpus(_manifest(rows), chunk=3000)
    with pytest.raises(RuntimeError, match="More than"):
        build_corpus(plan, tmp_path / "corpus")


def test_metadata_records_the_geometry_and_the_quality_columns(tmp_path, fake_pod5):
    rows = _reads(20, group="g1", quality_score=70.0, quality_margin=9.0)
    for r in rows:
        fake_pod5[r["read_id"]] = np.arange(4000, dtype=np.float32)
    plan = plan_corpus(_manifest(rows), chunk=3000)
    build_corpus(plan, tmp_path / "corpus", state_len=4)

    meta = load_corpus_meta(tmp_path / "corpus")
    assert int(meta["chunk"]) == 3000
    assert int(meta["target_len"]) == 48
    assert int(meta["state_len"]) == 4
    assert np.allclose(meta["gate_score"], 70.0)
    assert np.allclose(meta["gate_margin"], 9.0)


def test_absent_quality_columns_become_nan_not_zero(tmp_path, fake_pod5):
    """Zero is a real score; NaN is 'unscored', and the training-time gate has
    to be able to tell them apart."""
    rows = _reads(20, group="g1")
    for r in rows:
        fake_pod5[r["read_id"]] = np.arange(4000, dtype=np.float32)
    build_corpus(plan_corpus(_manifest(rows), chunk=3000), tmp_path / "corpus")
    meta = load_corpus_meta(tmp_path / "corpus")
    assert np.isnan(meta["gate_score"]).all()


def test_a_missing_corpus_names_both_layouts(tmp_path):
    """`--corpus` takes a stem, so reporting whichever suffix was tried last
    points at a path the caller never wrote."""
    with pytest.raises(FileNotFoundError) as exc:
        load_corpus(tmp_path / "absent")
    message = str(exc.value)
    assert "absent_X.npy" in message and "absent.npz" in message
    assert "STEM" in message


def test_load_corpus_meta_is_empty_for_a_legacy_layout(tmp_path):
    np.savez(
        tmp_path / "old.npz",
        X=np.zeros((2, 4), np.float32),
        y=np.array(["AC", "AC"]),
        code=np.array(["g", "g"]),
        read_id=np.array(["r0", "r1"]),
    )
    assert load_corpus_meta(tmp_path / "old.npz") == {}
    signal, y, groups, rids, split = load_corpus(tmp_path / "old.npz")
    assert signal.shape == (2, 4) and split is None
