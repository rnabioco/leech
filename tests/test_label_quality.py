"""label_quality: a numeric per-chunk column, gated at train time (issue #398).

Label quality travels as a number and never as a ``keep`` boolean, so the gate
stays sweepable at train time. An unscored chunk cannot pass a gate, so it
would be dropped without a word: coverage is checked and partial coverage
raises. The column has to survive every writer and reader the corpus goes
through, and corpora written before it existed must still load.
"""

import numpy as np
import pytest
from test_dataset_streaming import KMER_LEN, STORED_SIGNAL_LEN
from test_dataset_streaming import make_chunks as make_stream_chunks
from test_splitter_merge import make_chunks

from leech.chunking import ChunkSpool, ChunkTable, load_chunks, save_chunks
from leech.dataset import LeechDataset
from leech.label_quality import (
    apply_label_quality,
    gate_mask,
    label_quality_coverage,
    load_quality_table,
)
from leech.splitting import merge_and_split_multiclass


def scored(chunks, values=None):
    for i, chunk in enumerate(chunks):
        chunk["label_quality"] = float(values[i]) if values is not None else 0.5 + 0.01 * (i % 40)
    return chunks


# --------------------------------------------------------------------------
# format: save / load / table / spool / merge / old corpora
# --------------------------------------------------------------------------


def test_round_trip_through_save_and_load(tmp_path):
    chunks = scored(make_chunks(4, "r", "A"))
    chunks[1]["label_quality"] = None  # unscored
    path = tmp_path / "c.npz"
    save_chunks(chunks, path)

    with np.load(path, allow_pickle=False) as data:
        assert data["label_qualities"].dtype == np.float32
        assert np.isnan(data["label_qualities"][1])

    loaded = load_chunks(path)
    assert loaded[0]["label_quality"] == pytest.approx(chunks[0]["label_quality"])
    assert loaded[1]["label_quality"] is None

    table = ChunkTable.from_npz(path)
    assert table[0]["label_quality"] == pytest.approx(chunks[0]["label_quality"])
    assert table[1]["label_quality"] is None

    again = tmp_path / "again.npz"
    save_chunks(loaded, again)
    with np.load(path) as a, np.load(again) as b:
        np.testing.assert_array_equal(a["label_qualities"], b["label_qualities"])


def test_spool_writes_the_same_member_as_save_chunks(tmp_path):
    chunks = scored(make_chunks(5, "r", "A"))
    direct, spooled = tmp_path / "direct.npz", tmp_path / "spooled.npz"
    save_chunks(chunks, direct)
    with ChunkSpool(tmp_path) as spool:
        spool.append(chunks[:7])
        spool.append(chunks[7:])
        spool.write_npz(spooled)
    with np.load(direct) as a, np.load(spooled) as b:
        np.testing.assert_array_equal(a["label_qualities"], b["label_qualities"])


def test_corpus_without_the_member_still_loads(tmp_path):
    path = tmp_path / "old.npz"
    save_chunks(make_chunks(3, "r", "A"), path)
    with np.load(path, allow_pickle=False) as data:
        kept = {k: data[k] for k in data.files if k != "label_qualities"}
    np.savez(path, **kept)

    assert load_chunks(path)[0]["label_quality"] is None
    assert ChunkTable.from_npz(path)[0]["label_quality"] is None


def test_merge_carries_label_quality_one_row_per_chunk(tmp_path):
    inputs, labels = [], []
    for name in ("A", "B"):
        path = tmp_path / name / "all.npz"
        save_chunks(scored(make_chunks(12, name, name, seed=3)), path)
        inputs.append(path)
        labels.append(name)
    out = tmp_path / "merged"
    merge_and_split_multiclass(inputs, labels, out, seed=2)
    total = 0
    for split in ("train", "val", "test"):
        with np.load(out / f"{split}.npz", allow_pickle=False) as data:
            n = len(data["read_ids"])
            total += n
            assert len(data["label_qualities"]) == n
            assert not np.isnan(data["label_qualities"]).any()
    assert total == 2 * 12 * 3


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def test_coverage_counts_non_nan():
    q = np.array([0.9, np.nan, 0.1, np.nan], dtype=np.float32)
    assert label_quality_coverage(q) == 0.5
    assert label_quality_coverage(None) == 0.0


def test_gate_mask_keeps_at_or_above_threshold():
    q = np.array([0.2, 0.5, 0.9], dtype=np.float32)
    assert gate_mask(q, 0.5).tolist() == [False, True, True]


def test_gate_raises_on_partial_coverage_and_reports_it():
    q = np.array([0.9, np.nan, 0.8, 0.7], dtype=np.float32)
    with pytest.raises(ValueError, match=r"75\.0%|3 of 4"):
        gate_mask(q, 0.5)


def test_gate_raises_when_nothing_is_scored():
    with pytest.raises(ValueError, match="no label_quality"):
        gate_mask(None, 0.5)
    with pytest.raises(ValueError, match="no label_quality"):
        gate_mask(np.full(3, np.nan, dtype=np.float32), 0.5)


def _dataset(path=None, chunks=None, **kw):
    return LeechDataset(
        chunk_path=path,
        chunks=chunks,
        signal_len=STORED_SIGNAL_LEN,
        kmer_len=KMER_LEN,
        model_type="ConvLSTMDwell",
        seq_encoding="base_onehot",
        **kw,
    )


@pytest.mark.parametrize("streamed", [True, False])
def test_dataset_gate_filters_rows_and_keeps_alignment(tmp_path, streamed):
    chunks = make_stream_chunks(12)
    scored(chunks, values=[i / 12 for i in range(12)])
    path = tmp_path / "c.npz"
    save_chunks(chunks, path)

    ds = _dataset(
        path=path if streamed else None,
        chunks=None if streamed else load_chunks(path),
        min_label_quality=0.5,
    )
    kept = [c["read_id"] for c in (ds.chunks[i] for i in range(len(ds)))]
    assert kept == [f"read_{i:03d}" for i in range(6, 12)]
    assert len(ds) == 6
    # tensors stay paired with the rows that survived
    full = _dataset(path=path)
    assert np.array_equal(ds[0]["signal"].numpy(), full[6]["signal"].numpy())


def test_dataset_gate_unset_changes_nothing(tmp_path):
    path = tmp_path / "c.npz"
    save_chunks(scored(make_stream_chunks(8)), path)
    assert len(_dataset(path=path)) == 8


def test_dataset_gate_raises_on_corpus_without_scores(tmp_path):
    path = tmp_path / "c.npz"
    save_chunks(make_stream_chunks(8), path)
    with pytest.raises(ValueError, match="no label_quality"):
        _dataset(path=path, min_label_quality=0.5)


def test_dataset_gate_raises_on_partial_coverage(tmp_path):
    chunks = scored(make_stream_chunks(8))
    chunks[2]["label_quality"] = None
    path = tmp_path / "c.npz"
    save_chunks(chunks, path)
    with pytest.raises(ValueError, match="partial|coverage"):
        _dataset(path=path, min_label_quality=0.5)


# --------------------------------------------------------------------------
# where the numbers come from: an external table keyed by read_id
# --------------------------------------------------------------------------


def test_load_table_and_apply(tmp_path):
    table = tmp_path / "q.tsv"
    table.write_text("read_id\tlabel_quality\nr1\t0.9\nr2\t0.2\n")
    lookup = load_quality_table(table)
    assert lookup == {"r1": pytest.approx(0.9), "r2": pytest.approx(0.2)}

    chunks = [{"read_id": "r1"}, {"read_id": "r2"}, {"read_id": "r3"}]
    n_scored = apply_label_quality(chunks, lookup)
    assert n_scored == 2
    assert chunks[0]["label_quality"] == pytest.approx(0.9)
    assert chunks[2]["label_quality"] is None


def test_load_table_csv_gz_and_missing_column(tmp_path):
    import gzip

    good = tmp_path / "q.csv.gz"
    with gzip.open(good, "wt") as f:
        f.write("read_id,label_quality\nr1,0.5\n")
    assert load_quality_table(good) == {"r1": 0.5}

    bad = tmp_path / "bad.tsv"
    bad.write_text("read_id\tscore\nr1\t0.5\n")
    with pytest.raises(ValueError, match="label_quality"):
        load_quality_table(bad)


def test_prepare_exposes_the_table_option():
    from click.testing import CliRunner

    from leech.cli import cli

    result = CliRunner().invoke(cli, ["data", "prepare", "--help"])
    assert "--label-quality-table" in result.output
    result = CliRunner().invoke(cli, ["model", "train", "--help"])
    assert "--min-label-quality" in result.output


# --------------------------------------------------------------------------
# the noise-rate diagnostic
# --------------------------------------------------------------------------


def test_implied_flip_rate_recovers_a_planted_rate():
    from leech.label_noise import implied_flip_rates

    rng = np.random.default_rng(0)
    n, flip = 4000, 0.15
    # Group "Ala" is labelled positive; 15% of its reads sit in the negative mode.
    flipped = rng.random(n) < flip
    logits = np.where(flipped, rng.normal(-3, 1, n), rng.normal(3, 1, n))
    labels = np.ones(n, dtype=int)
    clean = rng.normal(-3, 1, 1000)  # a clean negative group
    out = implied_flip_rates(
        np.concatenate([logits, clean]),
        np.concatenate([labels, np.zeros(1000, dtype=int)]),
        np.array(["Ala"] * n + ["Neg"] * 1000),
        supplied={"Ala": 0.05},
    )
    assert out["Ala"]["implied_flip_rate"] == pytest.approx(flip, abs=0.03)
    assert out["Ala"]["supplied_rate"] == 0.05
    assert out["Neg"]["implied_flip_rate"] < 0.03
    assert out["Neg"]["supplied_rate"] is None
    assert out["Ala"]["n"] == n


def test_implied_flip_rate_skips_small_groups():
    from leech.label_noise import implied_flip_rates

    out = implied_flip_rates(np.zeros(10), np.ones(10, dtype=int), np.array(["x"] * 10))
    assert out["x"]["implied_flip_rate"] is None
