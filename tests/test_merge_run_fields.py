"""run_id / flow_cell_id on the chunk format (issue #394).

The two fields are per-chunk POD5 run identity. They have to survive every
writer and reader the corpus goes through -- ``save_chunks``, ``ChunkSpool``,
``load_chunks``, ``ChunkTable`` and the merge -- and an unknown run must be the
empty string everywhere, never the four-character string "None" (a run called
"None" is a perfectly good split group, and `_text()` exists to prevent it).
Corpora written before the fields existed must still load.
"""

import numpy as np
from test_splitter_merge import make_chunks

from leech.chunking import ChunkSpool, ChunkTable, load_chunks, save_chunks
from leech.splitting import merge_and_split_multiclass


def with_runs(chunks, run_id, flow_cell_id):
    for chunk in chunks:
        chunk["run_id"] = run_id
        chunk["flow_cell_id"] = flow_cell_id
    return chunks


def test_run_fields_round_trip_through_save_and_load(tmp_path):
    chunks = with_runs(make_chunks(4, "r", "A"), "acq-1", "FAZ123")
    path = tmp_path / "c.npz"
    save_chunks(chunks, path)

    loaded = load_chunks(path)
    assert {c["run_id"] for c in loaded} == {"acq-1"}
    assert {c["flow_cell_id"] for c in loaded} == {"FAZ123"}

    table = ChunkTable.from_npz(path)
    assert table[0]["run_id"] == "acq-1"
    assert table[0]["flow_cell_id"] == "FAZ123"


def test_empty_run_id_round_trips_as_empty_string_never_none_text(tmp_path):
    chunks = make_chunks(3, "r", "A")
    chunks[0]["run_id"] = None  # key present, value None: what --label-less paths produce
    chunks[0]["flow_cell_id"] = None
    # the remaining chunks carry no key at all
    path = tmp_path / "c.npz"
    save_chunks(chunks, path)

    with np.load(path, allow_pickle=False) as data:
        assert set(data["run_ids"].tolist()) == {""}
        assert set(data["flow_cell_ids"].tolist()) == {""}

    # Re-save a loaded corpus: the round trip must be idempotent.
    again = tmp_path / "again.npz"
    save_chunks(load_chunks(path), again)
    with np.load(again, allow_pickle=False) as data:
        assert set(data["run_ids"].tolist()) == {""}
        assert "None" not in data["run_ids"].tolist()
    assert ChunkTable.from_npz(again)[0]["run_id"] == ""


def test_spool_writes_the_same_run_members_as_save_chunks(tmp_path):
    chunks = with_runs(make_chunks(5, "r", "A"), "acq-9", "FC9")
    direct = tmp_path / "direct.npz"
    spooled = tmp_path / "spooled.npz"
    save_chunks(chunks, direct)
    with ChunkSpool(tmp_path) as spool:
        spool.append(chunks[:7])
        spool.append(chunks[7:])
        spool.write_npz(spooled)

    with np.load(direct, allow_pickle=False) as a, np.load(spooled, allow_pickle=False) as b:
        for member in ("run_ids", "flow_cell_ids"):
            assert member in b.files
            np.testing.assert_array_equal(a[member], b[member])


def test_corpus_without_run_members_still_loads(tmp_path):
    path = tmp_path / "old.npz"
    save_chunks(make_chunks(3, "r", "A"), path)
    with np.load(path, allow_pickle=False) as data:
        kept = {k: data[k] for k in data.files if k not in ("run_ids", "flow_cell_ids")}
    np.savez(path, **kept)

    loaded = load_chunks(path)
    assert loaded and "run_id" not in loaded[0]
    table = ChunkTable.from_npz(path)
    assert "run_id" not in table.columns
    assert "run_id" not in table[0]


def test_merge_carries_run_fields_one_row_per_chunk(tmp_path):
    inputs, labels = [], []
    for name, label, runs in (("A", "A", ("rA1", "rA2")), ("B", "B", ("rB1", "rB2"))):
        chunks = []
        for k, run in enumerate(runs):
            chunks += with_runs(make_chunks(10, f"{name}{k}", label, seed=k), run, f"FC_{run}")
        path = tmp_path / name / "all.npz"
        save_chunks(chunks, path)
        inputs.append(path)
        labels.append(label)

    out = tmp_path / "merged"
    merge_and_split_multiclass(inputs, labels, out, seed=2, split_by="run_id")

    for split in ("train", "val", "test"):
        with np.load(out / f"{split}.npz", allow_pickle=False) as data:
            n = len(data["read_ids"])
            assert len(data["run_ids"]) == n
            assert len(data["flow_cell_ids"]) == n
            # run and flow cell stay paired row for row
            for run, fc in zip(data["run_ids"], data["flow_cell_ids"], strict=True):
                assert fc == f"FC_{run}"
