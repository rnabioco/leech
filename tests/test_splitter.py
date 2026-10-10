"""Run-held-out splitting (issue #394): whole sequencing runs become the test split.

A read-level split puts every flowcell on both sides of the train/test line, so a
run-level property that tracks the class scores as accuracy. These tests pin the
properties that make the run-level split mean something: no run appears in two
of {train, test}, a class with fewer than two runs is refused (naming it), an
unknown run is refused rather than pooled into a pseudo-run, and the sidecar
`eval test` reads records which runs trained.
"""

import json
import random

import numpy as np
import pytest
from test_splitter_merge import make_chunks

from leech.chunking import ChunkTable, save_chunks
from leech.splitting import merge_and_split_chunks, merge_and_split_multiclass
from leech.splitting.splitter import SPLIT_RUNS_FILENAME, _split_by_run, is_run_split


def corpus_with_runs(tmp_path, name, label, runs, *, reads_per_run=12, seed=0):
    """One input file whose reads are spread over ``runs`` (a list of run ids)."""
    chunks = []
    for k, run in enumerate(runs):
        part = make_chunks(reads_per_run, f"{name}_{run}", label, seed=seed + k)
        for chunk in part:
            chunk["run_id"] = run
            chunk["flow_cell_id"] = f"FC_{run}"
        chunks.extend(part)
    path = tmp_path / name / "all.npz"
    save_chunks(chunks, path)
    return path


def runs_by_split(out_dir):
    """``{split: set(run ids)}`` read back from a merged directory."""
    result = {}
    for split in ("train", "val", "test"):
        table = ChunkTable.from_npz(out_dir / f"{split}.npz")
        result[split] = {str(r) for r in np.unique(table.require_values("run_id").astype(str))}
    return result


@pytest.fixture
def replicate_inputs(tmp_path):
    """Two classes with two runs each (Cys-style replicates)."""
    a = corpus_with_runs(tmp_path, "A", "A", ["rA1", "rA2"], seed=1)
    b = corpus_with_runs(tmp_path, "B", "B", ["rB1", "rB2"], seed=2)
    return a, b


def test_holdout_run_split_by_run_id_keeps_test_runs_out_of_train(tmp_path, replicate_inputs):
    a, b = replicate_inputs
    out = tmp_path / "merged"
    merge_and_split_multiclass([a, b], ["A", "B"], out, seed=7, split_by="run_id")

    runs = runs_by_split(out)
    assert runs["test"], "no test split"
    assert not runs["test"] & runs["train"], "a held-out run leaked into train"
    assert not runs["test"] & runs["val"], "a held-out run leaked into val"
    # Every class contributes a held-out run: one of its two, not both.
    assert len(runs["test"] & {"rA1", "rA2"}) == 1
    assert len(runs["test"] & {"rB1", "rB2"}) == 1
    # Train and val still see the remaining run of each class.
    assert runs["train"] == {"rA1", "rA2", "rB1", "rB2"} - runs["test"]


def test_holdout_run_explicit_runs_become_test(tmp_path, replicate_inputs):
    a, b = replicate_inputs
    out = tmp_path / "merged"
    merge_and_split_multiclass([a, b], ["A", "B"], out, seed=7, holdout_runs=["rA2", "rB1"])

    runs = runs_by_split(out)
    assert runs["test"] == {"rA2", "rB1"}
    assert runs["train"] == {"rA1", "rB2"}


def test_holdout_run_writes_sidecar_naming_train_and_heldout_runs(tmp_path, replicate_inputs):
    a, b = replicate_inputs
    out = tmp_path / "merged"
    merge_and_split_multiclass([a, b], ["A", "B"], out, seed=7, holdout_runs=["rA2", "rB1"])

    info = json.loads((out / SPLIT_RUNS_FILENAME).read_text())
    assert info["holdout_runs"] == {"A": ["rA2"], "B": ["rB1"]}
    assert info["train_runs"] == {"A": ["rA1"], "B": ["rB2"]}
    assert info["read_level_classes"] == []


def test_holdout_run_is_reproducible_for_a_seed(tmp_path, replicate_inputs):
    a, b = replicate_inputs
    first = tmp_path / "one"
    second = tmp_path / "two"
    merge_and_split_multiclass([a, b], ["A", "B"], first, seed=11, split_by="run_id")
    merge_and_split_multiclass([a, b], ["A", "B"], second, seed=11, split_by="run_id")
    assert runs_by_split(first) == runs_by_split(second)


def test_holdout_run_pairwise_entry_point(tmp_path, replicate_inputs):
    a, b = replicate_inputs
    out = tmp_path / "pair"
    stats = merge_and_split_chunks(
        [a, b],
        output_dir=out,
        seed=3,
        relabel_by_file={a: (0, "A"), b: (1, "B")},
        split_by="run_id",
    )
    runs = runs_by_split(out)
    assert not runs["test"] & runs["train"]
    assert stats["n_test"] > 0
    labels = ChunkTable.from_npz(out / "test.npz").require_values("label_int")
    assert set(np.unique(labels)) == {0, 1}, "both classes need a held-out run in test"


def test_holdout_run_refuses_single_run_class_naming_it(tmp_path):
    a = corpus_with_runs(tmp_path, "A", "A", ["rA1"], seed=1)
    b = corpus_with_runs(tmp_path, "B", "B", ["rB1", "rB2"], seed=2)
    with pytest.raises(ValueError, match=r"A: only 1 run\(s\)"):
        merge_and_split_multiclass([a, b], ["A", "B"], tmp_path / "out", seed=1, split_by="run_id")


def test_holdout_run_override_falls_back_to_read_level_for_that_class(tmp_path):
    a = corpus_with_runs(tmp_path, "A", "A", ["rA1"], seed=1)
    b = corpus_with_runs(tmp_path, "B", "B", ["rB1", "rB2"], seed=2)
    out = tmp_path / "out"
    merge_and_split_multiclass(
        [a, b],
        ["A", "B"],
        out,
        seed=1,
        split_by="run_id",
        allow_single_run_classes=True,
    )
    info = json.loads((out / SPLIT_RUNS_FILENAME).read_text())
    assert info["read_level_classes"] == ["A"]
    # The single-run class still reaches test, via a read-level split.
    test_labels = ChunkTable.from_npz(out / "test.npz").require_values("label_int")
    assert set(np.unique(test_labels)) == {0, 1}


def test_holdout_run_refuses_unknown_run_ids(tmp_path):
    # Corpus written without run info: every run id is "".
    unlabelled = tmp_path / "A" / "all.npz"
    save_chunks(make_chunks(10, "A", "A"), unlabelled)
    b = corpus_with_runs(tmp_path, "B", "B", ["rB1", "rB2"])
    with pytest.raises(ValueError, match="carry no run id"):
        merge_and_split_multiclass(
            [unlabelled, b], ["A", "B"], tmp_path / "out", seed=1, split_by="run_id"
        )


def test_holdout_run_refuses_corpus_predating_run_ids(tmp_path):
    path = tmp_path / "old" / "all.npz"
    save_chunks(make_chunks(6, "old", "A"), path)
    # Drop the members, as an older corpus would lack them.
    with np.load(path, allow_pickle=False) as data:
        kept = {k: data[k] for k in data.files if k not in ("run_ids", "flow_cell_ids")}
    np.savez(path, **kept)
    b = corpus_with_runs(tmp_path, "B", "B", ["rB1", "rB2"])
    with pytest.raises(ValueError, match="no 'run_ids' member"):
        merge_and_split_multiclass(
            [path, b], ["A", "B"], tmp_path / "out", seed=1, split_by="run_id"
        )


def test_holdout_run_unmatched_holdout_run_is_an_error(tmp_path, replicate_inputs):
    a, b = replicate_inputs
    with pytest.raises(ValueError, match="matches no run"):
        merge_and_split_multiclass(
            [a, b], ["A", "B"], tmp_path / "out", seed=1, holdout_runs=["typo"]
        )


def test_holdout_run_holds_out_a_shared_run_for_every_class_in_it():
    # One run carries both classes (multiplexed barcodes). Auto-selection must
    # hold out the same run for both classes, not a different one per class.
    read_to_run, read_to_label = {}, {}
    for run in ("r1", "r2"):
        for label in ("A", "B"):
            for i in range(10):
                rid = f"{run}_{label}_{i}"
                read_to_run[rid] = run
                read_to_label[rid] = label
    for seed in range(8):
        random.seed(seed)
        splits, info = _split_by_run(read_to_run, read_to_label)
        assert info["holdout_runs"]["A"] == info["holdout_runs"]["B"]
        held = info["holdout_runs"]["A"]
        assert {read_to_run[r] for r in splits["test"]} == set(held)
        assert not {read_to_run[r] for r in splits["train"]} & set(held)


def test_holdout_run_read_level_val_is_drawn_from_training_runs():
    read_to_run = {f"a{i}": "r1" for i in range(40)} | {f"b{i}": "r2" for i in range(40)}
    read_to_label = dict.fromkeys(read_to_run, "A")
    random.seed(0)
    splits, _ = _split_by_run(
        read_to_run, read_to_label, holdout_runs=["r2"], train_frac=0.7, val_frac=0.15
    )
    assert splits["test"] == {f"b{i}" for i in range(40)}
    assert splits["train"] | splits["val"] == {f"a{i}" for i in range(40)}
    assert splits["val"], "val must exist so checkpoint selection has data"
    assert not splits["train"] & splits["val"]


def test_is_run_split_recognises_both_spellings():
    assert is_run_split("run_id", None)
    assert is_run_split("run_ids", None)
    assert is_run_split(None, ["r1"])
    assert not is_run_split(None, None)
    assert not is_run_split("reference_names", None)
    assert not is_run_split(None, [])
