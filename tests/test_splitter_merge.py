"""Tests for the merge-and-split path (``leech data merge``).

Every one of these functions writes the corpus a training run reads, and until
now none of them had a test. That is how ``merge_and_kfold_split_multiclass``
shipped with an inline merge that had never learned about the CSR
``seq_to_sig_values``/``seq_to_sig_offsets`` pair — it raised ``IndexError`` on
any corpus the current ``save_chunks`` writes — and how a merge of inputs with
different member sets could write a short column beside full ones.

The contract under test is that a merged split is a corpus: every member has
one row per chunk, the rows line up, no read appears in two splits, and the
variable-length base-to-signal map of each chunk comes out the way it went in.
"""

import tracemalloc

import numpy as np
import pytest

from leech.chunking import ChunkTable, load_chunks, save_chunks
from leech.splitting import (
    merge_and_kfold_split_chunks,
    merge_and_kfold_split_multiclass,
    merge_and_split_chunks,
    merge_and_split_multiclass,
)
from leech.splitting.splitter import _split_codes

SIGNAL_LEN = 48
KMER_LEN = 11
FEAT = (4, 13)
CHUNKS_PER_READ = 3


def make_chunks(
    n_reads: int,
    prefix: str,
    label: str,
    *,
    seed: int = 0,
    with_maps: bool = True,
    with_focus: bool = True,
) -> list[dict]:
    """A synthetic corpus: several chunks per read, ragged base-to-signal maps.

    Map lengths vary per chunk on purpose — a merge that masks the flat CSR
    values array instead of gathering its rows only shows up when the rows are
    not all the same width.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for read in range(n_reads):
        for j in range(CHUNKS_PER_READ):
            i = read * CHUNKS_PER_READ + j
            chunk = {
                "signal": rng.standard_normal(SIGNAL_LEN).astype(np.float32),
                "sequence": "".join(rng.choice(list("ACGT"), KMER_LEN)),
                "dwell": rng.integers(1, 9, FEAT[1]).astype(np.float32),
                "features": rng.standard_normal(FEAT).astype(np.float32),
                "label": label,
                "label_int": i % 2,
                "read_id": f"{prefix}_{read:06d}",
                "base_idx": 100 + i,
                "source_group": f"{label}_src",
                "reference_name": f"tRNA-{label}",
                "feature_start": -6,
                "feature_end": 6,
                "cl_value": (i % 7) - 1,
            }
            if with_maps:
                n_bases = 5 + (i % 6)
                chunk["seq_to_sig_map"] = np.sort(
                    rng.choice(SIGNAL_LEN, n_bases + 1, replace=False)
                ).astype(np.int64)
                chunk["sequence_with_kmer_context"] = "".join(rng.choice(list("ACGT"), n_bases))
            if with_focus:
                chunk["focus_signal_pos"] = SIGNAL_LEN // 2 + (i % 5)
            chunks.append(chunk)
    return chunks


def write_corpus(tmp_path, name, **kwargs):
    """Write one input file under its own directory (source_group comes from it)."""
    path = tmp_path / name / "all.npz"
    save_chunks(make_chunks(**kwargs), path)
    return path


@pytest.fixture
def two_inputs(tmp_path):
    """Two single-label corpora, 12 reads each, no read id shared between them."""
    ala = write_corpus(tmp_path, "Ala", n_reads=12, prefix="ala", label="Ala", seed=1)
    gly = write_corpus(tmp_path, "Gly", n_reads=12, prefix="gly", label="Gly", seed=2)
    return ala, gly


def index_by_chunk(chunks: list[dict]) -> dict[tuple[str, int], dict]:
    """Chunks keyed by (read_id, base_idx), which is unique within a corpus."""
    return {(c["read_id"], c["base_idx"]): c for c in chunks}


def assert_is_a_corpus(path):
    """Every member of a merged file must agree on how many chunks it holds."""
    with np.load(path, allow_pickle=True) as data:
        n_chunks = len(data["read_ids"])
        for name in data.files:
            if name == "seq_to_sig_values":
                continue  # flat CSR payload, addressed through the offsets
            expected = n_chunks + 1 if name == "seq_to_sig_offsets" else n_chunks
            assert len(data[name]) == expected, (
                f"{path.name}: member '{name}' has {len(data[name])} rows, expected {expected}"
            )
        if "seq_to_sig_offsets" in data:
            assert data["seq_to_sig_offsets"][-1] == len(data["seq_to_sig_values"])
    # ChunkTable reads the members independently of load_chunks, so it catches
    # a column that is short in a way the npz itself does not complain about.
    assert len(ChunkTable.from_npz(path)) == n_chunks
    return n_chunks


def assert_round_trip(inputs, split_paths, *, expect_labels_int=None):
    """The union of the splits is the union of the inputs, chunk for chunk."""
    source = {}
    for path in inputs:
        source.update(index_by_chunk(load_chunks(path)))

    seen: dict[str, str] = {}  # read_id -> split it appeared in
    total = 0
    for split_name, path in split_paths.items():
        if not path.exists():
            continue
        merged = load_chunks(path)
        total += len(merged)
        assert_is_a_corpus(path)
        for chunk in merged:
            key = (chunk["read_id"], chunk["base_idx"])
            assert key in source, f"{key} is not in any input"
            original = source[key]

            # Read-level splitting: a read must land in exactly one split.
            previous = seen.setdefault(chunk["read_id"], split_name)
            assert previous == split_name, (
                f"read {chunk['read_id']} is in both {previous} and {split_name}"
            )

            np.testing.assert_array_equal(chunk["signal"], original["signal"])
            np.testing.assert_array_equal(chunk["features"], original["features"])
            np.testing.assert_array_equal(chunk["dwell"], original["dwell"])
            assert chunk["sequence"] == original["sequence"]
            # The variable-length map is stored CSR; a merge that masks it
            # instead of gathering its rows scrambles or truncates this.
            if original["seq_to_sig_map"] is None:
                assert chunk["seq_to_sig_map"] is None
            else:
                np.testing.assert_array_equal(chunk["seq_to_sig_map"], original["seq_to_sig_map"])
            assert chunk["sequence_with_kmer_context"] == original["sequence_with_kmer_context"]
            assert chunk["focus_signal_pos"] == original["focus_signal_pos"]
            assert chunk["cl_value"] == original["cl_value"]
            if expect_labels_int is not None:
                assert chunk["label_int"] == expect_labels_int[original["label"]]

    assert total == len(source), f"merged {total} chunks, inputs hold {len(source)}"
    return total


class TestMergeAndSplit:
    """merge_and_split_chunks: the pairwise / single-split entry point."""

    def test_round_trip(self, two_inputs, tmp_path):
        out = tmp_path / "merged"
        result = merge_and_split_chunks(list(two_inputs), output_dir=out, seed=42)

        assert result["n_total"] == 24 * CHUNKS_PER_READ
        assert result["n_total"] == result["n_train"] + result["n_val"] + result["n_test"]
        total = assert_round_trip(two_inputs, result["output_files"])
        assert total == result["n_total"]

    def test_source_group_comes_from_the_input_directory(self, two_inputs, tmp_path):
        out = tmp_path / "merged"
        result = merge_and_split_chunks(list(two_inputs), output_dir=out, seed=42)
        groups = set()
        for path in result["output_files"].values():
            if path.exists():
                groups.update(str(g) for g in np.load(path, allow_pickle=True)["source_groups"])
        assert groups == {"Ala", "Gly"}

    def test_relabel_pairwise(self, two_inputs, tmp_path):
        out = tmp_path / "merged"
        result = merge_and_split_chunks(
            list(two_inputs), output_dir=out, seed=42, relabel_pairwise=("Ala", "Gly")
        )
        assert_round_trip(
            two_inputs, result["output_files"], expect_labels_int={"Ala": 0, "Gly": 1}
        )

    def test_an_input_with_no_selected_reads_contributes_nothing(self, two_inputs, tmp_path):
        """Not even its dtypes: a wider column there must not widen the output."""
        from leech.splitting.splitter import _merge_arrays_by_split

        ala, gly = two_inputs
        unused = write_corpus(
            tmp_path,
            "Asparagine",
            n_reads=6,
            prefix="a_very_long_unused_read_prefix",
            label="Asparagine",
            seed=9,
        )
        reads = {str(r) for r in np.load(ala, allow_pickle=True)["read_ids"]}
        reads |= {str(r) for r in np.load(gly, allow_pickle=True)["read_ids"]}

        outputs = {"train": tmp_path / "out" / "train.npz"}
        counts = _merge_arrays_by_split([ala, gly, unused], {"train": reads}, outputs)
        assert counts["train"] == 24 * CHUNKS_PER_READ

        with (
            np.load(outputs["train"], allow_pickle=True) as merged,
            np.load(ala, allow_pickle=True) as source,
        ):
            for name in ("read_ids", "labels", "source_groups"):
                assert merged[name].dtype == source[name].dtype

    def test_no_output_dir_returns_chunk_lists(self, two_inputs):
        train, val, test = merge_and_split_chunks(list(two_inputs), seed=42)
        assert len(train) + len(val) + len(test) == 24 * CHUNKS_PER_READ
        reads = [{c["read_id"] for c in split} for split in (train, val, test)]
        assert reads[0].isdisjoint(reads[1])
        assert reads[0].isdisjoint(reads[2])
        assert reads[1].isdisjoint(reads[2])

    def test_seed_is_the_only_input_to_the_assignment(self, two_inputs, tmp_path):
        def read_ids(seed, tag):
            result = merge_and_split_chunks(list(two_inputs), output_dir=tmp_path / tag, seed=seed)
            return {
                name: sorted({str(r) for r in np.load(path, allow_pickle=True)["read_ids"]})
                for name, path in result["output_files"].items()
                if path.exists()
            }

        assert read_ids(42, "a") == read_ids(42, "b")
        assert read_ids(42, "a") != read_ids(7, "c")


class TestKFoldSplit:
    """merge_and_kfold_split_chunks: every fold holds the whole corpus."""

    def test_round_trip_per_fold(self, two_inputs, tmp_path):
        out = tmp_path / "kfold"
        result = merge_and_kfold_split_chunks(list(two_inputs), out, k_fold=3, seed=42)

        assert result["k_fold"] == 3
        assert result["n_total"] == 24 * CHUNKS_PER_READ
        assert len(result["folds"]) == 3
        for fold in result["folds"]:
            total = assert_round_trip(two_inputs, fold["output_files"])
            assert total == result["n_total"]

    def test_test_partitions_are_disjoint_across_folds(self, two_inputs, tmp_path):
        result = merge_and_kfold_split_chunks(
            list(two_inputs), tmp_path / "kfold", k_fold=3, seed=42
        )
        seen: set[str] = set()
        for fold in result["folds"]:
            path = fold["output_files"]["test"]
            reads = {str(r) for r in np.load(path, allow_pickle=True)["read_ids"]}
            assert seen.isdisjoint(reads)
            seen |= reads

    def test_rejects_fewer_than_three_folds(self, two_inputs, tmp_path):
        with pytest.raises(ValueError, match="k_fold must be >= 3"):
            merge_and_kfold_split_chunks(list(two_inputs), tmp_path / "kfold", k_fold=2)


class TestMulticlass:
    """merge_and_split_multiclass: label_int 0..N-1 from the per-file labels."""

    @pytest.fixture
    def three_inputs(self, tmp_path):
        return [
            write_corpus(tmp_path, name, n_reads=9, prefix=name.lower(), label=name, seed=i)
            for i, name in enumerate(("Ala", "Gly", "Ser"))
        ]

    def test_round_trip(self, three_inputs, tmp_path):
        out = tmp_path / "multi"
        labels = ["Ala", "Gly", "Ser"]
        result = merge_and_split_multiclass(three_inputs, labels, out, seed=42)

        assert result["label_map"] == {"Ala": 0, "Gly": 1, "Ser": 2}
        assert result["n_total"] == 27 * CHUNKS_PER_READ
        assert_round_trip(
            three_inputs, result["output_files"], expect_labels_int=result["label_map"]
        )
        assert (out / "label_map.json").exists()

    def test_split_by_group(self, three_inputs, tmp_path):
        result = merge_and_split_multiclass(
            three_inputs,
            ["Ala", "Gly", "Ser"],
            tmp_path / "multi",
            seed=42,
            split_by="reference_names",
        )
        # One reference per label, so every label has a single group -> train.
        assert result["n_train"] == 27 * CHUNKS_PER_READ
        assert_round_trip(three_inputs, result["output_files"])

    def test_split_by_rejects_a_missing_field(self, three_inputs, tmp_path):
        with pytest.raises(ValueError, match="not found in"):
            merge_and_split_multiclass(
                three_inputs, ["Ala", "Gly", "Ser"], tmp_path / "multi", split_by="nope"
            )


class TestMulticlassKFold:
    """The regression that motivated this file.

    ``merge_and_kfold_split_multiclass`` carried its own copy of the merge, and
    that copy masked *every* member with the per-chunk boolean mask — including
    the flat CSR values array, whose length is the total number of map entries.
    On any corpus written by the current ``save_chunks`` it raised::

        IndexError: boolean index did not match indexed array along axis 0;
        size of axis is 84 but size of corresponding boolean axis is 12
    """

    @pytest.fixture
    def three_inputs(self, tmp_path):
        return [
            write_corpus(tmp_path, name, n_reads=9, prefix=name.lower(), label=name, seed=i)
            for i, name in enumerate(("Ala", "Gly", "Ser"))
        ]

    def test_does_not_raise_on_a_current_format_corpus(self, three_inputs, tmp_path):
        result = merge_and_kfold_split_multiclass(
            three_inputs, ["Ala", "Gly", "Ser"], tmp_path / "kfold", k_fold=3, seed=42
        )
        assert result["k_fold"] == 3
        assert result["label_map"] == {"Ala": 0, "Gly": 1, "Ser": 2}
        assert result["n_total"] == 27 * CHUNKS_PER_READ

    def test_round_trip_per_fold(self, three_inputs, tmp_path):
        result = merge_and_kfold_split_multiclass(
            three_inputs, ["Ala", "Gly", "Ser"], tmp_path / "kfold", k_fold=3, seed=42
        )
        for fold in result["folds"]:
            total = assert_round_trip(
                three_inputs, fold["output_files"], expect_labels_int=result["label_map"]
            )
            assert total == result["n_total"]

    def test_matches_the_shared_kfold_partitioning(self, three_inputs, tmp_path):
        """Both k-fold entry points must assign reads the same way at one seed."""
        multi = merge_and_kfold_split_multiclass(
            three_inputs, ["Ala", "Gly", "Ser"], tmp_path / "multi", k_fold=3, seed=42
        )
        pairwise = merge_and_kfold_split_chunks(three_inputs, tmp_path / "pair", k_fold=3, seed=42)
        for a, b in zip(multi["folds"], pairwise["folds"], strict=True):
            for split in ("train", "val", "test"):
                left = {str(r) for r in np.load(a["output_files"][split])["read_ids"]}
                right = {str(r) for r in np.load(b["output_files"][split])["read_ids"]}
                assert left == right


class TestMemberSetMismatch:
    """Merging inputs whose member sets differ used to misalign a column.

    A file written before ``focus_signal_pos`` contributes no rows for that
    member but full rows for every other one, so the merged file carried
    ``focus_signal_pos`` with half as many rows as ``read_ids``: reads past the
    short column's end raise, and with the inputs in the other order every
    affected chunk silently gets another read's focus position — the wrong
    asymmetric signal crop.
    """

    @pytest.fixture
    def mismatched(self, tmp_path):
        with_focus = write_corpus(
            tmp_path, "Ala", n_reads=8, prefix="ala", label="Ala", seed=1, with_focus=True
        )
        without = write_corpus(
            tmp_path, "Gly", n_reads=8, prefix="gly", label="Gly", seed=2, with_focus=False
        )
        return with_focus, without

    def test_refuses_and_names_the_member_and_the_file(self, mismatched, tmp_path):
        with_focus, without = mismatched
        with pytest.raises(ValueError, match="focus_signal_pos") as exc:
            merge_and_split_chunks([with_focus, without], output_dir=tmp_path / "out", seed=42)
        assert str(without) in str(exc.value)

    def test_refuses_in_either_input_order(self, mismatched, tmp_path):
        """The reversed order is the silent one, so it must fail too."""
        with_focus, without = mismatched
        with pytest.raises(ValueError, match="focus_signal_pos"):
            merge_and_split_chunks([without, with_focus], output_dir=tmp_path / "out", seed=42)

    def test_never_writes_a_short_member(self, mismatched, tmp_path):
        out = tmp_path / "out"
        with_focus, without = mismatched
        with pytest.raises(ValueError):
            merge_and_split_chunks([with_focus, without], output_dir=out, seed=42)
        for path in out.glob("*.npz"):
            assert_is_a_corpus(path)

    def test_chunks_without_a_map_merge_beside_chunks_with_one(self, tmp_path):
        """``save_chunks`` writes an empty CSR row rather than omitting the member.

        So this is not a member-set mismatch — but it is the case where the CSR
        row lengths of one input are all zero, and the merged offsets have to
        stay one-per-chunk anyway or every row after the first file shifts.
        """
        with_maps = write_corpus(
            tmp_path, "Ala", n_reads=8, prefix="ala", label="Ala", seed=1, with_maps=True
        )
        without = write_corpus(
            tmp_path, "Gly", n_reads=8, prefix="gly", label="Gly", seed=2, with_maps=False
        )
        result = merge_and_split_chunks([with_maps, without], output_dir=tmp_path / "out", seed=42)
        assert result["n_total"] == 16 * CHUNKS_PER_READ
        assert_round_trip([with_maps, without], result["output_files"])

    def test_kfold_multiclass_refuses_too(self, mismatched, tmp_path):
        with_focus, without = mismatched
        with pytest.raises(ValueError, match="different member sets"):
            merge_and_kfold_split_multiclass(
                [with_focus, without], ["Ala", "Gly"], tmp_path / "out", k_fold=3, seed=42
            )

    def test_matching_inputs_still_merge(self, two_inputs, tmp_path):
        """The guard must not reject the ordinary case."""
        result = merge_and_split_chunks(list(two_inputs), output_dir=tmp_path / "out", seed=42)
        assert result["n_total"] == 24 * CHUNKS_PER_READ


class TestLegacyMapFormat:
    """Corpora written before v0.6.8 store the maps as a pickled object array."""

    def test_legacy_maps_normalize_to_csr(self, tmp_path):
        source = write_corpus(tmp_path, "Ala", n_reads=8, prefix="ala", label="Ala", seed=1)
        legacy = tmp_path / "Legacy" / "all.npz"
        legacy.parent.mkdir(parents=True)
        with np.load(source, allow_pickle=True) as data:
            members = {
                k: data[k]
                for k in data.files
                if k not in ("seq_to_sig_values", "seq_to_sig_offsets")
            }
            values, offsets = data["seq_to_sig_values"], data["seq_to_sig_offsets"]
        members["seq_to_sig_maps"] = np.array(
            [values[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)],
            dtype=object,
        )
        np.savez(legacy, **members)

        result = merge_and_split_chunks([legacy], output_dir=tmp_path / "out", seed=42)
        assert result["n_total"] == 8 * CHUNKS_PER_READ
        assert_round_trip([source], result["output_files"])
        for path in result["output_files"].values():
            if path.exists():
                with np.load(path, allow_pickle=True) as data:
                    assert "seq_to_sig_values" in data.files
                    assert "seq_to_sig_maps" not in data.files


class TestSplitCodes:
    """The vectorized split-mask build must agree with the loop it replaced."""

    def test_matches_the_membership_comprehension(self):
        rng = np.random.default_rng(0)
        read_ids = np.array([f"read_{i:06d}" for i in rng.integers(0, 200, 900)], dtype=str)
        ids = sorted({str(r) for r in read_ids})
        splits = {
            "train": set(ids[:100]),
            "val": set(ids[100:150]),
            "test": set(ids[150:190]),
        }
        codes = _split_codes(read_ids, splits)
        for code, (_name, rid_set) in enumerate(splits.items()):
            expected = np.array([str(r) in rid_set for r in read_ids], dtype=bool)
            np.testing.assert_array_equal(codes == code, expected)
        # The 10 ids in no split must be marked as such, not folded into one.
        unassigned = np.array(
            [not any(str(r) in s for s in splits.values()) for r in read_ids], dtype=bool
        )
        np.testing.assert_array_equal(codes == -1, unassigned)

    def test_handles_a_column_that_is_not_unicode(self):
        read_ids = np.array(["a", "b", "c"], dtype=object)
        codes = _split_codes(read_ids, {"train": {"a", "c"}, "val": {"b"}})
        np.testing.assert_array_equal(codes, [0, 1, 0])


def write_wide_corpus(path, n_chunks, tag, signal_len=540):
    """A corpus with production-width signals, written without building dicts.

    ``save_chunks`` would need a chunk dict and five ndarrays per row to write
    this; the memory test only cares about the members on disk.
    """
    rng = np.random.default_rng(abs(hash(tag)) % 2**32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        signals_flat=rng.standard_normal((n_chunks, signal_len)).astype(np.float32),
        dwells_flat=rng.integers(1, 9, (n_chunks, KMER_LEN)).astype(np.float32),
        features_flat=rng.standard_normal((n_chunks, *FEAT)).astype(np.float32),
        sequences=np.array(["ACGTACGTACG"] * n_chunks, dtype=str),
        labels=np.array([tag] * n_chunks, dtype=str),
        labels_int=np.zeros(n_chunks, dtype=np.int64),
        read_ids=np.array([f"{tag}_{i // 3:032d}" for i in range(n_chunks)], dtype=str),
        base_indices=np.arange(n_chunks, dtype=np.int64),
        feature_starts=np.full(n_chunks, -6, dtype=np.int64),
        feature_ends=np.full(n_chunks, 6, dtype=np.int64),
        source_groups=np.array([tag] * n_chunks, dtype=str),
        reference_names=np.array(["tRNA-X"] * n_chunks, dtype=str),
        cl_values=np.full(n_chunks, -1, dtype=np.int16),
        seq_to_sig_values=np.tile(np.arange(12, dtype=np.int32), n_chunks),
        seq_to_sig_offsets=np.arange(n_chunks + 1, dtype=np.int64) * 12,
        sequences_with_kmer_context=np.array(["ACGT" * 3] * n_chunks, dtype=str),
        focus_signal_pos=np.full(n_chunks, signal_len // 2, dtype=np.int64),
    )
    with np.load(path, allow_pickle=True) as data:
        return sum(data[name].nbytes for name in data.files)


class TestMergeMemory:
    """The merge must not hold the corpus twice.

    Accumulate-into-lists-then-concatenate cannot peak below roughly the merged
    corpus plus the largest split, because the source slices are still alive
    when the destination is allocated: 2.03x the payload on the corpus below,
    and 1.80x of peak RSS on a 661 MB one. Preallocate-and-fill peaks at the
    outputs plus one block buffer (8 MB by default) — 1.36x here — which is
    why the corpus has to be wide enough for that buffer not to dominate.
    """

    def test_peak_stays_near_the_payload(self, tmp_path):
        payload = sum(
            write_wide_corpus(tmp_path / f"L{i}" / "all.npz", 15_000, f"L{i}") for i in range(2)
        )
        inputs = [tmp_path / f"L{i}" / "all.npz" for i in range(2)]

        # merge_and_split_chunks imports leech.model_loading lazily, which
        # pulls in torch. Do it now: 60 MB of import machinery inside the
        # measured region makes the result depend on what ran before this test.
        import leech.model_loading  # noqa: F401

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            result = merge_and_split_chunks(inputs, output_dir=tmp_path / "out", seed=42)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        assert result["n_total"] == 30_000
        assert peak < 1.5 * payload, (
            f"merge peaked at {peak / 1e6:.1f} MB for a {payload / 1e6:.1f} MB payload "
            f"({peak / payload:.2f}x); the accumulate-then-concatenate shape this "
            f"replaced could not get below ~1.8x"
        )
