"""Tests for the incremental chunk writer (#211).

``ChunkSpool``/``ChunkNpzWriter`` take chunks a batch at a time and never hold
the corpus, but they must produce exactly what :func:`save_chunks` produces
from the same chunks — every member, dtype, shape and value, including the CSR
``seq_to_sig_values``/``seq_to_sig_offsets`` pair and the object-array
fallbacks for ragged chunks. These tests are what keeps the two writers from
drifting apart, and they check that every existing reader still reads the
result.
"""

import gc
import tracemalloc
import zipfile

import numpy as np
import pytest

from leech.chunking import (
    ChunkNpzWriter,
    ChunkSpool,
    ChunkTable,
    iter_npz_row_blocks,
    load_chunks,
    npz_array_members,
    save_chunks,
)
from leech.chunking import serialization as ser

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_chunks(
    n,
    *,
    start=0,
    ragged=False,
    residual=True,
    focus=True,
    empty_features=False,
    empty_maps=False,
    siglen=40,
    seed=0,
):
    """A batch of chunks shaped like what the extractors emit."""
    rng = np.random.default_rng(seed + start)
    chunks = []
    for offset in range(n):
        i = start + offset
        length = siglen + (i % 3 if ragged else 0)
        klen = 11
        chunk = {
            "signal": rng.standard_normal(length).astype(np.float32),
            "sequence": "ACGT" * (2 + i % 2),
            "dwell": rng.standard_normal(klen).astype(np.float32),
            "features": (
                np.empty((0,), dtype=np.float32)
                if empty_features
                else rng.standard_normal((5, klen)).astype(np.float32)
            ),
            "label": ["Ala", "Gly", "Trp"][i % 3],
            "label_int": i % 3,
            "read_id": f"read-{i % 7:04d}",
            "base_idx": i,
            "feature_start": -5,
            "feature_end": 5,
            "source_group": f"grp{i % 2}",
            "reference_name": f"ref{i % 4}",
            "cl_value": None if i % 5 == 0 else i % 7,
            "seq_to_sig_map": (
                np.array([], dtype=np.int64)
                if empty_maps
                else np.arange(3 + i % 4, dtype=np.int64) * 7
            ),
            "sequence_with_kmer_context": "ACGTACGTACG",
        }
        if residual:
            chunk["signal_residual"] = rng.standard_normal(length).astype(np.float32)
        if focus:
            chunk["focus_signal_pos"] = 20 + i % 5
        chunks.append(chunk)
    return chunks


def npz_members(path):
    with np.load(path, allow_pickle=True) as data:
        return {name: data[name] for name in data.files}


def assert_npz_identical(expected_path, actual_path):
    """Same members, in the same order, with the same dtypes/shapes/values."""
    with zipfile.ZipFile(expected_path) as a, zipfile.ZipFile(actual_path) as b:
        assert a.namelist() == b.namelist()

    expected = npz_members(expected_path)
    actual = npz_members(actual_path)
    assert set(expected) == set(actual)
    for name, want in expected.items():
        got = actual[name]
        assert got.dtype == want.dtype, f"{name}: dtype {got.dtype} != {want.dtype}"
        assert got.shape == want.shape, f"{name}: shape {got.shape} != {want.shape}"
        if want.dtype.hasobject:
            assert len(got) == len(want)
            for u, v in zip(want.ravel(), got.ravel(), strict=True):
                assert np.array_equal(u, v), f"{name}: ragged row differs"
        else:
            assert np.array_equal(got, want), f"{name}: values differ"


CASES = {
    "plain": {},
    "ragged": {"ragged": True},
    "no-residual": {"residual": False},
    "no-focus": {"focus": False},
    "empty-features": {"empty_features": True},
    "empty-maps": {"empty_maps": True},
}


# ---------------------------------------------------------------------------
# The core guarantee: the writer's output == save_chunks' output
# ---------------------------------------------------------------------------


class TestWriterMatchesSaveChunks:
    @pytest.mark.parametrize("case", sorted(CASES))
    @pytest.mark.parametrize("compressed", [False, True])
    def test_single_batch(self, tmp_path, case, compressed):
        chunks = make_chunks(13, **CASES[case])
        reference = tmp_path / "reference.npz"
        streamed = tmp_path / "streamed.npz"

        save_chunks(chunks, reference, compressed=compressed)
        with ChunkNpzWriter(streamed, compressed=compressed) as writer:
            writer.append(chunks)

        assert_npz_identical(reference, streamed)

    @pytest.mark.parametrize("case", sorted(CASES))
    @pytest.mark.parametrize("compressed", [False, True])
    def test_many_batches(self, tmp_path, case, compressed):
        """Batch boundaries must not show up anywhere in the file."""
        batches = [make_chunks(7, start=start, **CASES[case]) for start in range(0, 35, 7)]
        chunks = [chunk for batch in batches for chunk in batch]
        reference = tmp_path / "reference.npz"
        streamed = tmp_path / "streamed.npz"

        save_chunks(chunks, reference, compressed=compressed)
        with ChunkNpzWriter(streamed, compressed=compressed) as writer:
            for batch in batches:
                writer.append(batch)

        assert_npz_identical(reference, streamed)

    def test_widening_text_across_batches(self, tmp_path):
        """A later batch with longer strings must widen the member, not truncate it."""
        first = make_chunks(3)
        for chunk in first:
            chunk["label"] = "A"
            chunk["source_group"] = "g"
        second = make_chunks(3, start=3)
        for chunk in second:
            chunk["label"] = "Methionine"
            chunk["source_group"] = "a-much-longer-group"

        reference = tmp_path / "reference.npz"
        streamed = tmp_path / "streamed.npz"
        save_chunks(first + second, reference)
        with ChunkNpzWriter(streamed) as writer:
            writer.append(first)
            writer.append(second)

        assert_npz_identical(reference, streamed)
        with np.load(streamed) as data:
            assert list(data["labels"]) == ["A", "A", "A"] + ["Methionine"] * 3

    def test_appends_npz_suffix_like_savez(self, tmp_path):
        with ChunkNpzWriter(tmp_path / "corpus") as writer:
            writer.append(make_chunks(4))
        assert (tmp_path / "corpus.npz").exists()

    def test_empty_batches_are_ignored(self, tmp_path):
        reference = tmp_path / "reference.npz"
        streamed = tmp_path / "streamed.npz"
        chunks = make_chunks(5)
        save_chunks(chunks, reference)
        with ChunkNpzWriter(streamed) as writer:
            writer.append([])
            writer.append(chunks)
            writer.append([])
        assert_npz_identical(reference, streamed)

    def test_nothing_written_when_the_body_raises(self, tmp_path):
        out = tmp_path / "corpus.npz"
        with pytest.raises(RuntimeError):
            with ChunkNpzWriter(out) as writer:
                writer.append(make_chunks(3))
                raise RuntimeError("boom")
        assert not out.exists()

    def test_mismatched_batches_raise(self, tmp_path):
        """Batches that spill separately must agree on the member layout."""
        with pytest.raises(ValueError, match="disagree|changed layout"):
            with ChunkNpzWriter(tmp_path / "corpus.npz", batch_rows=3) as writer:
                writer.append(make_chunks(3, siglen=40))
                writer.append(make_chunks(3, start=3, siglen=50))

    def test_batching_is_invisible_in_the_output(self, tmp_path):
        """batch_rows only controls when the spool flushes, never the file."""
        batches = [make_chunks(5, start=start) for start in (0, 5, 10)]
        chunks = [chunk for batch in batches for chunk in batch]
        reference = tmp_path / "reference.npz"
        save_chunks(chunks, reference, compressed=False)
        for batch_rows in (1, 2, 7, 1000):
            out = tmp_path / f"rows{batch_rows}.npz"
            with ChunkNpzWriter(out, compressed=False, batch_rows=batch_rows) as writer:
                for batch in batches:
                    writer.append(batch)
            assert_npz_identical(reference, out)

    def test_empty_spool_refuses_to_write(self, tmp_path):
        with ChunkSpool(tmp_path) as spool:
            with pytest.raises(ValueError, match="No chunks to save"):
                spool.write_npz(tmp_path / "corpus.npz")


# ---------------------------------------------------------------------------
# Row selection: the same file save_chunks would write for a subset
# ---------------------------------------------------------------------------


class TestRowSelection:
    @pytest.mark.parametrize("case", sorted(CASES))
    def test_subset_matches_save_chunks_of_that_subset(self, tmp_path, case):
        batches = [make_chunks(6, start=start, **CASES[case]) for start in range(0, 24, 6)]
        chunks = [chunk for batch in batches for chunk in batch]
        # Deliberately not sorted and not contiguous: the prepare split hands
        # back rows grouped by read, which is neither.
        rows = np.array([17, 3, 4, 5, 22, 0, 11], dtype=np.int64)

        reference = tmp_path / "reference.npz"
        selected = tmp_path / "selected.npz"
        save_chunks([chunks[i] for i in rows], reference, compressed=False)
        with ChunkSpool(tmp_path, compressed=False) as spool:
            for batch in batches:
                spool.append(batch)
            n = spool.write_npz(selected, rows=rows)
        assert n == len(rows)

        assert_npz_identical(reference, selected)

    def test_disjoint_splits_cover_the_corpus(self, tmp_path):
        chunks = make_chunks(20)
        rows = np.arange(20)
        parts = [rows[:9], rows[9:15], rows[15:]]
        with ChunkSpool(tmp_path, compressed=False) as spool:
            spool.append(chunks)
            for i, part in enumerate(parts):
                spool.write_npz(tmp_path / f"part{i}.npz", rows=part)

        seen = []
        for i in range(3):
            seen.extend(
                c["read_id"] + str(c["base_idx"]) for c in load_chunks(tmp_path / f"part{i}.npz")
            )
        assert seen == [c["read_id"] + str(c["base_idx"]) for c in chunks]

    def test_text_width_is_the_subset_s_own(self, tmp_path):
        """A split file's text members must be as wide as that split needs, no wider."""
        chunks = make_chunks(6)
        for i, chunk in enumerate(chunks):
            chunk["label"] = "Methionine" if i < 3 else "A"
        rows = np.array([3, 4, 5], dtype=np.int64)

        reference = tmp_path / "reference.npz"
        selected = tmp_path / "selected.npz"
        save_chunks([chunks[i] for i in rows], reference, compressed=False)
        with ChunkSpool(tmp_path, compressed=False) as spool:
            spool.append(chunks)
            spool.write_npz(selected, rows=rows)

        with np.load(selected) as data:
            assert data["labels"].dtype == np.dtype("<U1")
        assert_npz_identical(reference, selected)

    def test_prepare_split_matches_the_chunk_level_split(self, tmp_path):
        """The spooled prepare split must write what the list-based one wrote.

        ``data prepare`` used to split a list of chunk dicts and hand each
        split to ``save_chunks``; it now splits row indices and writes them out
        of the spool. Same rule, same rows, same order, same files.
        """
        from leech.preparation.orchestrator import split_rows_by_read
        from leech.splitting import split_chunks_by_read

        batches = [make_chunks(9, start=start) for start in range(0, 45, 9)]
        chunks = [chunk for batch in batches for chunk in batch]

        # Reference: the old path.
        reference = split_chunks_by_read(chunks, train_frac=0.7, val_frac=0.15, seed=42)
        for name, split in zip(("train", "val", "test"), reference, strict=True):
            if split:
                save_chunks(split, tmp_path / f"ref-{name}.npz", compressed=False)

        # New path: spool, then row indices from the same rule.
        with ChunkSpool(tmp_path, compressed=False) as spool:
            for batch in batches:
                spool.append(batch)
            rows = split_rows_by_read(spool.read_ids(), 0.7, 0.15, 42)
            for name, part in zip(("train", "val", "test"), rows, strict=True):
                if len(part):
                    spool.write_npz(tmp_path / f"new-{name}.npz", rows=part)

        assert [len(split) for split in reference] == [len(part) for part in rows]
        assert sum(len(part) for part in rows) == len(chunks)
        for name, split in zip(("train", "val", "test"), reference, strict=True):
            if split:
                assert_npz_identical(tmp_path / f"ref-{name}.npz", tmp_path / f"new-{name}.npz")

    def test_falls_back_when_the_spill_cannot_be_mapped(self, tmp_path, monkeypatch):
        """A filesystem that refuses mmap must not fail the write."""

        def no_mmap(*args, **kwargs):
            raise OSError("mmap not supported here")

        chunks = make_chunks(12)
        rows = np.array([7, 1, 2, 11], dtype=np.int64)
        reference = tmp_path / "reference.npz"
        selected = tmp_path / "selected.npz"
        save_chunks([chunks[i] for i in rows], reference, compressed=False)

        monkeypatch.setattr(ser.np, "memmap", no_mmap)
        with ChunkSpool(tmp_path, compressed=False) as spool:
            spool.append(chunks)
            spool.write_npz(selected, rows=rows)

        assert_npz_identical(reference, selected)

    def test_read_ids_column(self, tmp_path):
        batches = [make_chunks(5, start=start) for start in (0, 5)]
        with ChunkSpool(tmp_path) as spool:
            for batch in batches:
                spool.append(batch)
            read_ids = spool.read_ids()
        expected = [c["read_id"] for batch in batches for c in batch]
        assert list(read_ids) == expected
        assert len(read_ids) == 10


# ---------------------------------------------------------------------------
# Every existing reader must read what the writer produced
# ---------------------------------------------------------------------------


class TestReadersUnchanged:
    @pytest.mark.parametrize("compressed", [False, True])
    def test_load_chunks_round_trip(self, tmp_path, compressed):
        batches = [make_chunks(6, start=start) for start in (0, 6, 12)]
        chunks = [chunk for batch in batches for chunk in batch]
        out = tmp_path / "corpus.npz"
        with ChunkNpzWriter(out, compressed=compressed) as writer:
            for batch in batches:
                writer.append(batch)

        loaded = load_chunks(out)
        assert len(loaded) == len(chunks)
        for got, want in zip(loaded, chunks, strict=True):
            assert got["read_id"] == want["read_id"]
            assert got["label"] == want["label"]
            assert got["base_idx"] == want["base_idx"]
            assert np.array_equal(got["signal"], want["signal"])
            assert np.array_equal(got["dwell"], want["dwell"])
            assert np.array_equal(got["features"], want["features"])
            assert np.array_equal(got["seq_to_sig_map"], want["seq_to_sig_map"])
            assert got["focus_signal_pos"] == want["focus_signal_pos"]

    def test_chunk_table_and_row_blocks(self, tmp_path):
        batches = [make_chunks(8, start=start) for start in (0, 8)]
        out = tmp_path / "corpus.npz"
        with ChunkNpzWriter(out, compressed=False) as writer:
            for batch in batches:
                writer.append(batch)

        table = ChunkTable.from_npz(out)
        assert len(table) == 16
        assert [row["read_id"] for row in table] == [
            c["read_id"] for batch in batches for c in batch
        ]

        members = npz_array_members(out)
        assert members["signals_flat"][0] == (16, 40)

        seen = 0
        for start, blocks in iter_npz_row_blocks(out, ["signals_flat"], block_rows=5):
            assert blocks["signals_flat"].shape[1] == 40
            seen += len(blocks["signals_flat"])
            assert start + len(blocks["signals_flat"]) <= 16
        assert seen == 16


# ---------------------------------------------------------------------------
# The point of all this: peak memory
# ---------------------------------------------------------------------------


class TestMemory:
    def test_save_chunks_does_not_copy_when_casting(self, monkeypatch):
        """M3: the chunks are already float32, so every astype must be copy=False."""
        copies = []
        real_stack = np.stack

        class _Recorder(np.ndarray):
            def astype(self, dtype, *args, **kwargs):
                if args:
                    copies.append(args[2] if len(args) > 2 else True)
                else:
                    copies.append(kwargs.get("copy", True))
                return np.ndarray.astype(self, dtype, *args, **kwargs)

        def spy(*args, **kwargs):
            return real_stack(*args, **kwargs).view(_Recorder)

        monkeypatch.setattr(ser.np, "stack", spy)
        list(ser.iter_chunk_columns(make_chunks(6)))

        assert copies, "no stacked member was cast — the spy did not fire"
        assert all(flag is False for flag in copies), f"astype copied: {copies}"

    def test_writer_peak_is_a_batch_not_the_corpus(self, tmp_path):
        """M4: appending batches must not scale peak allocation with the corpus."""
        n_batches, batch = 12, 400
        siglen = 512
        # signal + residual + dwell + features, per chunk
        payload = n_batches * batch * (2 * siglen + 11 + 5 * 11) * 4

        gc.collect()
        tracemalloc.start()
        try:
            with ChunkNpzWriter(
                tmp_path / "corpus.npz", compressed=False, batch_rows=batch
            ) as writer:
                for start in range(n_batches):
                    chunks = make_chunks(batch, start=start * batch, siglen=siglen)
                    writer.append(chunks)
                    del chunks
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # save_chunks needs the whole corpus resident plus a stacked copy; the
        # writer needs one batch. Half the payload is a wide margin either way.
        assert peak < payload * 0.5, f"peak {peak} of payload {payload}"


class TestOptionalTextRoundTrip:
    """`None` text fields are stored as `""`, not as the string `"None"`.

    `--label` defaults to `None`, so a `data prepare` run that does not pass it
    gave every chunk the literal label `"None"` -- and because `load_chunks`
    maps `""` back to `None`, a plain load/save round trip *renamed* an empty
    source group to a group called `"None"`. `--balance-groups` then weighted
    that bucket like a real one, and pairwise relabelling (which matches on the
    stored label) silently matched nothing.
    """

    @staticmethod
    def _chunks(label, source_group, reference_name, n=3):
        return [
            {
                "signal": np.zeros(8, np.float32),
                "sequence": "ACGTACGTACG",
                "dwell": np.ones(5, np.float32),
                "features": np.ones((2, 5), np.float32),
                "label": label,
                "label_int": None,
                "read_id": f"r{i}",
                "base_idx": i,
                "source_group": source_group,
                "reference_name": reference_name,
                "feature_start": -2,
                "feature_end": 2,
                "cl_value": None,
                "seq_to_sig_map": np.arange(6, dtype=np.int64),
                "sequence_with_kmer_context": "ACGT" * 4,
                "focus_signal_pos": 4,
            }
            for i in range(n)
        ]

    def test_none_is_stored_as_empty_not_the_string_none(self, tmp_path):
        path = tmp_path / "unlabelled.npz"
        save_chunks(self._chunks(None, None, None), path, compressed=False)

        with np.load(path) as data:
            for member in ("labels", "source_groups", "reference_names"):
                stored = data[member].tolist()
                assert stored == ["", "", ""], f"{member} stored as {stored!r}"

    def test_none_loads_back_as_the_readers_absent_value(self, tmp_path):
        """Whatever `load_chunks` calls "absent" -- never the string "None".

        The reader is not uniform and this pins that rather than papering over
        it: `label` and `source_group` come back as `None`, while
        `reference_name` deliberately comes back as `""` (see `load_chunks`).
        Both are fine; the round trip is idempotent either way. What matters is
        that none of them is the four-character string "None".
        """
        path = tmp_path / "unlabelled.npz"
        save_chunks(self._chunks(None, None, None), path, compressed=False)

        chunk = load_chunks(path)[0]
        assert chunk["label"] is None
        assert chunk["source_group"] is None
        assert chunk["reference_name"] == ""
        assert "None" not in (chunk["label"], chunk["source_group"], chunk["reference_name"])

    def test_round_trip_is_idempotent(self, tmp_path):
        """save(load(x)) == x. It was "" -> None -> "None" before."""
        first, second = tmp_path / "one.npz", tmp_path / "two.npz"
        save_chunks(self._chunks("", "", ""), first, compressed=False)
        save_chunks(load_chunks(first), second, compressed=False)

        with np.load(first) as a, np.load(second) as b:
            for member in ("labels", "source_groups", "reference_names"):
                assert a[member].tolist() == b[member].tolist() == ["", "", ""]

    def test_a_real_label_still_survives(self, tmp_path):
        path = tmp_path / "labelled.npz"
        save_chunks(self._chunks("Ala", "ThrRS_thr_b1", "tRNA-Ala-AGC"), path, compressed=False)

        chunk = load_chunks(path)[0]
        assert chunk["label"] == "Ala"
        assert chunk["source_group"] == "ThrRS_thr_b1"
        assert chunk["reference_name"] == "tRNA-Ala-AGC"

    def test_the_spooled_writer_agrees(self, tmp_path):
        """Both write paths share `iter_chunk_columns`; prove it stays that way."""
        listed, spooled = tmp_path / "listed.npz", tmp_path / "spooled.npz"
        chunks = self._chunks(None, None, None)
        save_chunks(chunks, listed, compressed=False)

        writer = ChunkNpzWriter(spooled, compressed=False)
        writer.append(chunks)
        writer.close()

        with np.load(listed) as a, np.load(spooled) as b:
            for member in ("labels", "source_groups", "reference_names"):
                assert a[member].tolist() == b[member].tolist() == ["", "", ""]
