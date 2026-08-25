"""Tests for the columnar chunk-metadata store.

``ChunkTable`` replaces one dict per chunk — 780 bytes each, measured — with
columns plus a row view. Everything downstream reads chunks as mappings, so the
contract under test is that a row is indistinguishable from the dict
``load_chunks`` would have built, including which keys are absent and which
values come back as None.
"""

import numpy as np
import pytest

from leech.chunking import ChunkTable, load_chunks, save_chunks

READ_ID_LEN = 36  # UUID-shaped, as dorado writes them


def make_chunks(n: int = 8, *, labelled: bool = True) -> list[dict]:
    rng = np.random.default_rng(7)
    chunks = []
    for i in range(n):
        chunks.append(
            {
                "signal": rng.standard_normal(32).astype(np.float32),
                "sequence": "".join(rng.choice(list("ACGT"), 11)),
                "dwell": rng.integers(1, 9, 13).astype(np.float32),
                "features": rng.standard_normal((4, 13)).astype(np.float32),
                # Empty label and -1 label_int are how load_chunks spells None.
                "label": ("charged" if i % 2 else "uncharged") if labelled else "",
                "label_int": (i % 2) if labelled else None,
                "read_id": f"{i:0{READ_ID_LEN}d}",
                "base_idx": 1000 + i,
                "source_group": "Ala" if i % 3 else "",
                "reference_name": "tRNA-Ala-AGC-1-1" if i % 2 else "",
                "feature_start": -6,
                "feature_end": 6,
                "cl_value": i if i % 4 else -1,
                "focus_signal_pos": 8 + i,
                "seq_to_sig_map": np.arange(12, dtype=np.int64),
                "sequence_with_kmer_context": "ACGT" * 3,
            }
        )
    return chunks


@pytest.fixture
def corpus(tmp_path):
    chunks = make_chunks(8)
    path = tmp_path / "chunks.npz"
    save_chunks(chunks, path)
    return path, chunks


class TestRowsMatchLoadChunks:
    """A row must read exactly like the dict load_chunks builds."""

    def test_every_metadata_field_matches(self, corpus):
        path, _ = corpus
        table = ChunkTable.from_npz(path)
        dicts = load_chunks(path)

        assert len(table) == len(dicts)
        array_fields = {
            "signal",
            "signal_residual",
            "dwell",
            "features",
            "seq_to_sig_map",
        }
        for row, chunk in zip(table, dicts, strict=True):
            for key in set(chunk) - array_fields:
                assert row.get(key) == chunk[key], key

    def test_missing_values_read_back_as_none(self, corpus):
        path, chunks = corpus
        table = ChunkTable.from_npz(path)

        # Sentinels: "" for text, -1 for ints — except reference_name, which
        # load_chunks reports as "" rather than None.
        assert table[0]["source_group"] is None  # i % 3 == 0 wrote ""
        assert table[1]["source_group"] == "Ala"
        assert table[0]["reference_name"] == ""
        assert table[1]["reference_name"] == "tRNA-Ala-AGC-1-1"
        assert table[4]["cl_value"] is None  # i % 4 == 0 wrote -1
        assert table[5]["cl_value"] == 5

    def test_unlabelled_chunks_read_as_none(self, tmp_path):
        path = tmp_path / "unlabelled.npz"
        save_chunks(make_chunks(4, labelled=False), path)
        table = ChunkTable.from_npz(path)
        assert [row["label_int"] for row in table] == [None] * 4
        assert [row["label"] for row in table] == [None] * 4


class TestMappingContract:
    def test_absent_field_is_absent_not_none(self, corpus):
        path, _ = corpus
        table = ChunkTable.from_npz(path)
        row = table[0]

        # `"feature_start" in chunk` is how training.py picks a feature-window
        # convention, so absence has to mean absence.
        assert "feature_start" in row
        assert "dwell_margin_left" not in row
        assert row.get("dwell_margin_left") is None
        with pytest.raises(KeyError):
            row["dwell_margin_left"]

    def test_row_is_a_mapping(self, corpus):
        path, _ = corpus
        table = ChunkTable.from_npz(path)
        row = table[2]
        assert dict(row) == {key: row[key] for key in row}
        assert set(row.keys()) == set(row)
        assert len(row) == len(list(row))
        assert row["read_id"] in repr(row)

    def test_indexing(self, corpus):
        path, chunks = corpus
        table = ChunkTable.from_npz(path)
        assert table[-1]["read_id"] == chunks[-1]["read_id"]
        with pytest.raises(IndexError):
            table[len(chunks)]
        with pytest.raises(TypeError, match="one chunk at a time"):
            table[0:2]

    def test_skip_leaves_a_field_out(self, corpus):
        path, _ = corpus
        full = ChunkTable.from_npz(path)
        trimmed = ChunkTable.from_npz(path, skip=("sequence_with_kmer_context",))
        assert "sequence_with_kmer_context" in full[0]
        assert "sequence_with_kmer_context" not in trimmed[0]
        assert trimmed[0]["sequence"] == full[0]["sequence"]


class TestSelect:
    def test_select_keeps_the_masked_rows(self, corpus):
        path, chunks = corpus
        table = ChunkTable.from_npz(path)
        mask = np.array([i % 3 == 0 for i in range(len(chunks))])

        selected = table.select(mask)
        expected = [c for c, keep in zip(chunks, mask, strict=True) if keep]
        assert len(selected) == len(expected)
        for row, chunk in zip(selected, expected, strict=True):
            assert row["read_id"] == chunk["read_id"]
            assert row["base_idx"] == chunk["base_idx"]
            assert row["cl_value"] == (chunk["cl_value"] if chunk["cl_value"] >= 0 else None)

    def test_select_none(self, corpus):
        path, chunks = corpus
        table = ChunkTable.from_npz(path)
        empty = table.select(np.zeros(len(chunks), dtype=bool))
        assert len(empty) == 0
        assert list(empty) == []


class TestStorage:
    """The two mechanisms that make it small, asserted directly."""

    def test_integers_are_narrowed_without_changing_values(self, corpus):
        path, chunks = corpus
        table = ChunkTable.from_npz(path)

        assert table.values("label_int").dtype == np.int8
        assert table.values("feature_start").dtype == np.int8
        assert table.values("base_idx").dtype == np.int16  # 1000..1007
        # Negative sentinels survive narrowing.
        assert table.values("cl_value").min() == -1
        assert [row["base_idx"] for row in table] == [c["base_idx"] for c in chunks]

    def test_text_is_stored_as_bytes(self, corpus):
        path, _ = corpus
        table = ChunkTable.from_npz(path)
        assert table.values("read_id").dtype.kind == "S"
        assert table.values("sequence").dtype.kind == "S"
        assert isinstance(table[0]["read_id"], str)

    def test_non_ascii_text_still_loads(self, tmp_path):
        chunks = make_chunks(4)
        for chunk in chunks:
            chunk["source_group"] = "Ångström"
        path = tmp_path / "unicode.npz"
        save_chunks(chunks, path)

        table = ChunkTable.from_npz(path)
        assert table.values("source_group").dtype.kind == "U"  # kept, not dropped
        assert table[0]["source_group"] == "Ångström"

    def test_columns_are_far_smaller_than_dicts(self, tmp_path):
        n = 2000
        path = tmp_path / "big.npz"
        save_chunks(make_chunks(n), path)

        table = ChunkTable.from_npz(path)
        per_chunk = table.nbytes() / n
        # The dicts this replaces measured 780 B/chunk on a corpus with these
        # same fields; the columns hold the same values in ~100.
        assert per_chunk < 200, f"{per_chunk:.0f} B/chunk"

    def test_values_returns_none_for_an_absent_field(self, corpus):
        path, _ = corpus
        assert ChunkTable.from_npz(path).values("dwell_margin_left") is None


class TestLegacyCorpora:
    def test_dwell_margin_left_replaces_the_signed_window(self, tmp_path):
        """Pre-feature_start corpora carry dwell_margin_lefts instead."""
        chunks = make_chunks(4)
        path = tmp_path / "legacy.npz"
        np.savez(
            path,
            sequences=np.array([c["sequence"] for c in chunks], dtype=str),
            labels=np.array([c["label"] for c in chunks], dtype=str),
            labels_int=np.array([c["label_int"] for c in chunks], dtype=np.int64),
            read_ids=np.array([c["read_id"] for c in chunks], dtype=str),
            base_indices=np.array([c["base_idx"] for c in chunks], dtype=np.int64),
            dwell_margin_lefts=np.full(len(chunks), 4, dtype=np.int64),
        )

        table = ChunkTable.from_npz(path)
        row = table[0]
        assert row["dwell_margin_left"] == 4
        assert "feature_start" not in row
        # cl_value predates this format but callers read it unguarded.
        assert row["cl_value"] is None


class TestPickling:
    """A DataLoader that spawns workers pickles the dataset, table and all."""

    def test_round_trips(self, corpus):
        import pickle

        path, chunks = corpus
        table = ChunkTable.from_npz(path)
        restored = pickle.loads(pickle.dumps(table))

        assert len(restored) == len(table)
        for row, original in zip(restored, chunks, strict=True):
            assert row["read_id"] == original["read_id"]
            assert row["label_int"] == original["label_int"]
            assert row["sequence"] == original["sequence"]
