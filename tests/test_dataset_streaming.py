"""Tests for streaming chunk arrays out of an npz instead of loading them whole.

``LeechDataset`` used to hold three copies of the corpus at once: the numpy
members from ``load_chunks``, one tensor per chunk, and the contiguous output
``torch.stack`` built while that list was still alive (#211). It now fills a
preallocated tensor from row blocks read straight off the npz.

The hard gate here is parity: a dataset built from a path (streaming) must be
bit-identical to one built from pre-loaded chunks (eager), field by field, over
the option matrix that changes how chunks are prepared.

Streaming now also prepares a whole row block at a time rather than a chunk at
a time (the loop cost 72 us per chunk, ~8 minutes of startup on a 6.7M-chunk
corpus). ``TestBlockFillParity`` is the gate on that: the block-wise filler and
the per-chunk one — which is the old loop, moved but not altered — must produce
identical tensors over the same matrix.
"""

import contextlib

import numpy as np
import pytest
import torch

from leech.chunking import (
    ChunkTable,
    csr_gather_index,
    iter_npz_row_blocks,
    load_chunks,
    npz_array_members,
    npz_member_names,
    save_chunks,
)
from leech.chunking.table import ChunkRow
from leech.dataset import LeechDataset

#: Per-chunk arrays: streamed or deferred, so they are not metadata and the
#: columnar store does not carry them.
ARRAY_FIELDS = frozenset(
    {
        "signal",
        "signal_residual",
        "dwell",
        "features",
        "seq_to_sig_map",
        "sequence_with_kmer_context",
    }
)

STORED_SIGNAL_LEN = 64
FEAT_WIDTH = 13
KMER_LEN = 11
NUM_FEATURES = 4


def make_chunks(
    n: int = 12,
    *,
    signal_len: int = STORED_SIGNAL_LEN,
    with_residual: bool = True,
    with_maps: bool = True,
    unlabeled: tuple[int, ...] = (),
    seed: int = 0,
    ragged_sequences: bool = False,
    varying_feature_start: bool = False,
    without_focus: bool = False,
    classes: int = 2,
) -> list[dict]:
    """Build a small synthetic corpus.

    Focus positions vary per chunk so an asymmetric crop lands inside the
    stored signal for some rows and overhangs it (the zero-pad branch) for
    others — the per-row gather the streaming path has to reproduce.

    ``ragged_sequences``, ``varying_feature_start`` and ``without_focus`` each
    put one column into a shape the block-wise filler has to notice: a text
    column whose rows are not all the same length, a per-chunk feature window,
    and a corpus with no focus position at all.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for i in range(n):
        sequence = "".join(rng.choice(list("ACGT"), KMER_LEN))
        chunk = {
            "signal": rng.standard_normal(signal_len).astype(np.float32),
            "sequence": sequence[: KMER_LEN - (i % 2)] if ragged_sequences else sequence,
            "dwell": rng.integers(1, 9, FEAT_WIDTH).astype(np.float32),
            "features": rng.standard_normal((NUM_FEATURES, FEAT_WIDTH)).astype(np.float32),
            "label": f"class{i % classes}"
            if classes > 2
            else ("charged" if i % 2 else "uncharged"),
            "label_int": None if i in unlabeled else i % classes,
            "read_id": f"read_{i:03d}",
            "base_idx": 100 + i,
            "source_group": "Ala" if i % 3 else "Gly",
            "reference_name": "tRNA-Ala-AGC",
            "feature_start": -(FEAT_WIDTH // 2) + (i % 2 if varying_feature_start else 0),
            "feature_end": FEAT_WIDTH // 2,
            "cl_value": i % 5,
            # 8, 22, 36, 50, 8, ... : the first and last overhang a 20/24 crop.
            "focus_signal_pos": 8 + 14 * (i % 4),
        }
        if without_focus:
            del chunk["focus_signal_pos"]
        if with_residual:
            chunk["signal_residual"] = rng.standard_normal(signal_len).astype(np.float32)
        if with_maps:
            n_bases = 9 + (i % 4)
            chunk["seq_to_sig_map"] = np.sort(
                rng.choice(min(signal_len, STORED_SIGNAL_LEN), n_bases + 1, replace=False)
            ).astype(np.int64)
            chunk["sequence_with_kmer_context"] = "".join(rng.choice(list("ACGT"), n_bases))
        chunks.append(chunk)
    return chunks


def build(path, **kwargs) -> tuple[LeechDataset, LeechDataset]:
    """Return (streaming, eager) datasets built from the same file and options."""
    streamed = LeechDataset(chunk_path=path, **kwargs)
    eager = LeechDataset(chunks=load_chunks(path), **kwargs)
    return streamed, eager


@contextlib.contextmanager
def forced_row_fill(monkeypatch):
    """Make every dataset built inside take the per-chunk filler.

    ``_block_fill_supported`` returning False is exactly the pre-block-wise
    behaviour: ``_fill_from_rows`` is the original loop, moved but not altered.
    """
    monkeypatch.setattr(LeechDataset, "_block_fill_supported", lambda self: False)
    try:
        yield
    finally:
        monkeypatch.undo()


def build_both_fillers(monkeypatch, path, **kwargs) -> tuple[LeechDataset, LeechDataset]:
    """Return (block-filled, row-filled) datasets from the same file and options."""
    blocked = LeechDataset(chunk_path=path, **kwargs)
    with forced_row_fill(monkeypatch):
        rowed = LeechDataset(chunk_path=path, **kwargs)
    return blocked, rowed


#: (name, make_chunks kwargs, LeechDataset kwargs). Every option that changes
#: how a chunk is prepared, so the block-wise filler is held to the per-chunk
#: one everywhere it is allowed to run.
BASE_OPTIONS = {
    "signal_len": STORED_SIGNAL_LEN,
    "kmer_len": KMER_LEN,
    "model_type": "ConvLSTMDwell",
    "seq_encoding": "base_onehot",
}
ASYM_OPTIONS = {**BASE_OPTIONS, "signal_len": 44, "left_context": 20, "right_context": 24}
KMER_OPTIONS = {**BASE_OPTIONS, "seq_encoding": "signal_kmer", "signal_kmer_context": (2, 2)}

FILL_MATRIX = [
    ("signal_mode_both", {}, BASE_OPTIONS),
    ("signal_mode_signal", {}, {**BASE_OPTIONS, "signal_mode": "signal"}),
    ("signal_mode_residual", {}, {**BASE_OPTIONS, "signal_mode": "residual"}),
    ("no_residual", {"with_residual": False}, BASE_OPTIONS),
    ("asymmetric_crop", {}, ASYM_OPTIONS),
    ("asymmetric_wide_features", {}, {**ASYM_OPTIONS, "model_type": "TCNDwellResidualLN"}),
    ("no_feature_branch", {}, {**BASE_OPTIONS, "model_type": "ConvLSTMBase"}),
    ("wide_features", {}, {**BASE_OPTIONS, "model_type": "TCNDwellResidualLN"}),
    ("dwell_offset", {}, {**BASE_OPTIONS, "dwell_offset": 1}),
    ("signal_kmer", {}, KMER_OPTIONS),
    ("signal_kmer_asymmetric", {}, {**KMER_OPTIONS, **ASYM_OPTIONS, "seq_encoding": "signal_kmer"}),
    ("signal_kmer_without_maps", {"with_maps": False}, KMER_OPTIONS),
    ("cl_regression", {}, {**BASE_OPTIONS, "cl_regression": True}),
    ("multiclass", {"classes": 4}, BASE_OPTIONS),
    ("unlabeled_rows", {"unlabeled": (0, 1, 6, 11)}, BASE_OPTIONS),
    ("varying_feature_start", {"varying_feature_start": True}, BASE_OPTIONS),
    ("without_focus", {"without_focus": True}, BASE_OPTIONS),
    ("without_focus_asymmetric", {"without_focus": True}, ASYM_OPTIONS),
    (
        "without_focus_signal_kmer",
        {"without_focus": True},
        {**KMER_OPTIONS, **ASYM_OPTIONS, "seq_encoding": "signal_kmer"},
    ),
    ("short_stored_signal", {"signal_len": 40}, BASE_OPTIONS),
    ("long_stored_signal", {"signal_len": 100}, BASE_OPTIONS),
    ("many_blocks", {"n": 900}, BASE_OPTIONS),
]


def assert_datasets_equal(streamed: LeechDataset, eager: LeechDataset) -> None:
    """Every tensor the two datasets expose must match bit for bit."""
    assert len(streamed) == len(eager)
    assert streamed.signal_channels == eager.signal_channels
    assert streamed._effective_seq_encoding == eager._effective_seq_encoding
    assert streamed._has_signal_residual == eager._has_signal_residual

    for name in (
        "_signals_tensor",
        "_features_tensor",
        "_labels_tensor",
        "_encoded_seqs_tensor",
        "_seq_ints_tensor",
        "_seq_to_sig_tensor",
        "_confound_labels_tensor",
        "_cl_targets_tensor",
    ):
        a, b = getattr(streamed, name), getattr(eager, name)
        assert (a is None) == (b is None), f"{name}: one path produced a tensor, the other did not"
        if a is not None:
            assert a.shape == b.shape, f"{name}: {a.shape} != {b.shape}"
            assert torch.equal(a, b), f"{name}: values differ"

    # And the assembled samples, which is what training actually consumes.
    for idx in (0, len(streamed) // 2, len(streamed) - 1):
        left, right = streamed[idx], eager[idx]
        assert left.keys() == right.keys()
        for key in left:
            assert torch.equal(left[key], right[key]), f"item {idx} field {key}"

    # Every metadata field, however it is stored: the columnar path must be
    # indistinguishable from the chunk dicts to samplers, the label tally and
    # the training-config introspection that read them.
    for i, (a, b) in enumerate(zip(streamed.chunks, eager.chunks, strict=True)):
        for key in (set(a) | set(b)) - ARRAY_FIELDS:
            assert a.get(key) == b.get(key), f"chunk {i} metadata {key}"


class TestNpzStreaming:
    """The row-block reader underneath the dataset."""

    @pytest.mark.parametrize("compressed", [True, False])
    @pytest.mark.parametrize("block_rows", [1, 5, 12, 100])
    def test_roundtrip_matches_np_load(self, tmp_path, compressed, block_rows):
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(12), path, compressed=compressed)

        names = ["signals_flat", "features_flat", "labels_int"]
        blocks: dict[str, list[np.ndarray]] = {name: [] for name in names}
        starts = []
        for start, block in iter_npz_row_blocks(path, names, block_rows):
            starts.append(start)
            for name in names:
                blocks[name].append(block[name].copy())

        assert starts == list(range(0, 12, block_rows))
        with np.load(path) as data:
            for name in names:
                np.testing.assert_array_equal(np.concatenate(blocks[name]), data[name])

    def test_blocks_are_recycled(self, tmp_path):
        """Documented contract: the yielded arrays are views into one buffer."""
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(8), path)
        seen = [
            block["signals_flat"] for _, block in iter_npz_row_blocks(path, ["signals_flat"], 4)
        ]
        assert seen[0].base is seen[1].base

    def test_pickled_member_raises(self, tmp_path):
        path = tmp_path / "objects.npz"
        np.savez(path, ragged=np.array([np.arange(3), np.arange(5)], dtype=object))
        with pytest.raises(ValueError, match="not row-streamable"):
            list(iter_npz_row_blocks(path, ["ragged"], 2))

    def test_block_size_defaults_to_a_byte_budget(self, tmp_path):
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(64, signal_len=4096), path, compressed=False)
        rows = [
            len(block["signals_flat"])
            for _, block in iter_npz_row_blocks(path, ["signals_flat"], block_bytes=1 << 18)
        ]
        # 4096 float32 = 16 KiB per row, so 16 rows fit the 256 KiB budget.
        assert rows[0] == 16
        assert sum(rows) == 64

    def test_members_exclude_pickled(self, tmp_path):
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(4), path)
        members = npz_array_members(path)
        assert members["signals_flat"][0] == (4, STORED_SIGNAL_LEN)
        assert members["features_flat"][0] == (4, NUM_FEATURES, FEAT_WIDTH)
        # CSR replaced the pickled object member entirely.
        assert "seq_to_sig_maps" not in npz_member_names(path)
        assert {"seq_to_sig_values", "seq_to_sig_offsets"} <= set(members)

    def test_csr_gather_index(self, tmp_path):
        path = tmp_path / "chunks.npz"
        chunks = make_chunks(6)
        save_chunks(chunks, path)
        with np.load(path) as data:
            values, offsets = data["seq_to_sig_values"], data["seq_to_sig_offsets"]
        rows = np.array([1, 3, 5])
        lens, _col, src = csr_gather_index(offsets, rows)
        np.testing.assert_array_equal(lens, [len(chunks[i]["seq_to_sig_map"]) for i in rows])
        np.testing.assert_array_equal(
            values[src], np.concatenate([chunks[i]["seq_to_sig_map"] for i in rows])
        )


class TestStreamingParity:
    """Streaming and eager construction must agree exactly."""

    @pytest.mark.parametrize("signal_mode", ["both", "signal", "residual"])
    def test_signal_modes(self, tmp_path, signal_mode):
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(12), path)
        assert_datasets_equal(
            *build(
                path,
                signal_len=STORED_SIGNAL_LEN,
                kmer_len=KMER_LEN,
                model_type="ConvLSTMDwell",
                signal_mode=signal_mode,
                seq_encoding="base_onehot",
            )
        )

    def test_streaming_path_stores_metadata_columnar(self, tmp_path):
        """The chunk dicts are the last per-chunk Python object at this scale."""
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(12), path)
        streamed = LeechDataset(
            chunk_path=path,
            signal_len=STORED_SIGNAL_LEN,
            kmer_len=KMER_LEN,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )
        assert isinstance(streamed.chunks, ChunkTable)
        # Text the run never reads is not loaded at all.
        assert "sequence_with_kmer_context" not in streamed.chunks[0]

    def test_asymmetric_crop(self, tmp_path):
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(12), path)
        assert_datasets_equal(
            *build(
                path,
                signal_len=44,
                kmer_len=KMER_LEN,
                model_type="ConvLSTMDwell",
                left_context=20,
                right_context=24,
                seq_encoding="base_onehot",
            )
        )

    @pytest.mark.parametrize("model_type", ["ConvLSTMDwell", "ConvLSTMBase"])
    def test_feature_and_non_feature_models(self, tmp_path, model_type):
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(12), path)
        streamed, eager = build(
            path,
            signal_len=STORED_SIGNAL_LEN,
            kmer_len=KMER_LEN,
            model_type=model_type,
            seq_encoding="base_onehot",
        )
        assert_datasets_equal(streamed, eager)
        if model_type == "ConvLSTMBase":
            # Nothing reads the features, so that member is never decompressed.
            assert "features" not in streamed._array_stream.members

    @pytest.mark.parametrize("dwell_offset", [0, 1])
    def test_dwell_offset(self, tmp_path, dwell_offset):
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(12), path)
        assert_datasets_equal(
            *build(
                path,
                signal_len=STORED_SIGNAL_LEN,
                kmer_len=KMER_LEN,
                model_type="ConvLSTMDwell",
                dwell_offset=dwell_offset,
                seq_encoding="base_onehot",
            )
        )

    @pytest.mark.parametrize("compressed", [True, False])
    @pytest.mark.parametrize("left_right", [None, (20, 24)])
    def test_signal_kmer_encoding(self, tmp_path, compressed, left_right):
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(12), path, compressed=compressed)
        kwargs = {
            "signal_len": STORED_SIGNAL_LEN if left_right is None else 44,
            "kmer_len": KMER_LEN,
            "model_type": "ConvLSTMDwell",
            "seq_encoding": "signal_kmer",
            "signal_kmer_context": (2, 2),
        }
        if left_right is not None:
            kwargs["left_context"], kwargs["right_context"] = left_right
        streamed, eager = build(path, **kwargs)
        assert streamed._effective_seq_encoding == "signal_kmer"
        assert_datasets_equal(streamed, eager)

    def test_signal_kmer_falls_back_without_maps(self, tmp_path):
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(8, with_maps=False), path)
        streamed, eager = build(
            path,
            signal_len=STORED_SIGNAL_LEN,
            kmer_len=KMER_LEN,
            model_type="ConvLSTMDwell",
            seq_encoding="signal_kmer",
        )
        assert streamed._effective_seq_encoding == "base_onehot"
        assert_datasets_equal(streamed, eager)

    def test_no_residual_channel(self, tmp_path):
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(10, with_residual=False), path)
        streamed, eager = build(
            path,
            signal_len=STORED_SIGNAL_LEN,
            kmer_len=KMER_LEN,
            model_type="ConvLSTMDwell",
            signal_mode="both",
            seq_encoding="base_onehot",
        )
        assert streamed.signal_channels == 1
        assert_datasets_equal(streamed, eager)

    def test_cl_regression_targets(self, tmp_path):
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(10), path)
        assert_datasets_equal(
            *build(
                path,
                signal_len=STORED_SIGNAL_LEN,
                kmer_len=KMER_LEN,
                model_type="ConvLSTMDwell",
                seq_encoding="base_onehot",
                cl_regression=True,
            )
        )

    def test_dwell_template_channels(self, tmp_path):
        table = tmp_path / "templates.tsv"
        rows = ["aa\tposition\tdwell_mean"]
        for aa in ("Ala", "Gly"):
            for pos in range(-6, 7):
                rows.append(f"{aa}\t{pos}\t{4.0 + pos * 0.1}")
        table.write_text("\n".join(rows) + "\n")

        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(10), path)
        streamed, eager = build(
            path,
            signal_len=STORED_SIGNAL_LEN,
            kmer_len=KMER_LEN,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
            dwell_template_table=table,
        )
        assert streamed._features_tensor.shape[1] == NUM_FEATURES + 2
        assert_datasets_equal(streamed, eager)


class TestBlockFillParity:
    """Preparing a row block at a time must equal preparing a chunk at a time."""

    @pytest.mark.parametrize("name,corpus,options", FILL_MATRIX, ids=[c[0] for c in FILL_MATRIX])
    @pytest.mark.parametrize("compressed", [True, False])
    def test_block_filler_matches_row_filler(
        self, tmp_path, monkeypatch, name, corpus, options, compressed
    ):
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(**corpus), path, compressed=compressed)
        blocked, rowed = build_both_fillers(monkeypatch, path, **options)
        assert_datasets_equal(blocked, rowed)

    @pytest.mark.parametrize("name,corpus,options", FILL_MATRIX, ids=[c[0] for c in FILL_MATRIX])
    def test_block_filler_matches_preloaded_chunks(self, tmp_path, name, corpus, options):
        """And the whole point: the block-wise path still equals the eager one."""
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(**corpus), path)
        streamed, eager = build(path, **options)
        assert streamed._block_fill_supported()
        assert_datasets_equal(streamed, eager)

    def test_block_fill_declines_dwell_templates(self, tmp_path):
        """The per-AA template append reads one chunk at a time; say so."""
        table = tmp_path / "templates.tsv"
        rows = ["aa\tposition\tdwell_mean"]
        for aa in ("Ala", "Gly"):
            for pos in range(-6, 7):
                rows.append(f"{aa}\t{pos}\t{4.0 + pos * 0.1}")
        table.write_text("\n".join(rows) + "\n")

        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(12), path)
        dataset = LeechDataset(chunk_path=path, dwell_template_table=table, **BASE_OPTIONS)
        assert not dataset._block_fill_supported()

    def test_block_fill_declines_ragged_sequences(self, tmp_path):
        """A NUL-padded row in the text column is a short sequence, not a base."""
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(10, ragged_sequences=True), path)
        dataset = LeechDataset(chunk_path=path, **BASE_OPTIONS)
        assert not dataset._block_fill_supported()
        assert dataset._encoded_seqs_tensor is None  # degraded, as before

    def test_block_fill_declines_a_feature_window_off_the_array(self, tmp_path):
        """The documented ValueError must still come from the row path."""
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(8), path)
        with pytest.raises(ValueError, match="exceeds feature width"):
            LeechDataset(chunk_path=path, dwell_offset=99, **BASE_OPTIONS)

    def test_block_fill_declines_a_ragged_asymmetric_crop(self, tmp_path):
        """An in-bounds width that differs from signal_len makes the output ragged.

        ``left + right`` is the width of the in-bounds branch while the
        overhang branch pads to ``signal_len``; when they disagree and some
        row overhangs, only the per-chunk path can produce the result.
        """
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(12), path)
        options = {**BASE_OPTIONS, "signal_len": 50, "left_context": 20, "right_context": 24}
        dataset = LeechDataset(chunk_path=path, **options)
        assert not dataset._block_fill_supported()

    def test_construction_does_not_build_a_row_view_per_chunk(self, tmp_path, monkeypatch):
        """No pass over the chunks may materialise a ChunkRow per chunk.

        Three did: the tensorize loop itself, the multi-class label tally and
        the asymmetric-crop focus lookup. Each cost ~150 ms per 200k chunks
        and ~5 s on the corpus in #211, before a single batch was served.
        """
        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(600), path)

        built = []
        original = ChunkRow.__init__

        def counting(self, table, index):
            built.append(index)
            original(self, table, index)

        monkeypatch.setattr(ChunkRow, "__init__", counting)
        LeechDataset(
            chunk_path=path,
            signal_len=44,
            kmer_len=KMER_LEN,
            model_type="ConvLSTMDwell",
            seq_encoding="signal_kmer",
            signal_kmer_context=(2, 2),
            left_context=20,
            right_context=24,
        )
        # A couple of probes (does the first chunk have a map?) are fine; a
        # count that scales with the corpus is the regression.
        assert len(built) < 10, f"{len(built)} row views built for 600 chunks"


class TestRowAlignment:
    """Rows dropped by label filtering must not shift the stream."""

    @pytest.mark.parametrize(
        "unlabeled",
        [(0,), (5,), (11,), (0, 1, 6, 11), tuple(range(1, 12, 2))],
    )
    def test_unlabeled_rows_are_skipped_not_shifted(self, tmp_path, unlabeled):
        path = tmp_path / "chunks.npz"
        chunks = make_chunks(12, unlabeled=unlabeled)
        save_chunks(chunks, path)

        streamed, eager = build(
            path,
            signal_len=STORED_SIGNAL_LEN,
            kmer_len=KMER_LEN,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )
        assert len(streamed) == 12 - len(unlabeled)
        assert_datasets_equal(streamed, eager)

        # Independently of the eager path: chunk i must carry the signal of the
        # source chunk with the same read_id, not of some earlier row.
        by_read = {c["read_id"]: c for c in chunks}
        for i, chunk in enumerate(streamed.chunks):
            expected = by_read[chunk["read_id"]]["signal"]
            np.testing.assert_array_equal(streamed._signals_tensor[i, 0].numpy(), expected)


class TestShapeMismatchFallback:
    """A field whose per-chunk shapes disagree still works, via list access."""

    def test_ragged_sequences_fall_back_to_a_list(self, tmp_path, caplog):
        chunks = make_chunks(10)
        for i, chunk in enumerate(chunks):  # 11, 10, 11, 10, ... bases
            chunk["sequence"] = chunk["sequence"][: KMER_LEN - (i % 2)]

        path = tmp_path / "chunks.npz"
        save_chunks(chunks, path)
        with caplog.at_level("WARNING", logger="leech.dataset"):
            streamed, eager = build(
                path,
                signal_len=STORED_SIGNAL_LEN,
                kmer_len=KMER_LEN,
                model_type="ConvLSTMDwell",
                seq_encoding="base_onehot",
            )

        assert "shapes differ" in caplog.text
        assert streamed._encoded_seqs_tensor is None
        assert len(streamed._encoded_seqs) == len(chunks)
        # The rows filled before the mismatch survive it.
        for i, chunk in enumerate(chunks):
            assert streamed._encoded_seqs[i].shape == (4, len(chunk["sequence"]))
            assert torch.equal(streamed._encoded_seqs[i], eager._encoded_seqs[i])
        # Everything else still fills a contiguous tensor.
        assert streamed._signals_tensor is not None
        assert torch.equal(streamed._signals_tensor, eager._signals_tensor)


class TestLegacyFormats:
    """Old corpora keep working — they simply do not stream."""

    def _write_object_format(self, path, chunks):
        """Write the pre-flat-array npz layout, pickled object members and all."""
        np.savez_compressed(
            path,
            signals=np.array([c["signal"] for c in chunks], dtype=object),
            sequences=np.array([c["sequence"] for c in chunks], dtype=str),
            dwells=np.array([c["dwell"] for c in chunks], dtype=object),
            features=np.array([c["features"] for c in chunks], dtype=object),
            labels=np.array([c["label"] for c in chunks], dtype=str),
            labels_int=np.array([c["label_int"] for c in chunks], dtype=np.int64),
            read_ids=np.array([c["read_id"] for c in chunks], dtype=str),
            base_indices=np.array([c["base_idx"] for c in chunks], dtype=np.int64),
            feature_starts=np.array([c["feature_start"] for c in chunks], dtype=np.int64),
            feature_ends=np.array([c["feature_end"] for c in chunks], dtype=np.int64),
            source_groups=np.array([c["source_group"] for c in chunks], dtype=str),
            reference_names=np.array([c["reference_name"] for c in chunks], dtype=str),
            seq_to_sig_maps=np.array([c["seq_to_sig_map"] for c in chunks], dtype=object),
            sequences_with_kmer_context=np.array(
                [c["sequence_with_kmer_context"] for c in chunks], dtype=str
            ),
            cl_values=np.array([c["cl_value"] for c in chunks], dtype=np.int16),
            focus_signal_pos=np.array([c["focus_signal_pos"] for c in chunks], dtype=np.int64),
        )

    def test_object_array_corpus_takes_eager_path(self, tmp_path):
        path = tmp_path / "legacy.npz"
        self._write_object_format(path, make_chunks(8))

        streamed, eager = build(
            path,
            signal_len=STORED_SIGNAL_LEN,
            kmer_len=KMER_LEN,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )
        assert streamed._array_stream is None  # not streamable, still correct
        assert_datasets_equal(streamed, eager)

    def test_legacy_maps_match_csr_maps(self, tmp_path):
        """A pickled seq_to_sig_maps member yields the same tensor as CSR."""
        chunks = make_chunks(8)
        legacy, modern = tmp_path / "legacy.npz", tmp_path / "modern.npz"
        self._write_object_format(legacy, chunks)
        save_chunks(chunks, modern)

        kwargs = {
            "signal_len": 44,
            "kmer_len": KMER_LEN,
            "model_type": "ConvLSTMDwell",
            "seq_encoding": "signal_kmer",
            "signal_kmer_context": (2, 2),
            "left_context": 20,
            "right_context": 24,
        }
        from_legacy = LeechDataset(chunk_path=legacy, **kwargs)
        from_csr = LeechDataset(chunk_path=modern, **kwargs)
        assert from_csr._effective_seq_encoding == "signal_kmer"
        assert torch.equal(from_legacy._seq_to_sig_tensor, from_csr._seq_to_sig_tensor)
        assert torch.equal(from_legacy._seq_ints_tensor, from_csr._seq_ints_tensor)


class TestPeakMemory:
    """Guards on the two copies #211 removed."""

    @pytest.mark.parametrize("filler", ["block", "row"])
    def test_init_never_allocates_a_second_copy(self, tmp_path, monkeypatch, filler):
        """Nothing during init allocates a copy of the output.

        ``torch.stack(items)`` allocates the whole result while ``items`` is
        still alive — that was the third copy in #211. The per-chunk filler
        replaced it with ``out=`` writes in bounded batches; the block-wise one
        copies a prepared block straight into the buffer and does not stack at
        all. Either way, no stack may allocate.
        """
        from leech.dataset import _TensorFill

        path = tmp_path / "chunks.npz"
        save_chunks(make_chunks(600), path)

        calls = []
        real_stack = torch.stack

        def record(tensors, *args, **kwargs):
            calls.append((len(tensors), kwargs.get("out") is not None))
            return real_stack(tensors, *args, **kwargs)

        monkeypatch.setattr(torch, "stack", record)
        if filler == "row":
            monkeypatch.setattr(LeechDataset, "_block_fill_supported", lambda self: False)
        dataset = LeechDataset(
            chunk_path=path,
            signal_len=STORED_SIGNAL_LEN,
            kmer_len=KMER_LEN,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )
        assert dataset._signals_tensor.shape[0] == 600
        allocating = [n for n, has_out in calls if not has_out]
        assert not allocating, f"{len(allocating)} stacks allocated instead of writing to out="
        if filler == "row":
            assert calls, "expected the per-chunk fill to stack in batches"
            assert max(n for n, _ in calls) <= _TensorFill._BATCH_ROWS
        else:
            assert not calls, "the block-wise fill has nothing to stack"

    def test_block_fill_holds_one_block_of_transient(self, tmp_path):
        """Preparing a block must not scale the transient with the corpus.

        The block-wise filler vectorizes, and vectorizing is exactly how a
        transient the size of the output gets allocated by accident. tracemalloc
        sees numpy but not torch, so this measures the prepared blocks.
        """
        import tracemalloc

        options = {**BASE_OPTIONS, "signal_len": 2048}

        def peak(n_chunks):
            path = tmp_path / f"chunks_{n_chunks}.npz"
            save_chunks(make_chunks(n_chunks, signal_len=2048), path, compressed=False)
            tracemalloc.start()
            dataset = LeechDataset(chunk_path=path, **options)
            assert dataset._block_fill_supported()
            _, measured = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            del dataset
            return measured

        small, large = peak(750), peak(1500)
        assert large < small * 1.5, (
            f"block-wise peak grew from {small / 1e6:.1f} MB to {large / 1e6:.1f} MB "
            f"when the corpus doubled — a block is being sized by the corpus"
        )

    def test_streaming_peak_does_not_scale_with_corpus(self, tmp_path):
        """The point of streaming: the array copy is bounded by the block size.

        tracemalloc sees numpy and Python allocations but not torch, so this
        measures exactly the thing #211 was about — whether the npz members
        get materialised alongside the tensors built from them. Doubling the
        corpus must not double what the load holds.
        """
        import tracemalloc

        kwargs = {
            "signal_len": 2048,
            "kmer_len": KMER_LEN,
            "model_type": "ConvLSTMDwell",
            "seq_encoding": "base_onehot",
        }

        def peaks(n_chunks):
            path = tmp_path / f"chunks_{n_chunks}.npz"
            # Wide signals so the members, not the chunk dicts, dominate.
            save_chunks(make_chunks(n_chunks, signal_len=2048), path, compressed=False)
            measured = {}
            for name, build_one in (
                ("eager", lambda: LeechDataset(chunks=load_chunks(path), **kwargs)),
                ("streamed", lambda: LeechDataset(chunk_path=path, **kwargs)),
            ):
                tracemalloc.start()
                dataset = build_one()
                _, measured[name] = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                del dataset
            return measured

        small, large = peaks(750), peaks(1500)

        eager_growth = large["eager"] - small["eager"]
        streamed_growth = large["streamed"] - small["streamed"]
        assert eager_growth > 0, "test corpus too small to measure growth"
        assert streamed_growth < eager_growth * 0.25, (
            f"streaming peak grew {streamed_growth / 1e6:.1f} MB when the corpus "
            f"doubled, against {eager_growth / 1e6:.1f} MB eager — the arrays are "
            f"still being materialised"
        )
        assert large["streamed"] < large["eager"]
