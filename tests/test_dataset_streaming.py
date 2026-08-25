"""Tests for streaming chunk arrays out of an npz instead of loading them whole.

``LeechDataset`` used to hold three copies of the corpus at once: the numpy
members from ``load_chunks``, one tensor per chunk, and the contiguous output
``torch.stack`` built while that list was still alive (#211). It now fills a
preallocated tensor from row blocks read straight off the npz.

The hard gate here is parity: a dataset built from a path (streaming) must be
bit-identical to one built from pre-loaded chunks (eager), field by field, over
the option matrix that changes how chunks are prepared.
"""

import numpy as np
import pytest
import torch

from leech.chunking import (
    csr_gather_index,
    iter_npz_row_blocks,
    load_chunks,
    npz_array_members,
    npz_member_names,
    save_chunks,
)
from leech.dataset import LeechDataset

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
) -> list[dict]:
    """Build a small synthetic corpus.

    Focus positions vary per chunk so an asymmetric crop lands inside the
    stored signal for some rows and overhangs it (the zero-pad branch) for
    others — the per-row gather the streaming path has to reproduce.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for i in range(n):
        chunk = {
            "signal": rng.standard_normal(signal_len).astype(np.float32),
            "sequence": "".join(rng.choice(list("ACGT"), KMER_LEN)),
            "dwell": rng.integers(1, 9, FEAT_WIDTH).astype(np.float32),
            "features": rng.standard_normal((NUM_FEATURES, FEAT_WIDTH)).astype(np.float32),
            "label": "charged" if i % 2 else "uncharged",
            "label_int": None if i in unlabeled else i % 2,
            "read_id": f"read_{i:03d}",
            "base_idx": 100 + i,
            "source_group": "Ala" if i % 3 else "Gly",
            "reference_name": "tRNA-Ala-AGC",
            "feature_start": -(FEAT_WIDTH // 2),
            "feature_end": FEAT_WIDTH // 2,
            "cl_value": i % 5,
            # 8, 22, 36, 50, 8, ... : the first and last overhang a 20/24 crop.
            "focus_signal_pos": 8 + 14 * (i % 4),
        }
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

    # Metadata the samplers and training config read off the chunk dicts.
    for a, b in zip(streamed.chunks, eager.chunks, strict=True):
        for key in ("read_id", "label_int", "source_group", "base_idx", "feature_start"):
            assert a.get(key) == b.get(key), f"chunk metadata {key}"


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

    def test_init_never_allocates_a_second_copy(self, tmp_path, monkeypatch):
        """Every stack during init writes into the preallocated output.

        ``torch.stack(items)`` allocates the whole result while ``items`` is
        still alive — that was the third copy in #211. Writing through ``out=``
        in bounded batches is what replaced it.
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
        dataset = LeechDataset(
            chunk_path=path,
            signal_len=STORED_SIGNAL_LEN,
            kmer_len=KMER_LEN,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )
        assert dataset._signals_tensor.shape[0] == 600
        assert calls, "expected the fill to stack in batches"
        allocating = [n for n, has_out in calls if not has_out]
        assert not allocating, f"{len(allocating)} stacks allocated instead of writing to out="
        assert max(n for n, _ in calls) <= _TensorFill._BATCH_ROWS

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
