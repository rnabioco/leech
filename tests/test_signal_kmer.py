"""
Tests for signal-level kmer encoding and seq_encoding support.

Covers:
- sequence_to_int() base mapping
- encode_signal_kmer() shape, one-hot properties, boundary handling
- All model architectures with seq_encoding="signal_kmer"
- LeechDataset with signal_kmer encoding + fallback to base_onehot
"""

import numpy as np
import pytest
import torch

from leech.features import encode_signal_kmer, sequence_to_int
from leech.models import MODEL_REGISTRY, get_model

# ---------------------------------------------------------------------------
# sequence_to_int
# ---------------------------------------------------------------------------


class TestSequenceToInt:
    def test_basic_mapping(self):
        result = sequence_to_int("ACGT")
        np.testing.assert_array_equal(result, [0, 1, 2, 3])

    def test_lowercase(self):
        result = sequence_to_int("acgt")
        np.testing.assert_array_equal(result, [0, 1, 2, 3])

    def test_uracil(self):
        result = sequence_to_int("U")
        np.testing.assert_array_equal(result, [3])

    def test_unknown_bases(self):
        result = sequence_to_int("ANX")
        assert result[0] == 0
        assert result[1] == -1
        assert result[2] == -1

    def test_empty(self):
        result = sequence_to_int("")
        assert len(result) == 0

    def test_dtype(self):
        result = sequence_to_int("ACGT")
        assert result.dtype == np.int8


# ---------------------------------------------------------------------------
# encode_signal_kmer
# ---------------------------------------------------------------------------


class TestEncodeSignalKmer:
    """Tests for the signal-level kmer encoding function."""

    @pytest.fixture
    def simple_encoding_inputs(self):
        """3 bases (ACG), each occupying 10 signal samples, kmer_context=(1,1)."""
        seq_ints = sequence_to_int("AACGG")  # 1 before + 3 core + 1 after
        seq_to_sig_map = np.array([0, 10, 20, 30])  # 3 core bases
        signal_len = 30
        kmer_context = (1, 1)  # kmer_len = 3
        return seq_ints, seq_to_sig_map, signal_len, kmer_context

    def test_output_shape(self, simple_encoding_inputs):
        seq_ints, seq_to_sig_map, signal_len, kmer_context = simple_encoding_inputs
        enc = encode_signal_kmer(seq_ints, seq_to_sig_map, signal_len, kmer_context)
        kmer_len = sum(kmer_context) + 1  # 3
        assert enc.shape == (4 * kmer_len, signal_len)

    def test_output_dtype(self, simple_encoding_inputs):
        enc = encode_signal_kmer(*simple_encoding_inputs)
        assert enc.dtype == np.float32

    def test_default_context_shape(self):
        """Default kmer_context=(4,4) gives 36 channels."""
        seq_len = 5
        seq_ints = sequence_to_int("A" * (seq_len + 4 + 4))
        seq_to_sig_map = np.arange(seq_len + 1) * 10
        signal_len = seq_len * 10
        enc = encode_signal_kmer(seq_ints, seq_to_sig_map, signal_len)
        assert enc.shape == (36, signal_len)

    def test_values_are_zero_or_one(self, simple_encoding_inputs):
        enc = encode_signal_kmer(*simple_encoding_inputs)
        assert np.all((enc == 0.0) | (enc == 1.0))

    def test_one_hot_per_kmer_position(self, simple_encoding_inputs):
        """For each kmer position, at most one base channel is hot per signal sample."""
        enc = encode_signal_kmer(*simple_encoding_inputs)
        kmer_len = 3
        for kp in range(kmer_len):
            block = enc[4 * kp : 4 * (kp + 1), :]  # (4, signal_len)
            # Each column should have at most one 1
            assert np.all(block.sum(axis=0) <= 1.0)

    def test_coverage(self, simple_encoding_inputs):
        """Every signal sample within bounds should be encoded for the center kmer position."""
        seq_ints, seq_to_sig_map, signal_len, kmer_context = simple_encoding_inputs
        enc = encode_signal_kmer(seq_ints, seq_to_sig_map, signal_len, kmer_context)
        # Center kmer position (index 1 for kmer_context=(1,1))
        center_block = enc[4:8, :]  # 4 channels for center position
        # Every signal sample 0..29 is covered by one of the 3 core bases
        assert np.all(center_block.sum(axis=0) == 1.0)

    def test_specific_encoding(self):
        """Verify specific base→channel mapping for a known input."""
        # 1 core base 'C' (int=1), kmer_context=(0,0) → kmer_len=1
        seq_ints = np.array([1], dtype=np.int8)
        seq_to_sig_map = np.array([0, 5])
        enc = encode_signal_kmer(seq_ints, seq_to_sig_map, signal_len=5, kmer_context=(0, 0))
        assert enc.shape == (4, 5)
        # Channel 1 (C) should be all 1s, others all 0s
        expected = np.zeros((4, 5), dtype=np.float32)
        expected[1, :] = 1.0
        np.testing.assert_array_equal(enc, expected)

    def test_unknown_bases_skipped(self):
        """Bases with int < 0 should be skipped (no encoding)."""
        seq_ints = np.array([-1], dtype=np.int8)
        seq_to_sig_map = np.array([0, 10])
        enc = encode_signal_kmer(seq_ints, seq_to_sig_map, signal_len=10, kmer_context=(0, 0))
        assert np.all(enc == 0.0)

    def test_signal_len_clipping(self):
        """Signal regions beyond signal_len should be clipped."""
        seq_ints = np.array([0], dtype=np.int8)  # A
        seq_to_sig_map = np.array([0, 20])  # base spans 0..20
        enc = encode_signal_kmer(seq_ints, seq_to_sig_map, signal_len=10, kmer_context=(0, 0))
        # Only first 10 samples should be filled
        assert enc.shape == (4, 10)
        assert np.all(enc[0, :] == 1.0)  # A channel filled for 0..9

    def test_empty_seq(self):
        """Zero-length sequence should produce all-zeros encoding."""
        seq_ints = np.array([], dtype=np.int8)
        seq_to_sig_map = np.array([0])
        enc = encode_signal_kmer(seq_ints, seq_to_sig_map, signal_len=10, kmer_context=(0, 0))
        assert np.all(enc == 0.0)


# ---------------------------------------------------------------------------
# Model forward pass with seq_encoding="signal_kmer"
# ---------------------------------------------------------------------------


class TestModelsSignalKmer:
    """Test all model architectures accept signal_kmer encoding."""

    SIGNAL_LEN = 100
    KMER_LEN = 11
    KMER_CONTEXT = (4, 4)
    SEQ_IN_CHANNELS = 4 * (4 + 1 + 4)  # 36
    BATCH_SIZE = 2

    def _model_config(self, model_name: str) -> dict:
        """Return minimal config for a model with signal_kmer."""
        base = {
            "signal_len": self.SIGNAL_LEN,
            "kmer_len": self.KMER_LEN,
            "seq_encoding": "signal_kmer",
            "signal_kmer_context": self.KMER_CONTEXT,
        }
        if model_name == "ConvLSTMBase":
            base.update({"conv_channels": [4, 16, 32], "lstm_hidden": 16})
        elif model_name == "ConvLSTMDwell":
            base.update({"num_features": 5, "conv_channels": [4, 16, 32], "lstm_hidden": 16})
        elif model_name == "TransformerDwell":
            base.update({"num_features": 5, "d_model": 32, "nhead": 4, "num_layers": 1})
        elif model_name == "ConvOnly":
            base.update({"num_features": 5, "base_channels": 4})
        elif model_name == "ResNetDwell":
            base.update({"num_features": 5, "base_channels": 4})
        elif model_name == "TCNDwell":
            base.update(
                {"num_features": 5, "hidden_channels": 16, "num_layers": 2, "kernel_size": 3}
            )
        return base

    def _make_inputs(self, requires_features: bool):
        """Create signal_kmer-shaped input tensors."""
        signal = torch.randn(self.BATCH_SIZE, self.SIGNAL_LEN)
        # signal_kmer sequence: (B, 36, signal_len) instead of (B, 4, kmer_len)
        sequence = torch.randn(self.BATCH_SIZE, self.SEQ_IN_CHANNELS, self.SIGNAL_LEN)
        features = torch.randn(self.BATCH_SIZE, 5, self.KMER_LEN) if requires_features else None
        return signal, sequence, features

    @pytest.mark.parametrize(
        "model_name,requires_features",
        [
            ("ConvLSTMBase", False),
            ("ConvLSTMDwell", True),
            ("TransformerDwell", True),
            ("ConvOnly", True),
            ("TCNDwell", True),
            ("ResNetDwell", True),
        ],
    )
    def test_forward_pass(self, model_name, requires_features):
        config = self._model_config(model_name)
        model = get_model(model_name, **config)
        model.eval()

        signal, sequence, features = self._make_inputs(requires_features)

        with torch.no_grad():
            if requires_features:
                output = model(signal, sequence, features)
            else:
                output = model(signal, sequence)

        assert output.shape == (self.BATCH_SIZE, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_stores_seq_encoding(self, model_name):
        """signal_kmer models should store their seq_encoding attribute."""
        config = self._model_config(model_name)
        model = get_model(model_name, **config)
        assert getattr(model, "seq_encoding", None) == "signal_kmer"

    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_base_onehot_still_works(self, model_name):
        """Ensure base_onehot (default) still works as before."""
        config = self._model_config(model_name)
        config["seq_encoding"] = "base_onehot"
        config.pop("signal_kmer_context", None)
        model = get_model(model_name, **config)
        assert getattr(model, "seq_encoding", "base_onehot") == "base_onehot"


# ---------------------------------------------------------------------------
# LeechDataset with signal_kmer encoding
# ---------------------------------------------------------------------------


class TestDatasetSignalKmer:
    """Test LeechDataset with seq_encoding='signal_kmer'."""

    @pytest.fixture
    def chunks_with_signal_kmer(self, sample_leech_read):
        """Create chunks that include seq_to_sig_map and sequence_with_kmer_context."""
        chunks = []
        kmer_context = 5
        for base_idx in range(kmer_context, min(15, sample_leech_read.num_bases - kmer_context)):
            chunk = sample_leech_read.get_chunk(
                base_idx,
                signal_context=(200, 200),
                kmer_context=kmer_context,
            )
            if chunk is None:
                continue
            # get_chunk now includes seq_to_sig_map and sequence_with_kmer_context
            if chunk.get("sequence_with_kmer_context") is None:
                continue
            chunk["read_id"] = sample_leech_read.read_id
            chunk["label_int"] = base_idx % 2
            chunk["label"] = "charged" if (base_idx % 2) == 1 else "uncharged"
            chunks.append(chunk)
        return chunks

    def test_signal_kmer_encoding_path(self, chunks_with_signal_kmer):
        """Dataset with signal_kmer should produce (36, signal_len) sequences."""
        from leech.dataset import LeechDataset

        if not chunks_with_signal_kmer:
            pytest.skip("No chunks with signal_kmer context available")

        dataset = LeechDataset(
            chunks=chunks_with_signal_kmer,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="signal_kmer",
            signal_kmer_context=(4, 4),
        )

        item = dataset[0]
        # signal_kmer: (36, signal_len)
        assert item["sequence"].shape == (36, 400)

    def test_signal_kmer_values(self, chunks_with_signal_kmer):
        """signal_kmer encoded sequences should only contain 0s and 1s."""
        from leech.dataset import LeechDataset

        if not chunks_with_signal_kmer:
            pytest.skip("No chunks with signal_kmer context available")

        dataset = LeechDataset(
            chunks=chunks_with_signal_kmer,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="signal_kmer",
            signal_kmer_context=(4, 4),
        )

        item = dataset[0]
        seq = item["sequence"]
        assert torch.all((seq == 0.0) | (seq == 1.0))

    def test_fallback_to_base_onehot(self, sample_chunks):
        """If chunks lack signal_kmer fields, dataset should fall back to base_onehot."""
        from leech.dataset import LeechDataset

        # sample_chunks from conftest may not have seq_to_sig_map (old format)
        # Remove signal_kmer fields to force fallback
        for c in sample_chunks:
            c.pop("seq_to_sig_map", None)
            c.pop("sequence_with_kmer_context", None)

        dataset = LeechDataset(
            chunks=sample_chunks,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="signal_kmer",  # request signal_kmer
            signal_kmer_context=(4, 4),
        )

        # Should fall back to base_onehot: shape (4, kmer_len)
        assert dataset._effective_seq_encoding == "base_onehot"
        item = dataset[0]
        assert item["sequence"].shape == (4, 11)

    def test_base_onehot_unchanged(self, sample_chunks):
        """Explicit base_onehot should produce (4, kmer_len) as before."""
        from leech.dataset import LeechDataset

        dataset = LeechDataset(
            chunks=sample_chunks,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        item = dataset[0]
        assert item["sequence"].shape == (4, 11)

    def test_collate_signal_kmer(self, chunks_with_signal_kmer):
        """Collating signal_kmer batches should stack correctly."""
        from leech.dataset import LeechDataset, collate_fn

        if not chunks_with_signal_kmer:
            pytest.skip("No chunks with signal_kmer context available")

        dataset = LeechDataset(
            chunks=chunks_with_signal_kmer,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="signal_kmer",
            signal_kmer_context=(4, 4),
        )

        batch_size = min(2, len(dataset))
        batch = [dataset[i] for i in range(batch_size)]
        collated = collate_fn(batch)

        assert collated["sequence"].shape == (batch_size, 36, 400)


# ---------------------------------------------------------------------------
# The signal_kmer -> base_onehot fallback: whole corpus, and refusable (#230)
# ---------------------------------------------------------------------------


def _blank_signal_kmer(chunk: dict) -> dict:
    """Strip a chunk's signal_kmer inputs, the way a corpus that lacks them does."""
    chunk["seq_to_sig_map"] = np.zeros(0, dtype=np.int64)
    chunk["sequence_with_kmer_context"] = ""
    return chunk


@pytest.fixture
def signal_kmer_chunks(sample_leech_read):
    """Chunks carrying both signal_kmer inputs: the map and the context string."""
    chunks = []
    kmer_context = 5
    for base_idx in range(kmer_context, min(15, sample_leech_read.num_bases - kmer_context)):
        chunk = sample_leech_read.get_chunk(
            base_idx, signal_context=(200, 200), kmer_context=kmer_context
        )
        if chunk is None or chunk.get("sequence_with_kmer_context") is None:
            continue
        chunk["read_id"] = sample_leech_read.read_id
        chunk["label_int"] = base_idx % 2
        chunk["label"] = "charged" if (base_idx % 2) == 1 else "uncharged"
        chunks.append(chunk)
    if len(chunks) < 3:
        pytest.skip("Need at least 3 chunks with signal_kmer context")
    return chunks


class TestEncodingFallbackPolicy:
    """The fallback is a whole-corpus decision, and it can be refused.

    Chunk 0 decided it for every chunk until #230, which is wrong in both
    directions: one empty first row flipped a whole corpus to ``base_onehot``,
    and a corpus whose first row was fine encoded every later row that had no
    map from data that isn't there — silently, as all-zero sequence channels.
    """

    RAGGED = "carries base-to-signal maps for every chunk or for none"

    KWARGS = {
        "signal_len": 400,
        "kmer_len": 11,
        "model_type": "ConvLSTMDwell",
        "seq_encoding": "signal_kmer",
        "signal_kmer_context": (4, 4),
    }

    @staticmethod
    def _corpus(chunks, tmp_path, name="chunks.npz"):
        from leech.chunking.serialization import save_chunks

        path = tmp_path / name
        save_chunks(chunks, path)
        return path

    @staticmethod
    def _all_but_first(chunks):
        """Chunk 0 keeps its signal_kmer inputs; every other chunk loses them."""
        copies = [dict(c) for c in chunks]
        for chunk in copies[1:]:
            _blank_signal_kmer(chunk)
        return copies

    def test_a_ragged_corpus_stops_the_run(self, signal_kmer_chunks, tmp_path):
        """Chunk 0 keeps its map and the rest do not — neither answer is right.

        Encoding the uncovered rows anyway gives them all-zero sequence
        channels; switching the whole corpus to ``base_onehot`` on the strength
        of a few bad rows throws the encoding away for every good one. Both are
        the silent representation change this guard exists to prevent, so a
        ragged corpus is reported rather than resolved.
        """
        from leech.dataset import LeechDataset

        chunks = self._all_but_first(signal_kmer_chunks)
        path = self._corpus(chunks, tmp_path)
        with pytest.raises(ValueError, match=self.RAGGED):
            LeechDataset(chunk_path=path, **self.KWARGS)

    def test_a_ragged_corpus_stops_the_run_even_where_the_fallback_is_allowed(
        self, signal_kmer_chunks, tmp_path
    ):
        """Permission to fall back is not permission to reinterpret a damaged corpus."""
        from leech.dataset import LeechDataset

        chunks = self._all_but_first(signal_kmer_chunks)
        path = self._corpus(chunks, tmp_path)
        with pytest.raises(ValueError, match=self.RAGGED):
            LeechDataset(chunk_path=path, allow_encoding_fallback=True, **self.KWARGS)

    def test_first_chunk_missing_is_reported_as_one_chunk(self, signal_kmer_chunks, tmp_path):
        """One empty row is one row, not the whole corpus — the other direction."""
        from leech.dataset import LeechDataset

        chunks = [dict(c) for c in signal_kmer_chunks]
        _blank_signal_kmer(chunks[0])
        path = self._corpus(chunks, tmp_path)

        with pytest.raises(ValueError, match=rf"\b1 of {len(chunks)} chunks"):
            LeechDataset(chunk_path=path, **self.KWARGS)

    def test_a_ragged_corpus_can_still_be_read_as_base_onehot(self, signal_kmer_chunks, tmp_path):
        """The escape hatch is naming the encoding, not letting one be chosen."""
        from leech.dataset import LeechDataset

        chunks = self._all_but_first(signal_kmer_chunks)
        path = self._corpus(chunks, tmp_path)
        kwargs = {**self.KWARGS, "seq_encoding": "base_onehot"}
        dataset = LeechDataset(chunk_path=path, **kwargs)
        assert dataset.effective_seq_encoding == "base_onehot"
        for i in range(len(dataset)):
            assert torch.count_nonzero(dataset[i]["sequence"]) > 0

    def test_a_corpus_with_no_maps_at_all_falls_back(self, signal_kmer_chunks, tmp_path, caplog):
        """The version-skew case, and the one the fallback exists for."""
        from leech.dataset import LeechDataset

        chunks = [_blank_signal_kmer(dict(c)) for c in signal_kmer_chunks]
        path = self._corpus(chunks, tmp_path)
        with caplog.at_level("WARNING", logger="leech.dataset"):
            dataset = LeechDataset(chunk_path=path, **self.KWARGS)

        assert dataset.effective_seq_encoding == "base_onehot"
        # The fraction is the point: a warning that does not say how much of the
        # corpus is affected is the one that got read past for four releases.
        assert f"{len(chunks)} of {len(chunks)}" in caplog.text

    def test_fallback_can_be_refused(self, signal_kmer_chunks, tmp_path):
        """An encoding asked for by name is not silently substituted."""
        from leech.dataset import LeechDataset

        chunks = [_blank_signal_kmer(dict(c)) for c in signal_kmer_chunks]
        path = self._corpus(chunks, tmp_path)
        with pytest.raises(ValueError, match="signal_kmer"):
            LeechDataset(chunk_path=path, allow_encoding_fallback=False, **self.KWARGS)

    def test_refusing_the_fallback_leaves_a_healthy_corpus_alone(
        self, signal_kmer_chunks, tmp_path
    ):
        from leech.dataset import LeechDataset

        path = self._corpus([dict(c) for c in signal_kmer_chunks], tmp_path)
        dataset = LeechDataset(chunk_path=path, allow_encoding_fallback=False, **self.KWARGS)
        assert dataset.effective_seq_encoding == "signal_kmer"
        assert dataset[0]["sequence"].shape == (36, 400)

    def test_streamed_and_preloaded_paths_agree(self, signal_kmer_chunks, tmp_path):
        """The CSR coverage count and the per-dict one answer the same question."""
        from leech.dataset import LeechDataset

        chunks = self._all_but_first(signal_kmer_chunks)
        with pytest.raises(ValueError, match=self.RAGGED):
            LeechDataset(chunk_path=self._corpus(chunks, tmp_path), **self.KWARGS)
        with pytest.raises(ValueError, match=self.RAGGED):
            LeechDataset(chunks=[dict(c) for c in chunks], **self.KWARGS)
