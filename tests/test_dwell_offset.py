"""
Tests for dwell_offset and dwell_margin functionality.

Tests the motor-sensor offset compensation feature:
- Wider dwell extraction with dwell_margin in get_chunk()
- Runtime offset slicing in LeechDataset.__getitem__()
- Backward compatibility when margin=0 and offset=0
- Boundary error when offset exceeds margin
"""

import numpy as np
import pytest

from leech.chunking import LeechRead, save_chunks
from leech.dataset import LeechDataset


@pytest.fixture
def read_with_known_dwells():
    """Create a LeechRead with predictable dwell values for offset testing.

    30 bases, each base's dwell value equals its index (0-29).
    This makes it easy to verify which bases are in the extracted window.
    """
    num_bases = 50
    stride = 5
    samples_per_base = 6  # 6 stride blocks = 30 signal samples per base
    trim_offset = 100

    # Build sequence
    bases = "ACGT"
    sequence = "".join(bases[i % 4] for i in range(num_bases))

    # Build signal (enough for context)
    total_signal = trim_offset + num_bases * samples_per_base * stride + 200
    np.random.seed(42)
    signal = np.random.randn(total_signal).astype(np.float32)

    # Build seq_to_sig_map: each base starts at trim_offset + i * 30
    sig_per_base = samples_per_base * stride
    seq_to_sig_map = np.array(
        [trim_offset + i * sig_per_base for i in range(num_bases)]
        + [trim_offset + num_bases * sig_per_base],
        dtype=np.int64,
    )

    # Dwells equal the base index for easy verification
    dwells = np.arange(num_bases, dtype=np.float32)

    # Feature arrays also use base index for easy verification
    dwell_features = {
        "dwell": dwells.copy(),
        "dwell_log": np.log(dwells + 1).astype(np.float32),
    }
    signal_features = {
        "level_mean": dwells.copy() * 10,
        "level_median": dwells.copy() * 100,
        "level_std": dwells.copy() * 0.1,
    }

    return LeechRead(
        read_id="test_offset_read",
        sequence=sequence,
        signal=signal,
        seq_to_sig_map=seq_to_sig_map,
        dwells=dwells,
        dwell_features=dwell_features,
        signal_features=signal_features,
        labels=None,
        metadata={"test": True},
    )


class TestGetChunkDwellMargin:
    """Test get_chunk() with dwell_margin parameter."""

    def test_default_no_margin(self, read_with_known_dwells):
        """dwell_margin=0 gives kmer_len-sized dwell array (current behavior)."""
        read = read_with_known_dwells
        kmer_context = 5
        kmer_len = 2 * kmer_context + 1  # 11

        chunk = read.get_chunk(base_idx=15, kmer_context=kmer_context, dwell_margin=0)
        assert chunk is not None
        assert len(chunk["dwell"]) == kmer_len
        assert chunk["features"].shape[1] == kmer_len

    def test_wider_dwell_with_margin(self, read_with_known_dwells):
        """dwell_margin=5 gives (kmer_len + 2*5)-sized dwell array."""
        read = read_with_known_dwells
        kmer_context = 5
        margin = 5
        expected_len = 2 * kmer_context + 1 + 2 * margin  # 21

        chunk = read.get_chunk(base_idx=15, kmer_context=kmer_context, dwell_margin=margin)
        assert chunk is not None
        assert len(chunk["dwell"]) == expected_len
        assert chunk["features"].shape[1] == expected_len

    def test_sequence_unchanged_by_margin(self, read_with_known_dwells):
        """Sequence length is always kmer_len regardless of dwell_margin."""
        read = read_with_known_dwells
        kmer_context = 5
        kmer_len = 2 * kmer_context + 1

        chunk_no_margin = read.get_chunk(base_idx=15, kmer_context=kmer_context, dwell_margin=0)
        chunk_with_margin = read.get_chunk(base_idx=15, kmer_context=kmer_context, dwell_margin=5)

        assert chunk_no_margin is not None
        assert chunk_with_margin is not None
        assert len(chunk_no_margin["sequence"]) == kmer_len
        assert len(chunk_with_margin["sequence"]) == kmer_len
        assert chunk_no_margin["sequence"] == chunk_with_margin["sequence"]

    def test_dwell_values_correct(self, read_with_known_dwells):
        """Verify dwell values match expected base indices."""
        read = read_with_known_dwells
        kmer_context = 5
        margin = 5
        base_idx = 15

        chunk = read.get_chunk(base_idx=base_idx, kmer_context=kmer_context, dwell_margin=margin)
        assert chunk is not None

        # Expected dwell range: base_idx - kmer_context - margin to base_idx + kmer_context + margin
        expected_start = base_idx - kmer_context - margin  # 5
        expected_end = base_idx + kmer_context + 1 + margin  # 26
        expected_dwells = np.arange(expected_start, expected_end, dtype=np.float32)
        np.testing.assert_array_equal(chunk["dwell"], expected_dwells)

    def test_boundary_rejection_with_margin(self, read_with_known_dwells):
        """Bases near edges are rejected when margin requires more room."""
        read = read_with_known_dwells
        kmer_context = 5

        # base_idx=5 works with margin=0 (needs 5 bases on each side)
        chunk = read.get_chunk(base_idx=5, kmer_context=kmer_context, dwell_margin=0)
        assert chunk is not None

        # base_idx=5 fails with margin=1 (needs 6 bases on left, only 5 available)
        chunk = read.get_chunk(base_idx=5, kmer_context=kmer_context, dwell_margin=1)
        assert chunk is None


class TestDatasetDwellOffset:
    """Test LeechDataset dwell_offset slicing."""

    def _make_chunks_file(self, tmp_path, read, kmer_context=5, dwell_margin=10):
        """Helper to create chunks file with known dwell_margin."""
        # Set labels
        read.labels = np.zeros(read.num_bases, dtype=np.int64)
        read.labels[25] = 1

        chunks = []
        for base_idx in [25]:
            chunk = read.get_chunk(
                base_idx, kmer_context=kmer_context, dwell_margin=dwell_margin
            )
            if chunk is not None:
                chunk["read_id"] = read.read_id
                chunk["label_int"] = 1
                chunk["label"] = "charged"
                chunks.append(chunk)

        chunks_file = tmp_path / "test_offset_chunks.npz"
        save_chunks(chunks, chunks_file)
        return chunks_file

    def test_offset_zero_matches_center(self, read_with_known_dwells, tmp_path):
        """dwell_offset=0 extracts the center of the wider dwell array."""
        kmer_context = 5
        kmer_len = 2 * kmer_context + 1
        dwell_margin = 10

        chunks_file = self._make_chunks_file(
            tmp_path, read_with_known_dwells, kmer_context=kmer_context, dwell_margin=dwell_margin
        )

        dataset = LeechDataset(
            chunks_file, signal_len=400, kmer_len=kmer_len,
            model_type="ConvLSTMDwell", dwell_offset=0,
        )
        item = dataset[0]

        # Features should be kmer_len wide
        assert item["features"].shape[1] == kmer_len

    def test_offset_shifts_features(self, read_with_known_dwells, tmp_path):
        """dwell_offset=5 shifts features 5 bases toward 3' end."""
        kmer_context = 5
        kmer_len = 2 * kmer_context + 1
        dwell_margin = 10

        chunks_file = self._make_chunks_file(
            tmp_path, read_with_known_dwells, kmer_context=kmer_context, dwell_margin=dwell_margin
        )

        # Offset=0
        ds0 = LeechDataset(
            chunks_file, signal_len=400, kmer_len=kmer_len,
            model_type="ConvLSTMDwell", dwell_offset=0,
        )
        item0 = ds0[0]

        # Offset=5
        ds5 = LeechDataset(
            chunks_file, signal_len=400, kmer_len=kmer_len,
            model_type="ConvLSTMDwell", dwell_offset=5,
        )
        item5 = ds5[0]

        # Features should differ (shifted by 5 bases)
        assert not np.allclose(item0["features"].numpy(), item5["features"].numpy())

        # The shapes should be the same
        assert item0["features"].shape == item5["features"].shape

        # Sequence should be identical (not affected by dwell_offset)
        assert item0["sequence"].equal(item5["sequence"])

    def test_offset_exceeds_margin_raises(self, read_with_known_dwells, tmp_path):
        """dwell_offset > margin raises ValueError."""
        kmer_context = 5
        kmer_len = 2 * kmer_context + 1
        dwell_margin = 5

        chunks_file = self._make_chunks_file(
            tmp_path, read_with_known_dwells, kmer_context=kmer_context, dwell_margin=dwell_margin
        )

        dataset = LeechDataset(
            chunks_file, signal_len=400, kmer_len=kmer_len,
            model_type="ConvLSTMDwell", dwell_offset=6,
        )
        with pytest.raises(ValueError, match="dwell_offset.*exceeds.*dwell_margin"):
            dataset[0]

    def test_backward_compat_no_margin(self, temp_chunks_file):
        """Existing chunks (no margin) work fine with dwell_offset=0."""
        dataset = LeechDataset(
            temp_chunks_file, signal_len=400, kmer_len=11,
            model_type="ConvLSTMDwell", dwell_offset=0,
        )
        item = dataset[0]
        assert item["features"].shape[1] == 11

    def test_base_model_ignores_offset(self, read_with_known_dwells, tmp_path):
        """ConvLSTMBase doesn't include features, so offset is effectively ignored."""
        kmer_context = 5
        kmer_len = 2 * kmer_context + 1
        dwell_margin = 10

        chunks_file = self._make_chunks_file(
            tmp_path, read_with_known_dwells, kmer_context=kmer_context, dwell_margin=dwell_margin
        )

        dataset = LeechDataset(
            chunks_file, signal_len=400, kmer_len=kmer_len,
            model_type="ConvLSTMBase", dwell_offset=5,
        )
        item = dataset[0]
        # ConvLSTMBase should not include features
        assert "features" not in item
