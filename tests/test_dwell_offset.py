"""
Tests for feature_start/feature_end and dwell_offset functionality.

Tests the feature window and motor-sensor offset compensation:
- Feature window sizing with feature_start/feature_end in get_chunk()
- Runtime offset slicing in LeechDataset.__getitem__()
- Backward compatibility when feature_start/end are None and offset=0
- Boundary error when offset exceeds feature window
"""

import numpy as np
import pytest

from leech.chunking import LeechRead, resolve_feature_window, save_chunks
from leech.dataset import LeechDataset


class TestResolveFeatureWindow:
    """The one place the feature-window default lives.

    A truthiness test instead of `is None` here turns `feature_start=0` --
    features starting AT the focus base -- back into the default and widens
    every chunk's window by `kmer_context` bases (issue #189).
    """

    def test_zero_start_is_kept(self):
        assert resolve_feature_window(0, 20, kmer_context=5) == (0, 20, 21)

    def test_zero_end_is_kept(self):
        assert resolve_feature_window(-20, 0, kmer_context=5) == (-20, 0, 21)

    def test_none_falls_back_to_kmer_window(self):
        assert resolve_feature_window(None, None, kmer_context=5) == (-5, 5, 11)

    def test_one_sided_defaults(self):
        assert resolve_feature_window(0, None, kmer_context=5) == (0, 5, 6)
        assert resolve_feature_window(None, 0, kmer_context=5) == (-5, 0, 6)

    def test_width_follows_kmer_context(self):
        assert resolve_feature_window(None, None, kmer_context=9) == (-9, 9, 19)


@pytest.fixture
def read_with_known_dwells():
    """Create a LeechRead with predictable dwell values for offset testing.

    50 bases, each base's dwell value equals its index (0-49).
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


class TestGetChunkFeatureWindow:
    """Test get_chunk() with feature_start/feature_end parameters."""

    def test_default_no_extra(self, read_with_known_dwells):
        """Default (None) gives kmer_len-sized dwell array."""
        read = read_with_known_dwells
        kmer_context = 5
        kmer_len = 2 * kmer_context + 1  # 11

        chunk = read.get_chunk(base_idx=15, kmer_context=kmer_context)
        assert chunk is not None
        assert len(chunk["dwell"]) == kmer_len
        assert chunk["features"].shape[1] == kmer_len
        assert chunk["feature_start"] == -kmer_context
        assert chunk["feature_end"] == kmer_context

    def test_wider_feature_window(self, read_with_known_dwells):
        """feature_start=-10, feature_end=10 gives width=21 dwell array."""
        read = read_with_known_dwells
        kmer_context = 5
        fs, fe = -10, 10
        expected_len = fe - fs + 1  # 21

        chunk = read.get_chunk(
            base_idx=15, kmer_context=kmer_context, feature_start=fs, feature_end=fe
        )
        assert chunk is not None
        assert len(chunk["dwell"]) == expected_len
        assert chunk["features"].shape[1] == expected_len
        assert chunk["feature_start"] == fs
        assert chunk["feature_end"] == fe

    def test_right_only_features(self, read_with_known_dwells):
        """feature_start=0, feature_end=20 gives right-only features."""
        read = read_with_known_dwells
        kmer_context = 5
        fs, fe = 0, 20
        expected_len = fe - fs + 1  # 21

        chunk = read.get_chunk(
            base_idx=15, kmer_context=kmer_context, feature_start=fs, feature_end=fe
        )
        assert chunk is not None
        assert len(chunk["dwell"]) == expected_len
        assert chunk["features"].shape[1] == expected_len
        assert chunk["feature_start"] == 0
        assert chunk["feature_end"] == 20

    def test_shifted_right_window(self, read_with_known_dwells):
        """feature_start=5, feature_end=25 gives window entirely past focus."""
        read = read_with_known_dwells
        kmer_context = 5
        fs, fe = 5, 25
        expected_len = fe - fs + 1  # 21

        chunk = read.get_chunk(
            base_idx=15, kmer_context=kmer_context, feature_start=fs, feature_end=fe
        )
        assert chunk is not None
        assert len(chunk["dwell"]) == expected_len
        # Dwells should start at base_idx + 5 = 20
        assert chunk["dwell"][0] == 20.0
        assert chunk["dwell"][-1] == 40.0

    def test_sequence_unchanged_by_feature_window(self, read_with_known_dwells):
        """Sequence length is always kmer_len regardless of feature window."""
        read = read_with_known_dwells
        kmer_context = 5
        kmer_len = 2 * kmer_context + 1

        chunk_default = read.get_chunk(base_idx=15, kmer_context=kmer_context)
        chunk_wide = read.get_chunk(
            base_idx=15, kmer_context=kmer_context, feature_start=-10, feature_end=10
        )

        assert chunk_default is not None
        assert chunk_wide is not None
        assert len(chunk_default["sequence"]) == kmer_len
        assert len(chunk_wide["sequence"]) == kmer_len
        assert chunk_default["sequence"] == chunk_wide["sequence"]

    def test_dwell_values_correct(self, read_with_known_dwells):
        """Verify dwell values match expected base indices."""
        read = read_with_known_dwells
        kmer_context = 5
        fs, fe = -10, 10
        base_idx = 15

        chunk = read.get_chunk(
            base_idx=base_idx, kmer_context=kmer_context, feature_start=fs, feature_end=fe
        )
        assert chunk is not None

        # Expected dwell range: [base_idx + fs, base_idx + fe] = [5, 25]
        expected_start = base_idx + fs  # 5
        expected_end = base_idx + fe + 1  # 26
        expected_dwells = np.arange(expected_start, expected_end, dtype=np.float32)
        np.testing.assert_array_equal(chunk["dwell"], expected_dwells)

    def test_boundary_with_wide_window_zero_pads(self, read_with_known_dwells):
        """Bases near edges with wide feature window get zero-padded dwells."""
        read = read_with_known_dwells
        kmer_context = 5

        # base_idx=5 with feature_start=-6: needs dwell at index -1, gets zero-padded
        chunk = read.get_chunk(
            base_idx=5, kmer_context=kmer_context, feature_start=-6, feature_end=6
        )
        assert chunk is not None
        # First element should be zero (padded), rest are real values
        assert chunk["dwell"][0] == 0.0


class TestDatasetDwellOffset:
    """Test LeechDataset dwell_offset slicing."""

    def _make_chunks_file(self, tmp_path, read, kmer_context=5, feature_start=-15, feature_end=15):
        """Helper to create chunks file with known feature window."""
        # Set labels
        read.labels = np.zeros(read.num_bases, dtype=np.int64)
        read.labels[25] = 1

        chunks = []
        for base_idx in [25]:
            chunk = read.get_chunk(
                base_idx,
                kmer_context=kmer_context,
                feature_start=feature_start,
                feature_end=feature_end,
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

        chunks_file = self._make_chunks_file(
            tmp_path,
            read_with_known_dwells,
            kmer_context=kmer_context,
            feature_start=-15,
            feature_end=15,
        )

        dataset = LeechDataset(
            chunks_file,
            signal_len=400,
            kmer_len=kmer_len,
            model_type="ConvLSTMDwell",
            dwell_offset=0,
        )
        item = dataset[0]

        # Features should be kmer_len wide
        assert item["features"].shape[1] == kmer_len

    def test_offset_shifts_features(self, read_with_known_dwells, tmp_path):
        """dwell_offset=5 shifts features 5 bases toward 3' end."""
        kmer_context = 5
        kmer_len = 2 * kmer_context + 1

        chunks_file = self._make_chunks_file(
            tmp_path,
            read_with_known_dwells,
            kmer_context=kmer_context,
            feature_start=-15,
            feature_end=15,
        )

        # Offset=0
        ds0 = LeechDataset(
            chunks_file,
            signal_len=400,
            kmer_len=kmer_len,
            model_type="ConvLSTMDwell",
            dwell_offset=0,
        )
        item0 = ds0[0]

        # Offset=5
        ds5 = LeechDataset(
            chunks_file,
            signal_len=400,
            kmer_len=kmer_len,
            model_type="ConvLSTMDwell",
            dwell_offset=5,
        )
        item5 = ds5[0]

        # Features should differ (shifted by 5 bases)
        assert not np.allclose(item0["features"].numpy(), item5["features"].numpy())

        # The shapes should be the same
        assert item0["features"].shape == item5["features"].shape

        # Sequence should be identical (not affected by dwell_offset)
        assert item0["sequence"].equal(item5["sequence"])

    def test_offset_exceeds_window_raises(self, read_with_known_dwells, tmp_path):
        """dwell_offset exceeding feature window raises ValueError."""
        kmer_context = 5
        kmer_len = 2 * kmer_context + 1
        # feature_start=-10: kmer_start_in_features = -5 - (-10) = 5
        # width = 10+1+10 = 21, max offset = 21-11-5 = 5
        chunks_file = self._make_chunks_file(
            tmp_path,
            read_with_known_dwells,
            kmer_context=kmer_context,
            feature_start=-10,
            feature_end=10,
        )

        with pytest.raises(ValueError, match="dwell_offset.*exceeds.*feature width"):
            LeechDataset(
                chunks_file,
                signal_len=400,
                kmer_len=kmer_len,
                model_type="ConvLSTMDwell",
                dwell_offset=6,
            )

    def test_backward_compat_no_margin(self, temp_chunks_file):
        """Existing chunks (no margin) work fine with dwell_offset=0."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            dwell_offset=0,
        )
        item = dataset[0]
        assert item["features"].shape[1] == 11

    def test_base_model_ignores_offset(self, read_with_known_dwells, tmp_path):
        """ConvLSTMBase doesn't include features, so offset is effectively ignored."""
        kmer_context = 5
        kmer_len = 2 * kmer_context + 1

        chunks_file = self._make_chunks_file(
            tmp_path,
            read_with_known_dwells,
            kmer_context=kmer_context,
            feature_start=-15,
            feature_end=15,
        )

        dataset = LeechDataset(
            chunks_file,
            signal_len=400,
            kmer_len=kmer_len,
            model_type="ConvLSTMBase",
            dwell_offset=5,
        )
        item = dataset[0]
        # ConvLSTMBase should not include features
        assert "features" not in item
