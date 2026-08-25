"""
Tests for feature extraction module.
"""

import numpy as np
import pytest

from leech.features import (
    MoveTable,
    compute_dwell_features,
    compute_dwell_times,
    compute_signal_levels,
    normalize_read_signal,
)


def test_move_table_basic():
    """Test basic MoveTable functionality."""
    stride = 5
    moves = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype=np.int8)

    mt = MoveTable(stride=stride, moves=moves, read_id="test_read", num_samples=1000, trim_offset=0)

    assert mt.num_bases == 4  # Four 1s in the move array
    assert mt.stride == 5


def test_seq_to_sig_map():
    """Test conversion of move table to sequence-to-signal mapping.

    Follows Remora convention: move=1 at position i means a new base
    starts at signal index i * stride. The last entry is num_samples.
    """
    stride = 5
    moves = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype=np.int8)

    mt = MoveTable(stride=stride, moves=moves, read_id="test_read", num_samples=40, trim_offset=0)

    seq_to_sig = mt.to_seq_to_sig_map()

    # move_positions = [0, 1, 3, 7]
    # Base 0 starts at 0*5=0, Base 1 at 1*5=5, Base 2 at 3*5=15, Base 3 at 7*5=35
    # Last entry = num_samples = 40
    expected = np.array([0, 5, 15, 35, 40])
    np.testing.assert_array_equal(seq_to_sig, expected)


def test_compute_dwell_times():
    """Test dwell time computation."""
    stride = 5
    moves = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype=np.int8)

    mt = MoveTable(stride=stride, moves=moves, read_id="test_read", num_samples=40, trim_offset=0)

    dwells = compute_dwell_times(mt)

    # move_positions = [0, 1, 3, 7], boundaries = [0, 5, 15, 35, 40]
    # Base 0: [0,5) → 5, Base 1: [5,15) → 10, Base 2: [15,35) → 20, Base 3: [35,40) → 5
    expected = np.array([5, 10, 20, 5])
    np.testing.assert_array_equal(dwells, expected)


def test_normalize_read_signal_median_mad():
    """Test median-MAD normalization."""
    # Create signal with known statistics
    signal = np.array([100, 102, 98, 105, 95, 101, 99, 103, 97], dtype=np.float32)

    norm_signal, params = normalize_read_signal(signal, method="median_mad")

    # Check that normalization parameters are computed
    assert "median" in params
    assert "mad" in params

    # Normalized signal should have median ~0
    assert abs(np.median(norm_signal)) < 0.1


def test_normalize_read_signal_zscore():
    """Test z-score normalization."""
    signal = np.array([100, 102, 98, 105, 95, 101, 99, 103, 97], dtype=np.float32)

    norm_signal, params = normalize_read_signal(signal, method="zscore")

    # Check parameters
    assert "mean" in params
    assert "std" in params

    # Normalized signal should have mean ~0 and std ~1
    assert abs(np.mean(norm_signal)) < 1e-6
    assert abs(np.std(norm_signal) - 1.0) < 1e-6


def test_compute_signal_levels():
    """Test per-base signal level computation."""
    # Create synthetic signal
    signal = np.array([1.0, 1.1, 0.9, 2.0, 2.1, 1.9, 3.0, 3.1, 2.9], dtype=np.float32)
    seq_to_sig = np.array([0, 3, 6, 9])  # 3 bases

    levels_mean = compute_signal_levels(signal, seq_to_sig, stat="mean")

    # Expected means: [1.0, 2.0, 3.0]
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(levels_mean, expected, decimal=1)


def test_compute_dwell_features():
    """Test dwell feature extraction."""
    dwells = np.array([5, 10, 8, 12, 6, 9], dtype=np.float32)

    features = compute_dwell_features(dwells, window=3)

    # Check that all expected features are present
    assert "dwell" in features
    assert "dwell_log" in features
    assert "dwell_mean" in features
    assert "dwell_std" in features
    assert "dwell_ratio" in features

    # Check shapes
    assert len(features["dwell"]) == len(dwells)
    assert len(features["dwell_log"]) == len(dwells)
    assert len(features["dwell_mean"]) == len(dwells)

    # Check that log transform is correct
    np.testing.assert_array_almost_equal(features["dwell_log"], np.log(dwells + 1e-6), decimal=5)


def test_compute_signal_features():
    """Test comprehensive signal feature extraction."""
    from leech.features import compute_signal_features

    # Create signal with known statistics per base
    signal = np.concatenate(
        [
            np.array([1.0, 1.0, 1.0]),  # Base 0: constant
            np.array([2.0, 3.0, 4.0]),  # Base 1: varying
            np.array([0.5, 0.5, 0.5]),  # Base 2: constant
        ]
    )
    seq_to_sig = np.array([0, 3, 6, 9])

    features = compute_signal_features(signal, seq_to_sig)

    # Check features
    assert "level_mean" in features
    assert "level_median" in features
    assert "level_std" in features
    assert "level_range" in features

    # Base 0 should have mean ~1.0, std ~0
    assert abs(features["level_mean"][0] - 1.0) < 1e-5
    assert features["level_std"][0] < 1e-5

    # Base 1 should have mean ~3.0, std > 0
    assert abs(features["level_mean"][1] - 3.0) < 1e-5
    assert features["level_std"][1] > 0.5


class TestMeanFastPathTail:
    """The per-base mean must not absorb signal past the end of the map.

    `np.add.reduceat` sums its *final* segment to the end of the array, so the
    Python fallback charged everything after the last mapped base to that base
    (audit C3). Median/std/range come from the explicit loop and never had the
    bug, which is why nothing caught it. These tests pin the Python fallback
    specifically -- `monkeypatch`, not "hope the extension is missing", because
    with `leech-core` installed the Rust path answers first and is correct.
    """

    # Base 0 -> [1, 1, 1], base 1 -> [2, 2, 2], and a tail belonging to no base.
    SIGNAL = np.array([1, 1, 1, 2, 2, 2, 100, 100, 100, 100], dtype=np.float32)
    MAP = np.array([0, 3, 6], dtype=np.int64)
    EXPECTED = np.array([1.0, 2.0], dtype=np.float32)

    @pytest.fixture
    def no_rust(self, monkeypatch):
        """Force the pure-Python fallback (what `pip install leech` runs)."""
        import leech.features as features_mod

        monkeypatch.setattr(features_mod, "HAS_RUST", False)

    def test_compute_signal_features(self, no_rust):
        from leech.features import compute_signal_features

        feats = compute_signal_features(self.SIGNAL, self.MAP)
        np.testing.assert_array_equal(feats["level_mean"], self.EXPECTED)
        # The loop-computed stats were always right; they are the cross-check.
        np.testing.assert_array_equal(feats["level_median"], self.EXPECTED)

    def test_compute_signal_levels(self, no_rust):
        levels = compute_signal_levels(self.SIGNAL, self.MAP, stat="mean")
        np.testing.assert_array_equal(levels, self.EXPECTED)

    def test_compute_kmer_residual_features(self, no_rust):
        from leech.features import compute_kmer_residual_features

        # Empty level table -> every expected level is 0, so the residual is
        # exactly the observed per-base mean.
        feats = compute_kmer_residual_features(self.SIGNAL, self.MAP, "AC", {}, kmer_len=1)
        np.testing.assert_array_equal(feats["kmer_residual"], self.EXPECTED)

    @pytest.mark.parametrize(
        "seq_to_sig",
        [
            np.array([0, 3, 6], dtype=np.int64),  # map ends before the signal
            np.array([0, 3, 6, 10], dtype=np.int64),  # map covers the signal
            np.array([0, 3, 3, 6, 10], dtype=np.int64),  # zero-dwell base
            np.array([2, 5, 10], dtype=np.int64),  # map starts late
            np.array([0, 4, 14], dtype=np.int64),  # map runs past the signal
        ],
    )
    def test_matches_per_base_numpy_mean(self, no_rust, seq_to_sig):
        """The fast path must equal `np.mean` over each base's own slice."""
        from leech.features import compute_signal_features

        rng = np.random.default_rng(0)
        signal = rng.standard_normal(10).astype(np.float32)
        expected = np.array(
            [
                np.mean(signal[s:e]) if e > s and len(signal[s:e]) else 0.0
                for s, e in zip(seq_to_sig[:-1], seq_to_sig[1:], strict=True)
            ],
            dtype=np.float32,
        )
        got = compute_signal_features(signal, seq_to_sig)["level_mean"]
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)

    def test_full_coverage_is_bit_identical_to_the_old_expression(self, no_rust):
        """No value moves on the normal case -- chunks must not change.

        The reference is the pre-fix expression verbatim. It has to be: the
        rounding of a `reduceat` segment sum is not reproducible with
        `np.add.reduce` over the same slice (they accumulate in a different
        order and differ by an ulp), and "same to an ulp" is not the claim.
        The claim is that a map covering its signal produces the same bits it
        always did.
        """
        from leech.features import compute_signal_features

        rng = np.random.default_rng(7)
        seq_to_sig = np.array([0, 5, 9, 15, 20], dtype=np.int64)
        signal = rng.standard_normal(20).astype(np.float32)

        sums = np.add.reduceat(signal, seq_to_sig[:-1])
        expected = (sums / np.diff(seq_to_sig)).astype(np.float32)

        got = compute_signal_features(signal, seq_to_sig)["level_mean"]
        np.testing.assert_array_equal(got, expected)
        np.testing.assert_array_equal(compute_signal_levels(signal, seq_to_sig, "mean"), expected)

    def test_rust_and_python_agree_on_a_partial_map(self, monkeypatch):
        """Both backends must charge the tail to nobody."""
        from leech.features import HAS_RUST, compute_signal_features

        if not HAS_RUST:
            pytest.skip("leech_core not installed")
        rust = compute_signal_features(self.SIGNAL, self.MAP)["level_mean"]

        import leech.features as features_mod

        monkeypatch.setattr(features_mod, "HAS_RUST", False)
        python = compute_signal_features(self.SIGNAL, self.MAP)["level_mean"]
        np.testing.assert_allclose(python, rust, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
