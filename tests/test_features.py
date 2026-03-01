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
    normalize_signal,
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


def test_normalize_signal_median_mad():
    """Test median-MAD normalization."""
    # Create signal with known statistics
    signal = np.array([100, 102, 98, 105, 95, 101, 99, 103, 97], dtype=np.float32)

    norm_signal, params = normalize_signal(signal, method="median_mad")

    # Check that normalization parameters are computed
    assert "median" in params
    assert "mad" in params

    # Normalized signal should have median ~0
    assert abs(np.median(norm_signal)) < 0.1


def test_normalize_signal_zscore():
    """Test z-score normalization."""
    signal = np.array([100, 102, 98, 105, 95, 101, 99, 103, 97], dtype=np.float32)

    norm_signal, params = normalize_signal(signal, method="zscore")

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
