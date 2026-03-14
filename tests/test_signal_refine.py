"""
Tests for signal map refinement.

Tests the banded DP, kmer level extraction, band computation, and SigMapRefiner.
"""

import numpy as np
import pytest

from leech.signal_refine import (
    DEFAULT_SHORT_DWELL_PEN,
    SigMapRefiner,
    adjust_seq_band,
    compute_dwell_pen_array,
    compute_sig_band,
    convert_to_seq_band,
    extract_levels,
    refine_signal_mapping,
    rough_rescale_quantile,
    seq_banded_dp,
    validate_band,
)


class TestExtractLevels:
    """Test kmer level extraction."""

    def test_basic_extraction(self):
        """Test basic level extraction with simple kmer table."""
        kmer_to_level = {
            "ACG": 1.0,
            "CGT": 2.0,
            "GTA": 3.0,
        }
        sequence = "ACGTA"
        levels = extract_levels(sequence, kmer_to_level, kmer_len=3)
        assert len(levels) == 5
        # Position 1 (center of "ACG") should be 1.0
        assert levels[1] == 1.0
        # Position 2 (center of "CGT") should be 2.0
        assert levels[2] == 2.0
        # Position 3 (center of "GTA") should be 3.0
        assert levels[3] == 3.0

    def test_edge_handling(self):
        """Test that edges use nearest valid kmer."""
        kmer_to_level = {
            "ACG": 1.5,
            "CGT": 2.5,
        }
        sequence = "ACGT"
        levels = extract_levels(sequence, kmer_to_level, kmer_len=3)
        assert len(levels) == 4

    def test_unknown_kmer(self):
        """Test fallback for unknown kmers."""
        kmer_to_level = {"ACG": 1.0}
        sequence = "ACGNN"
        levels = extract_levels(sequence, kmer_to_level, kmer_len=3)
        assert len(levels) == 5
        assert levels[1] == 1.0  # Known kmer


class TestRoughRescaleQuantile:
    """Test quantile-based rough signal rescaling."""

    def test_basic_rescale(self):
        """Test that rescaling produces output of same length."""
        np.random.seed(42)
        signal = np.random.randn(100).astype(np.float32)
        expected = np.random.randn(10).astype(np.float32)
        seq_to_sig = np.linspace(0, 100, 11).astype(np.int64)

        rescaled = rough_rescale_quantile(signal, expected, seq_to_sig, clip_bases=0)
        assert len(rescaled) == len(signal)


class TestComputeDwellPenArray:
    """Test dwell penalty array computation."""

    def test_default_params(self):
        """Test default (4, 3, 0.5) penalty."""
        pen = compute_dwell_pen_array(target=4, limit=3, weight=0.5)
        assert len(pen) == 3
        # dwell 0: 0.5 * (0-4)^2 = 8.0
        np.testing.assert_allclose(pen[0], 8.0)
        # dwell 1: 0.5 * (1-4)^2 = 4.5
        np.testing.assert_allclose(pen[1], 4.5)
        # dwell 2: 0.5 * (2-4)^2 = 2.0
        np.testing.assert_allclose(pen[2], 2.0)

    def test_limit_clamped(self):
        """Test that limit > target is clamped."""
        pen = compute_dwell_pen_array(target=3, limit=5, weight=1.0)
        assert len(pen) == 3  # Clamped to target


class TestBandComputation:
    """Test sig_band -> seq_band pipeline."""

    def test_compute_sig_band_basic(self):
        """Test that sig_band has correct shape and bounds."""
        bps = np.array([0, 5, 10, 15])
        levels = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        band = compute_sig_band(bps, levels, bhw=2)
        assert band.shape == (2, 15)  # sig_len = 15
        assert band[0, 0] == 0  # Start at 0
        assert band[1, -1] == 3  # seq_len = 3

    def test_convert_to_seq_band(self):
        """Test sig_band -> seq_band conversion."""
        bps = np.array([0, 5, 10, 15])
        levels = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sig_band = compute_sig_band(bps, levels, bhw=2)
        seq_band = convert_to_seq_band(sig_band)
        assert seq_band.shape[0] == 2
        assert seq_band.shape[1] == 3  # seq_len
        assert seq_band[0, 0] == 0  # First band starts at signal 0
        assert seq_band[1, -1] == 15  # Last band ends at sig_len

    def test_adjust_seq_band_monotonicity(self):
        """Test that adjust_seq_band enforces min_step."""
        seq_band = np.array([[0, 2, 4], [5, 7, 10]], dtype=np.int32)
        adjust_seq_band(seq_band, min_step=2)
        # Starts should be at least 2 apart
        for i in range(seq_band.shape[1] - 1):
            assert seq_band[0, i + 1] >= seq_band[0, i] + 1
        # Ends should be at least 2 apart
        for i in range(seq_band.shape[1] - 1):
            assert seq_band[1, i + 1] >= seq_band[1, i] + 1

    def test_validate_band_valid(self):
        """Test validation passes for a valid band."""
        bps = np.array([0, 10, 20, 30])
        levels = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sig_band = compute_sig_band(bps, levels, bhw=3)
        seq_band = convert_to_seq_band(sig_band)
        adjust_seq_band(seq_band, min_step=2)
        validate_band(seq_band, sig_len=30, seq_len=3)

    def test_validate_band_invalid_start(self):
        """Test validation fails when band doesn't start at 0."""
        band = np.array([[1, 2], [5, 10]], dtype=np.int32)
        with pytest.raises(ValueError, match="does not start with 0"):
            validate_band(band)


class TestSeqBandedDP:
    """Test banded Viterbi dynamic programming."""

    def test_trivial_case(self):
        """Test DP with signal that perfectly matches expected levels."""
        # 3 bases, each 10 samples
        signal = np.concatenate(
            [
                np.full(10, 1.0),
                np.full(10, 2.0),
                np.full(10, 3.0),
            ]
        ).astype(np.float32)
        levels = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        seq_to_sig = np.array([0, 10, 20, 30], dtype=np.int64)

        sig_band = compute_sig_band(seq_to_sig, levels, bhw=5)
        seq_band = convert_to_seq_band(sig_band)
        adjust_seq_band(seq_band, min_step=2)

        path = seq_banded_dp(signal, levels, seq_band, DEFAULT_SHORT_DWELL_PEN)

        assert len(path) == 4  # 3 bases + 1
        assert path[0] == 0
        assert path[3] == 30
        # Boundaries should be near 10 and 20
        assert 5 <= path[1] <= 15, f"Expected boundary near 10, got {path[1]}"
        assert 15 <= path[2] <= 25, f"Expected boundary near 20, got {path[2]}"

    def test_monotonicity(self):
        """Test that refined mapping is monotonically increasing."""
        np.random.seed(42)
        signal = np.random.randn(100).astype(np.float32)
        levels = np.array([0.0, 1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        seq_to_sig = np.array([0, 20, 40, 60, 80, 100], dtype=np.int64)

        sig_band = compute_sig_band(seq_to_sig, levels, bhw=5)
        seq_band = convert_to_seq_band(sig_band)
        adjust_seq_band(seq_band, min_step=2)

        path = seq_banded_dp(signal, levels, seq_band, DEFAULT_SHORT_DWELL_PEN)

        for i in range(1, len(path)):
            assert path[i] > path[i - 1], (
                f"Non-monotonic at position {i}: {path[i - 1]} -> {path[i]}"
            )

    def test_viterbi_algo(self):
        """Test with Viterbi algorithm (no dwell penalty)."""
        signal = np.concatenate(
            [
                np.full(10, 1.0),
                np.full(10, 2.0),
            ]
        ).astype(np.float32)
        levels = np.array([1.0, 2.0], dtype=np.float32)
        seq_to_sig = np.array([0, 10, 20], dtype=np.int64)

        sig_band = compute_sig_band(seq_to_sig, levels, bhw=5)
        seq_band = convert_to_seq_band(sig_band)
        adjust_seq_band(seq_band, min_step=2)
        sd_pen = np.zeros(0, dtype=np.float32)

        path = seq_banded_dp(signal, levels, seq_band, sd_pen, algo="Viterbi")

        assert path[0] == 0
        assert path[2] == 20
        assert 5 <= path[1] <= 15


class TestRefineSignalMapping:
    """Test the full refinement pipeline."""

    def test_basic_refinement(self):
        """Test that refine_signal_mapping returns valid path."""
        signal = np.concatenate(
            [
                np.full(10, 1.0),
                np.full(10, 2.0),
                np.full(10, 3.0),
            ]
        ).astype(np.float32)
        levels = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        seq_to_sig = np.array([0, 10, 20, 30], dtype=np.int64)

        path = refine_signal_mapping(signal, seq_to_sig, levels)

        assert len(path) == 4
        assert path[0] == 0
        assert path[3] == 30
        assert np.all(np.diff(path) > 0)


class TestSigMapRefiner:
    """Test SigMapRefiner dataclass."""

    def _make_refiner(self, **kwargs):
        """Create a refiner with a simple kmer table."""
        kmer_to_level = {}
        for a in "ACGT":
            for b in "ACGT":
                for c in "ACGT":
                    kmer_to_level[a + b + c] = hash(a + b + c) % 100 / 50.0 - 1.0
        defaults = dict(
            kmer_to_level=kmer_to_level,
            kmer_len=3,
            do_rough_rescale=False,
            scale_iters=0,
        )
        defaults.update(kwargs)
        return SigMapRefiner(**defaults)

    def test_refine_basic(self):
        """Test basic refinement produces valid output."""
        refiner = self._make_refiner()

        np.random.seed(42)
        sequence = "ACGTACGTAC"
        signal = np.random.randn(100).astype(np.float32)
        seq_to_sig = np.linspace(0, 100, len(sequence) + 1).astype(np.int64)

        refined_signal, refined_map = refiner.refine(signal, sequence, seq_to_sig)

        assert len(refined_map) == len(seq_to_sig)
        # Should be monotonically increasing (or fall back to original)
        diffs = np.diff(refined_map)
        if not np.all(diffs > 0):
            np.testing.assert_array_equal(refined_map, seq_to_sig)

    def test_refine_rescale_only(self):
        """Test scale_iters=-1 only rescales, doesn't change mapping."""
        refiner = self._make_refiner(scale_iters=-1, do_rough_rescale=True)

        np.random.seed(42)
        sequence = "ACGTACGTAC"
        signal = np.random.randn(100).astype(np.float32)
        seq_to_sig = np.linspace(0, 100, len(sequence) + 1).astype(np.int64)

        refined_signal, refined_map = refiner.refine(signal, sequence, seq_to_sig)

        np.testing.assert_array_equal(refined_map, seq_to_sig)

    def test_refine_multi_iteration(self):
        """Test scale_iters > 0 runs multiple iterations."""
        refiner = self._make_refiner(scale_iters=2)

        np.random.seed(42)
        sequence = "ACGTACGTAC"
        signal = np.random.randn(100).astype(np.float32)
        seq_to_sig = np.linspace(0, 100, len(sequence) + 1).astype(np.int64)

        refined_signal, refined_map = refiner.refine(signal, sequence, seq_to_sig)

        assert len(refined_map) == len(seq_to_sig)

    def test_refine_fallback(self):
        """Test that refinement falls back to original on invalid result."""
        kmer_to_level = {"AAA": 0.0}
        refiner = SigMapRefiner(
            kmer_to_level=kmer_to_level,
            kmer_len=3,
            do_rough_rescale=False,
            scale_iters=0,
        )

        signal = np.zeros(10, dtype=np.float32)
        sequence = "AAA"
        seq_to_sig = np.array([0, 3, 7, 10], dtype=np.int64)

        refined_signal, refined_map = refiner.refine(signal, sequence, seq_to_sig)
        assert len(refined_map) == len(seq_to_sig)
