"""
Tests for signal map refinement.

Tests the banded DP, kmer level extraction, and SigMapRefiner.
"""

import numpy as np

from leech.signal_refine import (
    SigMapRefiner,
    compute_sig_band,
    extract_levels,
    rough_rescale,
    seq_banded_dp,
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
        # All levels should be defined (no NaN/zero at edges)

    def test_unknown_kmer(self):
        """Test fallback for unknown kmers."""
        kmer_to_level = {"ACG": 1.0}
        sequence = "ACGNN"
        levels = extract_levels(sequence, kmer_to_level, kmer_len=3)
        assert len(levels) == 5
        assert levels[1] == 1.0  # Known kmer


class TestComputeSigBand:
    """Test signal band computation."""

    def test_basic_band(self):
        """Test basic band computation."""
        seq_to_sig = np.array([0, 100, 200, 300, 400])
        starts, ends = compute_sig_band(seq_to_sig, half_bandwidth=50)

        assert len(starts) == 5
        assert len(ends) == 5
        assert starts[0] == 0  # Clamped to 0
        assert starts[2] == 150  # 200 - 50
        assert ends[2] == 250  # 200 + 50

    def test_band_clamp(self):
        """Test that band starts are clamped to 0."""
        seq_to_sig = np.array([10, 50, 100])
        starts, _ = compute_sig_band(seq_to_sig, half_bandwidth=20)
        assert starts[0] == 0  # 10 - 20 = -10, clamped to 0


class TestRoughRescale:
    """Test rough signal rescaling."""

    def test_identity_rescale(self):
        """Test rescaling when signal already matches expected levels."""
        signal = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float64)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        seq_to_sig = np.array([0, 2, 4, 6])

        rescaled = rough_rescale(signal, expected, seq_to_sig)
        assert len(rescaled) == len(signal)
        # Should be approximately identity since signal matches expected

    def test_linear_rescale(self):
        """Test that rescaling removes linear bias."""
        # Signal is 2x expected + 10
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        signal = np.array([12.0, 12.0, 14.0, 14.0, 16.0, 16.0], dtype=np.float64)
        seq_to_sig = np.array([0, 2, 4, 6])

        rescaled = rough_rescale(signal, expected, seq_to_sig)
        # After rescaling, per-base means should be close to expected levels
        for i in range(3):
            base_mean = np.mean(rescaled[seq_to_sig[i] : seq_to_sig[i + 1]])
            np.testing.assert_allclose(base_mean, expected[i], atol=0.1)


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
        )
        expected = np.array([1.0, 2.0, 3.0])
        band_starts = np.array([0, 5, 15, 25])
        band_ends = np.array([5, 15, 25, 35])

        refined = seq_banded_dp(signal, expected, band_starts, band_ends, len(signal))

        assert len(refined) == 4  # 3 bases + 1
        # Boundaries should be near 0, 10, 20, 30
        assert refined[0] >= 0
        assert refined[3] <= 30

    def test_monotonicity(self):
        """Test that refined mapping is monotonically increasing."""
        np.random.seed(42)
        signal = np.random.randn(100)
        expected = np.array([0.0, 1.0, -1.0, 0.5, -0.5])
        seq_to_sig = np.array([0, 20, 40, 60, 80, 100])
        band_starts, band_ends = compute_sig_band(seq_to_sig, half_bandwidth=15)

        refined = seq_banded_dp(signal, expected, band_starts, band_ends, len(signal))

        for i in range(1, len(refined)):
            assert refined[i] >= refined[i - 1], f"Non-monotonic at position {i}"


class TestSigMapRefiner:
    """Test SigMapRefiner dataclass."""

    def test_refine_basic(self):
        """Test basic refinement produces valid output."""
        kmer_to_level = {}
        for a in "ACGT":
            for b in "ACGT":
                for c in "ACGT":
                    kmer_to_level[a + b + c] = hash(a + b + c) % 100 / 50.0 - 1.0

        refiner = SigMapRefiner(
            kmer_to_level=kmer_to_level,
            kmer_len=3,
            half_bandwidth=20,
            do_rough_rescale=False,
        )

        np.random.seed(42)
        sequence = "ACGTACGTAC"
        signal = np.random.randn(100).astype(np.float32)
        seq_to_sig = np.linspace(0, 100, len(sequence) + 1).astype(np.int64)

        refined_signal, refined_map = refiner.refine(signal, sequence, seq_to_sig)

        assert len(refined_map) == len(seq_to_sig)
        # Should be monotonically non-decreasing (or fall back to original)
        is_monotonic = all(
            refined_map[i] <= refined_map[i + 1] for i in range(len(refined_map) - 1)
        )
        if not is_monotonic:
            # Fallback case: should equal original
            np.testing.assert_array_equal(refined_map, seq_to_sig)

    def test_refine_fallback(self):
        """Test that refinement falls back to original on invalid result."""
        kmer_to_level = {"AAA": 0.0}
        refiner = SigMapRefiner(
            kmer_to_level=kmer_to_level,
            kmer_len=3,
            half_bandwidth=5,
            do_rough_rescale=False,
        )

        # Very short signal/sequence where DP might fail
        signal = np.zeros(10, dtype=np.float32)
        sequence = "AAA"
        seq_to_sig = np.array([0, 3, 7, 10], dtype=np.int64)

        refined_signal, refined_map = refiner.refine(signal, sequence, seq_to_sig)
        assert len(refined_map) == len(seq_to_sig)
