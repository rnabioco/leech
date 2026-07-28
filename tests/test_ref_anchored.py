"""
Tests for reference-anchored mode functions.

Tests make_ref_to_query_mapping() and compute_ref_to_signal() with
known CIGAR strings.
"""

import numpy as np
import pytest

from leech.features import compute_ref_to_signal, make_ref_to_query_mapping, normalize_read_signal


class TestMakeRefToQueryMapping:
    """Test reference-to-query coordinate mapping from CIGAR."""

    def test_perfect_match(self):
        """Test simple perfect match (all M)."""
        # 10M: 10 bases, all match
        cigar = [(0, 10)]
        mapping = make_ref_to_query_mapping(cigar)
        expected = np.arange(11)  # 0..10
        np.testing.assert_array_equal(mapping, expected)

    def test_insertion(self):
        """Test CIGAR with insertion (remora knot algorithm)."""
        # 5M 2I 5M: 10 ref bases, 12 query bases
        cigar = [(0, 5), (1, 2), (0, 5)]
        mapping = make_ref_to_query_mapping(cigar)
        # Remora knot algorithm: knots at start and end-1 of each match block
        # Match1: ref 0-4, query 0-4 → knots (0,0) and (4,4)
        # Ins: query 5-6 (not in ref)
        # Match2: ref 5-9, query 7-11 → knots (5,7) and (9,11)
        # ref[4] = query[4] (end-1 of first match)
        # ref[5] = query[7] (start of second match, insertion jumped)
        assert len(mapping) == 11  # 10 ref positions + 1
        assert mapping[0] == 0
        assert mapping[4] == 4  # End-1 of first match block
        assert mapping[5] == 7  # Start of second match (after insertion)
        assert mapping[10] == 12

    def test_deletion(self):
        """Test CIGAR with deletion (remora knot algorithm)."""
        # 5M 3D 5M: 13 ref bases, 10 query bases
        cigar = [(0, 5), (2, 3), (0, 5)]
        mapping = make_ref_to_query_mapping(cigar)
        # Remora knot algorithm: knots at start and end-1 of each match block
        # Match1: ref 0-4, query 0-4 → knots (0,0) and (4,4)
        # Del: ref 5-7
        # Match2: ref 8-12, query 5-9 → knots (8,5) and (12,9)
        # Deletion region ref[5-7] interpolates between query 4 and 5
        assert len(mapping) == 14  # 13 ref positions + 1
        assert mapping[0] == 0
        assert mapping[4] == 4
        # Deletion region: interpolates between (4,4) and (8,5)
        np.testing.assert_allclose(mapping[5], 4.25, atol=0.01)
        np.testing.assert_allclose(mapping[6], 4.5, atol=0.01)
        np.testing.assert_allclose(mapping[7], 4.75, atol=0.01)
        assert mapping[8] == 5  # Start of second match block
        assert mapping[13] == 10

    def test_soft_clip(self):
        """Test CIGAR with soft clips (remora knot algorithm)."""
        # 3S 5M 2S: 5 ref bases, 10 query bases
        cigar = [(4, 3), (0, 5), (4, 2)]
        mapping = make_ref_to_query_mapping(cigar)
        # Remora strips trailing non-match ops (2S stripped)
        # Remaining: 3S 5M
        # Soft clip 3S: query advances to 3
        # Match 5M: ref 0-4, query 3-7 → knots (0,3),(4,7)
        # Final ref_knots[-1] = 5
        assert len(mapping) == 6  # 5 ref positions + 1
        assert mapping[0] == 3  # After 3S
        assert mapping[4] == 7
        assert mapping[5] == 8  # Final position

    def test_complex_cigar(self):
        """Test complex CIGAR with mix of operations (remora knot algorithm)."""
        # 3M 1I 2M 1D 4M
        cigar = [(0, 3), (1, 1), (0, 2), (2, 1), (0, 4)]
        mapping = make_ref_to_query_mapping(cigar)
        # ref_len = 3 + 2 + 1 + 4 = 10
        # query_len = 3 + 1 + 2 + 4 = 10
        # Match1(3M): ref 0-2, query 0-2 → knots (0,0),(2,2)
        # Ins(1I): query 3
        # Match2(2M): ref 3-4, query 4-5 → knots (3,4),(4,5)
        # Del(1D): ref 5
        # Match3(4M): ref 6-9, query 6-9 → knots (6,6),(9,9)
        assert len(mapping) == 11
        assert mapping[0] == 0
        assert mapping[2] == 2  # End-1 of first match
        assert mapping[3] == 4  # Start of second match (after 1I)
        assert mapping[4] == 5  # End-1 of second match
        # Deletion at ref 5: interpolates between (4,5) and (6,6)
        np.testing.assert_allclose(mapping[5], 5.5, atol=0.01)
        assert mapping[6] == 6  # Start of third match
        assert mapping[10] == 10

    def test_empty_cigar(self):
        """Test empty CIGAR."""
        cigar = []
        mapping = make_ref_to_query_mapping(cigar)
        assert len(mapping) == 1  # Just [0]

    def test_monotonicity(self):
        """Test that mapping is monotonically non-decreasing."""
        cigar = [(0, 10), (1, 3), (0, 5), (2, 2), (0, 8)]
        mapping = make_ref_to_query_mapping(cigar)
        for i in range(1, len(mapping)):
            assert mapping[i] >= mapping[i - 1], f"Non-monotonic at position {i}"


class TestComputeRefToSignal:
    """Test full ref->signal mapping pipeline."""

    def test_simple_case(self):
        """Test ref->signal with perfect match."""
        # 5 bases, each 10 signal samples
        query_to_signal = np.array([0, 10, 20, 30, 40, 50])
        cigar = [(0, 5)]  # 5M
        ref_to_signal = compute_ref_to_signal(query_to_signal, cigar)
        np.testing.assert_array_equal(ref_to_signal, query_to_signal)

    def test_with_insertion(self):
        """Test ref->signal with insertion in query."""
        # 4 query bases, each 10 signal samples
        query_to_signal = np.array([0, 10, 20, 30, 40])
        # 2M 1I 1M -> 3 ref bases, 4 query bases
        cigar = [(0, 2), (1, 1), (0, 1)]
        ref_to_signal = compute_ref_to_signal(query_to_signal, cigar)
        # ref[0] -> query[0] -> sig[0]
        # ref[1] -> query[1] -> sig[10]
        # ref[2] -> query[3] -> sig[30] (skipped insertion at query[2])
        # ref[3] -> query[4] -> sig[40]
        assert len(ref_to_signal) == 4  # 3 ref bases + 1
        assert ref_to_signal[0] == 0
        assert ref_to_signal[1] == 10
        assert ref_to_signal[2] == 30
        assert ref_to_signal[3] == 40

    def test_with_deletion(self):
        """Test ref->signal with deletion in query."""
        # 3 query bases
        query_to_signal = np.array([0, 10, 20, 30])
        # 2M 1D 1M -> 4 ref bases, 3 query bases
        cigar = [(0, 2), (2, 1), (0, 1)]
        ref_to_signal = compute_ref_to_signal(query_to_signal, cigar)
        # ref[0] -> query[0] -> sig[0]
        # ref[1] -> query[1] -> sig[10]
        # ref[2] -> query[2] -> sig[20] (deletion, query stays at 2)
        # ref[3] -> query[2] -> sig[20]
        # ref[4] -> query[3] -> sig[30]
        assert len(ref_to_signal) == 5  # 4 ref bases + 1
        assert ref_to_signal[0] == 0
        assert ref_to_signal[1] == 10


class TestPaScalingNormalization:
    """Test pa_scaling normalization."""

    def test_pa_scaling_basic(self):
        """Test basic pa_scaling normalization."""
        raw = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)
        # POD5 calibration: pA = (DAC + offset) * scale
        cal_offset = -50.0
        cal_scale = 2.0
        # pA = (100-50)*2=100, (200-50)*2=300, (300-50)*2=500, (400-50)*2=700
        # norm = (pA - mean) / stdev
        pa_mean = 400.0
        pa_stdev = 200.0
        # norm = (100-400)/200=-1.5, (300-400)/200=-0.5, (500-400)/200=0.5, (700-400)/200=1.5

        normalized, params = normalize_read_signal(
            raw,
            method="pa_scaling",
            pa_mean=pa_mean,
            pa_stdev=pa_stdev,
            cal_offset=cal_offset,
            cal_scale=cal_scale,
        )

        expected = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
        np.testing.assert_allclose(normalized, expected, rtol=1e-5)
        assert params["pa_mean"] == pa_mean
        assert params["pa_stdev"] == pa_stdev

    def test_pa_scaling_missing_params(self):
        """Test that pa_scaling raises without required params."""
        raw = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="pa_scaling requires pa_mean"):
            normalize_read_signal(raw, method="pa_scaling")

        with pytest.raises(ValueError, match="pa_scaling requires cal_offset"):
            normalize_read_signal(raw, method="pa_scaling", pa_mean=1.0, pa_stdev=1.0)

    def test_median_mad_invariance(self):
        """Verify median-MAD is invariant to affine transform (DAC vs pA)."""
        np.random.seed(42)
        raw_dac = np.random.randn(1000).astype(np.float32) * 100 + 500

        # Normalize DACs directly
        norm_dac, _ = normalize_read_signal(raw_dac, method="median_mad")

        # Convert to pA then normalize
        offset, scale = 50.0, 2.0
        pa = (raw_dac - offset) * scale
        norm_pa, _ = normalize_read_signal(pa, method="median_mad")

        np.testing.assert_allclose(norm_dac, norm_pa, rtol=1e-5)
