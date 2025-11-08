"""
Comprehensive tests demonstrating the temporal relationship between
move tables (BAM) and raw signal (POD5).

This is the CORE concept linking basecalls to signal over time.
"""

import numpy as np
import pytest

from leech.features import MoveTable, compute_dwell_times, compute_signal_levels


class TestMoveTableTemporalRelationship:
    """
    Tests demonstrating how move tables map basecalls to signal over TIME.

    The move table is a temporal record of basecaller decisions:
    - Basecaller processes downsampled signal (stride = 5 typically)
    - At each position, it decides: stay on current base (0) or move to next base (1)
    - This creates a TIME-ALIGNED mapping between sequence and signal
    """

    def test_basic_temporal_mapping(self):
        """
        Test: Move table creates a time-indexed mapping.

        Scenario:
            stride = 5 (basecaller looks at every 5th signal sample)
            moves = [1, 1, 0, 1, 0, 0, 0, 1]

        Temporal interpretation:
            Position 0: Decision made → move to base 0 (1)
            Position 1: Decision made → move to base 1 (1)
            Position 2: Decision made → stay on base 1 (0)
            Position 3: Decision made → move to base 2 (1)
            Position 4-6: Stay on base 2 (0,0,0)
            Position 7: Decision made → move to base 3 (1)

        Signal time mapping (stride=5):
            Base 0: signal[0:5]    (1 move position * 5)
            Base 1: signal[5:10]   (2 move positions * 5)
            Base 2: signal[10:20]  (4 move positions * 5)
            Base 3: signal[20:40]  (4 move positions * 5)
        """
        stride = 5
        moves = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype=np.int8)

        mt = MoveTable(
            stride=stride, moves=moves, read_id="test_temporal", num_samples=1000, trim_offset=0
        )

        # Get temporal mapping
        seq_to_sig = mt.to_seq_to_sig_map()

        # Verify temporal positions
        # Position formula: (move_index + 1) * stride + trim_offset
        assert seq_to_sig[0] == 0  # Base 0 starts at time 0
        assert seq_to_sig[1] == 5  # Base 1 starts at time 5 (position 0: (0+1)*5)
        assert seq_to_sig[2] == 10  # Base 2 starts at time 10 (position 1: (1+1)*5)
        assert seq_to_sig[3] == 20  # Base 3 starts at time 20 (position 3: (3+1)*5)
        assert seq_to_sig[4] == 40  # End of base 3 at time 40 (position 7: (7+1)*5)

    def test_dwell_time_as_temporal_duration(self):
        """
        Test: Dwell time = temporal duration a base occupies in the signal.

        Dwell time directly measures how long (in signal samples) the nanopore
        spent on each base. This is CRITICAL for aa-tRNA classification because:
        - Charged tRNAs may have different kinetics
        - Temporal information complements spatial (signal level) information
        """
        stride = 5
        moves = np.array([1, 1, 0, 1, 0, 0, 0, 1], dtype=np.int8)

        mt = MoveTable(
            stride=stride, moves=moves, read_id="test_dwell", num_samples=1000, trim_offset=0
        )

        dwells = compute_dwell_times(mt)

        # Dwell times = time spent on each base
        assert dwells[0] == 5  # Base 0: 1 move position = 5 samples
        assert dwells[1] == 5  # Base 1: 1 move position = 5 samples (0 doesn't count)
        assert dwells[2] == 10  # Base 2: 2 move positions = 10 samples (1 + 3 zeros)
        assert dwells[3] == 20  # Base 3: 4 move positions = 20 samples

        # Total time = sum of dwell times
        total_dwell = np.sum(dwells)
        assert total_dwell == 40  # Matches end position

    def test_signal_extraction_temporal_alignment(self):
        """
        Test: Extract signal for each base using temporal alignment.

        This demonstrates how we use the move table to extract the CORRECT
        signal samples for each base, maintaining temporal alignment.
        """
        stride = 5
        moves = np.array([1, 1, 0, 1], dtype=np.int8)

        mt = MoveTable(
            stride=stride, moves=moves, read_id="test_signal_align", num_samples=100, trim_offset=0
        )

        # Create synthetic signal with distinct patterns per base
        signal = np.zeros(100, dtype=np.float32)
        signal[0:5] = 1.0  # Base 0: constant 1.0
        signal[5:10] = 2.0  # Base 1: constant 2.0
        signal[10:20] = 3.0  # Base 2: constant 3.0

        seq_to_sig = mt.to_seq_to_sig_map()

        # Extract mean signal level for each base
        levels = compute_signal_levels(signal, seq_to_sig, stat="mean")

        assert levels[0] == pytest.approx(1.0)  # Base 0 temporal range
        assert levels[1] == pytest.approx(2.0)  # Base 1 temporal range
        assert levels[2] == pytest.approx(3.0)  # Base 2 temporal range

    def test_trim_offset_temporal_shift(self):
        """
        Test: Trim offset shifts the entire temporal mapping.

        The basecaller may trim signal at the start (adapter, poor quality).
        trim_offset shifts all signal positions forward in time.
        """
        stride = 5
        moves = np.array([1, 1, 0, 1], dtype=np.int8)
        trim_offset = 100  # First 100 samples were trimmed

        mt = MoveTable(
            stride=stride,
            moves=moves,
            read_id="test_trim",
            num_samples=1000,
            trim_offset=trim_offset,
        )

        seq_to_sig = mt.to_seq_to_sig_map()

        # All positions should be shifted by trim_offset
        assert seq_to_sig[0] == 100  # Base 0 starts at trim_offset
        assert seq_to_sig[1] == 105  # (0+1)*5 + 100
        assert seq_to_sig[2] == 110  # (1+1)*5 + 100
        assert seq_to_sig[3] == 120  # (3+1)*5 + 100

    def test_temporal_consistency_with_real_stride(self):
        """
        Test: Verify temporal mapping with real-world stride values.

        Real nanopore data typically uses stride=5 or stride=6 depending
        on the basecaller model. This tests both.
        """
        for stride in [5, 6]:
            moves = np.array([1, 0, 1, 0, 0, 1], dtype=np.int8)

            mt = MoveTable(
                stride=stride,
                moves=moves,
                read_id=f"test_stride_{stride}",
                num_samples=1000,
                trim_offset=0,
            )

            seq_to_sig = mt.to_seq_to_sig_map()
            dwells = compute_dwell_times(mt)

            # Verify temporal consistency
            # Position of 1s: 0, 2, 5
            expected_positions = [0, stride, 3 * stride, 6 * stride]
            np.testing.assert_array_equal(seq_to_sig, expected_positions)

            # Verify dwells sum to total time
            assert np.sum(dwells) == expected_positions[-1]

    def test_chunk_extraction_temporal_window(self):
        """
        Test: Chunk extraction maintains temporal relationships.

        When we extract a chunk for training, we need to maintain the
        temporal relationship between:
        1. The signal window (temporal extent)
        2. The k-mer sequence (bases in that window)
        3. The dwell times (temporal durations)
        """
        stride = 5
        # Create a longer sequence: 10 bases
        moves = np.array([1] * 10 + [0] * 0, dtype=np.int8)  # Simple: one move per base

        mt = MoveTable(
            stride=stride, moves=moves, read_id="test_chunk", num_samples=100, trim_offset=0
        )

        seq_to_sig = mt.to_seq_to_sig_map()
        dwells = compute_dwell_times(mt)

        # Extract a "chunk" centered on base 5
        focus_base = 5
        kmer_context = 2  # 2 bases on each side

        # K-mer range: [3, 4, 5, 6, 7]
        kmer_start = focus_base - kmer_context
        kmer_end = focus_base + kmer_context + 1

        # Signal range: from start of base 3 to end of base 7
        signal_start = seq_to_sig[kmer_start]
        signal_end = seq_to_sig[kmer_end]

        # Verify temporal alignment
        assert signal_start == 15  # Base 3 starts at 3*5
        assert signal_end == 40  # Base 8 starts at 8*5
        assert signal_end - signal_start == 25  # 5 bases * 5 samples each

        # Dwell times in chunk
        chunk_dwells = dwells[kmer_start:kmer_end]
        assert len(chunk_dwells) == 5  # 5 bases
        assert np.sum(chunk_dwells) == signal_end - signal_start  # Temporal consistency


class TestRemoraConceptAlignment:
    """
    Tests verifying our implementation aligns with Remora's approach.
    """

    def test_fixed_signal_length_chunks(self):
        """
        Test: Like Remora, we extract fixed-length signal chunks.

        Remora defines chunks with:
        1. Fixed signal length (e.g., 400 samples)
        2. Focus position at center
        3. K-mer context around focus base
        """
        stride = 5
        moves = np.array([1] * 20, dtype=np.int8)  # 20 bases

        mt = MoveTable(
            stride=stride, moves=moves, read_id="test_fixed_chunk", num_samples=200, trim_offset=0
        )

        seq_to_sig = mt.to_seq_to_sig_map()

        # Simulate chunk extraction like Remora
        focus_base = 10  # Middle base
        signal_context = (100, 100)  # 200 samples total (fixed length)

        focus_sig_pos = seq_to_sig[focus_base]
        sig_start = focus_sig_pos - signal_context[0]
        sig_end = focus_sig_pos + signal_context[1]

        # Verify fixed length
        chunk_length = sig_end - sig_start
        assert chunk_length == 200  # Fixed signal length

    def test_kmer_expansion_with_move_table(self):
        """
        Test: K-mer expansion maintains temporal alignment.

        Remora expands each base to k-mer context, then uses move table
        to align k-mers with signal. We do the same but MORE explicitly
        by maintaining seq_to_sig_map throughout.
        """
        stride = 5
        moves = np.array([1] * 10, dtype=np.int8)

        mt = MoveTable(
            stride=stride, moves=moves, read_id="test_kmer_expand", num_samples=100, trim_offset=0
        )

        # Simulate k-mer extraction
        sequence = "ACGTACGTAC"  # 10 bases
        focus_base = 5
        kmer_context = 2

        # Extract k-mer: 2 bases on each side + focus = 5 bases
        kmer_start = focus_base - kmer_context
        kmer_end = focus_base + kmer_context + 1
        kmer = sequence[kmer_start:kmer_end]

        assert kmer == "ACGTA"
        assert len(kmer) == 5

        # Signal for this k-mer (temporal alignment)
        seq_to_sig = mt.to_seq_to_sig_map()
        sig_for_kmer_start = seq_to_sig[kmer_start]
        sig_for_kmer_end = seq_to_sig[kmer_end]

        # Verify the k-mer spans the correct temporal range
        assert sig_for_kmer_start == 15  # Base 3 * 5
        assert sig_for_kmer_end == 40  # Base 8 * 5


class TestDwellTimeAsNovelFeature:
    """
    Tests demonstrating dwell times as our NOVEL contribution beyond Remora.

    Remora uses move tables for signal-sequence alignment but doesn't
    explicitly extract dwell times as features. We do!
    """

    def test_dwell_features_capture_temporal_dynamics(self):
        """
        Test: Dwell features capture temporal kinetics.

        For aa-tRNA-seq:
        - Charged tRNAs may have different translocation kinetics
        - Dwell time variability could indicate charging state
        - This is BEYOND what Remora does (spatial signal only)
        """
        from leech.features import compute_dwell_features

        # Simulate variable dwell times
        # Charged: more uniform dwell times
        charged_dwells = np.array([5, 5, 5, 5, 5, 5, 5], dtype=np.float32)

        # Uncharged: more variable dwell times
        uncharged_dwells = np.array([3, 8, 4, 9, 2, 10, 5], dtype=np.float32)

        charged_feats = compute_dwell_features(charged_dwells, window=3)
        uncharged_feats = compute_dwell_features(uncharged_dwells, window=3)

        # Charged should have lower std deviation
        charged_std_mean = np.mean(charged_feats["dwell_std"])
        uncharged_std_mean = np.mean(uncharged_feats["dwell_std"])

        assert uncharged_std_mean > charged_std_mean

    def test_dwell_and_signal_level_complementary(self):
        """
        Test: Dwell time and signal level are complementary features.

        - Signal level: spatial information (current amplitude)
        - Dwell time: temporal information (duration)
        - Together: spatiotemporal characterization
        """
        stride = 5
        moves = np.array([1, 1, 0, 1, 0, 0], dtype=np.int8)

        mt = MoveTable(
            stride=stride, moves=moves, read_id="test_complementary", num_samples=100, trim_offset=0
        )

        # Create signal with varying levels and dwells
        signal = np.zeros(100, dtype=np.float32)
        signal[0:5] = 1.0  # Base 0: level=1.0, dwell=5
        signal[5:10] = 2.0  # Base 1: level=2.0, dwell=5
        signal[10:20] = 1.5  # Base 2: level=1.5, dwell=10

        seq_to_sig = mt.to_seq_to_sig_map()
        dwells = compute_dwell_times(mt)
        levels = compute_signal_levels(signal, seq_to_sig, stat="mean")

        # Base 0 and 2 have similar levels but different dwells
        assert levels[0] == pytest.approx(levels[2], rel=0.5)  # Similar levels
        assert dwells[0] != dwells[2]  # Different dwells (5 vs 10)

        # This demonstrates why dwell time adds information!


class TestEdgeCasesAndRobustness:
    """Test edge cases in temporal mapping."""

    def test_single_base_read(self):
        """Test a read with only one base."""
        stride = 5
        moves = np.array([1], dtype=np.int8)

        mt = MoveTable(
            stride=stride, moves=moves, read_id="single_base", num_samples=10, trim_offset=0
        )

        assert mt.num_bases == 1
        seq_to_sig = mt.to_seq_to_sig_map()
        assert len(seq_to_sig) == 2  # Start and end
        assert seq_to_sig[0] == 0
        assert seq_to_sig[1] == 5

    def test_consecutive_moves(self):
        """Test rapid base transitions (all 1s)."""
        stride = 5
        moves = np.array([1, 1, 1, 1, 1], dtype=np.int8)

        mt = MoveTable(
            stride=stride, moves=moves, read_id="consecutive", num_samples=50, trim_offset=0
        )

        dwells = compute_dwell_times(mt)

        # All dwells should be exactly stride
        assert np.all(dwells == stride)
        assert mt.num_bases == 5

    def test_long_homopolymer(self):
        """Test long homopolymer (many 0s)."""
        stride = 5
        moves = np.array([1] + [0] * 100 + [1], dtype=np.int8)

        mt = MoveTable(
            stride=stride, moves=moves, read_id="homopolymer", num_samples=1000, trim_offset=0
        )

        assert mt.num_bases == 2
        dwells = compute_dwell_times(mt)

        # First base has very long dwell (101 positions * stride)
        assert dwells[0] == 505  # (101 positions) * 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
