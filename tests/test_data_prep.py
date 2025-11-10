"""
Tests for data preparation module.

Tests LeechRead, chunk extraction, and serialization.
"""

import numpy as np
import pytest
import torch

from leech.data_prep import (
    LeechRead,
    encode_kmer,
    extract_training_chunks,
    find_motif_in_reference,
    int_to_seq,
    load_chunks,
    map_reference_to_query_coords,
    one_hot_encode_sequence,
    save_chunks,
    seq_to_int,
)


class TestLeechRead:
    """Test LeechRead dataclass."""

    def test_initialization(self, sample_leech_read):
        """Test LeechRead initialization."""
        assert sample_leech_read.read_id == "test_read_001"
        assert isinstance(sample_leech_read.sequence, str)
        assert isinstance(sample_leech_read.signal, np.ndarray)
        assert isinstance(sample_leech_read.dwells, np.ndarray)

    def test_num_bases_property(self, sample_leech_read):
        """Test num_bases property."""
        assert sample_leech_read.num_bases == len(sample_leech_read.sequence)

    def test_num_samples_property(self, sample_leech_read):
        """Test num_samples property."""
        assert sample_leech_read.num_samples == len(sample_leech_read.signal)

    def test_get_chunk_valid(self, sample_leech_read):
        """Test getting a valid chunk."""
        chunk = sample_leech_read.get_chunk(base_idx=10, signal_context=(200, 200), kmer_context=5)

        assert chunk is not None
        assert "signal" in chunk
        assert "sequence" in chunk
        assert "dwell" in chunk
        assert "features" in chunk
        assert "base_idx" in chunk

    def test_get_chunk_invalid_base_idx(self, sample_leech_read):
        """Test that get_chunk returns None for invalid indices."""
        # Too close to start
        chunk = sample_leech_read.get_chunk(base_idx=2, kmer_context=5)
        assert chunk is None

        # Too close to end
        chunk = sample_leech_read.get_chunk(
            base_idx=sample_leech_read.num_bases - 2, kmer_context=5
        )
        assert chunk is None

    def test_get_chunk_signal_length(self, sample_leech_read):
        """Test that signal chunk has correct length."""
        left_context = 100
        right_context = 150

        chunk = sample_leech_read.get_chunk(
            base_idx=10, signal_context=(left_context, right_context), kmer_context=5
        )

        if chunk is not None:
            assert len(chunk["signal"]) == left_context + right_context

    def test_get_chunk_sequence_length(self, sample_leech_read):
        """Test that sequence chunk has correct length."""
        kmer_context = 5

        chunk = sample_leech_read.get_chunk(base_idx=10, kmer_context=kmer_context)

        if chunk is not None:
            assert len(chunk["sequence"]) == 2 * kmer_context + 1

    def test_get_chunk_dwell_length(self, sample_leech_read):
        """Test that dwell chunk has correct length."""
        kmer_context = 5

        chunk = sample_leech_read.get_chunk(base_idx=10, kmer_context=kmer_context)

        if chunk is not None:
            assert len(chunk["dwell"]) == 2 * kmer_context + 1

    def test_get_chunk_features_shape(self, sample_leech_read):
        """Test that features have correct shape."""
        kmer_context = 5

        chunk = sample_leech_read.get_chunk(base_idx=10, kmer_context=kmer_context)

        if chunk is not None and chunk["features"].size > 0:
            num_features = chunk["features"].shape[0]
            feature_len = chunk["features"].shape[1]
            assert feature_len == 2 * kmer_context + 1
            assert num_features > 0


class TestExtractTrainingChunks:
    """Test extract_training_chunks function."""

    def test_extract_without_motif(self, sample_leech_read):
        """Test extracting chunks without motif filtering."""
        chunks = extract_training_chunks(sample_leech_read, motif=None, label=1)

        assert len(chunks) > 0
        assert all(chunk["label"] == 1 for chunk in chunks)
        assert all(chunk["read_id"] == sample_leech_read.read_id for chunk in chunks)

    def test_extract_with_motif(self, sample_leech_read):
        """Test extracting chunks with motif filtering."""
        # Use a motif that exists in the sequence
        motif = sample_leech_read.sequence[7:10]  # 3-base motif

        chunks = extract_training_chunks(sample_leech_read, motif=motif, motif_offset=1, label=0)

        # Should find at least one occurrence
        assert len(chunks) >= 1
        assert all(chunk["label"] == 0 for chunk in chunks)

    def test_extract_with_nonexistent_motif(self, sample_leech_read):
        """Test extracting with a motif that doesn't exist."""
        motif = "ZZZZZ"  # Invalid motif

        chunks = extract_training_chunks(sample_leech_read, motif=motif, label=0)

        # Should find no chunks
        assert len(chunks) == 0

    def test_chunk_structure(self, sample_leech_read):
        """Test that extracted chunks have correct structure."""
        chunks = extract_training_chunks(sample_leech_read, motif=None, label=1)

        for chunk in chunks:
            assert "signal" in chunk
            assert "sequence" in chunk
            assert "dwell" in chunk
            assert "features" in chunk
            assert "base_idx" in chunk
            assert "label" in chunk
            assert "read_id" in chunk


class TestChunkSerialization:
    """Test save_chunks and load_chunks functions."""

    def test_save_and_load_chunks(self, sample_chunks, tmp_path):
        """Test saving and loading chunks."""
        output_file = tmp_path / "test_chunks.npz"

        # Save
        save_chunks(sample_chunks, output_file)
        assert output_file.exists()

        # Load
        loaded_chunks = load_chunks(output_file)
        assert len(loaded_chunks) == len(sample_chunks)

    def test_loaded_chunk_structure(self, sample_chunks, tmp_path):
        """Test that loaded chunks have correct structure."""
        output_file = tmp_path / "test_chunks.npz"

        save_chunks(sample_chunks, output_file)
        loaded_chunks = load_chunks(output_file)

        for chunk in loaded_chunks:
            assert "signal" in chunk
            assert "sequence" in chunk
            assert "dwell" in chunk
            assert "features" in chunk
            assert "label" in chunk
            assert "read_id" in chunk
            assert "base_idx" in chunk

    def test_loaded_chunk_values(self, sample_chunks, tmp_path):
        """Test that loaded chunk values match original."""
        output_file = tmp_path / "test_chunks.npz"

        save_chunks(sample_chunks, output_file)
        loaded_chunks = load_chunks(output_file)

        # Compare first chunk
        orig = sample_chunks[0]
        loaded = loaded_chunks[0]

        np.testing.assert_array_almost_equal(orig["signal"], loaded["signal"])
        assert orig["sequence"] == loaded["sequence"]
        np.testing.assert_array_almost_equal(orig["dwell"], loaded["dwell"])
        assert orig["label"] == loaded["label"]
        assert orig["read_id"] == loaded["read_id"]

    def test_save_empty_chunks_raises(self, tmp_path):
        """Test that saving empty chunks raises ValueError."""
        output_file = tmp_path / "empty.npz"

        with pytest.raises(ValueError, match="No chunks to save"):
            save_chunks([], output_file)

    def test_save_creates_directory(self, sample_chunks, tmp_path):
        """Test that save_chunks creates parent directories."""
        output_file = tmp_path / "subdir" / "test_chunks.npz"

        save_chunks(sample_chunks, output_file)
        assert output_file.exists()


class TestSequenceEncoding:
    """Test sequence encoding functions."""

    def test_seq_to_int_basic(self):
        """Test basic DNA sequence to integer encoding."""
        seq = "ACGT"
        encoded = seq_to_int(seq)

        expected = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(encoded, expected)

    def test_seq_to_int_with_n(self):
        """Test encoding with N (unknown base)."""
        seq = "ACGTN"
        encoded = seq_to_int(seq)

        expected = np.array([0, 1, 2, 3, 4])
        np.testing.assert_array_equal(encoded, expected)

    def test_seq_to_int_with_uracil(self):
        """Test that U (RNA) is treated as T."""
        seq = "ACGU"
        encoded = seq_to_int(seq)

        expected = np.array([0, 1, 2, 3])  # U -> 3 (same as T)
        np.testing.assert_array_equal(encoded, expected)

    def test_seq_to_int_lowercase(self):
        """Test that lowercase sequences work."""
        seq = "acgt"
        encoded = seq_to_int(seq)

        expected = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(encoded, expected)

    def test_int_to_seq_basic(self):
        """Test integer to sequence conversion."""
        int_seq = np.array([0, 1, 2, 3])
        seq = int_to_seq(int_seq)

        assert seq == "ACGT"

    def test_int_to_seq_with_n(self):
        """Test integer to sequence with N."""
        int_seq = np.array([0, 1, 2, 3, 4])
        seq = int_to_seq(int_seq)

        assert seq == "ACGTN"

    def test_seq_to_int_to_seq_roundtrip(self):
        """Test that encoding then decoding recovers original."""
        original = "ACGTACGT"
        encoded = seq_to_int(original)
        decoded = int_to_seq(encoded)

        assert decoded == original

    def test_one_hot_encode_sequence_basic(self):
        """Test one-hot encoding of sequence."""
        seq = "ACG"
        kmer_len = 1

        encoded = one_hot_encode_sequence(seq, kmer_len)

        # Should be shape (kmer_len * 4, seq_len) = (4, 3)
        assert encoded.shape == (4, 3)

        # Check specific positions
        assert encoded[0, 0] == 1.0  # A at position 0
        assert encoded[1, 1] == 1.0  # C at position 1
        assert encoded[2, 2] == 1.0  # G at position 2

    def test_encode_kmer_basic(self):
        """Test encode_kmer function."""
        seq = "ACGT"
        encoded = encode_kmer(seq)

        assert isinstance(encoded, torch.Tensor)
        assert encoded.shape == (4, 4)  # 4 bases x 4 positions

        # Check one-hot property: each column should sum to 1
        assert torch.all(encoded.sum(dim=0) == 1.0)

    def test_encode_kmer_with_n(self):
        """Test encode_kmer with N (unknown base)."""
        seq = "ACGTN"
        encoded = encode_kmer(seq)

        # N should be encoded as all zeros
        assert torch.all(encoded[:, 4] == 0.0)

    def test_encode_kmer_dtype(self):
        """Test that encode_kmer returns float32."""
        seq = "ACGT"
        encoded = encode_kmer(seq)

        assert encoded.dtype == torch.float32


class TestDataPrepEdgeCases:
    """Test edge cases in data preparation."""

    def test_leech_read_with_no_labels(self, sample_leech_read):
        """Test LeechRead without labels."""
        assert sample_leech_read.labels is None

        # Getting chunk without labels should work
        chunk = sample_leech_read.get_chunk(base_idx=10)
        assert chunk is not None
        assert chunk["label"] is None

    def test_leech_read_with_labels(self, sample_leech_read):
        """Test LeechRead with labels."""
        labels = np.ones(sample_leech_read.num_bases, dtype=np.int64)
        sample_leech_read.labels = labels

        chunk = sample_leech_read.get_chunk(base_idx=10)
        assert chunk is not None
        assert chunk["label"] == 1

    def test_get_chunk_at_signal_boundary(self, sample_leech_read):
        """Test get_chunk when signal context extends beyond boundaries."""
        # Try to get chunk near the end with large context
        base_idx = sample_leech_read.num_bases - 10
        chunk = sample_leech_read.get_chunk(
            base_idx=base_idx, signal_context=(5000, 5000), kmer_context=5
        )

        # Should return None due to insufficient signal context
        assert chunk is None

    def test_chunks_with_empty_features(self, sample_signal, sample_sequence, sample_move_table):
        """Test chunks with empty feature dict."""
        seq_to_sig_map = sample_move_table.to_seq_to_sig_map()
        dwells = np.diff(seq_to_sig_map)

        # Create LeechRead with no features
        leech_read = LeechRead(
            read_id="test_read_no_features",
            sequence=sample_sequence,
            signal=sample_signal,
            seq_to_sig_map=seq_to_sig_map,
            dwells=dwells,
            dwell_features={},  # Empty
            signal_features={},  # Empty
        )

        chunk = leech_read.get_chunk(base_idx=10)
        if chunk is not None:
            assert chunk["features"].size == 0  # Empty array


class TestReferenceBasedMotifSearch:
    """Test reference-based motif search functions."""

    def test_find_motif_in_reference_basic(self):
        """Test finding a motif in a reference sequence."""
        ref_seq = "ACGTACGTCCAACGT"
        motif = "CCA"
        ref_start = 0
        ref_end = len(ref_seq)

        positions = find_motif_in_reference(ref_seq, motif, ref_start, ref_end)

        # Should find "CCA" at position 8
        assert 8 in positions
        assert len(positions) >= 1

    def test_find_motif_in_reference_multiple(self):
        """Test finding multiple occurrences of motif."""
        ref_seq = "CCAACGTCCAACGTCCA"
        motif = "CCA"
        ref_start = 0
        ref_end = len(ref_seq)

        positions = find_motif_in_reference(ref_seq, motif, ref_start, ref_end)

        # Should find at positions 0, 7, 14
        assert len(positions) == 3
        assert 0 in positions
        assert 7 in positions
        assert 14 in positions

    def test_find_motif_in_reference_region(self):
        """Test finding motif only within specified region."""
        ref_seq = "ACGTCCAACGTCCA"
        motif = "CCA"
        ref_start = 5  # Start after first CCA
        ref_end = len(ref_seq)

        positions = find_motif_in_reference(ref_seq, motif, ref_start, ref_end)

        # Should only find the second CCA at position 11
        assert 11 in positions
        assert 4 not in positions  # First CCA is before ref_start

    def test_find_motif_in_reference_not_found(self):
        """Test when motif is not found."""
        ref_seq = "ACGTACGTACGT"
        motif = "TTT"
        ref_start = 0
        ref_end = len(ref_seq)

        positions = find_motif_in_reference(ref_seq, motif, ref_start, ref_end)

        assert len(positions) == 0


class TestCIGARMapping:
    """Test CIGAR string coordinate mapping."""

    def test_map_reference_to_query_all_matches(self):
        """Test mapping with simple match CIGAR (all M operations)."""
        # Create mock alignment with CIGAR: 20M
        import pysam

        # Create a mock alignment
        aln = pysam.AlignedSegment()
        aln.reference_start = 100
        aln.query_alignment_start = 0
        aln.cigartuples = [(0, 20)]  # 20M (BAM_CMATCH)
        # reference_end is automatically calculated from CIGAR: 100 + 20 = 120

        # Map reference positions 105-110 to query
        result = map_reference_to_query_coords(aln, 105, 110, skip_indels=False)

        assert result is not None
        query_start, query_end = result
        # With all matches, query positions = ref positions - ref_start
        assert query_start == 5  # 105 - 100
        assert query_end == 10  # 110 - 100

    def test_map_reference_to_query_with_insertion(self):
        """Test mapping when there's an insertion in the motif region."""
        import pysam

        # CIGAR: 10M 3I 10M (insertion in the middle)
        aln = pysam.AlignedSegment()
        aln.reference_start = 100
        aln.query_alignment_start = 0
        aln.cigartuples = [(0, 10), (1, 3), (0, 10)]  # 10M 3I 10M
        # reference_end = 100 + 10 + 10 = 120 (insertion doesn't consume ref)

        # Map across the insertion (ref 108-112 includes the insertion point)
        result = map_reference_to_query_coords(aln, 108, 112, skip_indels=True)

        # Should return None when skip_indels=True
        assert result is None

    def test_map_reference_to_query_with_deletion(self):
        """Test mapping when there's a deletion in the motif region."""
        import pysam

        # CIGAR: 10M 3D 10M (deletion in the middle)
        aln = pysam.AlignedSegment()
        aln.reference_start = 100
        aln.query_alignment_start = 0
        aln.cigartuples = [(0, 10), (2, 3), (0, 10)]  # 10M 3D 10M
        # reference_end = 100 + 10 + 3 + 10 = 123 (deletion consumes ref)

        # Map across the deletion (ref 108-115 includes the deletion)
        result = map_reference_to_query_coords(aln, 108, 115, skip_indels=True)

        # Should return None when skip_indels=True
        assert result is None

    def test_map_reference_to_query_skip_indels_false(self):
        """Test that mapping works with skip_indels=False even with indels."""
        import pysam

        # CIGAR: 10M 3I 10M
        aln = pysam.AlignedSegment()
        aln.reference_start = 100
        aln.query_alignment_start = 0
        aln.cigartuples = [(0, 10), (1, 3), (0, 10)]  # 10M 3I 10M
        # reference_end = 120

        # Map a region before the insertion
        result = map_reference_to_query_coords(aln, 102, 105, skip_indels=False)

        assert result is not None
        query_start, query_end = result
        assert query_start == 2  # 102 - 100
        assert query_end == 5  # 105 - 100

    def test_map_reference_to_query_out_of_bounds(self):
        """Test mapping when requested region is outside aligned region."""
        import pysam

        aln = pysam.AlignedSegment()
        aln.reference_start = 100
        aln.query_alignment_start = 0
        aln.cigartuples = [(0, 20)]  # 20M
        # reference_end = 120

        # Try to map region before alignment starts
        result = map_reference_to_query_coords(aln, 50, 55, skip_indels=False)
        assert result is None

        # Try to map region after alignment ends
        result = map_reference_to_query_coords(aln, 150, 155, skip_indels=False)
        assert result is None

    def test_map_reference_to_query_with_soft_clip(self):
        """Test mapping with soft-clipped bases."""
        import pysam

        # CIGAR: 5S 15M 5S (soft clips at both ends)
        aln = pysam.AlignedSegment()
        aln.reference_start = 100
        aln.query_alignment_start = 5  # Starts after 5S
        aln.cigartuples = [(4, 5), (0, 15), (4, 5)]  # 5S 15M 5S
        # reference_end = 100 + 15 = 115 (only 15M consumes reference)

        # Map within the matched region
        result = map_reference_to_query_coords(aln, 105, 110, skip_indels=False)

        assert result is not None
        query_start, query_end = result
        # Query positions account for soft clip offset
        assert query_start == 10  # 5 (soft clip) + 5 (105-100)
        assert query_end == 15  # 5 (soft clip) + 10 (110-100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
