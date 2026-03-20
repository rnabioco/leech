"""
Tests for data preparation module.

Tests LeechRead, chunk extraction, and serialization.
"""

import numpy as np
import pytest
import torch

from leech.chunking import LeechRead, extract_training_chunks, load_chunks, save_chunks
from leech.configs import ChunkConfig, LabelConfig, MotifConfig
from leech.io.motif_search import find_motif_in_sequence, map_reference_to_query_coords
from leech.preparation import encode_kmer, int_to_seq, one_hot_encode_sequence, seq_to_int
from leech.splitting import split_chunks_by_read


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
        """Test that get_chunk returns None for truly invalid indices."""
        # Negative index
        chunk = sample_leech_read.get_chunk(base_idx=-1, kmer_context=5)
        assert chunk is None

        # Index beyond sequence length
        chunk = sample_leech_read.get_chunk(base_idx=sample_leech_read.num_bases, kmer_context=5)
        assert chunk is None

    def test_get_chunk_edge_base_idx(self, sample_leech_read):
        """Test that get_chunk handles edge positions with padding."""
        # Near start - should return chunk with N-padded kmer
        chunk = sample_leech_read.get_chunk(base_idx=2, kmer_context=5)
        assert chunk is not None
        assert "N" in chunk["sequence"]

        # Near end - should return chunk with N-padded kmer
        chunk = sample_leech_read.get_chunk(
            base_idx=sample_leech_read.num_bases - 2, kmer_context=5
        )
        assert chunk is not None
        assert "N" in chunk["sequence"]

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
        chunks = extract_training_chunks(
            sample_leech_read,
            motif_config=MotifConfig(),
            chunk_config=ChunkConfig(),
            labeling=LabelConfig(label="charged", label_int=1),
        )

        assert len(chunks) > 0
        assert all(chunk["label"] == "charged" for chunk in chunks)
        assert all(chunk["label_int"] == 1 for chunk in chunks)
        assert all(chunk["read_id"] == sample_leech_read.read_id for chunk in chunks)

    def test_extract_with_motif(self, sample_leech_read):
        """Test extracting chunks with motif filtering."""
        from leech.io.motif_search import get_motif_searcher

        # Use a motif that exists in the sequence
        motif = sample_leech_read.sequence[7:10]  # 3-base motif

        # Create motif searcher
        motif_searcher = get_motif_searcher(mode="bam")

        chunks = extract_training_chunks(
            sample_leech_read,
            motif_config=MotifConfig(motif=motif, motif_offset=1),
            chunk_config=ChunkConfig(),
            labeling=LabelConfig(label="uncharged", label_int=0),
            motif_searcher=motif_searcher,
        )

        # Should find at least one occurrence
        assert len(chunks) >= 1
        assert all(chunk["label"] == "uncharged" for chunk in chunks)
        assert all(chunk["label_int"] == 0 for chunk in chunks)

    def test_extract_with_nonexistent_motif(self, sample_leech_read):
        """Test extracting with a motif that doesn't exist."""
        from leech.io.motif_search import get_motif_searcher

        motif = "ZZZZZ"  # Invalid motif

        # Create motif searcher
        motif_searcher = get_motif_searcher(mode="bam")

        chunks = extract_training_chunks(
            sample_leech_read,
            motif_config=MotifConfig(motif=motif),
            chunk_config=ChunkConfig(),
            labeling=LabelConfig(label="uncharged", label_int=0),
            motif_searcher=motif_searcher,
        )

        # Should find no chunks
        assert len(chunks) == 0

    def test_chunk_structure(self, sample_leech_read):
        """Test that extracted chunks have correct structure."""
        chunks = extract_training_chunks(
            sample_leech_read,
            motif_config=MotifConfig(),
            chunk_config=ChunkConfig(),
            labeling=LabelConfig(label="charged", label_int=1),
        )

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
        assert orig["label"] == loaded["label"]  # String label
        assert orig["label_int"] == loaded["label_int"]  # Numeric label
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

        # Should return zero-padded chunk (remora convention)
        assert chunk is not None
        assert len(chunk["signal"]) == 10000
        # Signal should be mostly zeros (padded beyond boundaries)
        assert np.sum(chunk["signal"] == 0.0) > 0

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

    def test_find_motif_in_sequence_basic(self):
        """Test finding a motif in a reference sequence."""
        ref_seq = "ACGTACGTCCAACGT"
        motif = "CCA"

        positions = find_motif_in_sequence(ref_seq, motif)

        # Should find "CCA" at position 8
        assert 8 in positions
        assert len(positions) >= 1

    def test_find_motif_in_sequence_multiple(self):
        """Test finding multiple occurrences of motif."""
        ref_seq = "CCAACGTCCAACGTCCA"
        motif = "CCA"

        positions = find_motif_in_sequence(ref_seq, motif)

        # Should find at positions 0, 7, 14
        assert len(positions) == 3
        assert 0 in positions
        assert 7 in positions
        assert 14 in positions

    def test_find_motif_in_sequence_region(self):
        """Test finding motif only within specified region."""
        ref_seq = "ACGTCCAACGTCCA"
        motif = "CCA"
        ref_start = 5  # Start after first CCA
        ref_end = len(ref_seq)

        positions = find_motif_in_sequence(ref_seq, motif, ref_start, ref_end)

        # Should only find the second CCA at position 11
        assert 11 in positions
        assert 4 not in positions  # First CCA is before ref_start

    def test_find_motif_in_sequence_not_found(self):
        """Test when motif is not found."""
        ref_seq = "ACGTACGTACGT"
        motif = "TTT"

        positions = find_motif_in_sequence(ref_seq, motif)

        assert len(positions) == 0


class TestSplitChunksByRead:
    """Test split_chunks_by_read function for preventing data leakage."""

    def test_split_chunks_by_read_no_overlap(self):
        """Test that no read ID appears in multiple splits."""
        # Create chunks from multiple reads
        chunks = []
        for read_num in range(100):
            read_id = f"read_{read_num:03d}"
            # Each read has 5 chunks
            for chunk_num in range(5):
                chunks.append(
                    {
                        "read_id": read_id,
                        "signal": np.random.randn(400),
                        "sequence": "ACGT" * 3,
                        "dwell": np.random.randn(11),
                        "features": np.random.randn(3, 11),
                        "label": chunk_num % 2,
                        "base_idx": chunk_num,
                    }
                )

        train, val, test = split_chunks_by_read(chunks, train_frac=0.7, val_frac=0.15, seed=42)

        # Extract read IDs from each split
        train_reads = {c["read_id"] for c in train}
        val_reads = {c["read_id"] for c in val}
        test_reads = {c["read_id"] for c in test}

        # Verify no overlap
        assert len(train_reads & val_reads) == 0, "Train and val share read IDs"
        assert len(train_reads & test_reads) == 0, "Train and test share read IDs"
        assert len(val_reads & test_reads) == 0, "Val and test share read IDs"

    def test_split_chunks_by_read_all_chunks_assigned(self):
        """Test that all chunks are assigned to exactly one split."""
        chunks = []
        for read_num in range(50):
            read_id = f"read_{read_num:03d}"
            for chunk_num in range(3):
                chunks.append(
                    {
                        "read_id": read_id,
                        "signal": np.random.randn(400),
                        "sequence": "ACGT" * 3,
                        "dwell": np.random.randn(11),
                        "features": np.random.randn(3, 11),
                        "label": 0,
                        "base_idx": chunk_num,
                    }
                )

        train, val, test = split_chunks_by_read(chunks, train_frac=0.6, val_frac=0.2, seed=42)

        # Verify all chunks are assigned
        assert len(train) + len(val) + len(test) == len(chunks)

    def test_split_chunks_by_read_fractions(self):
        """Test that splits have approximately correct proportions."""
        chunks = []
        for read_num in range(100):
            read_id = f"read_{read_num:03d}"
            # Single chunk per read for easier proportion checking
            chunks.append(
                {
                    "read_id": read_id,
                    "signal": np.random.randn(400),
                    "sequence": "ACGT" * 3,
                    "dwell": np.random.randn(11),
                    "features": np.random.randn(3, 11),
                    "label": 0,
                    "base_idx": 0,
                }
            )

        train, val, test = split_chunks_by_read(chunks, train_frac=0.7, val_frac=0.15, seed=42)

        # With 100 reads: expect ~70 train, ~15 val, ~15 test
        assert 65 <= len(train) <= 75, f"Train has {len(train)}, expected ~70"
        assert 10 <= len(val) <= 20, f"Val has {len(val)}, expected ~15"
        assert 10 <= len(test) <= 20, f"Test has {len(test)}, expected ~15"

    def test_split_chunks_by_read_keeps_read_chunks_together(self):
        """Test that all chunks from same read go to same split."""
        chunks = []
        for read_num in range(30):
            read_id = f"read_{read_num:03d}"
            # Each read has 10 chunks
            for chunk_num in range(10):
                chunks.append(
                    {
                        "read_id": read_id,
                        "signal": np.random.randn(400),
                        "sequence": "ACGT" * 3,
                        "dwell": np.random.randn(11),
                        "features": np.random.randn(3, 11),
                        "label": 0,
                        "base_idx": chunk_num,
                    }
                )

        train, val, test = split_chunks_by_read(chunks, train_frac=0.6, val_frac=0.2, seed=42)

        # Count chunks per read in each split
        from collections import Counter

        train_read_counts = Counter(c["read_id"] for c in train)
        val_read_counts = Counter(c["read_id"] for c in val)
        test_read_counts = Counter(c["read_id"] for c in test)

        # Every read should have all 10 chunks in one split only
        for read_id, count in train_read_counts.items():
            assert count == 10, f"Read {read_id} has {count} chunks in train, expected 10"
            assert read_id not in val_read_counts
            assert read_id not in test_read_counts

        for read_id, count in val_read_counts.items():
            assert count == 10
            assert read_id not in train_read_counts
            assert read_id not in test_read_counts

        for read_id, count in test_read_counts.items():
            assert count == 10
            assert read_id not in train_read_counts
            assert read_id not in val_read_counts

    def test_split_chunks_by_read_reproducibility(self):
        """Test that same seed produces same split."""
        chunks = []
        for read_num in range(50):
            read_id = f"read_{read_num:03d}"
            chunks.append(
                {
                    "read_id": read_id,
                    "signal": np.random.randn(400),
                    "sequence": "ACGT" * 3,
                    "dwell": np.random.randn(11),
                    "features": np.random.randn(3, 11),
                    "label": 0,
                    "base_idx": 0,
                }
            )

        # Split twice with same seed
        train1, val1, test1 = split_chunks_by_read(chunks, train_frac=0.7, val_frac=0.15, seed=123)
        train2, val2, test2 = split_chunks_by_read(chunks, train_frac=0.7, val_frac=0.15, seed=123)

        # Should be identical
        train_ids1 = {c["read_id"] for c in train1}
        train_ids2 = {c["read_id"] for c in train2}
        assert train_ids1 == train_ids2

    def test_split_chunks_by_read_invalid_fractions(self):
        """Test that invalid fractions raise ValueError."""
        chunks = [{"read_id": "read_1", "signal": np.random.randn(400)}]

        with pytest.raises(ValueError, match="must be <= 1.0"):
            split_chunks_by_read(chunks, train_frac=0.8, val_frac=0.3, seed=42)


class TestCIGARMapping:
    """Test CIGAR string coordinate mapping."""

    def test_map_reference_to_query_all_matches(self):
        """Test mapping with simple match CIGAR (all M operations)."""
        # Create mock alignment with CIGAR: 20M
        import pysam

        # Create a properly constructed alignment
        aln = pysam.AlignedSegment()
        aln.query_sequence = "A" * 20  # 20 bases
        aln.reference_start = 100
        aln.cigartuples = [(0, 20)]  # 20M (BAM_CMATCH)
        # reference_end is automatically calculated from CIGAR: 100 + 20 = 120
        # query_alignment_start is automatically 0 (no soft clips)

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
        aln.query_sequence = "A" * 23  # 10 + 3 + 10 = 23 bases
        aln.reference_start = 100
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
        aln.query_sequence = "A" * 20  # 10 + 10 = 20 bases (deletion doesn't consume query)
        aln.reference_start = 100
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
        aln.query_sequence = "A" * 23  # 10 + 3 + 10 = 23 bases
        aln.reference_start = 100
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
        aln.query_sequence = "A" * 20  # 20 bases
        aln.reference_start = 100
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
        aln.query_sequence = "A" * 25  # 5 + 15 + 5 = 25 bases
        aln.reference_start = 100
        aln.cigartuples = [(4, 5), (0, 15), (4, 5)]  # 5S 15M 5S
        # reference_end = 100 + 15 = 115 (only 15M consumes reference)
        # query_alignment_start = 5 (automatically computed from 5S)

        # Map within the matched region
        result = map_reference_to_query_coords(aln, 105, 110, skip_indels=False)

        assert result is not None
        query_start, query_end = result
        # Query positions account for soft clip offset
        # CIGAR processes: 5S (query 0->5), then 15M (ref 100-114 maps to query 5-19)
        # At ref 105 (5 bases into match): query = 5 + 5 = 10
        # At ref 110 (10 bases into match): query = 5 + 10 = 15
        assert query_start == 10
        assert query_end == 15

    def test_map_reference_to_query_insertion_at_boundary(self):
        """Test insertion exactly at the start boundary of region."""
        import pysam

        # CIGAR: 10M 3I 10M - insertion at position 110
        aln = pysam.AlignedSegment()
        aln.query_sequence = "A" * 23
        aln.reference_start = 100
        aln.cigartuples = [(0, 10), (1, 3), (0, 10)]  # 10M 3I 10M

        # Map region that starts right at the insertion point
        result = map_reference_to_query_coords(aln, 110, 115, skip_indels=True)
        # Should return None because insertion is at the boundary
        assert result is None

        # With skip_indels=False, should work
        result = map_reference_to_query_coords(aln, 110, 115, skip_indels=False)
        assert result is not None

    def test_map_reference_to_query_deletion_at_boundary(self):
        """Test deletion exactly at the start boundary of region."""
        import pysam

        # CIGAR: 10M 3D 10M - deletion at position 110-112
        aln = pysam.AlignedSegment()
        aln.query_sequence = "A" * 20
        aln.reference_start = 100
        aln.cigartuples = [(0, 10), (2, 3), (0, 10)]  # 10M 3D 10M

        # Map region starting right after deletion
        result = map_reference_to_query_coords(aln, 113, 118, skip_indels=False)
        assert result is not None

        # Map region that includes the deletion
        result = map_reference_to_query_coords(aln, 110, 115, skip_indels=True)
        assert result is None

    def test_map_reference_to_query_multiple_indels(self):
        """Test multiple insertions and deletions in the region."""
        import pysam

        # CIGAR: 5M 2I 5M 2D 5M - complex pattern
        aln = pysam.AlignedSegment()
        aln.query_sequence = "A" * 12  # 5 + 2 + 5 = 12 (no deletion in query)
        aln.reference_start = 100
        aln.cigartuples = [(0, 5), (1, 2), (0, 5), (2, 2), (0, 5)]

        # Map region that includes both insertion and deletion
        result = map_reference_to_query_coords(aln, 102, 115, skip_indels=True)
        # Should return None due to indels
        assert result is None

    def test_map_reference_to_query_with_hard_clip(self):
        """Test that hard clips are handled correctly (ignored)."""
        import pysam

        # CIGAR: 5H 20M 5H - hard clips at both ends
        aln = pysam.AlignedSegment()
        aln.query_sequence = "A" * 20  # Hard clips not in sequence
        aln.reference_start = 100
        aln.cigartuples = [(5, 5), (0, 20), (5, 5)]  # 5H 20M 5H

        # Map normally - hard clips should be ignored
        result = map_reference_to_query_coords(aln, 105, 110, skip_indels=False)
        assert result is not None
        query_start, query_end = result
        # Hard clips don't affect query positions
        assert query_start == 5
        assert query_end == 10

    def test_map_reference_to_query_complex_with_soft_and_hard(self):
        """Test complex CIGAR with both soft and hard clips."""
        import pysam

        # CIGAR: 5H 3S 15M 3S 5H
        aln = pysam.AlignedSegment()
        aln.query_sequence = "A" * 21  # 3 + 15 + 3 (no hard clips in seq)
        aln.reference_start = 100
        aln.cigartuples = [(5, 5), (4, 3), (0, 15), (4, 3), (5, 5)]

        result = map_reference_to_query_coords(aln, 105, 110, skip_indels=False)
        assert result is not None
        query_start, query_end = result
        # 5H (ignored) + 3S (query 0->3) + 5M (query 3->8)
        assert query_start == 8  # 3 (soft) + 5 (offset)
        assert query_end == 13  # 3 (soft) + 10 (offset)

    def test_map_reference_to_query_exact_alignment_boundaries(self):
        """Test mapping at exact start and end of alignment."""
        import pysam

        # CIGAR: 20M
        aln = pysam.AlignedSegment()
        aln.query_sequence = "A" * 20
        aln.reference_start = 100
        aln.cigartuples = [(0, 20)]  # 20M

        # Map from exact start
        result = map_reference_to_query_coords(aln, 100, 105, skip_indels=False)
        assert result is not None
        assert result == (0, 5)

        # Map to exact end
        result = map_reference_to_query_coords(aln, 115, 120, skip_indels=False)
        assert result is not None
        assert result == (15, 20)

        # Map entire alignment
        result = map_reference_to_query_coords(aln, 100, 120, skip_indels=False)
        assert result is not None
        assert result == (0, 20)

    def test_map_reference_to_query_single_base(self):
        """Test mapping a single base."""
        import pysam

        aln = pysam.AlignedSegment()
        aln.query_sequence = "A" * 20
        aln.reference_start = 100
        aln.cigartuples = [(0, 20)]  # 20M

        # Map single base at position 105
        result = map_reference_to_query_coords(aln, 105, 106, skip_indels=False)
        assert result is not None
        assert result == (5, 6)


class TestSaveChunksUncompressed:
    """Test save_chunks with compressed=False."""

    def test_uncompressed_round_trip(self, sample_chunks, tmp_path):
        """Test that uncompressed save/load round-trips correctly."""
        output_file = tmp_path / "uncompressed.npz"

        save_chunks(sample_chunks, output_file, compressed=False)
        assert output_file.exists()

        loaded = load_chunks(output_file)
        assert len(loaded) == len(sample_chunks)

        # Verify values match
        for orig, loaded_chunk in zip(sample_chunks, loaded, strict=True):
            np.testing.assert_array_almost_equal(orig["signal"], loaded_chunk["signal"])
            assert orig["sequence"] == loaded_chunk["sequence"]
            assert orig["read_id"] == loaded_chunk["read_id"]

    def test_uncompressed_larger_file(self, sample_chunks, tmp_path):
        """Test that uncompressed files are >= compressed files."""
        compressed_file = tmp_path / "compressed.npz"
        uncompressed_file = tmp_path / "uncompressed.npz"

        save_chunks(sample_chunks, compressed_file, compressed=True)
        save_chunks(sample_chunks, uncompressed_file, compressed=False)

        assert uncompressed_file.stat().st_size >= compressed_file.stat().st_size


class TestMergeArraysBySplit:
    """Test _merge_arrays_by_split helper."""

    def _make_chunk_file(self, tmp_path, name, chunks):
        """Helper to save chunks to a file."""
        path = tmp_path / f"{name}.npz"
        save_chunks(chunks, path)
        return path

    def _make_chunks(self, read_prefix, label, n_reads=10, chunks_per_read=3):
        """Helper to create test chunks."""
        chunks = []
        for r in range(n_reads):
            for c in range(chunks_per_read):
                chunks.append(
                    {
                        "read_id": f"{read_prefix}_read_{r:03d}",
                        "signal": np.random.randn(400).astype(np.float32),
                        "sequence": "ACGT" * 3,
                        "dwell": np.random.randn(11).astype(np.float32),
                        "features": np.random.randn(3, 11).astype(np.float32),
                        "label": label,
                        "label_int": 0,
                        "base_idx": c,
                        "feature_start": -5,
                        "feature_end": 5,
                        "source_group": "",
                    }
                )
        return chunks

    def test_basic_merge(self, tmp_path):
        """Test basic array-level merge without overrides."""
        from leech.splitting.splitter import _merge_arrays_by_split

        chunks_a = self._make_chunks("a", "Ala", n_reads=10)
        chunks_b = self._make_chunks("b", "Gly", n_reads=10)

        path_a = self._make_chunk_file(tmp_path, "ala", chunks_a)
        path_b = self._make_chunk_file(tmp_path, "gly", chunks_b)

        # Collect all read IDs and split
        all_reads = sorted({c["read_id"] for c in chunks_a + chunks_b})
        n = len(all_reads)
        train_ids = set(all_reads[: int(n * 0.7)])
        val_ids = set(all_reads[int(n * 0.7) : int(n * 0.85)])
        test_ids = set(all_reads[int(n * 0.85) :])

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        counts = _merge_arrays_by_split(
            input_paths=[path_a, path_b],
            split_read_ids={"train": train_ids, "val": val_ids, "test": test_ids},
            output_paths={
                "train": out_dir / "train.npz",
                "val": out_dir / "val.npz",
                "test": out_dir / "test.npz",
            },
        )

        total = counts["train"] + counts["val"] + counts["test"]
        assert total == len(chunks_a) + len(chunks_b)

        # Verify output files are loadable
        loaded_train = load_chunks(out_dir / "train.npz")
        assert len(loaded_train) == counts["train"]

    def test_label_overrides(self, tmp_path):
        """Test that label_overrides correctly relabels arrays."""
        from leech.splitting.splitter import _merge_arrays_by_split

        chunks_a = self._make_chunks("a", "Ala", n_reads=5)
        chunks_b = self._make_chunks("b", "Gly", n_reads=5)

        path_a = self._make_chunk_file(tmp_path, "ala", chunks_a)
        path_b = self._make_chunk_file(tmp_path, "gly", chunks_b)

        all_reads = sorted({c["read_id"] for c in chunks_a + chunks_b})
        train_ids = set(all_reads)  # Put all in train for simplicity

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        _merge_arrays_by_split(
            input_paths=[path_a, path_b],
            split_read_ids={"train": train_ids},
            output_paths={"train": out_dir / "train.npz"},
            label_overrides={
                path_a: (0, "Ala"),
                path_b: (1, "Gly"),
            },
        )

        # Load and check labels
        loaded = load_chunks(out_dir / "train.npz")
        labels_int = {c["label_int"] for c in loaded}
        assert 0 in labels_int
        assert 1 in labels_int

        # Check that Ala chunks got label_int=0 and Gly got label_int=1
        for c in loaded:
            if "a_read" in c["read_id"]:
                assert c["label_int"] == 0
            elif "b_read" in c["read_id"]:
                assert c["label_int"] == 1

    def test_source_group_overrides(self, tmp_path):
        """Test that source_group_overrides correctly sets source groups."""
        from leech.splitting.splitter import _merge_arrays_by_split

        chunks_a = self._make_chunks("a", "Ala", n_reads=5)
        path_a = self._make_chunk_file(tmp_path, "ala", chunks_a)

        all_reads = {c["read_id"] for c in chunks_a}
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        _merge_arrays_by_split(
            input_paths=[path_a],
            split_read_ids={"train": all_reads},
            output_paths={"train": out_dir / "train.npz"},
            source_group_overrides={path_a: "my_source"},
        )

        loaded = load_chunks(out_dir / "train.npz")
        for c in loaded:
            assert c["source_group"] == "my_source"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
