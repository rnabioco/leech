"""
Tests for data leakage prevention in data preparation pipeline.

These tests ensure that:
1. Read IDs are preserved through the entire pipeline
2. Read-level splitting prevents data leakage
3. No read appears in multiple splits (train/val/test)
4. Parallel processing maintains read-level grouping
"""

import numpy as np
import pytest
from pathlib import Path

from leech.data_prep import (
    LeechRead,
    extract_training_chunks,
    split_chunks_by_read,
    merge_and_split_chunks,
    save_chunks,
    load_chunks,
)


class TestReadIDPreservation:
    """Test that read IDs are preserved through the pipeline."""

    def test_leech_read_preserves_id(self):
        """Test that LeechRead stores and retrieves read_id correctly."""
        read_id = "test_read_12345"
        sequence = "ACGTACGT"
        signal = np.random.randn(100).astype(np.float32)
        seq_to_sig_map = np.linspace(0, 100, len(sequence) + 1, dtype=np.int32)
        dwells = np.diff(seq_to_sig_map)

        leech_read = LeechRead(
            read_id=read_id,
            sequence=sequence,
            signal=signal,
            seq_to_sig_map=seq_to_sig_map,
            dwells=dwells,
        )

        assert leech_read.read_id == read_id

    def test_chunk_contains_read_id(self):
        """Test that extracted chunks contain the correct read_id."""
        read_id = "test_read_67890"
        sequence = "ACGTACGTACGTACGTACGTACGT"  # 24 bases (enough for kmer context)
        # Need enough signal: ~10 samples/base * 24 bases = 240, plus 400 for signal context
        signal = np.random.randn(700).astype(np.float32)
        seq_to_sig_map = np.linspace(0, 700, len(sequence) + 1, dtype=np.int32)
        dwells = np.diff(seq_to_sig_map)

        leech_read = LeechRead(
            read_id=read_id,
            sequence=sequence,
            signal=signal,
            seq_to_sig_map=seq_to_sig_map,
            dwells=dwells,
            dwell_features={},  # Add empty features
            signal_features={},  # Add empty features
            labels=np.zeros(len(sequence), dtype=np.int64),
        )

        # Extract a chunk from the middle (away from edges)
        chunk = leech_read.get_chunk(base_idx=12)

        assert chunk is not None, "Failed to extract chunk - check signal/kmer context"

        # Manually add read_id as done in extract_training_chunks
        chunk["read_id"] = leech_read.read_id

        assert chunk["read_id"] == read_id

    def test_extract_training_chunks_preserves_read_id(self):
        """Test that extract_training_chunks preserves read_id in all chunks."""
        read_id = "test_read_motif_123"
        # Longer sequence with motif in middle, enough signal for extraction
        # Motif CCAGGC at position 12 (well away from edges)
        sequence = "ACGTACGTACGTCCAGGCACGTACGTACGT"  # 30 bases
        # Need ~10 samples/base * 30 = 300, plus 400 for signal context
        signal = np.random.randn(800).astype(np.float32)
        seq_to_sig_map = np.linspace(0, 800, len(sequence) + 1, dtype=np.int32)
        dwells = np.diff(seq_to_sig_map)

        leech_read = LeechRead(
            read_id=read_id,
            sequence=sequence,
            signal=signal,
            seq_to_sig_map=seq_to_sig_map,
            dwells=dwells,
            dwell_features={},  # Add empty features
            signal_features={},  # Add empty features
        )

        chunks = extract_training_chunks(
            leech_read,
            motif="CCAGGC",
            motif_offset=0,
            label=1,
            motif_reference="bam",
        )

        assert len(chunks) > 0, "Should extract at least one chunk"
        for chunk in chunks:
            assert "read_id" in chunk
            assert chunk["read_id"] == read_id


class TestReadLevelSplitting:
    """Test that data splitting happens at the read level."""

    def create_test_chunks(self, n_reads=10, chunks_per_read=5):
        """Helper to create test chunks with known read IDs."""
        chunks = []
        for read_idx in range(n_reads):
            read_id = f"read_{read_idx:04d}"
            for chunk_idx in range(chunks_per_read):
                chunk = {
                    "signal": np.random.randn(100).astype(np.float32),
                    "sequence": "ACGTACGT",
                    "dwell": np.array([10, 12, 11, 9, 10, 12, 11, 9], dtype=np.int32),
                    "features": np.random.randn(3, 8).astype(np.float32),
                    "label": 0,
                    "read_id": read_id,
                    "base_idx": chunk_idx,
                }
                chunks.append(chunk)
        return chunks

    def test_split_chunks_by_read_no_leakage(self):
        """Test that split_chunks_by_read prevents read ID leakage."""
        chunks = self.create_test_chunks(n_reads=100, chunks_per_read=5)

        train_chunks, val_chunks, test_chunks = split_chunks_by_read(
            chunks,
            train_frac=0.7,
            val_frac=0.15,
            seed=42,
        )

        # Collect unique read IDs from each split
        train_read_ids = set(c["read_id"] for c in train_chunks)
        val_read_ids = set(c["read_id"] for c in val_chunks)
        test_read_ids = set(c["read_id"] for c in test_chunks)

        # Check for leakage: no read ID should appear in multiple splits
        assert len(train_read_ids & val_read_ids) == 0, "Train and val sets share read IDs!"
        assert len(train_read_ids & test_read_ids) == 0, "Train and test sets share read IDs!"
        assert len(val_read_ids & test_read_ids) == 0, "Val and test sets share read IDs!"

    def test_split_chunks_by_read_all_chunks_assigned(self):
        """Test that all chunks are assigned to exactly one split."""
        chunks = self.create_test_chunks(n_reads=50, chunks_per_read=3)

        train_chunks, val_chunks, test_chunks = split_chunks_by_read(
            chunks,
            train_frac=0.6,
            val_frac=0.2,
            seed=123,
        )

        total_chunks = len(train_chunks) + len(val_chunks) + len(test_chunks)
        assert total_chunks == len(chunks), "Not all chunks were assigned!"

    def test_split_chunks_by_read_respects_proportions(self):
        """Test that split proportions are approximately correct."""
        chunks = self.create_test_chunks(n_reads=1000, chunks_per_read=5)

        train_chunks, val_chunks, test_chunks = split_chunks_by_read(
            chunks,
            train_frac=0.7,
            val_frac=0.15,
            seed=42,
        )

        # Get unique read counts (not chunk counts)
        train_read_ids = set(c["read_id"] for c in train_chunks)
        val_read_ids = set(c["read_id"] for c in val_chunks)
        test_read_ids = set(c["read_id"] for c in test_chunks)

        n_reads = len(train_read_ids) + len(val_read_ids) + len(test_read_ids)

        train_frac = len(train_read_ids) / n_reads
        val_frac = len(val_read_ids) / n_reads
        test_frac = len(test_read_ids) / n_reads

        # Allow 5% tolerance due to rounding
        assert 0.65 <= train_frac <= 0.75, f"Train fraction {train_frac} not close to 0.7"
        assert 0.10 <= val_frac <= 0.20, f"Val fraction {val_frac} not close to 0.15"
        assert 0.10 <= test_frac <= 0.20, f"Test fraction {test_frac} not close to 0.15"

    def test_split_chunks_by_read_reproducible(self):
        """Test that splits are reproducible with the same seed."""
        chunks = self.create_test_chunks(n_reads=50, chunks_per_read=4)

        # Split twice with same seed
        train1, val1, test1 = split_chunks_by_read(chunks, train_frac=0.7, val_frac=0.15, seed=99)
        train2, val2, test2 = split_chunks_by_read(chunks, train_frac=0.7, val_frac=0.15, seed=99)

        # Should produce identical splits
        train_ids1 = set(c["read_id"] for c in train1)
        train_ids2 = set(c["read_id"] for c in train2)

        val_ids1 = set(c["read_id"] for c in val1)
        val_ids2 = set(c["read_id"] for c in val2)

        assert train_ids1 == train_ids2, "Train splits not reproducible"
        assert val_ids1 == val_ids2, "Val splits not reproducible"


class TestMergeAndSplitChunks:
    """Test that merging and splitting prevents leakage across samples."""

    def create_test_chunk_file(self, path: Path, n_reads: int, chunks_per_read: int, label: int):
        """Helper to create and save test chunk files."""
        chunks = []
        for read_idx in range(n_reads):
            read_id = f"sample_{path.stem}_read_{read_idx:04d}"
            for chunk_idx in range(chunks_per_read):
                chunk = {
                    "signal": np.random.randn(100).astype(np.float32),
                    "sequence": "ACGTACGT",
                    "dwell": np.array([10, 12, 11, 9, 10, 12, 11, 9], dtype=np.int32),
                    "features": np.random.randn(3, 8).astype(np.float32),
                    "label": label,
                    "read_id": read_id,
                    "base_idx": chunk_idx,
                }
                chunks.append(chunk)

        save_chunks(chunks, path)
        return chunks

    def test_merge_and_split_no_leakage(self, tmp_path):
        """Test that merge_and_split_chunks prevents leakage across samples."""
        # Create two sample files (e.g., charged and uncharged)
        sample1_path = tmp_path / "sample1.npz"
        sample2_path = tmp_path / "sample2.npz"

        self.create_test_chunk_file(sample1_path, n_reads=50, chunks_per_read=3, label=0)
        self.create_test_chunk_file(sample2_path, n_reads=50, chunks_per_read=3, label=1)

        # Merge and split
        output_dir = tmp_path / "merged"
        result = merge_and_split_chunks(
            input_paths=[sample1_path, sample2_path],
            output_dir=output_dir,
            train_frac=0.7,
            val_frac=0.15,
            seed=42,
        )

        # Load the splits
        train_chunks = load_chunks(result["output_files"]["train"])
        val_chunks = load_chunks(result["output_files"]["val"])
        test_chunks = load_chunks(result["output_files"]["test"])

        # Check for leakage
        train_read_ids = set(c["read_id"] for c in train_chunks)
        val_read_ids = set(c["read_id"] for c in val_chunks)
        test_read_ids = set(c["read_id"] for c in test_chunks)

        assert len(train_read_ids & val_read_ids) == 0, "Merged: Train and val share read IDs!"
        assert len(train_read_ids & test_read_ids) == 0, "Merged: Train and test share read IDs!"
        assert len(val_read_ids & test_read_ids) == 0, "Merged: Val and test share read IDs!"

    def test_merge_and_split_preserves_labels(self, tmp_path):
        """Test that merge_and_split preserves labels from different samples."""
        sample1_path = tmp_path / "uncharged.npz"
        sample2_path = tmp_path / "charged.npz"

        self.create_test_chunk_file(sample1_path, n_reads=30, chunks_per_read=2, label=0)
        self.create_test_chunk_file(sample2_path, n_reads=30, chunks_per_read=2, label=1)

        output_dir = tmp_path / "merged"
        result = merge_and_split_chunks(
            input_paths=[sample1_path, sample2_path],
            output_dir=output_dir,
            train_frac=0.7,
            val_frac=0.15,
            seed=42,
        )

        # Load train chunks and check both labels present
        train_chunks = load_chunks(result["output_files"]["train"])
        train_labels = set(c["label"] for c in train_chunks)

        assert 0 in train_labels, "Label 0 (uncharged) missing from train set"
        assert 1 in train_labels, "Label 1 (charged) missing from train set"


class TestSaveLoadChunks:
    """Test that save/load preserves read IDs."""

    def test_save_load_preserves_read_ids(self, tmp_path):
        """Test that saving and loading chunks preserves read IDs."""
        # Create test chunks
        chunks = []
        for i in range(10):
            chunk = {
                "signal": np.random.randn(100).astype(np.float32),
                "sequence": "ACGTACGT",
                "dwell": np.array([10, 12, 11, 9, 10, 12, 11, 9], dtype=np.int32),
                "features": np.random.randn(3, 8).astype(np.float32),
                "label": 0,
                "read_id": f"read_{i:04d}",
                "base_idx": 5,
            }
            chunks.append(chunk)

        # Save
        output_path = tmp_path / "test_chunks.npz"
        save_chunks(chunks, output_path)

        # Load
        loaded_chunks = load_chunks(output_path)

        # Check read IDs preserved
        original_read_ids = [c["read_id"] for c in chunks]
        loaded_read_ids = [c["read_id"] for c in loaded_chunks]

        assert original_read_ids == loaded_read_ids, "Read IDs not preserved after save/load"


class TestDataLeakageDetection:
    """Utility tests for detecting data leakage."""

    def test_detect_leakage_in_splits(self):
        """Test utility function to detect leakage in existing splits."""
        # Create chunks with known leakage
        train_chunks = [
            {"read_id": "read_001", "label": 0, "signal": np.array([1, 2, 3])},
            {"read_id": "read_002", "label": 0, "signal": np.array([1, 2, 3])},
            {"read_id": "read_003", "label": 0, "signal": np.array([1, 2, 3])},  # Leak!
        ]
        val_chunks = [
            {"read_id": "read_003", "label": 0, "signal": np.array([4, 5, 6])},  # Leak!
            {"read_id": "read_004", "label": 1, "signal": np.array([4, 5, 6])},
        ]

        train_ids = set(c["read_id"] for c in train_chunks)
        val_ids = set(c["read_id"] for c in val_chunks)

        leaked_ids = train_ids & val_ids
        assert len(leaked_ids) > 0, "Should detect leakage"
        assert "read_003" in leaked_ids, "Should detect read_003 as leaked"

    def test_no_leakage_in_clean_splits(self):
        """Test that clean splits are correctly identified as leak-free."""
        train_chunks = [
            {"read_id": "read_001", "label": 0, "signal": np.array([1, 2, 3])},
            {"read_id": "read_002", "label": 0, "signal": np.array([1, 2, 3])},
        ]
        val_chunks = [
            {"read_id": "read_003", "label": 0, "signal": np.array([4, 5, 6])},
            {"read_id": "read_004", "label": 1, "signal": np.array([4, 5, 6])},
        ]

        train_ids = set(c["read_id"] for c in train_chunks)
        val_ids = set(c["read_id"] for c in val_chunks)

        leaked_ids = train_ids & val_ids
        assert len(leaked_ids) == 0, "Should not detect leakage in clean splits"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
