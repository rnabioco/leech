"""Tests for comparison spec parsing and batch processing."""

import json

import numpy as np
import pytest

from leech.chunking import save_chunks
from leech.splitting import merge_and_split_chunks, parse_comparison_spec, process_comparison_spec


@pytest.fixture
def sample_tsv(tmp_path):
    """Create a sample comparison spec TSV file."""
    tsv_path = tmp_path / "comparisons.tsv"
    with open(tsv_path, "w") as f:
        f.write("basic\tLys,Arg\tacidic\tAsp,Glu\n")
        f.write("polar\tSer,Thr\tnonpolar\tAla,Val\n")
        f.write("aromatic\tPhe,Tyr,Trp\taliphatic\tGly,Ala,Val\n")
    return tsv_path


@pytest.fixture
def sample_chunks(tmp_path):
    """Create sample chunk files for testing."""
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()

    # Create chunks for different labels
    labels = ["Lys", "Arg", "Asp", "Glu", "Ser", "Thr", "Ala", "Val"]
    chunk_files = []

    for label in labels:
        # Create dummy chunks
        chunks = []
        for i in range(10):
            chunk = {
                "signal": np.random.randn(100),
                "sequence": "ACGT" * 10,
                "dwell": np.random.randint(1, 10, size=40),
                "features": np.random.randn(5, 40),
                "read_id": f"{label}_read_{i}",
                "base_idx": i * 10,
                "label": label,
                "label_int": None,
            }
            chunks.append(chunk)

        chunk_file = chunk_dir / f"{label}_chunks.npz"
        save_chunks(chunks, chunk_file)
        chunk_files.append(chunk_file)

    return chunk_files


def test_parse_comparison_spec(sample_tsv):
    """Test parsing of comparison spec TSV file."""
    comparisons = parse_comparison_spec(sample_tsv)

    assert len(comparisons) == 3

    # Check first comparison
    meta1, labels1, meta2, labels2 = comparisons[0]
    assert meta1 == "basic"
    assert labels1 == ["Lys", "Arg"]
    assert meta2 == "acidic"
    assert labels2 == ["Asp", "Glu"]

    # Check second comparison
    meta1, labels1, meta2, labels2 = comparisons[1]
    assert meta1 == "polar"
    assert labels1 == ["Ser", "Thr"]
    assert meta2 == "nonpolar"
    assert labels2 == ["Ala", "Val"]

    # Check third comparison (with 3 labels in one group)
    meta1, labels1, meta2, labels2 = comparisons[2]
    assert meta1 == "aromatic"
    assert labels1 == ["Phe", "Tyr", "Trp"]
    assert meta2 == "aliphatic"
    assert labels2 == ["Gly", "Ala", "Val"]


def test_parse_comparison_spec_with_comments(tmp_path):
    """Test that comments and empty lines are skipped."""
    tsv_path = tmp_path / "comparisons_with_comments.tsv"
    with open(tsv_path, "w") as f:
        f.write("# This is a comment\n")
        f.write("\n")
        f.write("basic\tLys,Arg\tacidic\tAsp,Glu\n")
        f.write("# Another comment\n")
        f.write("polar\tSer,Thr\tnonpolar\tAla,Val\n")

    comparisons = parse_comparison_spec(tsv_path)
    assert len(comparisons) == 2


def test_parse_comparison_spec_invalid_format(tmp_path):
    """Test that invalid TSV format raises error."""
    tsv_path = tmp_path / "invalid.tsv"
    with open(tsv_path, "w") as f:
        f.write("basic\tLys,Arg\tacidic\n")  # Only 3 columns

    with pytest.raises(ValueError, match="expected 4 tab-separated columns"):
        parse_comparison_spec(tsv_path)


def test_merge_and_split_with_label_lists(sample_chunks, tmp_path):
    """Test merge_and_split_chunks with lists of labels for grouping."""
    output_dir = tmp_path / "merged"

    # Test with multi-label groups (basic vs acidic)
    result = merge_and_split_chunks(
        input_paths=sample_chunks,
        output_dir=output_dir,
        train_frac=0.7,
        val_frac=0.15,
        seed=42,
        relabel_pairwise=(["Lys", "Arg"], ["Asp", "Glu"]),
    )

    assert isinstance(result, dict)
    assert result["n_train"] > 0
    assert result["n_val"] > 0
    assert result["n_test"] > 0

    # Load and check train chunks
    with np.load(output_dir / "train.npz", allow_pickle=True) as data:
        labels_int = data["labels_int"]
        # Should have both 0s (basic) and 1s (acidic)
        assert 0 in labels_int
        assert 1 in labels_int


def test_merge_and_split_with_single_labels(sample_chunks, tmp_path):
    """Test merge_and_split_chunks with single labels (backward compatible)."""
    output_dir = tmp_path / "merged_single"

    result = merge_and_split_chunks(
        input_paths=sample_chunks,
        output_dir=output_dir,
        train_frac=0.7,
        val_frac=0.15,
        seed=42,
        relabel_pairwise=("Lys", "Arg"),
    )

    assert isinstance(result, dict)
    assert result["n_train"] > 0

    # Load and check train chunks
    with np.load(output_dir / "train.npz", allow_pickle=True) as data:
        labels_int = data["labels_int"]
        # Should have both 0s (Lys) and 1s (Arg)
        assert 0 in labels_int
        assert 1 in labels_int


def test_process_comparison_spec(sample_chunks, sample_tsv, tmp_path):
    """Test batch processing of multiple comparisons from spec file."""
    output_dir = tmp_path / "comparisons"

    # Extract chunk directories from chunk files
    chunk_dirs = [tmp_path / "chunks"]

    result = process_comparison_spec(
        chunk_dirs=chunk_dirs,
        comparison_spec=sample_tsv,
        output_dir=output_dir,
        train_frac=0.7,
        val_frac=0.15,
        seed=42,
    )

    assert result["n_comparisons"] == 3
    assert "basic_vs_acidic" in result["comparisons"]
    assert "polar_vs_nonpolar" in result["comparisons"]
    assert "aromatic_vs_aliphatic" in result["comparisons"]

    # Check that output directories were created
    assert (output_dir / "basic_vs_acidic").exists()
    assert (output_dir / "polar_vs_nonpolar").exists()
    assert (output_dir / "aromatic_vs_aliphatic").exists()

    # Check that splits were saved
    assert (output_dir / "basic_vs_acidic" / "train.npz").exists()
    assert (output_dir / "basic_vs_acidic" / "val.npz").exists()
    assert (output_dir / "basic_vs_acidic" / "test.npz").exists()

    # Check metadata file
    metadata_file = output_dir / "basic_vs_acidic" / "metadata.json"
    assert metadata_file.exists()

    with open(metadata_file) as f:
        metadata = json.load(f)

    assert metadata["comparison"] == "basic_vs_acidic"
    assert metadata["group_0"]["meta_label"] == "basic"
    assert metadata["group_0"]["labels"] == ["Lys", "Arg"]
    assert metadata["group_1"]["meta_label"] == "acidic"
    assert metadata["group_1"]["labels"] == ["Asp", "Glu"]
    assert metadata["seed"] == 42


def test_process_comparison_spec_empty_chunks_dir(sample_tsv, tmp_path):
    """Test that process_comparison_spec handles missing chunk dirs."""
    output_dir = tmp_path / "comparisons"
    nonexistent_dir = tmp_path / "nonexistent"

    with pytest.raises(ValueError, match="No .npz chunk files found"):
        process_comparison_spec(
            chunk_dirs=[nonexistent_dir],
            comparison_spec=sample_tsv,
            output_dir=output_dir,
            seed=42,
        )
