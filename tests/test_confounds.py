"""Tests for confound mappings, in particular the per-AA disc_base extractor."""

from pathlib import Path

import pytest

from leech.confounds import (
    DISC_BASE_TO_INT,
    build_confound_map,
    extract_disc_bases_from_fasta,
)


@pytest.fixture
def fasta_path(tmp_path: Path) -> Path:
    """Three tRNAs where one AA (Arg) has two isoacceptors with conflicting disc_bases."""
    path = tmp_path / "test.fa"
    path.write_text(
        ">tRNA-Ala-GGC-1-1\n"
        "AAAAAAAACCCAGGCTTT\n"  # disc_base 'C' (one nt before CCAGGC)
        ">tRNA-Arg-ACG-1-1\n"
        "AAAAAAAAGCCAGGCTTT\n"  # disc_base 'G'
        ">tRNA-Arg-CCG-1-1\n"
        "AAAAAAAAGCCAGGCTTT\n"  # disc_base 'G' (same so Arg mode is unambiguous)
        ">tRNA-Cys-GCA-1-1\n"
        "AAAAAAAATCCAGGCTTT\n"  # disc_base 'T'
    )
    return path


def test_extract_disc_bases_basic(fasta_path: Path) -> None:
    disc_map = extract_disc_bases_from_fasta(fasta_path)
    assert disc_map == {"Ala": "C", "Arg": "G", "Cys": "T"}


def test_extract_disc_bases_modal_vote(tmp_path: Path) -> None:
    """When isoacceptors of one AA disagree, the modal base wins."""
    path = tmp_path / "mixed.fa"
    path.write_text(
        ">tRNA-Arg-ACG-1-1\n"
        "AAAAAAAACCCAGGC\n"  # C
        ">tRNA-Arg-CCG-1-1\n"
        "AAAAAAAATCCAGGC\n"  # T
        ">tRNA-Arg-CCT-1-1\n"
        "AAAAAAAACCCAGGC\n"  # C (modal)
    )
    disc_map = extract_disc_bases_from_fasta(path)
    assert disc_map == {"Arg": "C"}


def test_extract_disc_bases_rna_to_dna(tmp_path: Path) -> None:
    """U is silently mapped to T so RNA-style FASTAs work."""
    path = tmp_path / "rna.fa"
    path.write_text(">tRNA-Phe-AAA-1-1\nAAAAAAAAUCCAGGC\n")
    disc_map = extract_disc_bases_from_fasta(path)
    assert disc_map == {"Phe": "T"}


def test_extract_disc_bases_missing_motif(tmp_path: Path) -> None:
    path = tmp_path / "no_motif.fa"
    path.write_text(">tRNA-Lys-CTT-1-1\nAAAAAAAAGGGG\n")
    with pytest.raises(ValueError, match="not found"):
        extract_disc_bases_from_fasta(path)


def test_extract_disc_bases_motif_at_start(tmp_path: Path) -> None:
    path = tmp_path / "start.fa"
    path.write_text(">tRNA-Lys-CTT-1-1\nCCAGGCAAA\n")
    with pytest.raises(ValueError, match="position 0"):
        extract_disc_bases_from_fasta(path)


def test_extract_disc_bases_bad_header(tmp_path: Path) -> None:
    path = tmp_path / "bad.fa"
    path.write_text(">NotAtRNA\nAAAACCCAGGC\n")
    with pytest.raises(ValueError, match="Cannot parse"):
        extract_disc_bases_from_fasta(path)


def test_extract_disc_bases_feeds_build_confound_map(fasta_path: Path) -> None:
    """End-to-end: extracted disc_map feeds build_confound_map → int classes."""
    disc_map = extract_disc_bases_from_fasta(fasta_path)
    label_map = {"Ala": 0, "Arg": 1, "Cys": 2}
    confound_map = build_confound_map(label_map, disc_map)
    assert confound_map == {
        0: DISC_BASE_TO_INT["C"],
        1: DISC_BASE_TO_INT["G"],
        2: DISC_BASE_TO_INT["T"],
    }
