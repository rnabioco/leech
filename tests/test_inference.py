"""
Tests for inference motif search logic.

Validates that inference uses reference-based motif search (ReferenceMotifSearcher)
instead of basecalled search (find_motif_in_sequence), which avoids missed motifs
caused by basecalling errors at modification sites.
"""

import pysam
import pytest

from leech.configs import InferenceConfig, MotifConfig, SignalConfig
from leech.io.bam_reader import MockAlignment
from leech.io.motif_search import (
    BasecalledMotifSearcher,
    ReferenceMotifSearcher,
    find_motif_in_sequence,
    get_motif_searcher,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# pysam.AlignedSegment requires a BAM header to resolve reference_name from
# reference_id. We create a header containing our synthetic reference so that
# tests using real pysam objects work correctly.
_BAM_HEADER = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "tRNA-Ala", "LN": 40}]})


def _make_pysam_alignment(
    query_name: str,
    query_sequence: str,
    reference_name: str,
    reference_start: int,
    cigartuples: list[tuple[int, int]],
) -> pysam.AlignedSegment:
    """Create a minimal pysam.AlignedSegment for motif search tests."""
    aln = pysam.AlignedSegment(header=_BAM_HEADER)
    aln.query_name = query_name
    aln.query_sequence = query_sequence
    aln.reference_id = _BAM_HEADER.get_tid(reference_name)
    aln.reference_start = reference_start
    aln.cigartuples = cigartuples
    return aln


def _make_mock_alignment(
    reference_name: str,
    reference_start: int,
    reference_end: int,
    cigartuples: list[tuple[int, int]],
    is_reverse: bool = False,
) -> MockAlignment:
    """Create a MockAlignment directly (bypass ReadInfo/pysam dependency)."""
    mock = MockAlignment.__new__(MockAlignment)
    mock.reference_name = reference_name
    mock.reference_start = reference_start
    mock.reference_end = reference_end
    mock.cigartuples = cigartuples
    mock.is_reverse = is_reverse
    return mock


# ---------------------------------------------------------------------------
# Reference data — 40bp synthetic tRNA reference with CCAGGC at position 20
# ---------------------------------------------------------------------------

REFERENCE_NAME = "tRNA-Ala"
REFERENCE_SEQ = "ACGTACGTACGTACGTACGTCCAGGCTTACGTACGTACGT"
#                0         10        20        30
#                                    ^^^^^^ CCAGGC at [20, 26)
REFERENCE_SEQUENCES = {REFERENCE_NAME: REFERENCE_SEQ}


# ---------------------------------------------------------------------------
# Tests: get_motif_searcher factory
# ---------------------------------------------------------------------------


class TestGetMotifSearcher:
    """Tests for the motif searcher factory function."""

    def test_fasta_mode_returns_reference_searcher(self):
        searcher = get_motif_searcher("fasta", reference_sequences=REFERENCE_SEQUENCES)
        assert isinstance(searcher, ReferenceMotifSearcher)

    def test_bam_mode_returns_basecalled_searcher(self):
        searcher = get_motif_searcher("bam")
        assert isinstance(searcher, BasecalledMotifSearcher)

    def test_fasta_mode_without_refs_raises(self):
        with pytest.raises(ValueError, match="reference_sequences required"):
            get_motif_searcher("fasta", reference_sequences=None)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid motif search mode"):
            get_motif_searcher("invalid")


# ---------------------------------------------------------------------------
# Tests: reference vs basecalled motif search
# ---------------------------------------------------------------------------


class TestReferenceVsBasecalledSearch:
    """Core bug test: basecalled search misses motifs with basecalling errors."""

    def test_basecalled_finds_motif_when_sequence_correct(self):
        """Basecalled search works when the basecalled sequence is correct."""
        # Basecalled sequence matches reference exactly
        sequence = "ACGTACGTACGTACGTACGTCCAGGCTTACGTACGTACGT"
        positions = find_motif_in_sequence(sequence, "CCAGGC")
        assert positions == [20]

    def test_basecalled_misses_motif_with_basecalling_error(self):
        """Basecalled search MISSES motif when basecalling error corrupts it."""
        # Basecalling error: CCAGGC → CCATGC (G→T substitution)
        sequence = "ACGTACGTACGTACGTACGTCCATGCTTACGTACGTACGT"
        positions = find_motif_in_sequence(sequence, "CCAGGC")
        assert positions == []  # Missed!

    def test_reference_finds_motif_despite_basecalling_error(self):
        """Reference search finds motif even when basecalled sequence is wrong."""
        # Basecalled sequence has error at motif, but reference is correct
        corrupted_sequence = "ACGTACGTACGTACGTACGTCCATGCTTACGTACGTACGT"

        aln = _make_pysam_alignment(
            query_name="read_001",
            query_sequence=corrupted_sequence,
            reference_name=REFERENCE_NAME,
            reference_start=0,
            cigartuples=[(0, 40)],  # Perfect alignment (40M)
        )

        searcher = ReferenceMotifSearcher(
            reference_sequences=REFERENCE_SEQUENCES,
            skip_indels=True,
            anchor="reference",
        )
        positions = searcher.find_motif_positions("read_001", corrupted_sequence, aln, "CCAGGC")
        assert positions == [20]

    def test_reference_search_with_mock_alignment(self):
        """ReferenceMotifSearcher works with MockAlignment (used in parallel worker)."""
        mock_aln = _make_mock_alignment(
            reference_name=REFERENCE_NAME,
            reference_start=0,
            reference_end=40,
            cigartuples=[(0, 40)],
        )

        searcher = ReferenceMotifSearcher(
            reference_sequences=REFERENCE_SEQUENCES,
            skip_indels=True,
            anchor="reference",
        )
        positions = searcher.find_motif_positions("read_001", "X" * 40, mock_aln, "CCAGGC")
        assert positions == [20]

    def test_reference_search_basecall_anchor(self):
        """With anchor='basecall', returns query coordinates instead of ref-relative."""
        sequence = "ACGTACGTACGTACGTACGTCCATGCTTACGTACGTACGT"

        aln = _make_pysam_alignment(
            query_name="read_001",
            query_sequence=sequence,
            reference_name=REFERENCE_NAME,
            reference_start=0,
            cigartuples=[(0, 40)],
        )

        searcher = ReferenceMotifSearcher(
            reference_sequences=REFERENCE_SEQUENCES,
            skip_indels=True,
            anchor="basecall",
        )
        positions = searcher.find_motif_positions("read_001", sequence, aln, "CCAGGC")
        # With perfect alignment and ref_start=0, query coords == ref coords
        assert positions == [20]

    def test_reference_search_with_offset_alignment(self):
        """Alignment that starts partway into the reference."""
        # Read aligns to ref positions [10, 40) — still covers CCAGGC at [20, 26)
        sequence = "ACGTACGTACCCAGGCTTACGTACGTACGT"  # 30 bases

        aln = _make_pysam_alignment(
            query_name="read_002",
            query_sequence=sequence,
            reference_name=REFERENCE_NAME,
            reference_start=10,
            cigartuples=[(0, 30)],  # 30M starting at ref pos 10
        )

        searcher = ReferenceMotifSearcher(
            reference_sequences=REFERENCE_SEQUENCES,
            skip_indels=True,
            anchor="reference",
        )
        positions = searcher.find_motif_positions("read_002", sequence, aln, "CCAGGC")
        # Motif is at ref pos 20; relative to alignment start (10) → position 10
        assert positions == [10]

    def test_reference_search_skips_indel_in_motif(self):
        """Reference search skips motif when indel overlaps the motif region."""
        # Insertion at ref position 22 (inside CCAGGC [20, 26))
        sequence = "ACGTACGTACGTACGTACGTCCAXGGCTTACGTACGTACGT"  # 41 bases (1 ins)

        aln = _make_pysam_alignment(
            query_name="read_003",
            query_sequence=sequence,
            reference_name=REFERENCE_NAME,
            reference_start=0,
            cigartuples=[(0, 22), (1, 1), (0, 18)],  # 22M 1I 18M
        )

        searcher = ReferenceMotifSearcher(
            reference_sequences=REFERENCE_SEQUENCES,
            skip_indels=True,
            anchor="reference",
        )
        positions = searcher.find_motif_positions("read_003", sequence, aln, "CCAGGC")
        assert positions == []  # Skipped due to indel

    def test_reference_search_allows_indel_when_disabled(self):
        """Reference search finds motif when skip_indels=False."""
        # Insertion at ref position 22 (inside CCAGGC)
        sequence = "ACGTACGTACGTACGTACGTCCAXGGCTTACGTACGTACGT"

        aln = _make_pysam_alignment(
            query_name="read_003",
            query_sequence=sequence,
            reference_name=REFERENCE_NAME,
            reference_start=0,
            cigartuples=[(0, 22), (1, 1), (0, 18)],
        )

        searcher = ReferenceMotifSearcher(
            reference_sequences=REFERENCE_SEQUENCES,
            skip_indels=False,
            anchor="reference",
        )
        positions = searcher.find_motif_positions("read_003", sequence, aln, "CCAGGC")
        assert positions == [20]

    def test_reference_search_no_alignment_returns_empty(self):
        """Reference search gracefully returns empty when alignment is None."""
        searcher = ReferenceMotifSearcher(
            reference_sequences=REFERENCE_SEQUENCES,
            skip_indels=True,
        )
        positions = searcher.find_motif_positions("read_004", "ACGT", None, "CCAGGC")
        assert positions == []

    def test_reference_search_unknown_reference_returns_empty(self):
        """Reference search returns empty when reference name not in dict."""
        aln = _make_pysam_alignment(
            query_name="read_005",
            query_sequence="A" * 40,
            reference_name="unknown_ref",
            reference_start=0,
            cigartuples=[(0, 40)],
        )

        searcher = ReferenceMotifSearcher(
            reference_sequences=REFERENCE_SEQUENCES,
            skip_indels=True,
        )
        positions = searcher.find_motif_positions("read_005", "A" * 40, aln, "CCAGGC")
        assert positions == []


# ---------------------------------------------------------------------------
# Tests: InferenceConfig carries reference_sequences
# ---------------------------------------------------------------------------


class TestInferenceConfigMotifSearch:
    """Tests that InferenceConfig properly propagates reference_sequences."""

    def test_motif_config_carries_reference_sequences(self):
        """MotifConfig stores reference_sequences for workers."""
        mc = MotifConfig(
            motif="CCAGGC",
            motif_offset=2,
            reference_sequences=REFERENCE_SEQUENCES,
        )
        assert mc.reference_sequences is not None
        assert mc.reference_sequences[REFERENCE_NAME] == REFERENCE_SEQ

    def test_motif_config_default_reference_sequences_is_none(self):
        mc = MotifConfig(motif="CCAGGC", motif_offset=2)
        assert mc.reference_sequences is None

    def test_inference_config_motif_reference_sequences(self):
        """InferenceConfig.motif.reference_sequences is accessible."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            cfg = InferenceConfig(
                pod5_path=Path(td) / "dummy.pod5",
                signal=SignalConfig(anchor="reference"),
                motif=MotifConfig(
                    motif="CCAGGC",
                    motif_offset=2,
                    reference_sequences=REFERENCE_SEQUENCES,
                ),
            )
            assert cfg.motif.reference_sequences is REFERENCE_SEQUENCES

    def test_inference_config_is_picklable_with_reference_sequences(self):
        """InferenceConfig with reference_sequences must survive pickling (multiprocessing)."""
        import pickle
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            cfg = InferenceConfig(
                pod5_path=Path(td) / "dummy.pod5",
                motif=MotifConfig(
                    motif="CCAGGC",
                    motif_offset=2,
                    reference_sequences=REFERENCE_SEQUENCES,
                ),
            )
            restored = pickle.loads(pickle.dumps(cfg))
            assert restored.motif.reference_sequences == REFERENCE_SEQUENCES


# ---------------------------------------------------------------------------
# Tests: motif searcher selection logic (mirrors inference.py wiring)
# ---------------------------------------------------------------------------


class TestMotifSearcherSelection:
    """Tests the searcher selection logic used in run_inference / run_bundle_inference."""

    def test_selects_reference_searcher_when_refs_available(self):
        """When reference_sequences is available, use fasta mode."""
        searcher = get_motif_searcher(
            mode="fasta" if REFERENCE_SEQUENCES else "bam",
            reference_sequences=REFERENCE_SEQUENCES,
            skip_indels=True,
            anchor="reference",
        )
        assert isinstance(searcher, ReferenceMotifSearcher)

    def test_selects_basecalled_searcher_when_refs_unavailable(self):
        """When reference_sequences is None, fall back to bam mode."""
        reference_sequences = None
        searcher = get_motif_searcher(
            mode="fasta" if reference_sequences else "bam",
            reference_sequences=reference_sequences,
        )
        assert isinstance(searcher, BasecalledMotifSearcher)

    def test_reference_searcher_with_motif_offset(self):
        """Motif offset is applied after motif search (not inside searcher)."""
        aln = _make_pysam_alignment(
            query_name="read_001",
            query_sequence="X" * 40,
            reference_name=REFERENCE_NAME,
            reference_start=0,
            cigartuples=[(0, 40)],
        )

        searcher = ReferenceMotifSearcher(
            reference_sequences=REFERENCE_SEQUENCES,
            skip_indels=True,
            anchor="reference",
        )
        motif_offset = 2  # Focus on 'A' in CCA (the aminoacylation site)
        raw_positions = searcher.find_motif_positions("read_001", "X" * 40, aln, "CCAGGC")
        positions = [pos + motif_offset for pos in raw_positions]
        # CCAGGC at position 20, offset 2 → focus base at 22
        assert positions == [22]


# ---------------------------------------------------------------------------
# Tests: debug statistics
# ---------------------------------------------------------------------------


class TestReferenceSearcherStats:
    """Tests for ReferenceMotifSearcher debug statistics."""

    def test_stats_count_successful(self):
        aln = _make_pysam_alignment(
            query_name="read_001",
            query_sequence="X" * 40,
            reference_name=REFERENCE_NAME,
            reference_start=0,
            cigartuples=[(0, 40)],
        )

        searcher = ReferenceMotifSearcher(
            reference_sequences=REFERENCE_SEQUENCES,
            skip_indels=True,
            debug=True,
        )
        searcher.find_motif_positions("read_001", "X" * 40, aln, "CCAGGC")
        stats = searcher.get_stats()
        assert stats["motifs_in_reference"] == 1
        assert stats["successful"] == 1
        assert stats["failed_cigar_mapping"] == 0
        assert stats["failed_indels"] == 0

    def test_stats_count_indel_failures(self):
        # Insertion inside motif region
        sequence = "ACGTACGTACGTACGTACGTCCAXGGCTTACGTACGTACGT"
        aln = _make_pysam_alignment(
            query_name="read_003",
            query_sequence=sequence,
            reference_name=REFERENCE_NAME,
            reference_start=0,
            cigartuples=[(0, 22), (1, 1), (0, 18)],
        )

        searcher = ReferenceMotifSearcher(
            reference_sequences=REFERENCE_SEQUENCES,
            skip_indels=True,
            debug=True,
        )
        searcher.find_motif_positions("read_003", sequence, aln, "CCAGGC")
        stats = searcher.get_stats()
        assert stats["motifs_in_reference"] == 1
        assert stats["failed_indels"] == 1
        assert stats["successful"] == 0

    def test_stats_reset(self):
        searcher = ReferenceMotifSearcher(
            reference_sequences=REFERENCE_SEQUENCES,
            skip_indels=True,
            debug=True,
        )
        aln = _make_mock_alignment(REFERENCE_NAME, 0, 40, [(0, 40)])
        searcher.find_motif_positions("read_001", "X" * 40, aln, "CCAGGC")
        searcher.reset_stats()
        stats = searcher.get_stats()
        assert all(v == 0 for v in stats.values())
