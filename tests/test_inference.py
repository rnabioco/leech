"""
Tests for inference logic.

Covers:
- _write_prediction_tags: compact/raw encoding, thresholds (min_confidence,
  min_margin), margin computation, edge cases
- Motif search: reference-based vs basecalled search
"""

import array

import pysam
import pytest
import torch
from conftest import TRNA_BAM, TRNA_FIXTURES_AVAILABLE, TRNA_POD5, TRNA_REF  # noqa: E402

from leech.configs import InferenceConfig, MotifConfig, SignalConfig
from leech.inference import _write_prediction_tags
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


# ---------------------------------------------------------------------------
# Tests: _write_prediction_tags
# ---------------------------------------------------------------------------


def _make_bare_alignment() -> pysam.AlignedSegment:
    """Create a minimal AlignedSegment for tag-writing tests."""
    aln = pysam.AlignedSegment(header=_BAM_HEADER)
    aln.query_name = "test_read"
    aln.query_sequence = "ACGT"
    aln.reference_id = 0
    aln.reference_start = 0
    aln.cigartuples = [(0, 4)]
    return aln


class TestWritePredictionTagsCompact:
    """Tests for compact uint8 tag encoding (default mode)."""

    def test_basic_tags_written(self):
        aln = _make_bare_alignment()
        _write_prediction_tags(
            aln,
            "Ala",
            0.93,
            "Ala,Gly,Met",
            [0.93, 0.05, 0.02],
            raw=False,
            min_confidence=0,
        )
        assert aln.get_tag("aa") == "Ala"
        assert aln.get_tag("pn") == "Ala,Gly,Met"
        # ac should be uint8
        ac = aln.get_tag("ac")
        assert isinstance(ac, int)
        assert ac == round(0.93 * 255)
        # am should be uint8 margin: 0.93 - 0.05 = 0.88
        am = aln.get_tag("am")
        assert isinstance(am, int)
        assert am == round(0.88 * 255)
        # pp should be uint8 array
        pp = list(aln.get_tag("pp"))
        assert len(pp) == 3
        assert all(isinstance(p, int) for p in pp)
        assert pp[0] == round(0.93 * 255)

    def test_pp_array_type(self):
        """pp tag should be a B:B (unsigned byte array) in compact mode."""
        aln = _make_bare_alignment()
        _write_prediction_tags(
            aln,
            "Gly",
            0.8,
            "Ala,Gly",
            [0.2, 0.8],
            raw=False,
            min_confidence=0,
        )
        pp = aln.get_tag("pp")
        assert isinstance(pp, array.array)


class TestWritePredictionTagsRaw:
    """Tests for raw float tag encoding (--raw mode)."""

    def test_raw_float_tags(self):
        aln = _make_bare_alignment()
        probs = [0.93, 0.05, 0.02]
        _write_prediction_tags(
            aln,
            "Ala",
            0.93,
            "Ala,Gly,Met",
            probs,
            raw=True,
            min_confidence=0,
        )
        assert aln.get_tag("aa") == "Ala"
        ac = aln.get_tag("ac")
        assert isinstance(ac, float)
        assert ac == pytest.approx(0.93)
        am = aln.get_tag("am")
        assert isinstance(am, float)
        assert am == pytest.approx(0.88)
        pp = list(aln.get_tag("pp"))
        assert len(pp) == 3
        assert pp == pytest.approx(probs, abs=1e-5)


class TestWritePredictionTagsMargin:
    """Tests for margin (am tag) computation."""

    def test_margin_two_classes(self):
        """Margin = max_prob - 2nd_prob for two-class case."""
        aln = _make_bare_alignment()
        _write_prediction_tags(
            aln,
            "Ala",
            0.7,
            "Ala,Gly",
            [0.7, 0.3],
            raw=True,
            min_confidence=0,
        )
        assert aln.get_tag("am") == pytest.approx(0.4)

    def test_margin_single_class(self):
        """Margin = 1.0 when only one class exists."""
        aln = _make_bare_alignment()
        _write_prediction_tags(
            aln,
            "Ala",
            1.0,
            "Ala",
            [1.0],
            raw=True,
            min_confidence=0,
        )
        assert aln.get_tag("am") == pytest.approx(1.0)

    def test_margin_equal_probs(self):
        """Margin = 0 when top two classes have identical probability."""
        aln = _make_bare_alignment()
        _write_prediction_tags(
            aln,
            "Ala",
            0.5,
            "Ala,Gly",
            [0.5, 0.5],
            raw=True,
            min_confidence=0,
        )
        assert aln.get_tag("am") == pytest.approx(0.0)

    def test_margin_uses_sorted_probs(self):
        """Margin uses the two highest probs regardless of ordering."""
        aln = _make_bare_alignment()
        # probs are not sorted by value — 2nd highest is 0.20 (Met), not 0.05 (Gly)
        _write_prediction_tags(
            aln,
            "Ala",
            0.75,
            "Ala,Gly,Met",
            [0.75, 0.05, 0.20],
            raw=True,
            min_confidence=0,
        )
        assert aln.get_tag("am") == pytest.approx(0.55)

    def test_margin_compact_roundtrip(self):
        """Compact uint8 margin survives encode/decode within 1/255 tolerance."""
        aln = _make_bare_alignment()
        _write_prediction_tags(
            aln,
            "Ala",
            0.85,
            "Ala,Gly",
            [0.85, 0.15],
            raw=False,
            min_confidence=0,
        )
        am = aln.get_tag("am")
        assert abs(am / 255.0 - 0.70) < 1.5 / 255.0


class TestWritePredictionTagsMinConfidence:
    """Tests for --min-confidence threshold gating."""

    def test_above_threshold_is_called(self):
        aln = _make_bare_alignment()
        # conf=0.9 → uint8=230, threshold=200
        _write_prediction_tags(
            aln,
            "Ala",
            0.9,
            "Ala,Gly",
            [0.9, 0.1],
            raw=False,
            min_confidence=200,
        )
        assert aln.get_tag("aa") == "Ala"
        # ac should encode the actual confidence
        assert aln.get_tag("ac") == round(0.9 * 255)

    def test_below_threshold_is_unc(self):
        aln = _make_bare_alignment()
        # conf=0.6 → uint8=153, threshold=200
        _write_prediction_tags(
            aln,
            "Ala",
            0.6,
            "Ala,Gly",
            [0.6, 0.4],
            raw=False,
            min_confidence=200,
        )
        assert aln.get_tag("aa") == "unc"
        # ac keeps the winning class probability; it is not inverted
        assert aln.get_tag("ac") == round(0.6 * 255)

    def test_unc_raw_mode(self):
        """In raw mode, filtered reads still report the winning probability."""
        aln = _make_bare_alignment()
        _write_prediction_tags(
            aln,
            "Met",
            0.3,
            "Met,Ile",
            [0.3, 0.7],
            raw=True,
            min_confidence=200,
        )
        assert aln.get_tag("aa") == "unc"
        assert aln.get_tag("ac") == pytest.approx(0.3)

    def test_below_threshold_multiclass_ac_is_not_inverted(self):
        """Regression for #139.

        With more than two classes, 1-conf is not the probability of anything.
        A filtered 5-class read must not report a *higher* ac than its own
        winning probability, or `ac` becomes unfilterable.
        """
        aln = _make_bare_alignment()
        probs = [0.4, 0.3, 0.15, 0.1, 0.05]
        _write_prediction_tags(
            aln,
            "ClassA",
            0.4,
            "ClassA,ClassB,ClassC,ClassD,ClassE",
            probs,
            raw=True,
            min_confidence=200,
        )
        assert aln.get_tag("aa") == "unc"
        assert aln.get_tag("ac") == pytest.approx(0.4)

    def test_threshold_zero_passes_all(self):
        """min_confidence=0 means every read is called."""
        aln = _make_bare_alignment()
        _write_prediction_tags(
            aln,
            "Gly",
            0.01,
            "Gly,Ala",
            [0.01, 0.99],
            raw=False,
            min_confidence=0,
        )
        assert aln.get_tag("aa") == "Gly"


class TestWritePredictionTagsMinMargin:
    """Tests for --min-margin threshold gating."""

    def test_above_margin_threshold_is_called(self):
        aln = _make_bare_alignment()
        # margin = 0.9 - 0.1 = 0.8 → uint8=204, threshold=128
        _write_prediction_tags(
            aln,
            "Ala",
            0.9,
            "Ala,Gly",
            [0.9, 0.1],
            raw=False,
            min_confidence=0,
            min_margin=128,
        )
        assert aln.get_tag("aa") == "Ala"

    def test_below_margin_threshold_is_unc(self):
        aln = _make_bare_alignment()
        # margin = 0.55 - 0.45 = 0.1 → uint8=26, threshold=128
        _write_prediction_tags(
            aln,
            "Ala",
            0.55,
            "Ala,Gly",
            [0.55, 0.45],
            raw=False,
            min_confidence=0,
            min_margin=128,
        )
        assert aln.get_tag("aa") == "unc"

    def test_margin_threshold_zero_passes_all(self):
        aln = _make_bare_alignment()
        _write_prediction_tags(
            aln,
            "Met",
            0.51,
            "Met,Ile",
            [0.51, 0.49],
            raw=False,
            min_confidence=0,
            min_margin=0,
        )
        assert aln.get_tag("aa") == "Met"


class TestWritePredictionTagsCombinedThresholds:
    """Tests for combined --min-confidence + --min-margin gating."""

    def test_passes_both_thresholds(self):
        aln = _make_bare_alignment()
        # conf=0.95 → uint8=242, margin=0.90 → uint8=230
        _write_prediction_tags(
            aln,
            "Ala",
            0.95,
            "Ala,Gly",
            [0.95, 0.05],
            raw=False,
            min_confidence=200,
            min_margin=200,
        )
        assert aln.get_tag("aa") == "Ala"

    def test_fails_confidence_passes_margin(self):
        aln = _make_bare_alignment()
        # conf=0.6 → uint8=153, margin=0.5 → uint8=128
        _write_prediction_tags(
            aln,
            "Ala",
            0.6,
            "Ala,Gly",
            [0.6, 0.1],
            raw=False,
            min_confidence=200,
            min_margin=50,
        )
        assert aln.get_tag("aa") == "unc"

    def test_passes_confidence_fails_margin(self):
        aln = _make_bare_alignment()
        # conf=0.55 → uint8=140, margin=0.1 → uint8=26
        _write_prediction_tags(
            aln,
            "Ala",
            0.55,
            "Ala,Gly",
            [0.55, 0.45],
            raw=False,
            min_confidence=100,
            min_margin=128,
        )
        assert aln.get_tag("aa") == "unc"

    def test_fails_both_thresholds(self):
        aln = _make_bare_alignment()
        _write_prediction_tags(
            aln,
            "Ala",
            0.4,
            "Ala,Gly",
            [0.4, 0.6],
            raw=False,
            min_confidence=200,
            min_margin=200,
        )
        assert aln.get_tag("aa") == "unc"

    def test_unc_still_writes_all_tags(self):
        """Even when unc, all 5 tags (aa, ac, am, pn, pp) are written."""
        aln = _make_bare_alignment()
        _write_prediction_tags(
            aln,
            "Ala",
            0.4,
            "Ala,Gly,Met",
            [0.4, 0.35, 0.25],
            raw=False,
            min_confidence=255,
            min_margin=255,
        )
        assert aln.get_tag("aa") == "unc"
        assert aln.has_tag("ac")
        assert aln.has_tag("am")
        assert aln.has_tag("pn")
        assert aln.has_tag("pp")
        # pp preserves the full distribution
        pp = list(aln.get_tag("pp"))
        assert len(pp) == 3


# ---------------------------------------------------------------------------
# Integration tests: run_inference / run_bundle_inference → output BAM tags
#
# Uses 20-read tRNA fixtures (AlaRS IVC) with CCAGGC motif and reference FASTA.
# Models are randomly initialized — outputs are garbage, but we verify that
# tags are written correctly, thresholds gate properly, and encodings work.
# ---------------------------------------------------------------------------

# Shared model architecture config (small, fast)
_ARCH_CONFIG = {
    "signal_len": 400,
    "kmer_len": 11,
    "num_features": 9,
    "conv_channels": [4, 16, 64],
    "lstm_hidden": 32,
    "dropout": 0.0,
    "refine_signal_map": False,
    "seq_encoding": "base_onehot",
}

# Keys in _ARCH_CONFIG that are config-level, not model constructor args
_NON_MODEL_KEYS = {"motif", "motif_offset", "model_name", "refine_signal_map", "seq_encoding"}


def _model_kwargs():
    """Filter _ARCH_CONFIG to only model constructor kwargs."""
    return {k: v for k, v in _ARCH_CONFIG.items() if k not in _NON_MODEL_KEYS}


def _create_multiclass_bundle(tmp_path, class_names):
    """Create a multiclass bundle with a randomly initialized model."""
    from leech.models import get_model

    num_out = len(class_names)
    model = get_model("ConvLSTMDwell", num_out=num_out, **_model_kwargs())

    config = {
        "model_name": "ConvLSTMDwell",
        **_ARCH_CONFIG,
        "num_out": num_out,
        "motif": "CCAGGC",
        "motif_offset": 2,
    }
    label_map = {name: i for i, name in enumerate(class_names)}
    bundle = {
        "metadata": {
            "format_version": 1,
            "bundle_version": "0.0.1-test",
            "architecture": "ConvLSTMDwell",
            "comparison_type": "multiclass",
            "num_classes": num_out,
            "label_map": label_map,
            "pairs": list(label_map.values()),
        },
        "config": config,
        "model": {"state_dict": model.state_dict()},
    }
    bundle_path = tmp_path / "mc_bundle.pt"
    torch.save(bundle, bundle_path)
    return bundle_path


def _create_pairwise_bundle(tmp_path, pair_names):
    """Create a pairwise bundle with randomly initialized models."""
    import json

    from leech.bundling import create_bundle
    from leech.models import get_model

    model_dirs = {}
    config = {
        "model_name": "ConvLSTMDwell",
        **_ARCH_CONFIG,
        "motif": "CCAGGC",
        "motif_offset": 2,
    }
    for pair in pair_names:
        pair_dir = tmp_path / "pw_models" / pair
        pair_dir.mkdir(parents=True)
        with open(pair_dir / "config.json", "w") as f:
            json.dump(config, f)
        model = get_model("ConvLSTMDwell", **_model_kwargs())
        torch.save({"model_state_dict": model.state_dict()}, pair_dir / "model_best.pt")
        model_dirs[pair] = pair_dir

    bundle_path = tmp_path / "pw_bundle.pt"
    create_bundle(model_dirs, bundle_path, "pairwise", "0.0.1-test")
    return bundle_path


def _run_multiclass_inference(tmp_path, class_names, *, raw=False, min_confidence=0, min_margin=0):
    """Run multiclass inference on tRNA fixtures and return output reads."""
    from leech.bundling import load_model_from_multiclass_bundle
    from leech.inference import run_inference

    bundle_path = _create_multiclass_bundle(tmp_path, class_names)
    model, config = load_model_from_multiclass_bundle(bundle_path, device="cpu")

    output_bam = tmp_path / "out.bam"
    run_inference(
        model_and_config=(model, config),
        pod5_path=TRNA_POD5,
        bam_path=TRNA_BAM,
        output_path=output_bam,
        device="cpu",
        batch_size=64,
        reverse_signal=True,  # RNA data
        reference_fasta=TRNA_REF,
        raw=raw,
        min_confidence=min_confidence,
        min_margin=min_margin,
    )

    with pysam.AlignmentFile(str(output_bam), "rb") as bam:
        return list(bam)


@pytest.mark.skipif(not TRNA_FIXTURES_AVAILABLE, reason="tRNA fixtures not available")
class TestRunInferenceTagsIntegration:
    """Integration: run_inference (multiclass) on tRNA fixtures → verify output BAM tags."""

    CLASS_NAMES = ["Ala", "Gly", "Met"]

    def test_all_five_tags_written(self, tmp_path):
        """Every predicted read has aa, ac, am, pn, pp tags."""
        reads = _run_multiclass_inference(tmp_path, self.CLASS_NAMES)
        tagged = [r for r in reads if r.has_tag("aa")]
        assert len(tagged) > 0, "No reads were tagged"
        for read in tagged:
            assert read.has_tag("ac"), f"Missing ac on {read.query_name}"
            assert read.has_tag("am"), f"Missing am on {read.query_name}"
            assert read.has_tag("pn"), f"Missing pn on {read.query_name}"
            assert read.has_tag("pp"), f"Missing pp on {read.query_name}"

    def test_compact_encoding_types(self, tmp_path):
        """Compact mode writes uint8 for ac/am/pp."""
        reads = _run_multiclass_inference(tmp_path, self.CLASS_NAMES, raw=False)
        tagged = [r for r in reads if r.has_tag("aa")]
        read = tagged[0]
        assert isinstance(read.get_tag("ac"), int)
        assert isinstance(read.get_tag("am"), int)
        assert 0 <= read.get_tag("ac") <= 255
        assert 0 <= read.get_tag("am") <= 255
        pp = read.get_tag("pp")
        assert isinstance(pp, array.array)
        assert all(0 <= p <= 255 for p in pp)

    def test_raw_encoding_types(self, tmp_path):
        """Raw mode writes floats for ac/am/pp."""
        reads = _run_multiclass_inference(tmp_path, self.CLASS_NAMES, raw=True)
        tagged = [r for r in reads if r.has_tag("aa")]
        read = tagged[0]
        assert isinstance(read.get_tag("ac"), float)
        assert isinstance(read.get_tag("am"), float)

    def test_pn_matches_pp_length(self, tmp_path):
        """pn class count matches pp probability count."""
        reads = _run_multiclass_inference(tmp_path, self.CLASS_NAMES)
        tagged = [r for r in reads if r.has_tag("aa")]
        for read in tagged:
            pn = read.get_tag("pn").split(",")
            pp = list(read.get_tag("pp"))
            assert len(pn) == len(pp) == len(self.CLASS_NAMES)

    def test_aa_is_valid_class_or_unc(self, tmp_path):
        """aa tag is one of the class names or 'unc'."""
        reads = _run_multiclass_inference(tmp_path, self.CLASS_NAMES)
        valid = set(self.CLASS_NAMES) | {"unc"}
        for read in reads:
            if read.has_tag("aa"):
                assert read.get_tag("aa") in valid

    def test_min_confidence_gates_to_unc(self, tmp_path):
        """With min_confidence=255, all reads should be 'unc'."""
        reads = _run_multiclass_inference(tmp_path, self.CLASS_NAMES, min_confidence=255)
        tagged = [r for r in reads if r.has_tag("aa")]
        assert len(tagged) > 0
        for read in tagged:
            assert read.get_tag("aa") == "unc"

    def test_min_margin_gates_to_unc(self, tmp_path):
        """With min_margin=255, all reads should be 'unc'."""
        reads = _run_multiclass_inference(tmp_path, self.CLASS_NAMES, min_margin=255)
        tagged = [r for r in reads if r.has_tag("aa")]
        assert len(tagged) > 0
        for read in tagged:
            assert read.get_tag("aa") == "unc"

    def test_ac_is_independent_of_threshold(self, tmp_path):
        """ac reports max_prob whether or not the read passed (#139).

        Previously a filtered read got 1-max_prob, so `ac` meant different
        things depending on the `aa` tag and could not be filtered on directly.
        """
        from leech.bundling import load_model_from_multiclass_bundle
        from leech.inference import run_inference

        # Use ONE bundle for both runs so same model weights → same predictions
        bundle_path = _create_multiclass_bundle(tmp_path, self.CLASS_NAMES)
        model, config = load_model_from_multiclass_bundle(bundle_path, device="cpu")

        def _run(subdir, min_confidence):
            d = tmp_path / subdir
            d.mkdir()
            out = d / "out.bam"
            run_inference(
                model_and_config=(model, config),
                pod5_path=TRNA_POD5,
                bam_path=TRNA_BAM,
                output_path=out,
                device="cpu",
                batch_size=64,
                reverse_signal=True,
                reference_fasta=TRNA_REF,
                raw=True,
                min_confidence=min_confidence,
            )
            with pysam.AlignmentFile(str(out), "rb") as bam:
                return {r.query_name: r for r in bam if r.has_tag("aa")}

        filtered_by_id = _run("filtered", min_confidence=255)
        called_by_id = _run("called", min_confidence=0)
        common = set(filtered_by_id) & set(called_by_id)
        assert len(common) > 0

        for rid in list(common)[:5]:
            filtered, called = filtered_by_id[rid], called_by_id[rid]
            assert filtered.get_tag("aa") == "unc"
            assert called.get_tag("aa") != "unc"
            # Same weights, same read -> same confidence regardless of threshold
            assert filtered.get_tag("ac") == pytest.approx(called.get_tag("ac"), abs=1e-5)
            # ...and it matches the winning class probability in pp
            assert filtered.get_tag("ac") == pytest.approx(max(filtered.get_tag("pp")), abs=1e-5)

    def test_margin_consistency(self, tmp_path):
        """Margin am = max_prob - 2nd_prob, verified against pp distribution."""
        reads = _run_multiclass_inference(tmp_path, self.CLASS_NAMES, raw=True)
        tagged = [r for r in reads if r.has_tag("aa")]
        for read in tagged[:5]:
            pp = sorted(read.get_tag("pp"), reverse=True)
            expected_margin = pp[0] - pp[1] if len(pp) > 1 else 1.0
            assert read.get_tag("am") == pytest.approx(expected_margin, abs=1e-5)


@pytest.mark.skipif(not TRNA_FIXTURES_AVAILABLE, reason="tRNA fixtures not available")
class TestRunBundleInferenceTagsIntegration:
    """Integration: run_bundle_inference (pairwise) on tRNA fixtures → verify output tags."""

    PAIR_NAMES = ["Ala_Gly", "Ala_Met", "Gly_Met"]

    def test_pairwise_bundle_writes_all_tags(self, tmp_path):
        """Pairwise bundle writes aa, ac, am, pn, pp on every predicted read."""
        from leech.inference import run_bundle_inference

        bundle_path = _create_pairwise_bundle(tmp_path, self.PAIR_NAMES)
        output_bam = tmp_path / "pw_out.bam"
        run_bundle_inference(
            bundle_path=bundle_path,
            pod5_path=TRNA_POD5,
            bam_path=TRNA_BAM,
            output_path=output_bam,
            device="cpu",
            batch_size=64,
            reverse_signal=True,
            reference_fasta=TRNA_REF,
            raw=False,
            min_confidence=0,
            min_margin=0,
        )

        with pysam.AlignmentFile(str(output_bam), "rb") as bam:
            tagged = [r for r in bam if r.has_tag("aa")]
        assert len(tagged) > 0
        for read in tagged:
            assert read.has_tag("ac")
            assert read.has_tag("am")
            assert read.has_tag("pn")
            assert read.has_tag("pp")
            pn = read.get_tag("pn").split(",")
            assert len(pn) == len(self.PAIR_NAMES)

    def test_pairwise_bundle_min_confidence_unc(self, tmp_path):
        """Pairwise bundle respects min_confidence threshold."""
        from leech.inference import run_bundle_inference

        bundle_path = _create_pairwise_bundle(tmp_path, self.PAIR_NAMES)
        output_bam = tmp_path / "pw_unc.bam"
        run_bundle_inference(
            bundle_path=bundle_path,
            pod5_path=TRNA_POD5,
            bam_path=TRNA_BAM,
            output_path=output_bam,
            device="cpu",
            batch_size=64,
            reverse_signal=True,
            reference_fasta=TRNA_REF,
            min_confidence=255,
        )

        with pysam.AlignmentFile(str(output_bam), "rb") as bam:
            tagged = [r for r in bam if r.has_tag("aa")]
        assert len(tagged) > 0
        assert all(r.get_tag("aa") == "unc" for r in tagged)


# ---------------------------------------------------------------------------
# BatchAccumulator: the one batching state machine (was written three times —
# multiprocessing workers and threaded Rust in single.py, and bundle.py)
# ---------------------------------------------------------------------------


class TestBatchAccumulator:
    """Unit tests for leech.inference.helpers.BatchAccumulator."""

    @staticmethod
    def _recording_accumulator(batch_size):
        from leech.inference.helpers import BatchAccumulator

        flushed = []

        def flush(sigs, seqs, feats, meta):
            flushed.append((sigs, seqs, feats, meta))

        return BatchAccumulator(batch_size, flush), flushed

    def test_flushes_exactly_at_batch_size(self):
        acc, flushed = self._recording_accumulator(3)
        for i in range(7):
            acc.add(f"s{i}", f"q{i}", f"f{i}", ("r", i))
        # 7 chunks, batch 3 -> two full flushes, one chunk still buffered
        assert [len(b[0]) for b in flushed] == [3, 3]
        assert len(acc) == 1
        acc.flush()
        assert [len(b[0]) for b in flushed] == [3, 3, 1]
        assert len(acc) == 0

    def test_buffers_stay_parallel(self):
        acc, flushed = self._recording_accumulator(2)
        acc.add("s0", "q0", None, ("r0", 0))
        acc.add("s1", "q1", "f1", ("r1", 1))
        sigs, seqs, feats, meta = flushed[0]
        assert sigs == ["s0", "s1"]
        assert seqs == ["q0", "q1"]
        assert feats == [None, "f1"]
        assert meta == [("r0", 0), ("r1", 1)]

    def test_flush_is_a_noop_when_empty(self):
        acc, flushed = self._recording_accumulator(4)
        acc.flush()
        acc.flush()
        assert flushed == []

    def test_flush_detaches_the_buffers(self):
        """The GPU thread keeps the flushed lists while extraction refills."""
        acc, flushed = self._recording_accumulator(2)
        acc.add("s0", "q0", "f0", ("r0", 0))
        acc.add("s1", "q1", "f1", ("r1", 1))
        held = flushed[0]
        acc.add("s2", "q2", "f2", ("r2", 2))
        # Appending after the flush must not mutate what the callback was given.
        assert held[0] == ["s0", "s1"]
        assert acc.signals == ["s2"]

    def test_add_flushes_when_batch_size_is_one(self):
        acc, flushed = self._recording_accumulator(1)
        acc.add("s0", "q0", "f0", ("r0", 0))
        assert [len(b[0]) for b in flushed] == [1]
        assert len(acc) == 0


# ---------------------------------------------------------------------------
# _stack_to_device: pinned staging buffers for the host->device batch copy
# ---------------------------------------------------------------------------


class TestStackToDevice:
    """The pinned staging path must return exactly what the pageable one did."""

    @staticmethod
    def _batch(n, shape, seed):
        import numpy as np

        rng = np.random.default_rng(seed)
        return [rng.standard_normal(shape, dtype=np.float32) for _ in range(n)]

    def test_cpu_matches_plain_stack(self):
        import numpy as np

        from leech.inference.helpers import _stack_to_device

        arrays = self._batch(5, (2, 7), 0)
        got = _stack_to_device(arrays, "cpu", "signal")
        assert torch.equal(got, torch.from_numpy(np.stack(arrays)))

    def test_cpu_preserves_dtype_and_shape(self):
        import numpy as np

        from leech.inference.helpers import _stack_to_device

        arrays = [np.arange(6, dtype=np.float32).reshape(2, 3) for _ in range(3)]
        got = _stack_to_device(arrays, "cpu", "features")
        assert got.shape == (3, 2, 3)
        assert got.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_matches_pageable_copy(self):
        import numpy as np

        from leech.inference.helpers import _stack_to_device

        arrays = self._batch(9, (2, 11), 1)
        got = _stack_to_device(arrays, "cuda", "signal")
        want = torch.from_numpy(np.stack(arrays)).to("cuda")
        assert got.is_cuda
        assert torch.equal(got, want)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_reuse_does_not_leak_stale_rows(self):
        """A reused staging buffer must never hand back the previous batch's rows.

        The buffer is grown, not reallocated, so a shorter batch reads a view
        over rows that still hold the longer batch's data.
        """
        import numpy as np

        from leech.inference.helpers import _stack_to_device

        big = self._batch(16, (2, 5), 2)
        _stack_to_device(big, "cuda", "signal")
        for n in (16, 4, 12, 1, 16):
            arrays = self._batch(n, (2, 5), 100 + n)
            got = _stack_to_device(arrays, "cuda", "signal").cpu()
            assert torch.equal(got, torch.from_numpy(np.stack(arrays)))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_slots_do_not_alias(self):
        """signal/sequence/features must not share one staging buffer."""
        import numpy as np

        from leech.inference.helpers import _stack_to_device

        a = self._batch(4, (2, 5), 3)
        b = self._batch(4, (2, 5), 4)
        ta = _stack_to_device(a, "cuda", "signal")
        tb = _stack_to_device(b, "cuda", "sequence")
        assert torch.equal(ta.cpu(), torch.from_numpy(np.stack(a)))
        assert torch.equal(tb.cpu(), torch.from_numpy(np.stack(b)))


# ---------------------------------------------------------------------------
# _run_batch / _run_batch_multiclass: probabilities reach `pending` unchanged
# ---------------------------------------------------------------------------


class _FixedLogitsWrapper:
    """Model stand-in returning preset logits, ignoring its inputs."""

    requires_features = False

    def __init__(self, logits):
        self._logits = logits

    def forward_batch(self, batch, device):
        return self._logits


class TestRunBatchProbabilities:
    """`tolist()` on the whole batch must equal `float(p)` per element."""

    def test_multiclass_probs_are_exact_float_promotions(self):
        import numpy as np

        from leech.inference.helpers import _run_batch_multiclass

        logits = torch.tensor([[0.1, -2.0, 3.5], [-1.0, 0.25, 0.75]], dtype=torch.float32)
        n = logits.shape[0]
        signals = [np.zeros((1, 4), dtype=np.float32) for _ in range(n)]
        seqs = [np.zeros((4, 3), dtype=np.float32) for _ in range(n)]
        meta = [("readA", 0), ("readB", 1)]
        pending: dict = {}
        _run_batch_multiclass(
            signals, seqs, [None, None], meta, _FixedLogitsWrapper(logits), False, "cpu", pending
        )

        probs = torch.softmax(logits, dim=-1).numpy()
        for i, (read_id, base_idx) in enumerate(meta):
            stored = pending[read_id][0]
            assert stored[0] == base_idx
            assert stored[1] == int(np.argmax(probs[i]))
            assert stored[2] == float(probs[i].max())
            # exactly what the per-element float() comprehension produced
            assert stored[3] == [float(p) for p in probs[i]]
            assert all(type(p) is float for p in stored[3])

    def test_binary_probs_are_exact_float_promotions(self):
        import numpy as np

        from leech.inference.helpers import _run_batch

        logits = torch.tensor([[0.3], [-1.7], [2.25]], dtype=torch.float32)
        n = logits.shape[0]
        signals = [np.zeros((1, 4), dtype=np.float32) for _ in range(n)]
        seqs = [np.zeros((4, 3), dtype=np.float32) for _ in range(n)]
        meta = [("readA", 0), ("readA", 5), ("readB", 2)]
        pending: dict = {}
        _run_batch(
            signals, seqs, [None] * n, meta, _FixedLogitsWrapper(logits), False, "cpu", pending
        )

        want = torch.sigmoid(logits).numpy().flatten()
        assert pending["readA"] == [(0, float(want[0])), (5, float(want[1]))]
        assert pending["readB"] == [(2, float(want[2]))]
        assert all(type(p) is float for _, p in pending["readA"])


# ---------------------------------------------------------------------------
# Extraction-path parity: every path through run_inference must agree
#
# `single.py` carries two complete extraction paths (multiprocessing workers
# and the threaded sequential path), and each backend feeds a different chunk
# producer into the same batching state machine. The paths are only allowed to
# differ in what produces chunks — never in what gets predicted.
# ---------------------------------------------------------------------------

_PREDICTION_TAGS = ("aa", "ac", "am", "pn", "pp", "MP", "ML")


def _dump_prediction_tags(bam_path):
    """Per-read prediction tags from an output BAM, as plain comparable Python."""
    rows = []
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for aln in bam:
            row = {"qname": aln.query_name}
            for tag in _PREDICTION_TAGS:
                if aln.has_tag(tag):
                    value = aln.get_tag(tag)
                    if isinstance(value, str):
                        row[tag] = value
                    elif hasattr(value, "__len__"):
                        row[tag] = list(value)
                    else:
                        row[tag] = value
            rows.append(row)
    return rows


# Driver comparing the multiprocessing path against the sequential one. It runs
# in its own interpreter, and runs `--workers 2` FIRST, because the sequential
# path leaves its ThreadPoolExecutors running (`shutdown(wait=False)`) and
# forking an `mp.Pool` after that deadlocks. That hazard predates this test,
# which is also why both runs live in one subprocess rather than two.
_WORKER_PATH_DRIVER = """
import json, sys
from pathlib import Path
import pysam
from leech.bundling import load_model_from_multiclass_bundle
from leech.inference import run_inference

bundle, pod5, bam, ref, workdir = sys.argv[1:6]
work = Path(workdir)


def dump(path):
    rows = []
    with pysam.AlignmentFile(str(path), "rb") as f:
        for aln in f:
            row = {"qname": aln.query_name}
            for tag in ("aa", "ac", "am", "pn", "pp", "MP", "ML"):
                if aln.has_tag(tag):
                    v = aln.get_tag(tag)
                    if isinstance(v, str):
                        row[tag] = v
                    elif hasattr(v, "__len__"):
                        row[tag] = list(v)
                    else:
                        row[tag] = v
            rows.append(row)
    return rows


for workers in (2, 0):
    model, config = load_model_from_multiclass_bundle(Path(bundle), device="cpu")
    out_bam = work / ("w%d.bam" % workers)
    run_inference(
        model_and_config=(model, config),
        pod5_path=Path(pod5),
        bam_path=Path(bam),
        output_path=out_bam,
        device="cpu",
        batch_size=8,
        reverse_signal=True,
        reference_fasta=Path(ref),
        num_workers=workers,
        chunk_size=4,
        backend="python",
        read_batch_size=7,
    )
    (work / ("w%d.json" % workers)).write_text(json.dumps(dump(out_bam), sort_keys=True))
"""


@pytest.mark.skipif(not TRNA_FIXTURES_AVAILABLE, reason="tRNA fixtures not available")
class TestExtractionPathParity:
    """Same input, same model, same predictions — whichever path extracted the chunks."""

    CLASS_NAMES = ["Ala", "Gly", "Met"]

    def _predict(self, tmp_path, bundle_path, name, *, backend, num_workers=0):
        from leech.bundling import load_model_from_multiclass_bundle
        from leech.inference import run_inference

        model, config = load_model_from_multiclass_bundle(bundle_path, device="cpu")
        out = tmp_path / f"{name}.bam"
        run_inference(
            model_and_config=(model, config),
            pod5_path=TRNA_POD5,
            bam_path=TRNA_BAM,
            output_path=out,
            device="cpu",
            batch_size=8,
            reverse_signal=True,
            reference_fasta=TRNA_REF,
            num_workers=num_workers,
            chunk_size=4,
            backend=backend,
            read_batch_size=7,
        )
        return _dump_prediction_tags(out)

    def test_rust_and_python_extraction_agree(self, tmp_path):
        """The two chunk producers on the sequential path predict identically.

        ``--backend rust`` consumes chunks straight from leech_core;
        ``--backend python`` builds them through ``_extract_one_read``. Both
        feed the same accumulator, so a divergence here is in the batching,
        not in the chunks.
        """
        pytest.importorskip("leech_core")
        bundle_path = _create_multiclass_bundle(tmp_path, self.CLASS_NAMES)
        py_rows = self._predict(tmp_path, bundle_path, "seq_py", backend="python")
        rs_rows = self._predict(tmp_path, bundle_path, "seq_rs", backend="rust")
        assert len(py_rows) > 0
        assert any("aa" in r for r in py_rows)
        assert py_rows == rs_rows

    def test_batch_boundaries_do_not_move_predictions(self, tmp_path):
        """The accumulator flushes at ``batch_size`` and drains the remainder.

        A batch of 3 therefore splits the chunk stream differently than one
        that swallows the whole file; the predictions must not notice.
        """
        from leech.bundling import load_model_from_multiclass_bundle
        from leech.inference import run_inference

        bundle_path = _create_multiclass_bundle(tmp_path, self.CLASS_NAMES)
        rows = []
        for batch_size in (3, 4096):
            model, config = load_model_from_multiclass_bundle(bundle_path, device="cpu")
            out = tmp_path / f"bs{batch_size}.bam"
            run_inference(
                model_and_config=(model, config),
                pod5_path=TRNA_POD5,
                bam_path=TRNA_BAM,
                output_path=out,
                device="cpu",
                batch_size=batch_size,
                reverse_signal=True,
                reference_fasta=TRNA_REF,
                backend="python",
            )
            rows.append(_dump_prediction_tags(out))
        assert any("aa" in r for r in rows[0])
        assert rows[0] == rows[1]

    def test_worker_path_matches_sequential_path(self, tmp_path):
        """``--workers N`` predicts exactly what the sequential path predicts.

        The two accumulate into batches differently — the worker path flushes
        at the end of every worker result set, the sequential path at the end
        of every mega-batch — and neither is allowed to move a prediction.

        Runs in a fresh interpreter (see ``_WORKER_PATH_DRIVER``).
        """
        import json
        import subprocess
        import sys

        bundle_path = _create_multiclass_bundle(tmp_path, self.CLASS_NAMES)
        subprocess.run(
            [
                sys.executable,
                "-c",
                _WORKER_PATH_DRIVER,
                str(bundle_path),
                str(TRNA_POD5),
                str(TRNA_BAM),
                str(TRNA_REF),
                str(tmp_path),
            ],
            check=True,
            timeout=900,
        )
        parallel = json.loads((tmp_path / "w2.json").read_text())
        sequential = json.loads((tmp_path / "w0.json").read_text())
        assert any("aa" in r for r in sequential)
        assert parallel == sequential


class TestSequentialThenParallelInProcess:
    """Two `run_inference` calls in one process must not deadlock.

    The sequential path shut its `ThreadPoolExecutor`s down with `wait=False`,
    leaving worker threads alive past the return. The parallel path then forks
    an `mp.Pool`, and a fork inherits the memory of a process with running
    threads -- including any lock those threads hold -- but not the threads
    themselves, so nothing ever releases it. The second call hung forever, with
    no error and no timeout. The CLI never hit this (one predict per process);
    anyone scripting `leech.run_inference` twice did.
    """

    SCRIPT = """
from concurrent.futures import ThreadPoolExecutor

# Stand-in for the sequential path's teardown: threads still running at fork.
pool = ThreadPoolExecutor(max_workers=2)
list(pool.map(lambda i: i * 2, range(8)))
pool.shutdown(wait={wait})

import multiprocessing as mp
with mp.get_context("fork").Pool(processes=2) as p:
    assert p.map(abs, [-1, -2, -3]) == [1, 2, 3]
print("OK")
"""

    def test_a_fork_after_a_waited_shutdown_completes(self, tmp_path):
        """Runs in a subprocess with a hard timeout: the bug under test IS a hang."""
        import subprocess
        import sys

        script = tmp_path / "fork_after_threads.py"
        script.write_text(self.SCRIPT.format(wait=True))
        done = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=120
        )
        assert done.stdout.strip() == "OK", done.stderr

    def test_the_sequential_teardown_waits(self):
        """The guard: `wait=False` here is what reintroduces the hang."""
        from pathlib import Path

        from leech.inference import single as inference_single

        source = Path(inference_single.__file__).read_text()
        teardown = source[source.index("_extract_pool.shutdown") :]
        teardown = teardown[: teardown.index("_bam_write_executor.shutdown") + 60]
        assert "wait=False" not in teardown, (
            "an executor in the sequential teardown shuts down with wait=False; "
            "the mp.Pool fork in the parallel path will inherit its held locks"
        )
