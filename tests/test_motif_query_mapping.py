"""`require_query_mapping=False` keeps reads whose motif basecalls badly.

Under ``anchor="reference"`` the query mapping computed in
``ReferenceMotifSearcher.find_motif_positions`` is used only to accept or reject:
the coordinate that comes back is ``ref_motif_start - reference_start``, and the
chunk is positioned from the reference through ``compute_ref_to_signal``, which
interpolates across indels. So the mapping is a quality gate, not a requirement
of window placement, and reads that fail it can be kept without moving anything.

That matters when the modification under study mis-calls the motif itself. On
aminoacyl-tRNA the adduct disrupts the basecall at the CCA junction badly enough
that 51.9% of charged reads carry a CIGAR indel across ``CCAGGC`` against 2.4% of
uncharged, and the gate drops 28% of charged reads against 6% of uncharged —
selection on the label, applied before any model sees the data.
"""

import pysam
import pytest

from leech.io.motif_search import ReferenceMotifSearcher, get_motif_searcher

MOTIF = "CCAGGC"
REF_NAME = "ref0"
# Motif starts at reference position 110; the alignment starts at 100.
REF_SEQ = "A" * 110 + MOTIF + "T" * 40
REFS = {REF_NAME: REF_SEQ}
# `reference_name` raises without a header, and the searcher reads it.
HEADER = pysam.AlignmentHeader.from_dict(
    {"HD": {"VN": "1.6"}, "SQ": [{"SN": REF_NAME, "LN": len(REF_SEQ)}]}
)


def _aln_with_insertion_in_motif() -> pysam.AlignedSegment:
    """An alignment whose CIGAR carries a 4 bp insertion inside the motif.

    4 bp rather than 1: the length check tolerates ±3 when ``skip_indels`` is
    False, which is how this project runs it, so a smaller indel would be
    accepted by both settings and the test could not fail.
    """
    aln = pysam.AlignedSegment(header=HEADER)
    aln.query_name = "read0"
    aln.reference_id = 0
    aln.query_sequence = "A" * 54
    aln.reference_start = 100
    # 12M 4I 38M -> the insertion lands at ref 112, inside CCAGGC (110..115).
    aln.cigartuples = [(0, 12), (1, 4), (0, 38)]
    return aln


def test_gate_on_rejects_a_badly_called_motif():
    s = ReferenceMotifSearcher(REFS, skip_indels=False, debug=True)
    assert s.require_query_mapping is True
    pos = s.find_motif_positions("read0", "", _aln_with_insertion_in_motif(), MOTIF)
    assert pos == []
    assert s.stats["failed_length_check"] == 1
    assert s.stats["successful"] == 0


def test_gate_off_keeps_it_and_returns_the_reference_coordinate():
    s = ReferenceMotifSearcher(REFS, skip_indels=False, debug=True,
                               require_query_mapping=False)
    pos = s.find_motif_positions("read0", "", _aln_with_insertion_in_motif(), MOTIF)
    # 110 - 100: reference-relative, exactly what anchor="reference" returns on
    # the clean path. The window therefore lands in the same place.
    assert pos == [10]
    assert s.stats["accepted_without_query_mapping"] == 1
    assert s.stats["successful"] == 1
    assert s.stats["failed_length_check"] == 0


def test_gate_off_agrees_with_gate_on_when_the_motif_is_clean():
    """Relaxing the gate must not MOVE anything — only keep more."""
    aln = pysam.AlignedSegment(header=HEADER)
    aln.query_name = "clean"
    aln.reference_id = 0
    aln.query_sequence = "A" * 50
    aln.reference_start = 100
    aln.cigartuples = [(0, 50)]

    strict = ReferenceMotifSearcher(REFS, skip_indels=False)
    lax = ReferenceMotifSearcher(REFS, skip_indels=False, require_query_mapping=False)
    assert strict.find_motif_positions("clean", "", aln, MOTIF) == [10]
    assert lax.find_motif_positions("clean", "", aln, MOTIF) == [10]


def test_gate_off_is_refused_under_basecall_anchoring():
    """Under anchor="basecall" the returned coordinate IS the query start."""
    with pytest.raises(ValueError, match="only valid with anchor='reference'"):
        ReferenceMotifSearcher(REFS, anchor="basecall", require_query_mapping=False)


def test_factory_threads_the_flag():
    s = get_motif_searcher("fasta", reference_sequences=REFS,
                           require_query_mapping=False)
    assert s.require_query_mapping is False
    assert get_motif_searcher("fasta", reference_sequences=REFS).require_query_mapping
