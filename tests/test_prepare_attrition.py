"""Per-class read attrition by cause, and the summariser's junction strata (#395)."""

import importlib.util
import logging
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pysam
import pytest

from leech.io.motif_search import ReferenceMotifSearcher
from leech.preparation import parallel as par

MOTIF = "CCAGGC"
REF_NAME = "ref0"
REF_SEQ = "A" * 110 + MOTIF + "T" * 40
REFS = {REF_NAME: REF_SEQ}
HEADER = pysam.AlignmentHeader.from_dict(
    {"HD": {"VN": "1.6"}, "SQ": [{"SN": REF_NAME, "LN": len(REF_SEQ)}]}
)


def _aln(cigar, start=100, qlen=54) -> pysam.AlignedSegment:
    aln = pysam.AlignedSegment(header=HEADER)
    aln.query_name = "read0"
    aln.reference_id = 0
    aln.query_sequence = "A" * qlen
    aln.reference_start = start
    aln.cigartuples = cigar
    return aln


INSERTION = [(0, 12), (1, 4), (0, 38)]  # 4I inside the motif


def test_motif_absent_from_aligned_span_is_no_motif():
    # Alignment stops before the motif at reference 110.
    s = ReferenceMotifSearcher(REFS)
    assert s.rejection_cause("read0", "", _aln([(0, 5)], qlen=5), MOTIF) == "no_motif"


def test_strict_gate_reports_indel():
    s = ReferenceMotifSearcher(REFS, skip_indels=True)
    aln = _aln(INSERTION)
    assert s.find_motif_positions("read0", "", aln, MOTIF) == []
    assert s.rejection_cause("read0", "", aln, MOTIF) == "indel"


def test_lenient_gate_reports_length_for_a_big_insertion():
    s = ReferenceMotifSearcher(REFS, skip_indels=False)
    aln = _aln(INSERTION)
    assert s.find_motif_positions("read0", "", aln, MOTIF) == []
    assert s.rejection_cause("read0", "", aln, MOTIF) == "length"


def test_gate_off_keeps_the_read():
    s = ReferenceMotifSearcher(REFS, skip_indels=False, require_query_mapping=False)
    assert s.find_motif_positions("read0", "", _aln(INSERTION), MOTIF)


def _read_info():
    return SimpleNamespace(
        read_id="read0",
        sequence="A" * 54,
        reference_sequence=REF_SEQ[100:150],
        cigar_tuples=INSERTION,
        to_mock_alignment=lambda: _aln(INSERTION),
    )


def _config():
    return SimpleNamespace(
        motif=SimpleNamespace(
            motif=MOTIF, motif_reference="fasta", reference_sequences=REFS, motif_offset=3
        ),
        signal=SimpleNamespace(anchor="reference"),
    )


def test_note_attrition_without_motif_config_counts_no_chunks():
    c = Counter()
    cfg = SimpleNamespace(motif=SimpleNamespace(motif=None))
    par._note_attrition(c, _read_info(), cfg, None, positions_known_empty=False)
    assert c == {"no_chunks": 1}


def test_note_attrition_records_the_gate(monkeypatch):
    searcher = ReferenceMotifSearcher(REFS, skip_indels=True)
    monkeypatch.setattr(par, "_extraction_sequence", lambda ri, cfg: "")
    c = Counter()
    par._note_attrition(c, _read_info(), _config(), searcher, positions_known_empty=True)
    assert c == {"indel": 1}


def test_attrition_summary_is_keyed_by_class_and_warns_on_gating(caplog):
    with caplog.at_level(logging.INFO, logger="leech.preparation.parallel"):
        out = par._attrition_summary("charged", 100, 40, 10, 0, Counter(indel=30, no_motif=20))
    lost = out["charged"]["lost"]
    assert out["charged"]["reads"] == 100
    assert out["charged"]["reads_with_chunks"] == 40
    assert lost["indel"] == 30 and lost["no_motif"] == 20 and lost["missing_from_pod5"] == 10
    assert set(lost) == set(par.ATTRITION_CAUSES)
    assert any("class-selective" in r.message for r in caplog.records)


def test_attrition_summary_quiet_when_no_gate_fired(caplog):
    with caplog.at_level(logging.WARNING, logger="leech.preparation.parallel"):
        par._attrition_summary(None, 10, 10, 0, 0, Counter())
    assert not caplog.records


def _load_summarizer():
    path = Path(__file__).parent.parent / "pipeline/workflow/scripts/summarize_metrics.py"
    spec = importlib.util.spec_from_file_location("summarize_metrics_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("strata", "status"),
    [
        (None, "absent"),
        ({"exact": {"n": 5, "auroc": 0.9}, "disrupted": {"n": 0}}, "disrupted_empty"),
        ({"exact": {"n": 5, "auroc": 0.9}, "disrupted": {"n": 3, "auroc": 0.8}}, "ok"),
    ],
)
def test_summarizer_flags_missing_disrupted_stratum(strata, status):
    pytest.importorskip("pandas")
    mod = _load_summarizer()
    data = {"accuracy": 0.9}
    if strata is not None:
        data["junction_stratified"] = strata
    out = mod.flatten_junction_strata(data)
    assert out["junction_status"] == status
    assert "junction_stratified" not in out
    if status == "ok":
        assert out["junction_disrupted_auroc"] == 0.8
        assert out["junction_exact_n"] == 5
