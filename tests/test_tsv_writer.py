"""Tests for TsvPredictionWriter."""

import gzip
from unittest.mock import MagicMock

import pytest

from leech.io.tsv_writer import TsvPredictionWriter


@pytest.fixture
def class_names():
    return ["Ala", "Gly", "Ser"]


@pytest.fixture
def int_to_label():
    return {0: "Ala", 1: "Gly", 2: "Ser"}


def _make_alignment(query_name: str, ref_name: str = "tRNA-Ala-AGC-1"):
    aln = MagicMock()
    aln.query_name = query_name
    aln.reference_name = ref_name
    return aln


class TestTsvPredictionWriterInit:
    """Test initialization and header generation."""

    def test_plain_output(self, tmp_path, class_names):
        path = tmp_path / "out.tsv"
        with TsvPredictionWriter(path, class_names, has_cl=False) as w:
            assert w.output_path == path
        header = path.read_text().strip()
        assert header == (
            "read_name\tref_name\tprob_Ala\tprob_Gly\tprob_Ser\tpredicted_aa\tconfidence\tmargin"
            "\tjunction_indel\tjunction_mapped"
        )

    def test_gzip_output(self, tmp_path, class_names):
        path = tmp_path / "out.tsv.gz"
        with TsvPredictionWriter(path, class_names, has_cl=False):
            pass
        with gzip.open(path, "rt") as f:
            header = f.readline().strip()
        assert "prob_Ala" in header

    def test_header_with_cl(self, tmp_path, class_names):
        path = tmp_path / "out.tsv"
        with TsvPredictionWriter(path, class_names, has_cl=True):
            pass
        header = path.read_text().strip()
        assert "predicted_cl" in header.split("\t")
        assert header.endswith("junction_indel\tjunction_mapped")


class TestWritePredictions:
    """Test write_predictions with various inputs."""

    def test_standard_write(self, tmp_path, class_names, int_to_label):
        path = tmp_path / "out.tsv"
        aln = _make_alignment("read_001")
        pending = {
            "read_001": [(0, 0, 0.95, [0.95, 0.03, 0.02])],
        }
        with TsvPredictionWriter(path, class_names, has_cl=False) as w:
            n = w.write_predictions([aln], pending, int_to_label)
        assert n == 1
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        fields = lines[1].split("\t")
        assert fields[0] == "read_001"
        assert fields[1] == "tRNA-Ala-AGC-1"
        assert fields[5] == "Ala"  # predicted_aa
        assert float(fields[6]) == pytest.approx(0.95)  # confidence
        assert float(fields[7]) == pytest.approx(0.95 - 0.03)  # margin

    def test_no_matching_predictions(self, tmp_path, class_names, int_to_label):
        path = tmp_path / "out.tsv"
        aln = _make_alignment("read_999")
        pending = {"read_001": [(0, 0, 0.5, [0.5, 0.3, 0.2])]}
        with TsvPredictionWriter(path, class_names, has_cl=False) as w:
            n = w.write_predictions([aln], pending, int_to_label)
        assert n == 0

    def test_write_with_cl(self, tmp_path, class_names, int_to_label):
        path = tmp_path / "out.tsv"
        aln = _make_alignment("read_002")
        pending = {
            "read_002": [(0, 1, 0.80, [0.10, 0.80, 0.10], 3.14)],
        }
        with TsvPredictionWriter(path, class_names, has_cl=True) as w:
            n = w.write_predictions([aln], pending, int_to_label)
        assert n == 1
        fields = path.read_text().strip().split("\n")[1].split("\t")
        assert fields[5] == "Gly"
        assert float(fields[-3]) == pytest.approx(3.14)  # predicted_cl
        assert fields[-2] == ""  # junction_indel: not measured for a 5-tuple
        assert fields[-1] == "False"  # junction_mapped


class TestEdgeCases:
    """Test confidence and margin edge cases."""

    def test_single_class_margin(self, tmp_path, int_to_label):
        """Single-class probability → margin = 1.0."""
        path = tmp_path / "out.tsv"
        aln = _make_alignment("read_single")
        pending = {
            "read_single": [(0, 0, 1.0, [1.0])],
        }
        with TsvPredictionWriter(path, ["Ala"], has_cl=False) as w:
            w.write_predictions([aln], pending, int_to_label)
        fields = path.read_text().strip().split("\n")[1].split("\t")
        assert float(fields[-3]) == pytest.approx(1.0)  # margin

    def test_multiple_reads(self, tmp_path, class_names, int_to_label):
        path = tmp_path / "out.tsv"
        alns = [_make_alignment(f"read_{i}") for i in range(5)]
        pending = {f"read_{i}": [(0, i % 3, 0.7, [0.7, 0.2, 0.1])] for i in range(5)}
        with TsvPredictionWriter(path, class_names, has_cl=False) as w:
            n = w.write_predictions(alns, pending, int_to_label)
        assert n == 5
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 6  # header + 5 data rows


class TestJunctionIndelColumn:
    """junction_indel/junction_mapped columns and --abstain-on-junction-indel (issue #282)."""

    def test_junction_fields_written_from_seven_tuple(self, tmp_path, class_names, int_to_label):
        path = tmp_path / "out.tsv"
        aln = _make_alignment("read_001")
        pending = {
            # (base_idx, cls_idx, conf, probs, cl_pred, junction_indel, junction_mapped)
            "read_001": [(0, 0, 0.95, [0.95, 0.03, 0.02], None, 2, True)],
        }
        with TsvPredictionWriter(path, class_names, has_cl=False) as w:
            w.write_predictions([aln], pending, int_to_label)
        fields = path.read_text().strip().split("\n")[1].split("\t")
        assert fields[-2] == "2"
        assert fields[-1] == "True"

    def test_abstain_on_junction_indel_marks_disrupted_low_margin_reads_unc(
        self, tmp_path, class_names, int_to_label
    ):
        path = tmp_path / "out.tsv"
        aln = _make_alignment("read_001")
        # margin = 0.55 - 0.45 = 0.1 -> uint8 26, well below min_margin=128;
        # junction_indel=1 (disrupted) and mapped.
        pending = {"read_001": [(0, 0, 0.55, [0.55, 0.45], None, 1, True)]}
        with TsvPredictionWriter(
            path, ["Ala", "Gly"], has_cl=False, min_margin=128, abstain_on_junction_indel=True
        ) as w:
            w.write_predictions([aln], pending, int_to_label)
        fields = path.read_text().strip().split("\n")[1].split("\t")
        assert fields[4] == "unc"  # predicted_aa (2 classes: no prob_Ser column)
        assert fields[-2] == "1"  # junction_indel still recorded, unaltered

    def test_abstain_on_junction_indel_leaves_intact_junction_alone(
        self, tmp_path, class_names, int_to_label
    ):
        """An intact junction bypasses the margin gate entirely under the flag."""
        path = tmp_path / "out.tsv"
        aln = _make_alignment("read_001")
        pending = {"read_001": [(0, 0, 0.55, [0.55, 0.45], None, 0, True)]}
        with TsvPredictionWriter(
            path, ["Ala", "Gly"], has_cl=False, min_margin=128, abstain_on_junction_indel=True
        ) as w:
            w.write_predictions([aln], pending, int_to_label)
        fields = path.read_text().strip().split("\n")[1].split("\t")
        assert fields[4] == "Ala"  # predicted_aa (2 classes: no prob_Ser column)

    def test_without_the_flag_junction_indel_never_triggers_unc(
        self, tmp_path, class_names, int_to_label
    ):
        """Default: predicted_aa is never overwritten, regardless of junction_indel."""
        path = tmp_path / "out.tsv"
        aln = _make_alignment("read_001")
        pending = {"read_001": [(0, 0, 0.55, [0.55, 0.45], None, 5, True)]}
        with TsvPredictionWriter(path, ["Ala", "Gly"], has_cl=False, min_margin=128) as w:
            w.write_predictions([aln], pending, int_to_label)
        fields = path.read_text().strip().split("\n")[1].split("\t")
        assert fields[4] == "Ala"  # predicted_aa (2 classes: no prob_Ser column)


class TestContextManager:
    """Test context manager protocol."""

    def test_enter_returns_self(self, tmp_path, class_names):
        path = tmp_path / "out.tsv"
        writer = TsvPredictionWriter(path, class_names, has_cl=False)
        with writer as w:
            assert w is writer

    def test_exit_closes_file(self, tmp_path, class_names):
        path = tmp_path / "out.tsv"
        with TsvPredictionWriter(path, class_names, has_cl=False):
            pass
        # File should be readable after close
        assert path.read_text().startswith("read_name")

    def test_close_idempotent(self, tmp_path, class_names):
        path = tmp_path / "out.tsv"
        w = TsvPredictionWriter(path, class_names, has_cl=False)
        w.close()
        # Second close should not raise
        w.close()
