"""Integration tests using real nanopore data fixtures."""

from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BAM_FILE = FIXTURES_DIR / "can_mappings.bam"
POD5_FILE = FIXTURES_DIR / "can_reads.pod5"
LEVELS_FILE = FIXTURES_DIR / "levels.txt"

# Skip all tests if fixtures not available
pytestmark = pytest.mark.skipif(not BAM_FILE.exists(), reason="Test fixtures not available")


class TestReadInfoCollection:
    """Test collecting read info from real BAM."""

    def test_collect_read_infos(self):
        from leech.io import collect_read_infos

        infos = collect_read_infos(BAM_FILE, min_mapq=0)
        assert len(infos) == 14
        for info in infos:
            assert info.read_id is not None
            assert len(info.sequence) > 0
            assert info.stride > 0

    def test_read_info_to_move_table(self):
        from leech.io import collect_read_infos

        infos = collect_read_infos(BAM_FILE)
        mt = infos[0].to_move_table()
        sig_map = mt.to_seq_to_sig_map()
        assert sig_map[-1] == mt.num_samples
        assert np.all(np.diff(sig_map) >= 0)


class TestBuildLeechRead:
    """Test building LeechRead from real data."""

    def test_build_from_real_data(self):
        from pod5 import DatasetReader

        from leech.io import collect_read_infos
        from leech.preparation.reader import build_leech_read

        infos = collect_read_infos(BAM_FILE)
        info = infos[0]

        with DatasetReader(POD5_FILE) as reader:
            for read in reader.reads([info.read_id]):
                raw_signal = read.signal
                break

        lr = build_leech_read(
            read_id=info.read_id,
            sequence=info.sequence,
            raw_signal=raw_signal,
            move_table=info.to_move_table(),
            reverse_signal=False,  # DNA data
        )

        assert lr.read_id == info.read_id
        assert len(lr.signal) > 0
        assert len(lr.dwells) == len(lr.sequence)
        assert lr.signal_features["level_mean"].shape == lr.dwells.shape


class TestSignalRefineIntegration:
    """Test signal refinement with real kmer table."""

    def test_load_real_kmer_table(self):
        from leech.signal_refine import load_kmer_table

        kmer_to_level, kmer_len = load_kmer_table(LEVELS_FILE)
        assert kmer_len == 9
        assert len(kmer_to_level) == 4**9  # 262144

    @pytest.mark.slow
    def test_refine_real_data(self):
        from pod5 import DatasetReader

        from leech.io import collect_read_infos
        from leech.preparation.reader import build_leech_read
        from leech.signal_refine import SigMapRefiner

        # Use small bandwidth to keep the pure-Python banded DP tractable on
        # real reads (~7000 bases).  half_bandwidth=5 -> O(n * 10^2) ~ 700K ops.
        refiner = SigMapRefiner.from_table(LEVELS_FILE, half_bandwidth=5)

        infos = collect_read_infos(BAM_FILE)
        info = infos[0]

        with DatasetReader(POD5_FILE) as reader:
            for read in reader.reads([info.read_id]):
                raw_signal = read.signal
                break

        lr = build_leech_read(
            read_id=info.read_id,
            sequence=info.sequence,
            raw_signal=raw_signal,
            move_table=info.to_move_table(),
            reverse_signal=False,
            refine_signal_map=True,
            signal_refiner=refiner,
        )

        # Refined mapping should still be valid
        assert len(lr.seq_to_sig_map) == len(lr.sequence) + 1
        assert np.all(np.diff(lr.seq_to_sig_map) >= 0)


class TestAggregationFunctions:
    """Test inference aggregation functions (pure functions, no fixtures needed)."""

    def test_aggregate_pairwise_basic(self):
        from leech.inference import aggregate_pairwise

        pairs = ["Ala_Gly", "Ala_Ser", "Gly_Ser"]
        # p=0.2 -> Ala gets 0.8, Gly gets 0.2
        # p=0.3 -> Ala gets 0.7, Ser gets 0.3
        # p=0.6 -> Gly gets 0.4, Ser gets 0.6
        probs = [0.2, 0.3, 0.6]
        predicted, confidence, votes = aggregate_pairwise(pairs, probs)
        assert predicted == "Ala"  # Ala: 0.8 + 0.7 = 1.5
        assert votes["Ala"] == pytest.approx(1.5)
        assert votes["Gly"] == pytest.approx(0.6)
        assert votes["Ser"] == pytest.approx(0.9)

    def test_aggregate_pairwise_tie(self):
        from leech.inference import aggregate_pairwise

        pairs = ["A_B"]
        probs = [0.5]
        predicted, confidence, votes = aggregate_pairwise(pairs, probs)
        assert votes["A"] == pytest.approx(0.5)
        assert votes["B"] == pytest.approx(0.5)

    def test_aggregate_one_vs_all_basic(self):
        from leech.inference import aggregate_one_vs_all

        pairs = ["Ala_notAla", "Gly_notGly", "Ser_notSer"]
        probs = [0.1, 0.8, 0.5]  # Ala most likely (lowest p)
        predicted, confidence, scores = aggregate_one_vs_all(pairs, probs)
        assert predicted == "Ala"
        assert scores["Ala"] == pytest.approx(0.9)
        assert scores["Gly"] == pytest.approx(0.2)
        assert scores["Ser"] == pytest.approx(0.5)

    def test_aggregate_one_vs_all_single(self):
        from leech.inference import aggregate_one_vs_all

        pairs = ["X_notX"]
        probs = [0.3]
        predicted, confidence, scores = aggregate_one_vs_all(pairs, probs)
        assert predicted == "X"
        assert confidence == pytest.approx(0.7)
