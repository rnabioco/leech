"""
Test Rust vs Python parity for chunk extraction and signal refinement.

Covers:
- Sequence encoding: base_onehot, signal_kmer
- End-to-end: extract_inference_chunks on real fixture data (no refinement)
- Rust monolithic refinement: runs, preserves chunk structure, has effect

Python-vs-Rust equality of the actual refinement *settings* -- across
anchors, ``scale_iters``, and the rest of the extraction matrix -- is
``tests/test_backend_parity.py``'s job; it compares full training corpora
field by field. This file used to also compare the low-level banded-DP
primitives (``extract_levels``, ``rough_rescale_quantile``, ``seq_banded_dp``)
against Rust-accelerated ``leech_core`` bindings of the same functions, but
those bindings only ever served that comparison -- ``SigMapRefiner.refine()``
delegates the real refinement entirely to escapepod-signal, on both backends
-- and were removed in #276 along with the reference implementation itself
(now a frozen oracle at ``tests/reference_signal_refine.py``, exercised by
``tests/test_signal_refine.py``).
"""

from pathlib import Path

import numpy as np
import pysam
import pytest

from leech.features import (
    MoveTable,
    compute_dwell_features,
    compute_signal_features,
    extract_move_table,
    normalize_read_signal,
    sequence_to_int,
)
from leech.signal_refine import load_kmer_table

# Skip entire module if leech_core not built
pytest.importorskip("leech_core")

from leech._rust_accel import (  # noqa: E402
    _rs_extract_inference_chunks,
    make_kmer_levels,
)

FIXTURES = Path(__file__).parent / "fixtures"
LEVELS_TABLE = FIXTURES / "levels.txt"
TRNA_BAM = FIXTURES / "trna_mappings.bam"
TRNA_POD5 = FIXTURES / "trna_reads.pod5"
TRNA_REF = FIXTURES / "trna_reference.fa"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_real_reads(max_reads: int = 5):
    """Load real reads from fixture data."""
    from escapepod import Reader

    reads = []
    with pysam.AlignmentFile(str(TRNA_BAM), "rb") as bam:
        for aln in bam.fetch(until_eof=True):
            if aln.query_name is None or aln.query_sequence is None:
                continue
            try:
                mt = extract_move_table(aln)
            except Exception:
                continue
            reads.append(
                {
                    "read_id": aln.query_name,
                    "sequence": aln.query_sequence,
                    "stride": mt.stride,
                    "moves": mt.moves,
                    "num_samples": mt.num_samples,
                    "trim_offset": mt.trim_offset,
                    "aln": aln,
                }
            )
            if len(reads) >= max_reads:
                break

    # Fetch signals
    rids = [r["read_id"] for r in reads]
    pod5 = Reader(str(TRNA_POD5))
    read_datas = pod5.get_reads(rids)
    for read_data in read_datas:
        rid = read_data.read_id
        signal = pod5.get_signal(read_data)
        for r in reads:
            if r["read_id"] == rid:
                r["raw_signal"] = signal
                break

    return [r for r in reads if "raw_signal" in r]


# ---------------------------------------------------------------------------
# Tests: Full monolithic pipeline on real data
# ---------------------------------------------------------------------------


class TestMonolithicPipelineParity:
    """Test extract_inference_chunks (Rust monolithic) vs Python path on real data."""

    @pytest.fixture(scope="class")
    def kmer_table(self):
        return load_kmer_table(LEVELS_TABLE)

    def _python_extract_one_read(self, r, kmer_to_level, kmer_len, motif_positions):
        """Run the Python extraction pipeline for one read."""
        raw = r["raw_signal"]
        ts = r["trim_offset"]
        ns = r["num_samples"]
        trimmed = raw[ts:ns].astype(np.float32)[::-1].copy()
        norm, _ = normalize_read_signal(trimmed, method="median_mad")

        mt = MoveTable(
            stride=r["stride"],
            moves=r["moves"],
            read_id=r["read_id"],
            num_samples=ns,
            trim_offset=ts,
        )
        sig_map = mt.to_seq_to_sig_map() - ts
        sig_len_val = ns - ts
        sig_map = sig_len_val - sig_map[::-1]

        seq = r["sequence"]
        num_bases = len(sig_map) - 1
        dwells = np.diff(sig_map).astype(np.float32)

        # Features
        dwell_feats = compute_dwell_features(dwells)
        sig_feats = compute_signal_features(norm, sig_map)

        feat_rows = []
        for key in ["dwell", "dwell_log", "dwell_mean", "dwell_std", "dwell_ratio"]:
            feat_rows.append(dwell_feats[key])
        for key in ["level_mean", "level_median", "level_std", "level_range"]:
            feat_rows.append(sig_feats[key])
        features = np.stack(feat_rows, axis=0)

        # Sequence encoding (base_onehot)
        seq_int = sequence_to_int(seq)

        # Extract chunks at motif positions
        signal_context_left = 200
        signal_context_right = 200
        kmer_ctx = 5
        chunks = []
        for pos in motif_positions:
            if pos < 0 or pos >= num_bases:
                continue
            sig_start = int(sig_map[pos])
            sig_end = int(sig_map[pos + 1]) if pos + 1 < len(sig_map) else int(sig_map[-1])
            sig_center = (sig_start + sig_end) // 2

            # Signal chunk
            chunk_start = max(0, sig_center - signal_context_left)
            chunk_end = min(len(norm), sig_center + signal_context_right)
            signal_len = signal_context_left + signal_context_right
            if chunk_end - chunk_start < signal_len:
                continue
            sig_chunk = norm[chunk_start : chunk_start + signal_len]

            # Kmer context
            kmer_start = max(0, pos - kmer_ctx)
            kmer_end = min(num_bases, pos + kmer_ctx + 1)
            kmer_win = 2 * kmer_ctx + 1
            if kmer_end - kmer_start < kmer_win:
                continue

            # Sequence encoding slice
            seq_slice = seq_int[kmer_start : kmer_start + kmer_win]
            seq_enc = np.zeros((4, kmer_win), dtype=np.float32)
            for j, base_val in enumerate(seq_slice):
                if 0 <= base_val < 4:
                    seq_enc[base_val, j] = 1.0

            # Feature slice
            feat_slice = features[:, kmer_start : kmer_start + kmer_win].copy()

            chunks.append(
                {
                    "signal": sig_chunk,
                    "seq_enc": seq_enc,
                    "features": feat_slice,
                    "read_id": r["read_id"],
                    "base_idx": pos,
                }
            )

        return chunks

    def test_extract_chunks_no_refinement(self, kmer_table):
        """Compare Rust vs Python chunk extraction without signal refinement."""
        reads = _load_real_reads(5)
        if not reads:
            pytest.skip("No fixture reads")

        for r in reads:
            # Find a motif position near the middle
            num_bases = len(r["sequence"])
            mid = num_bases // 2
            # Pick a position with enough context
            test_positions = [mid]

            py_chunks = self._python_extract_one_read(
                r, kmer_table[0], kmer_table[1], test_positions
            )

            if not py_chunks:
                continue

            # Rust monolithic extraction
            # Positional: pod5, read_ids, sequences, mv_strides, mv_arrays,
            #   num_samples, trim_offsets, sig_ctx_left, sig_ctx_right,
            #   kmer_ctx, motif_positions, signal_len, compute_features
            rs_chunks = _rs_extract_inference_chunks(
                str(TRNA_POD5),
                [r["read_id"]],
                [r["sequence"]],
                [r["stride"]],
                [r["moves"].view(np.uint8)],
                [r["num_samples"]],
                [r["trim_offset"]],
                200,  # signal_context_left
                200,  # signal_context_right
                5,  # kmer_context
                [test_positions],
                400,  # signal_len = left + right
                True,  # compute_features
                True,  # reverse_signal
            )

            assert len(rs_chunks) == len(py_chunks), (
                f"Chunk count mismatch for {r['read_id']}: "
                f"Rust={len(rs_chunks)}, Python={len(py_chunks)}"
            )

            for i, (rs, py) in enumerate(zip(rs_chunks, py_chunks, strict=True)):
                rs_sig, rs_seq, rs_feat, rs_rid, rs_bidx = rs

                np.testing.assert_allclose(
                    np.asarray(rs_sig),
                    py["signal"],
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Signal mismatch for {r['read_id']} chunk {i}",
                )

                np.testing.assert_array_equal(
                    np.asarray(rs_seq),
                    py["seq_enc"],
                    err_msg=f"Seq encoding mismatch for {r['read_id']} chunk {i}",
                )

                if rs_feat is not None:
                    np.testing.assert_allclose(
                        np.asarray(rs_feat),
                        py["features"],
                        rtol=1e-5,
                        atol=1e-6,
                        err_msg=f"Feature mismatch for {r['read_id']} chunk {i}",
                    )

    def test_extract_chunks_with_refinement(self, kmer_table):
        """Compare chunk extraction WITH signal refinement enabled."""
        kmer_to_level, kmer_len = kmer_table
        reads = _load_real_reads(3)
        if not reads:
            pytest.skip("No fixture reads")

        for r in reads:
            num_bases = len(r["sequence"])
            mid = num_bases // 2
            test_positions = [mid]

            # Rust monolithic with refinement
            rs_chunks_refined = _rs_extract_inference_chunks(
                str(TRNA_POD5),
                [r["read_id"]],
                [r["sequence"]],
                [r["stride"]],
                [r["moves"].view(np.uint8)],
                [r["num_samples"]],
                [r["trim_offset"]],
                200,
                200,
                5,  # signal_context_left/right, kmer_context
                [test_positions],
                400,
                True,
                True,  # signal_len, compute_features, reverse_signal
                refine_signal_map=True,
                kmer_table=make_kmer_levels(kmer_to_level),
                kmer_len=kmer_len,
                kmer_center_idx=kmer_len // 2,
                refine_half_bandwidth=5,
                refine_scale_iters=2,
            )

            # Rust monolithic WITHOUT refinement (control)
            rs_chunks_plain = _rs_extract_inference_chunks(
                str(TRNA_POD5),
                [r["read_id"]],
                [r["sequence"]],
                [r["stride"]],
                [r["moves"].view(np.uint8)],
                [r["num_samples"]],
                [r["trim_offset"]],
                200,
                200,
                5,
                [test_positions],
                400,
                True,
                True,
            )

            if not rs_chunks_refined or not rs_chunks_plain:
                continue

            # Refinement should change the signal (different normalization)
            # but not crash. The key test is that it runs without error
            # and produces valid output.
            rs_sig_r = np.asarray(rs_chunks_refined[0][0])
            rs_sig_p = np.asarray(rs_chunks_plain[0][0])

            assert rs_sig_r.shape == rs_sig_p.shape, "Shape mismatch after refinement"
            assert np.isfinite(rs_sig_r).all(), "Non-finite values in refined signal"

            # Features should differ after refinement (different sig_map → different dwells)
            if rs_chunks_refined[0][2] is not None:
                rs_feat_r = np.asarray(rs_chunks_refined[0][2])
                assert np.isfinite(rs_feat_r).all(), "Non-finite values in refined features"


# ---------------------------------------------------------------------------
# Tests: refinement-enabled monolithic pipeline (escapepod-backed refine)
# ---------------------------------------------------------------------------


class TestMonolithicRefinementPipeline:
    """Exercise the refinement-enabled Rust monolithic extractor.

    leech_core delegates its signal-map refinement to escapepod-signal's
    ``resquiggle::refine_signal_map``. The rest of the parity suite runs the
    monolithic extractor with refinement OFF, so this class covers the
    refinement path directly: it must run, preserve chunk structure, produce
    finite/bounded features, and measurably adjust the output vs no refinement.
    """

    @pytest.fixture(scope="class")
    def kmer_table(self):
        return load_kmer_table(LEVELS_TABLE)

    def _extract(self, reads, refine, kmer_to_level, kmer_len):
        rids = [r["read_id"] for r in reads]
        seqs = [r["sequence"] for r in reads]
        strides = [r["stride"] for r in reads]
        movs = [r["moves"].view(np.uint8) for r in reads]
        nss = [r["num_samples"] for r in reads]
        tss = [r["trim_offset"] for r in reads]
        motif_positions = [[len(r["sequence"]) // 2] for r in reads]
        return _rs_extract_inference_chunks(
            str(TRNA_POD5),
            rids,
            seqs,
            strides,
            movs,
            nss,
            tss,
            200,  # signal_context_left
            200,  # signal_context_right
            5,  # kmer_context
            motif_positions,
            400,  # signal_len
            True,  # compute_features
            reverse_signal=True,
            refine_signal_map=refine,
            kmer_table=(make_kmer_levels(kmer_to_level) if refine else None),
            kmer_len=kmer_len,
            refine_half_bandwidth=5,
            refine_scale_iters=2,
        )

    def test_refinement_runs_preserves_structure_and_has_effect(self, kmer_table):
        kmer_to_level, kmer_len = kmer_table
        reads = _load_real_reads(6)
        if not reads:
            pytest.skip("No fixture reads")

        off = self._extract(reads, False, kmer_to_level, kmer_len)
        on = self._extract(reads, True, kmer_to_level, kmer_len)

        assert len(on) == len(off) and len(on) > 0, "refinement changed chunk count"

        any_sig_changed = False
        for on_chunk, off_chunk in zip(on, off, strict=True):
            sig_on, _, feat_on, rid_on, _ = on_chunk
            sig_off, _, feat_off, rid_off, _ = off_chunk
            assert rid_on == rid_off
            sig_on = np.asarray(sig_on)
            feat_on = np.asarray(feat_on)

            # Signal chunk shape is fixed (signal_len) either way.
            assert sig_on.shape == np.asarray(sig_off).shape == (400,)
            # Refinement enables kmer-residual features (expected levels present):
            # 9 base features (5 dwell + 4 level) without refine, +3 (ke/kr/kra) with.
            assert np.asarray(feat_off).shape[0] == 9
            assert feat_on.shape[0] == 12
            assert feat_on.shape[1] == np.asarray(feat_off).shape[1]

            # Finite + bounded (normalized signal must stay sane)
            assert np.all(np.isfinite(sig_on)), f"non-finite signal for {rid_on}"
            assert np.all(np.isfinite(feat_on)), f"non-finite features for {rid_on}"
            assert np.abs(sig_on).max() < 50.0, f"signal out of range for {rid_on}"

            if not np.allclose(sig_on, np.asarray(sig_off), atol=1e-6):
                any_sig_changed = True

        assert any_sig_changed, "refinement had no measurable effect on the signal"

    def test_refinement_dwell_features_positive(self, kmer_table):
        """Refined per-base dwell (feature row 0) must stay positive."""
        kmer_to_level, kmer_len = kmer_table
        reads = _load_real_reads(6)
        if not reads:
            pytest.skip("No fixture reads")

        on = self._extract(reads, True, kmer_to_level, kmer_len)
        assert on, "no chunks produced with refinement"
        for _sig, _seq, feat, rid, _bidx in on:
            feat = np.asarray(feat)
            dwell_row = feat[0]  # raw dwell feature
            assert np.all(dwell_row >= 0.0), f"negative dwell after refinement for {rid}"
