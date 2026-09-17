"""Predict-path parity for a base-defined (``--signal-context-bases``) window.

``tests/test_backend_parity.py`` holds ``data prepare``'s two backends equal
for this option. Predict's two backends diverged from it: ``inference.rs``
had no base-defined window support at all, so ``check_rust_extraction_available``
always forced such a model through the Python path (issue #278 added it to
``training.rs`` only; issue #341 ported it to ``inference.rs``). Before #341
this comparison had nothing to check -- the Rust predict path could not run
this configuration.

Compares ``LeechRead.get_chunk`` (the Python predict path, shared with
``data prepare``) against ``_rs_extract_inference_chunks`` (the Rust predict
path) on the tRNA fixtures, across several ``(L, R, signal_len)`` shapes,
including one that forces the centre-crop branch (a requested window wider
than ``signal_len``).
"""

from __future__ import annotations

import numpy as np
import pysam
import pytest
from conftest import TRNA_BAM, TRNA_FIXTURES_AVAILABLE, TRNA_POD5, TRNA_REF

pytestmark = pytest.mark.skipif(not TRNA_FIXTURES_AVAILABLE, reason="tRNA fixtures not available")

MOTIF = "CCAGGC"
MOTIF_OFFSET = 2
ANCHOR = "reference"


@pytest.fixture(scope="module")
def _rust_available():
    pytest.importorskip("leech_core")
    from leech._rust_accel import HAS_RUST, _rs_extract_inference_chunks

    if not HAS_RUST or _rs_extract_inference_chunks is None:
        pytest.skip("leech_core Rust acceleration not available")


def _run_both_backends(
    *, left_bases: int, right_bases: int, signal_len: int, base_justify: str = "center"
) -> tuple[dict[tuple[str, int], np.ndarray], dict[tuple[str, int], np.ndarray]]:
    """Extract signal windows from both predict backends, keyed by (read_id, base_idx)."""
    from leech._rust_accel import _rs_extract_inference_chunks
    from leech.configs import ChunkConfig, SignalConfig
    from leech.features import extract_move_table
    from leech.inference.helpers import build_rust_extraction_kwargs, collect_bam_metadata_for_rust
    from leech.io import get_motif_searcher, get_reference_sequences
    from leech.io.pod5_reader import POD5Reader
    from leech.preparation.reader import build_leech_read

    reference_sequences = get_reference_sequences(TRNA_BAM, TRNA_REF)
    searcher = get_motif_searcher(
        mode="fasta",
        reference_sequences=reference_sequences,
        skip_indels=False,
        anchor=ANCHOR,
        require_query_mapping=True,
    )

    signal_config = SignalConfig(
        reverse_signal=True,
        anchor=ANCHOR,
        norm_method="median_mad",
        refine_signal_map=False,
    )
    chunk_config = ChunkConfig(
        base_justify=base_justify,
        signal_context_bases=(left_bases, right_bases),
        signal_len=signal_len,
    )

    py_signals: dict[tuple[str, int], np.ndarray] = {}
    with pysam.AlignmentFile(str(TRNA_BAM), "rb") as bam, POD5Reader(TRNA_POD5) as pod5_reader:
        aln_batch = [aln for aln in bam.fetch(until_eof=True) if not aln.is_unmapped]
        for aln in aln_batch:
            read_id = aln.query_name
            read_seq = aln.query_sequence
            if read_id is None or read_seq is None:
                continue
            move_table = extract_move_table(aln)
            raw_signal, pod5_metadata = pod5_reader.get_signal(read_id)
            full_ref = reference_sequences[aln.reference_name]
            ref_seq = full_ref[aln.reference_start : aln.reference_end]

            leech_read = build_leech_read(
                read_id=read_id,
                sequence=read_seq,
                raw_signal=raw_signal,
                move_table=move_table,
                signal_config=signal_config,
                metadata={},
                reference_sequence=ref_seq,
                cigar_tuples=aln.cigartuples,
                cal_offset=pod5_metadata.get("calibration_offset"),
                cal_scale=pod5_metadata.get("calibration_scale"),
            )
            matches = searcher.find_motif_positions(read_id, leech_read.sequence, aln, MOTIF)
            positions = [m.position + MOTIF_OFFSET for m in matches]
            for base_idx in positions:
                chunk = leech_read.get_chunk(base_idx, config=chunk_config)
                if chunk is None:
                    continue
                py_signals[(read_id, base_idx)] = np.asarray(chunk["signal"], dtype=np.float32)

        rs_meta = collect_bam_metadata_for_rust(
            aln_batch,
            motif=MOTIF,
            motif_offset=MOTIF_OFFSET,
            motif_searcher=searcher,
            anchor=ANCHOR,
            reference_sequences=reference_sequences,
        )
        rs_kwargs = build_rust_extraction_kwargs(
            signal_context=(200, 200),
            kmer_context=5,
            signal_len=signal_len,
            compute_features=False,
            reverse_signal=True,
            feature_start=None,
            feature_end=None,
            anchor=ANCHOR,
            seq_encoding="base_onehot",
            signal_kmer_context=(4, 4),
            refine_signal_map=False,
            signal_refiner=None,
            refine_half_bandwidth=5,
            refine_scale_iters=2,
            signal_in_channels=1,
            base_justify=base_justify,
            signal_context_bases=(left_bases, right_bases),
        )
        rs_chunks = _rs_extract_inference_chunks(
            str(TRNA_POD5),
            read_ids=rs_meta[0],
            sequences=rs_meta[1],
            mv_strides=rs_meta[2],
            mv_arrays=rs_meta[3],
            num_samples_list=rs_meta[4],
            trim_offsets=rs_meta[5],
            motif_positions=rs_meta[6],
            cigar_tuples=rs_meta[7],
            reference_sequences=rs_meta[8],
            **rs_kwargs,
        )

    rs_signals = {
        (read_id, int(base_idx)): np.asarray(sig, dtype=np.float32)
        for sig, _seq, _feat, read_id, base_idx in rs_chunks
    }
    return py_signals, rs_signals


class TestSignalContextBasesInferenceParity:
    """Both predict backends must cut the same base-defined signal window."""

    @pytest.mark.parametrize(
        "left_bases,right_bases,signal_len",
        [
            (20, 20, 300),  # narrower than signal_len: left-aligned, zero-padded
            (100, 100, 120),  # wider than signal_len: centre-cropped
        ],
    )
    def test_signals_match(self, _rust_available, left_bases, right_bases, signal_len):
        py_signals, rs_signals = _run_both_backends(
            left_bases=left_bases, right_bases=right_bases, signal_len=signal_len
        )
        assert py_signals, "python backend produced no chunks"
        assert rs_signals, "rust backend produced no chunks"
        assert set(py_signals) == set(rs_signals), (
            "backends disagreed on which chunks were extracted: "
            f"python-only {sorted(set(py_signals) - set(rs_signals))}, "
            f"rust-only {sorted(set(rs_signals) - set(py_signals))}"
        )
        for key, py_sig in py_signals.items():
            np.testing.assert_allclose(
                py_sig, rs_signals[key], atol=1e-3, rtol=1e-3, err_msg=f"signal mismatch at {key}"
            )

    @pytest.mark.parametrize("base_justify", ["start", "end"])
    def test_base_justify(self, _rust_available, base_justify):
        py_signals, rs_signals = _run_both_backends(
            left_bases=10, right_bases=10, signal_len=150, base_justify=base_justify
        )
        assert py_signals and rs_signals
        assert set(py_signals) == set(rs_signals)
        for key, py_sig in py_signals.items():
            np.testing.assert_allclose(py_sig, rs_signals[key], atol=1e-3, rtol=1e-3)
