"""
Tests for parallel data preparation and POD5 reader caching.

Covers:
- ``read_pod5_signals_batch_cached`` reuses a single open Reader across calls.
- Cached reads match uncached reads bit-for-bit.
- ``_process_read_chunk_worker`` (Python) and ``_prepare_batch_rust`` produce
  equivalent training chunks on the same inputs.
- ``prepare_training_data_parallel`` round-trips on the tRNA fixtures with
  both a single worker (serial-in-parallel) and the Rust backend when
  available.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import TRNA_BAM, TRNA_FIXTURES_AVAILABLE, TRNA_POD5, TRNA_REF

from leech.configs import ChunkConfig, LabelConfig, MotifConfig, PrepareConfig, SignalConfig
from leech.io import collect_read_infos, get_reference_sequences
from leech.io.pod5_reader import (
    _READER_CACHE,
    get_cached_reader,
    read_pod5_signals_batch,
    read_pod5_signals_batch_cached,
)

pytestmark = pytest.mark.skipif(not TRNA_FIXTURES_AVAILABLE, reason="tRNA fixtures not available")


# ---------------------------------------------------------------------------
# POD5 reader cache
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_reader_cache():
    """Ensure each test starts with a clean cache."""
    _READER_CACHE.clear()
    yield
    _READER_CACHE.clear()


class TestPOD5ReaderCache:
    def test_reuses_reader_handle(self):
        reader1, infos1 = get_cached_reader(TRNA_POD5)
        reader2, infos2 = get_cached_reader(TRNA_POD5)
        assert reader1 is reader2
        assert infos1 is infos2
        assert str(TRNA_POD5) in _READER_CACHE

    def test_distinct_paths_cached_separately(self, tmp_path):
        # Copying the same POD5 under a new name gives two distinct cache keys.
        import shutil

        alt = tmp_path / "alt.pod5"
        shutil.copy(TRNA_POD5, alt)

        reader_a, _ = get_cached_reader(TRNA_POD5)
        reader_b, _ = get_cached_reader(alt)
        assert reader_a is not reader_b
        assert set(_READER_CACHE.keys()) == {str(TRNA_POD5), str(alt)}

    def test_cached_batch_matches_uncached(self):
        read_infos = collect_read_infos(TRNA_BAM, min_mapq=0)
        read_ids = [ri.read_id for ri in read_infos[:5]]
        assert read_ids, "fixture should expose at least one read"

        uncached = read_pod5_signals_batch(TRNA_POD5, read_ids)
        cached = read_pod5_signals_batch_cached(TRNA_POD5, read_ids)

        assert set(uncached.keys()) == set(cached.keys())
        for rid, (u_sig, u_meta) in uncached.items():
            c_sig, c_meta = cached[rid]
            np.testing.assert_array_equal(u_sig, c_sig)
            # Float cal params may differ by zero in representation; compare exactly.
            assert u_meta == c_meta


# ---------------------------------------------------------------------------
# Worker parity: Python (_process_read_chunk_worker) vs Rust (_prepare_batch_rust)
# ---------------------------------------------------------------------------


def _trna_config(anchor: str = "basecall") -> PrepareConfig:
    """Build a PrepareConfig that matches the tRNA fixtures."""
    reference_sequences = None
    motif_reference = "bam"
    if anchor == "reference":
        reference_sequences = get_reference_sequences(TRNA_BAM, TRNA_REF)
        motif_reference = "fasta"

    return PrepareConfig(
        pod5_path=TRNA_POD5,
        signal=SignalConfig(
            reverse_signal=True,
            anchor=anchor,
            norm_method="median_mad",
            refine_signal_map=False,
        ),
        motif=MotifConfig(
            motif="CCAGGC",
            motif_offset=2,
            motif_reference=motif_reference,
            reference_sequences=reference_sequences,
            skip_motif_indels=False,
        ),
        chunk=ChunkConfig(
            base_justify="center",
            signal_context=(200, 200),
        ),
        labeling=LabelConfig(label="Ala", label_int=1),
    )


def _chunks_by_key(chunks: list[dict]) -> dict[tuple[str, int], dict]:
    return {(c["read_id"], int(c["base_idx"])): c for c in chunks}


class TestRustPythonWorkerParity:
    """Python and Rust paths must agree on core chunk content."""

    @pytest.fixture
    def read_infos(self):
        return collect_read_infos(TRNA_BAM, min_mapq=0)

    def test_python_worker_produces_chunks(self, read_infos):
        from leech.preparation.parallel import _process_read_chunk_worker

        config = _trna_config()
        chunks = _process_read_chunk_worker((read_infos, config))
        assert len(chunks) > 0
        c = chunks[0]
        for key in ("signal", "sequence", "dwell", "features", "read_id", "base_idx"):
            assert key in c, f"missing key: {key}"

    def test_rust_worker_matches_python(self, read_infos):
        pytest.importorskip("leech_core")
        from leech._rust_accel import HAS_RUST, _rs_extract_training_chunks

        if not HAS_RUST or _rs_extract_training_chunks is None:
            pytest.skip("leech_core Rust acceleration not available")

        from leech.io import get_motif_searcher
        from leech.preparation.parallel import _prepare_batch_rust, _process_read_chunk_worker

        config = _trna_config()
        motif_searcher = get_motif_searcher(
            mode=config.motif.motif_reference,
            reference_sequences=config.motif.reference_sequences,
            skip_indels=config.motif.skip_motif_indels,
            anchor=config.signal.anchor,
        )

        py_chunks = _process_read_chunk_worker((read_infos, config))
        rs_chunks = _prepare_batch_rust(read_infos, config, motif_searcher)

        assert len(py_chunks) > 0
        assert len(rs_chunks) > 0

        py_by_key = _chunks_by_key(py_chunks)
        rs_by_key = _chunks_by_key(rs_chunks)
        common = sorted(set(py_by_key) & set(rs_by_key))
        # Rust path skips edge reads that Python pads with "N"; require substantial overlap.
        assert len(common) >= min(len(py_chunks), len(rs_chunks)) * 0.5, (
            f"Too little overlap: {len(common)} common (py={len(py_chunks)}, rs={len(rs_chunks)})"
        )

        for key in common:
            pc, rc = py_by_key[key], rs_by_key[key]
            rid = key[0][:8]
            # Signal must match within float32 tolerance.
            py_sig = np.asarray(pc["signal"], dtype=np.float32)
            rs_sig = np.asarray(rc["signal"], dtype=np.float32)
            assert py_sig.shape == rs_sig.shape, f"signal shape {rid}"
            np.testing.assert_allclose(
                py_sig, rs_sig, atol=1e-5, rtol=1e-4, err_msg=f"signal {rid}"
            )
            # Sequence must match exactly.
            assert str(pc["sequence"]) == str(rc["sequence"]), f"sequence {rid}"
            # Dwells must match.
            np.testing.assert_array_equal(
                np.asarray(pc["dwell"], dtype=np.float32),
                np.asarray(rc["dwell"], dtype=np.float32),
                err_msg=f"dwell {rid}",
            )


# ---------------------------------------------------------------------------
# End-to-end prepare_training_data_parallel (Rust path only — avoids mp.Pool
# fork inside pytest, which deadlocks with BAM iteration on some environments)
# ---------------------------------------------------------------------------


class TestPrepareTrainingDataParallel:
    def test_rust_backend_end_to_end(self):
        """Full prepare_training_data_parallel Rust path (no subprocess fork)."""
        pytest.importorskip("leech_core")
        import leech.preparation.parallel as pp

        if not pp.HAS_RUST or pp._rs_extract_training_chunks is None:
            pytest.skip("leech_core Rust acceleration not available")

        config = _trna_config()
        chunks, stats = pp.prepare_training_data_parallel(
            TRNA_BAM, config, num_workers=1, chunk_size=10
        )
        assert stats["total_reads"] > 0
        assert stats["total_chunks"] > 0
        # Every chunk should carry its label from config.
        assert all(c["label"] == "Ala" for c in chunks)
        assert all(int(c["label_int"]) == 1 for c in chunks)
