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
from leech.constants import DEFAULT_KMER_CONTEXT
from leech.io import collect_read_infos, get_reference_sequences
from leech.io.pod5_reader import (
    _DATASET_CACHE,
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
    _DATASET_CACHE.clear()
    yield
    _DATASET_CACHE.clear()


class TestPOD5ReaderCache:
    def test_reuses_reader_handle(self):
        reader1, infos1 = get_cached_reader(TRNA_POD5)
        reader2, infos2 = get_cached_reader(TRNA_POD5)
        assert reader1 is reader2
        assert infos1 is infos2
        assert str(TRNA_POD5) in _DATASET_CACHE

    def test_distinct_paths_cached_separately(self, tmp_path):
        # Copying the same POD5 under a new name gives two distinct cache keys.
        import shutil

        alt = tmp_path / "alt.pod5"
        shutil.copy(TRNA_POD5, alt)

        reader_a, _ = get_cached_reader(TRNA_POD5)
        reader_b, _ = get_cached_reader(alt)
        assert reader_a is not reader_b
        assert set(_DATASET_CACHE.keys()) == {str(TRNA_POD5), str(alt)}

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


def _trna_config(
    anchor: str = "basecall",
    feature_start: int | None = None,
    feature_end: int | None = None,
) -> PrepareConfig:
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
            feature_start=feature_start,
            feature_end=feature_end,
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

    @pytest.mark.parametrize("anchor", ["basecall", "reference"])
    def test_rust_worker_matches_python(self, read_infos, anchor):
        """Both anchor modes must agree across backends.

        ``reference`` is the default and the one that crops the normalized
        signal to the aligned region before refinement, so it exercises
        ``compute_ref_to_signal`` and the crop arithmetic that ``basecall``
        never reaches.
        """
        pytest.importorskip("leech_core")
        from leech._rust_accel import HAS_RUST, _rs_extract_training_chunks

        if not HAS_RUST or _rs_extract_training_chunks is None:
            pytest.skip("leech_core Rust acceleration not available")

        from leech.io import get_motif_searcher
        from leech.preparation.parallel import _prepare_batch_rust, _process_read_chunk_worker

        config = _trna_config(anchor)
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
        # The backends must agree on the chunk SET, not merely overlap on it.
        # This used to allow a 50% overlap because the Rust path dropped focus
        # bases whose k-mer window ran off the end of the sequence while Python
        # padded them with "N"; on production data that was ~1% of reads,
        # silently, biased toward supplementary and indel-heavy alignments
        # (issue #185).
        assert set(py_by_key) == set(rs_by_key), (
            f"chunk sets differ: {len(set(py_by_key) - set(rs_by_key))} python-only, "
            f"{len(set(rs_by_key) - set(py_by_key))} rust-only "
            f"(py={len(py_chunks)}, rs={len(rs_chunks)})"
        )

        for key in sorted(py_by_key):
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
            # Features are what the model actually consumes, so hold them to
            # float32 rounding too.
            py_feat = np.asarray(pc["features"], dtype=np.float32)
            rs_feat = np.asarray(rc["features"], dtype=np.float32)
            assert py_feat.shape == rs_feat.shape, f"features shape {rid}"
            np.testing.assert_allclose(
                py_feat, rs_feat, atol=1e-3, rtol=1e-3, err_msg=f"features {rid}"
            )


class TestEdgeWindowParity:
    """A focus base near either end of the sequence still yields a chunk.

    The fixture motif sits mid-reference, so ordinary parity never reaches the
    k-mer margin. Shifting ``motif_offset`` walks the focus base into it, which
    is what a real read does when its aligned region stops near the motif --
    the supplementary-aligned and indel-heavy population that issue #185 found
    the Rust backend silently discarding.

    Python's rule (``LeechRead.get_chunk``) is that a k-mer window overhanging
    the sequence is padded with "N"; only a focus base with no signal
    boundaries is dropped. Rust must do the same.
    """

    @pytest.fixture
    def read_infos(self):
        return collect_read_infos(TRNA_BAM, min_mapq=0)

    @staticmethod
    def _both_backends(read_infos, anchor, motif_offset):
        from leech.io import get_motif_searcher
        from leech.preparation.parallel import _prepare_batch_rust, _process_read_chunk_worker

        config = _trna_config(anchor)
        config.motif.motif_offset = motif_offset
        motif_searcher = get_motif_searcher(
            mode=config.motif.motif_reference,
            reference_sequences=config.motif.reference_sequences,
            skip_indels=config.motif.skip_motif_indels,
            anchor=config.signal.anchor,
        )
        py = _chunks_by_key(_process_read_chunk_worker((read_infos, config)))
        rs = _chunks_by_key(_prepare_batch_rust(read_infos, config, motif_searcher))
        return py, rs

    # Offsets chosen to push the focus base off the right end (+38/+40) and
    # off the left end (-85) of the ~130 bp aligned reference slice.
    @pytest.mark.parametrize("anchor", ["basecall", "reference"])
    @pytest.mark.parametrize("motif_offset", [38, 40, -85])
    def test_edge_focus_bases_are_padded_not_dropped(self, read_infos, anchor, motif_offset):
        pytest.importorskip("leech_core")
        from leech._rust_accel import HAS_RUST, _rs_extract_training_chunks

        if not HAS_RUST or _rs_extract_training_chunks is None:
            pytest.skip("leech_core Rust acceleration not available")

        py, rs = self._both_backends(read_infos, anchor, motif_offset)

        # The offset must actually reach the margin, or the test proves nothing.
        padded = [k for k, c in py.items() if "N" in str(c["sequence"])]
        assert padded, f"motif_offset={motif_offset} produced no edge chunks to test"

        assert set(py) == set(rs), (
            f"anchor={anchor} motif_offset={motif_offset}: "
            f"{len(set(py) - set(rs))} chunks dropped by Rust that Python kept"
        )
        for key in padded:
            assert str(py[key]["sequence"]) == str(rs[key]["sequence"]), f"padding differs at {key}"
            np.testing.assert_allclose(
                np.asarray(py[key]["signal"], dtype=np.float32),
                np.asarray(rs[key]["signal"], dtype=np.float32),
                atol=1e-5,
                rtol=1e-4,
                err_msg=f"signal {key}",
            )
            np.testing.assert_array_equal(
                np.asarray(py[key]["dwell"], dtype=np.float32),
                np.asarray(rs[key]["dwell"], dtype=np.float32),
                err_msg=f"dwell {key}",
            )
            py_feat = np.asarray(py[key]["features"], dtype=np.float32)
            rs_feat = np.asarray(rs[key]["features"], dtype=np.float32)
            assert py_feat.shape == rs_feat.shape, f"features shape {key}"
            np.testing.assert_allclose(
                py_feat, rs_feat, atol=1e-3, rtol=1e-3, err_msg=f"features {key}"
            )

    def test_basecalled_search_under_reference_anchor(self, read_infos):
        """`--motif-reference bam --anchor reference` searches the reference.

        A legal but unusual pairing: the searcher takes whatever sequence it is
        handed, and under `anchor="reference"` the sequence chunks are cut from
        is the aligned reference slice, so that is what the motif must be found
        in for the returned position to mean anything. The Rust path used to
        hand it the basecall and index the result into the reference anyway.
        """
        pytest.importorskip("leech_core")
        from leech.io import get_motif_searcher, get_reference_sequences
        from leech.preparation.parallel import _prepare_batch_rust, _process_read_chunk_worker

        config = _trna_config("reference")
        # Reference anchoring (so chunks are cut from the reference slice) with
        # basecalled motif search.
        config.motif.motif_reference = "bam"
        config.motif.reference_sequences = get_reference_sequences(TRNA_BAM, TRNA_REF)
        motif_searcher = get_motif_searcher(mode="bam")

        py = _chunks_by_key(_process_read_chunk_worker((read_infos, config)))
        rs = _chunks_by_key(_prepare_batch_rust(read_infos, config, motif_searcher))

        assert py, "fixture should yield chunks in this mode"
        assert set(py) == set(rs), (
            f"{len(set(py) - set(rs))} python-only, {len(set(rs) - set(py))} rust-only"
        )
        for key in py:
            assert str(py[key]["sequence"]) == str(rs[key]["sequence"]), key

    def test_kmer_window_is_full_width_at_the_edge(self, read_infos):
        """An N-padded k-mer is padded, not truncated: width is invariant.

        Checked on both backends -- a Rust k-mer built by slicing a clamped
        range would come back short rather than absent, which the set-equality
        assertion above would not catch.
        """
        pytest.importorskip("leech_core")
        py, rs = self._both_backends(read_infos, "reference", 40)
        expected = {2 * DEFAULT_KMER_CONTEXT + 1}
        assert {len(str(c["sequence"])) for c in py.values()} == expected
        assert {len(str(c["sequence"])) for c in rs.values()} == expected


class TestSignalKmerFieldParity:
    """The two fields `--seq-encoding signal_kmer` runs on must agree too.

    `seq_to_sig_map` and `sequence_with_kmer_context` are derived from the
    SIGNAL window in Python (two `searchsorted` calls over the read's map) and
    used to be derived from the k-mer window in Rust. Those select different
    numbers of bases, so the backends disagreed on every chunk, not just edge
    ones -- issue #186. Nothing compared them, because the other parity tests
    check signal/sequence/dwell/features only.

    Asserted at three levels: the raw fields, their shapes, and the encoding
    they produce, which is what the model actually consumes.
    """

    @pytest.fixture
    def read_infos(self):
        return collect_read_infos(TRNA_BAM, min_mapq=0)

    @pytest.mark.parametrize("anchor", ["basecall", "reference"])
    @pytest.mark.parametrize("motif_offset", [2, 40, -85])
    def test_fields_and_encoding_match(self, read_infos, anchor, motif_offset):
        pytest.importorskip("leech_core")
        from leech._rust_accel import HAS_RUST, _rs_extract_training_chunks

        if not HAS_RUST or _rs_extract_training_chunks is None:
            pytest.skip("leech_core Rust acceleration not available")

        from leech.constants import DEFAULT_SIGNAL_KMER_CONTEXT
        from leech.features import encode_signal_kmer, sequence_to_int
        from leech.io import get_motif_searcher
        from leech.preparation.parallel import _prepare_batch_rust, _process_read_chunk_worker

        config = _trna_config(anchor)
        config.motif.motif_offset = motif_offset
        signal_len = sum(config.chunk.signal_context)
        motif_searcher = get_motif_searcher(
            mode=config.motif.motif_reference,
            reference_sequences=config.motif.reference_sequences,
            skip_indels=config.motif.skip_motif_indels,
            anchor=config.signal.anchor,
        )
        py = _chunks_by_key(_process_read_chunk_worker((read_infos, config)))
        rs = _chunks_by_key(_prepare_batch_rust(read_infos, config, motif_searcher))
        assert py and set(py) == set(rs)

        def encode(chunk):
            seq_ints = sequence_to_int(str(chunk["sequence_with_kmer_context"])).astype(np.int8)
            s2s = np.asarray(chunk["seq_to_sig_map"]).astype(np.int64)
            return encode_signal_kmer(seq_ints, s2s, signal_len, DEFAULT_SIGNAL_KMER_CONTEXT)

        for key in py:
            pc, rc = py[key], rs[key]
            assert str(pc["sequence_with_kmer_context"]) == str(rc["sequence_with_kmer_context"]), (
                f"context sequence {key}"
            )
            np.testing.assert_array_equal(
                np.asarray(pc["seq_to_sig_map"]),
                np.asarray(rc["seq_to_sig_map"]),
                err_msg=f"seq_to_sig_map {key}",
            )
            # The encoder's two inputs must stay dimensionally consistent:
            # len(context) == n + before + after and len(map) == n + 1.
            before, after = DEFAULT_SIGNAL_KMER_CONTEXT
            n_bases = len(np.asarray(rc["seq_to_sig_map"])) - 1
            assert len(str(rc["sequence_with_kmer_context"])) == n_bases + before + after, key
            # And the encoding itself, which is what reaches the model.
            np.testing.assert_array_equal(encode(pc), encode(rc), err_msg=f"encoding {key}")

    @pytest.mark.parametrize("anchor", ["basecall", "reference"])
    def test_inference_path_encodes_the_same(self, read_infos, anchor):
        """The Rust *inference* path uses the same helper — check it agrees.

        It builds the encoding in-process instead of returning the two fields,
        so it is compared against the encoding Python's inference path produces
        from `get_chunk`'s output. Same divergence lived here (#186).
        """
        pytest.importorskip("leech_core")
        from leech._rust_accel import HAS_RUST, _rs_extract_inference_chunks

        if not HAS_RUST or _rs_extract_inference_chunks is None:
            pytest.skip("leech_core Rust acceleration not available")

        from leech.constants import DEFAULT_SIGNAL_KMER_CONTEXT
        from leech.features import encode_signal_kmer, sequence_to_int
        from leech.io import get_motif_searcher
        from leech.preparation.parallel import (
            _extraction_sequence,
            _find_motif_positions,
            _process_read_chunk_worker,
        )

        config = _trna_config(anchor)
        signal_len = sum(config.chunk.signal_context)
        motif_searcher = get_motif_searcher(
            mode=config.motif.motif_reference,
            reference_sequences=config.motif.reference_sequences,
            skip_indels=config.motif.skip_motif_indels,
            anchor=config.signal.anchor,
        )

        # Reference: encode Python's get_chunk output, as inference/single.py does.
        py_enc = {}
        for chunk in _process_read_chunk_worker((read_infos, config)):
            seq_ints = sequence_to_int(str(chunk["sequence_with_kmer_context"])).astype(np.int8)
            s2s = np.asarray(chunk["seq_to_sig_map"]).astype(np.int64)
            py_enc[(chunk["read_id"], int(chunk["base_idx"]))] = encode_signal_kmer(
                seq_ints, s2s, signal_len, DEFAULT_SIGNAL_KMER_CONTEXT
            )

        # Rust inference, signal_kmer encoding, over the same reads.
        kept = [ri for ri in read_infos if _find_motif_positions(ri, motif_searcher, config)]
        assert kept
        mts = [ri.to_move_table() for ri in kept]
        rs_chunks = _rs_extract_inference_chunks(
            str(TRNA_POD5),
            read_ids=[ri.read_id for ri in kept],
            sequences=[_extraction_sequence(ri, config) for ri in kept],
            mv_strides=[mt.stride for mt in mts],
            mv_arrays=[mt.moves.tolist() for mt in mts],
            num_samples_list=[mt.num_samples for mt in mts],
            trim_offsets=[mt.trim_offset for mt in mts],
            motif_positions=[_find_motif_positions(ri, motif_searcher, config) for ri in kept],
            signal_context_left=config.chunk.signal_context[0],
            signal_context_right=config.chunk.signal_context[1],
            kmer_context=config.chunk.kmer_context,
            signal_len=signal_len,
            compute_features=config.signal.compute_features,
            reverse_signal=config.signal.reverse_signal,
            anchor=anchor,
            cigar_tuples=(
                [ri.cigar_tuples or [] for ri in kept] if anchor == "reference" else None
            ),
            reference_sequences=(
                [ri.reference_sequence for ri in kept] if anchor == "reference" else None
            ),
            seq_encoding="signal_kmer",
            base_justify=config.chunk.base_justify,
        )

        assert rs_chunks
        for _sig, seq_enc, _feat, read_id, base_idx in rs_chunks:
            key = (read_id, int(base_idx))
            assert key in py_enc, f"rust produced a chunk python did not: {key}"
            np.testing.assert_array_equal(
                np.asarray(seq_enc), py_enc[key], err_msg=f"signal_kmer encoding {key}"
            )

    def test_map_spans_the_whole_chunk(self, read_infos):
        """The map runs edge to edge: first entry 0, last the chunk width.

        Python snaps the partially-overlapping first and last bases to the
        window edges. A map built from k-mer bounds does not, and can even go
        negative -- which `encode_signal_kmer_inner` silently turns into a
        skipped base rather than a clamped one.
        """
        pytest.importorskip("leech_core")
        from leech.io import get_motif_searcher
        from leech.preparation.parallel import _prepare_batch_rust

        config = _trna_config("reference")
        signal_len = sum(config.chunk.signal_context)
        motif_searcher = get_motif_searcher(
            mode=config.motif.motif_reference,
            reference_sequences=config.motif.reference_sequences,
            skip_indels=config.motif.skip_motif_indels,
            anchor=config.signal.anchor,
        )
        rs = _prepare_batch_rust(read_infos, config, motif_searcher)
        assert rs
        for chunk in rs:
            s2s = np.asarray(chunk["seq_to_sig_map"])
            assert s2s[0] == 0, chunk["read_id"]
            assert s2s[-1] == signal_len, chunk["read_id"]
            assert np.all(s2s >= 0) and np.all(s2s <= signal_len), chunk["read_id"]
            assert np.all(np.diff(s2s) >= 0), chunk["read_id"]


# ---------------------------------------------------------------------------
# End-to-end prepare_training_data_parallel (Rust path only — avoids mp.Pool
# fork inside pytest, which deadlocks with BAM iteration on some environments)
# ---------------------------------------------------------------------------


class TestFeatureWindowParity:
    """A non-default `--feature-start`/`--feature-end` must survive to Rust.

    Every other parity test leaves the window unset, so the backends were only
    ever compared on the default k-mer window. `--feature-start 0` -- features
    beginning AT the focus base, the right-only window used for tRNA 3' ends
    -- is falsy, and the Rust batch path resolved the stored value with
    `config.chunk.feature_start or -5`, stamping -5 onto chunks whose arrays
    actually started at 0 (issue #189).

    Nothing about the array shapes gives that away: `dataset.py` slices the
    k-mer window out of the feature array using the stored `feature_start`, so
    the corpus trains on a window shifted by `kmer_context` bases in silence.
    Hence the metadata is asserted here alongside the arrays.
    """

    @pytest.fixture
    def read_infos(self):
        return collect_read_infos(TRNA_BAM, min_mapq=0)

    @pytest.mark.parametrize(
        ("feature_start", "feature_end"),
        [
            (0, 20),  # right-only, the falsy start that started #189
            (-15, 15),  # wide symmetric
            (5, 25),  # entirely past the focus base
            (None, None),  # default: resolves to +/- kmer_context
        ],
    )
    def test_window_matches_python(self, read_infos, feature_start, feature_end):
        pytest.importorskip("leech_core")
        from leech._rust_accel import HAS_RUST, _rs_extract_training_chunks

        if not HAS_RUST or _rs_extract_training_chunks is None:
            pytest.skip("leech_core Rust acceleration not available")

        from leech.io import get_motif_searcher
        from leech.preparation.parallel import _prepare_batch_rust, _process_read_chunk_worker

        config = _trna_config("reference", feature_start, feature_end)
        motif_searcher = get_motif_searcher(
            mode=config.motif.motif_reference,
            reference_sequences=config.motif.reference_sequences,
            skip_indels=config.motif.skip_motif_indels,
            anchor=config.signal.anchor,
        )
        py = _chunks_by_key(_process_read_chunk_worker((read_infos, config)))
        rs = _chunks_by_key(_prepare_batch_rust(read_infos, config, motif_searcher))
        assert py and set(py) == set(rs)

        exp_start = feature_start if feature_start is not None else -DEFAULT_KMER_CONTEXT
        exp_end = feature_end if feature_end is not None else DEFAULT_KMER_CONTEXT
        exp_width = exp_end - exp_start + 1

        for key in py:
            pc, rc = py[key], rs[key]
            # The window each backend reports, and the window it actually cut.
            assert int(pc["feature_start"]) == exp_start, f"python feature_start {key}"
            assert int(rc["feature_start"]) == exp_start, f"rust feature_start {key}"
            assert int(pc["feature_end"]) == exp_end, f"python feature_end {key}"
            assert int(rc["feature_end"]) == exp_end, f"rust feature_end {key}"
            assert np.asarray(pc["dwell"]).shape == (exp_width,), f"python dwell width {key}"
            assert np.asarray(rc["dwell"]).shape == (exp_width,), f"rust dwell width {key}"
            assert np.asarray(pc["features"]).shape[1] == exp_width, f"python feat width {key}"
            assert np.asarray(rc["features"]).shape[1] == exp_width, f"rust feat width {key}"
            # ... and agree on the contents of it.
            np.testing.assert_array_equal(
                np.asarray(pc["dwell"], dtype=np.float32),
                np.asarray(rc["dwell"], dtype=np.float32),
                err_msg=f"dwell {key}",
            )
            np.testing.assert_allclose(
                np.asarray(pc["features"], dtype=np.float32),
                np.asarray(rc["features"], dtype=np.float32),
                atol=1e-3,
                rtol=1e-3,
                err_msg=f"features {key}",
            )


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


# ---------------------------------------------------------------------------
# Rust backend capability gating
# ---------------------------------------------------------------------------


class TestRustNormMethodGate:
    """The Rust pipeline always applies median-MAD, so other norms must not
    silently reach it — see ``rust/src/inference_pipeline/processing.rs``,
    whose ``PipelineConfig`` carries no normalization field.
    """

    @pytest.mark.parametrize("norm", ["median_mad", None])
    def test_supported_norms(self, norm):
        from leech._rust_accel import rust_supports_norm_method

        assert rust_supports_norm_method(norm)

    @pytest.mark.parametrize("norm", ["zscore", "quantile", "pa_scaling"])
    def test_unsupported_norms(self, norm):
        from leech._rust_accel import rust_supports_norm_method

        assert not rust_supports_norm_method(norm)

    @pytest.mark.parametrize("norm", ["zscore", "quantile", "pa_scaling"])
    def test_prepare_bypasses_rust(self, norm):
        """A non-median_mad prepare run must not take the Rust path."""
        from leech.preparation.parallel import rust_prepare_unsupported_reason

        config = _trna_config()
        config.signal.norm_method = norm
        reason = rust_prepare_unsupported_reason(config)
        assert reason is not None
        assert "normalization" in reason

    def test_prepare_allows_rust_for_median_mad(self):
        from leech.preparation.parallel import rust_prepare_unsupported_reason

        assert rust_prepare_unsupported_reason(_trna_config()) is None

    def test_inference_backend_rust_rejects_unsupported_norm(self):
        """``--backend rust`` must fail loudly rather than mis-normalize."""
        pytest.importorskip("leech_core")
        from leech._rust_accel import HAS_RUST
        from leech.inference.helpers import check_rust_extraction_available

        if not HAS_RUST:
            pytest.skip("leech_core Rust acceleration not available")

        with pytest.raises(RuntimeError, match="only implements"):
            check_rust_extraction_available("rust", "zscore")

    def test_inference_auto_falls_back(self):
        """``--backend auto`` silently prefers Python for unsupported norms."""
        pytest.importorskip("leech_core")
        from leech._rust_accel import HAS_RUST
        from leech.inference.helpers import check_rust_extraction_available

        if not HAS_RUST:
            pytest.skip("leech_core Rust acceleration not available")

        use_rust, *_ = check_rust_extraction_available("auto", "zscore")
        assert use_rust is False
        use_rust, *_ = check_rust_extraction_available("auto", "median_mad")
        assert use_rust is True


class TestRustSoftclipRecoveryGate:
    """``recover_softclip_signal`` fills chunk-window samples outside the
    aligned region from the pre-crop signal. Rust's ``ProcessedRead`` never
    keeps that signal, so the flag must route work to Python instead of being
    silently dropped back to zero-padding.
    """

    def test_flag_off_is_supported(self):
        from leech._rust_accel import rust_supports_softclip_recovery

        assert rust_supports_softclip_recovery(False)

    def test_flag_on_is_unsupported(self):
        from leech._rust_accel import rust_supports_softclip_recovery

        assert not rust_supports_softclip_recovery(True)

    def test_prepare_bypasses_rust(self):
        from leech.preparation.parallel import rust_prepare_unsupported_reason

        config = _trna_config("reference")
        config.chunk.recover_softclip_signal = True
        reason = rust_prepare_unsupported_reason(config)
        assert reason is not None
        assert "recover_softclip_signal" in reason

    def test_focus_map_still_bypasses_rust(self):
        """The pre-existing focus_map bypass must survive the refactor."""
        from leech.preparation.parallel import rust_prepare_unsupported_reason

        config = _trna_config()
        config.labeling.focus_map = {"read-a": (1, 100)}
        reason = rust_prepare_unsupported_reason(config)
        assert reason is not None
        assert "focus_map" in reason

    def test_inference_backend_rust_rejects_flag(self):
        pytest.importorskip("leech_core")
        from leech._rust_accel import HAS_RUST
        from leech.inference.helpers import check_rust_extraction_available

        if not HAS_RUST:
            pytest.skip("leech_core Rust acceleration not available")

        with pytest.raises(RuntimeError, match="recover_softclip_signal"):
            check_rust_extraction_available("rust", "median_mad", True)

    def test_inference_auto_falls_back(self):
        pytest.importorskip("leech_core")
        from leech._rust_accel import HAS_RUST
        from leech.inference.helpers import check_rust_extraction_available

        if not HAS_RUST:
            pytest.skip("leech_core Rust acceleration not available")

        use_rust, *_ = check_rust_extraction_available("auto", "median_mad", True)
        assert use_rust is False
        use_rust, *_ = check_rust_extraction_available("auto", "median_mad", False)
        assert use_rust is True
