"""Field-by-field parity between the two ``data prepare`` backends.

Every divergence found so far was invisible to the check that caught the
previous one:

- #185 chunk **counts** differed; the check compared counts, so it caught that
  and nothing else.
- #186 the ``signal_kmer`` fields differed; counts matched, so nothing noticed
  until a full field diff was run by hand.
- #189/#190 the stored feature **window** was wrong; counts and shapes matched.
- #193 the feature **values** were wrong; counts, shapes and the stored window
  all matched.

The pattern is a hand-written check per field, extended one field at a time,
always one behind. This module inverts that: it serializes both backends
through ``save_chunks`` and compares **every array in the npz**, and fails on
any field it has not been told how to compare. Adding a field to the chunk
format without classifying it here is a test failure, not a silent gap.

Comparison is by field kind, not by name:

- integer, string and object fields must be **exactly** equal;
- float fields are compared with a tolerance, because the two backends
  genuinely accumulate differently (Python computes per-base statistics in
  float64 via numpy and casts to float32; Rust accumulates in float32). On the
  fixtures that shows up as e.g. ``dwell_ratio`` differing in 94% of elements
  at a maximum absolute difference of 1.3e-07. Requiring exact equality there
  is unfixable-red; requiring nothing is how #193 survived. The tolerance is
  tight enough that any real divergence -- all of the ones above moved values
  by 0.5% to 300% -- fails it.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import LEVELS_FILE, TRNA_BAM, TRNA_FIXTURES_AVAILABLE, TRNA_POD5, TRNA_REF

from leech.chunking import save_chunks
from leech.configs import ChunkConfig, LabelConfig, MotifConfig, PrepareConfig, SignalConfig
from leech.io import collect_read_infos, get_motif_searcher, get_reference_sequences

pytestmark = pytest.mark.skipif(not TRNA_FIXTURES_AVAILABLE, reason="tRNA fixtures not available")

#: Float fields compare with a tolerance; everything else must match exactly.
#: Listed by name rather than inferred from dtype so that a field silently
#: changing dtype does not silently relax its comparison.
FLOAT_FIELDS = {
    "signals_flat",
    "signals",
    "signal_residuals_flat",
    "signal_residuals",
    "features_flat",
    "features",
}

#: Fields compared exactly. `dwells` are counts of signal samples: integral
#: even though they are stored as float32, so they get no tolerance -- an
#: off-by-one dwell is a real boundary difference, which is exactly what #193
#: turned out to be.
EXACT_FIELDS = {
    "dwells_flat",
    "dwells",
    "sequences",
    "labels",
    "labels_int",
    "read_ids",
    "base_indices",
    "feature_starts",
    "feature_ends",
    "source_groups",
    "reference_names",
    "seq_to_sig_maps",
    "seq_to_sig_values",
    "seq_to_sig_offsets",
    "sequences_with_kmer_context",
    "cl_values",
    "focus_signal_pos",
}

#: Absolute/relative tolerance for float fields. Comfortably above float32
#: accumulation noise (~1e-05 worst case on the fixtures) and far below any
#: divergence this suite is meant to catch.
FLOAT_ATOL = 1e-3
FLOAT_RTOL = 1e-3


def _config(
    *,
    anchor: str = "reference",
    refine: bool = True,
    scale_iters: int = 2,
    base_justify: str = "center",
    feature_start: int | None = None,
    feature_end: int | None = None,
    signal_context: tuple[int, int] = (200, 200),
    require_query_mapping: bool = True,
) -> PrepareConfig:
    reference_sequences = None
    motif_reference = "bam"
    if anchor == "reference":
        reference_sequences = get_reference_sequences(TRNA_BAM, TRNA_REF)
        motif_reference = "fasta"

    signal_refiner = None
    if refine:
        from leech.signal_refine import SigMapRefiner

        signal_refiner = SigMapRefiner.from_table(LEVELS_FILE, scale_iters=scale_iters)

    return PrepareConfig(
        pod5_path=TRNA_POD5,
        signal=SignalConfig(
            reverse_signal=True,
            anchor=anchor,
            norm_method="median_mad",
            refine_signal_map=refine,
            refine_scale_iters=scale_iters,
            signal_refiner=signal_refiner,
        ),
        motif=MotifConfig(
            motif="CCAGGC",
            motif_offset=2,
            motif_reference=motif_reference,
            reference_sequences=reference_sequences,
            require_query_mapping=require_query_mapping,
        ),
        chunk=ChunkConfig(
            base_justify=base_justify,
            signal_context=signal_context,
            feature_start=feature_start,
            feature_end=feature_end,
        ),
        labeling=LabelConfig(label="Ala", label_int=1),
    )


def _run_both_backends(config: PrepareConfig, tmp_path) -> tuple[dict, dict]:
    """Serialize both backends' chunks and return the two npz payloads.

    Goes through ``save_chunks`` rather than comparing chunk dicts directly:
    that is the layer a corpus is actually consumed at, it is where defaults
    like ``source_groups`` and ``feature_starts`` are stamped, and it is the
    layer the production divergence reports compared.
    """
    from leech.preparation.parallel import _prepare_batch_rust, _process_read_chunk_worker

    read_infos = collect_read_infos(TRNA_BAM, min_mapq=0)
    searcher = get_motif_searcher(
        mode=config.motif.motif_reference,
        reference_sequences=config.motif.reference_sequences,
        skip_indels=config.motif.skip_motif_indels,
        anchor=config.signal.anchor,
        require_query_mapping=config.motif.require_query_mapping,
    )

    py_chunks = _process_read_chunk_worker((read_infos, config))
    rs_chunks = _prepare_batch_rust(read_infos, config, searcher)
    assert py_chunks, "python backend produced no chunks"
    assert rs_chunks, "rust backend produced no chunks"

    # Both backends emit in their own order; sort by the identity of a chunk.
    key = lambda c: (str(c["read_id"]), int(c["base_idx"]))  # noqa: E731
    py_chunks.sort(key=key)
    rs_chunks.sort(key=key)

    py_path, rs_path = tmp_path / "py.npz", tmp_path / "rs.npz"
    save_chunks(py_chunks, py_path)
    save_chunks(rs_chunks, rs_path)
    with np.load(py_path, allow_pickle=True) as py, np.load(rs_path, allow_pickle=True) as rs:
        return {k: py[k] for k in py.files}, {k: rs[k] for k in rs.files}


def _assert_field_equal(name: str, a: np.ndarray, b: np.ndarray) -> None:
    assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
    if a.dtype == object:
        # Variable-length rows (legacy object members). Compare row by row so a
        # length difference names the row rather than raising from numpy.
        for i, (ra, rb) in enumerate(zip(a, b, strict=True)):
            ra, rb = np.asarray(ra), np.asarray(rb)
            assert ra.shape == rb.shape, f"{name}[{i}]: shape {ra.shape} != {rb.shape}"
            if ra.dtype.kind == "f":
                np.testing.assert_allclose(
                    ra, rb, atol=FLOAT_ATOL, rtol=FLOAT_RTOL, err_msg=f"{name}[{i}]"
                )
            else:
                np.testing.assert_array_equal(ra, rb, err_msg=f"{name}[{i}]")
        return
    if name in FLOAT_FIELDS:
        np.testing.assert_allclose(a, b, atol=FLOAT_ATOL, rtol=FLOAT_RTOL, err_msg=name)
    else:
        np.testing.assert_array_equal(a, b, err_msg=name)


def _assert_npz_parity(py: dict, rs: dict) -> None:
    assert set(py) == set(rs), (
        f"backends wrote different fields: python-only {sorted(set(py) - set(rs))}, "
        f"rust-only {sorted(set(rs) - set(py))}"
    )

    unclassified = sorted(set(py) - FLOAT_FIELDS - EXACT_FIELDS)
    assert not unclassified, (
        f"unclassified chunk fields: {unclassified}. Add each to FLOAT_FIELDS or "
        f"EXACT_FIELDS in this module -- a field nothing compares is how every "
        f"backend divergence so far stayed hidden."
    )

    for name in sorted(py):
        _assert_field_equal(name, py[name], rs[name])


@pytest.fixture(scope="module")
def _rust_available():
    pytest.importorskip("leech_core")
    from leech._rust_accel import HAS_RUST, _rs_extract_training_chunks

    if not HAS_RUST or _rs_extract_training_chunks is None:
        pytest.skip("leech_core Rust acceleration not available")


class TestBackendFieldParity:
    """Both backends must write byte-identical corpora, field for field."""

    @pytest.mark.parametrize("anchor", ["reference", "basecall"])
    @pytest.mark.parametrize("refine", [True, False])
    def test_anchor_and_refinement(self, _rust_available, tmp_path, anchor, refine):
        py, rs = _run_both_backends(_config(anchor=anchor, refine=refine), tmp_path)
        _assert_npz_parity(py, rs)

    @pytest.mark.parametrize("base_justify", ["center", "start", "end"])
    def test_base_justify(self, _rust_available, tmp_path, base_justify):
        py, rs = _run_both_backends(_config(base_justify=base_justify), tmp_path)
        _assert_npz_parity(py, rs)

    @pytest.mark.parametrize(
        ("feature_start", "feature_end"),
        [(None, None), (0, 20), (-5, 5), (-10, 10), (0, 0), (2, 8)],
    )
    def test_feature_windows(self, _rust_available, tmp_path, feature_start, feature_end):
        """`feature_start=0` is the one that was mistaken for "unset" (#189)."""
        py, rs = _run_both_backends(
            _config(feature_start=feature_start, feature_end=feature_end), tmp_path
        )
        _assert_npz_parity(py, rs)

    @pytest.mark.parametrize("scale_iters", [-1, 0, 1, 2])
    def test_scale_iters(self, _rust_available, tmp_path, scale_iters):
        """`-1` meant "no DP" on one backend and "one DP pass" on the other,
        which is #193."""
        py, rs = _run_both_backends(_config(scale_iters=scale_iters), tmp_path)
        _assert_npz_parity(py, rs)

    @pytest.mark.parametrize("signal_context", [(200, 200), (90, 300), (400, 100)])
    def test_asymmetric_signal_context(self, _rust_available, tmp_path, signal_context):
        py, rs = _run_both_backends(_config(signal_context=signal_context), tmp_path)
        _assert_npz_parity(py, rs)

    def test_issue_193_production_config(self, _rust_available, tmp_path):
        """The exact flag set from the #193 report.

        `--feature-start 0 --feature-end 20 --signal-context 90 300
        --scale-iters -1 --no-require-query-mapping`, which diverged on 90% of
        dwell elements, every `seq_to_sig_map` length and every
        `sequence_with_kmer_context`.
        """
        py, rs = _run_both_backends(
            _config(
                feature_start=0,
                feature_end=20,
                signal_context=(90, 300),
                scale_iters=-1,
                require_query_mapping=False,
            ),
            tmp_path,
        )
        _assert_npz_parity(py, rs)


class TestParityHarnessItself:
    """The harness has to be able to fail, or it proves nothing."""

    def test_detects_a_perturbed_float_field(self):
        a = {"features_flat": np.ones((2, 3), dtype=np.float32)}
        b = {"features_flat": np.ones((2, 3), dtype=np.float32) * 1.5}
        with pytest.raises(AssertionError):
            _assert_npz_parity(a, b)

    def test_tolerates_float32_rounding(self):
        base = np.ones((2, 3), dtype=np.float32)
        a = {"features_flat": base}
        b = {"features_flat": base + 1e-6}
        _assert_npz_parity(a, b)

    def test_detects_an_off_by_one_dwell(self):
        """Dwells are exact: a single sample difference is a boundary
        difference, which is what #193 was."""
        a = {"dwells_flat": np.array([[12.0, 24.0]], dtype=np.float32)}
        b = {"dwells_flat": np.array([[13.0, 24.0]], dtype=np.float32)}
        with pytest.raises(AssertionError):
            _assert_npz_parity(a, b)

    def test_rejects_an_unclassified_field(self):
        payload = {"some_new_field": np.zeros(3)}
        with pytest.raises(AssertionError, match="unclassified chunk fields"):
            _assert_npz_parity(payload, payload)

    def test_detects_differing_field_sets(self):
        with pytest.raises(AssertionError, match="different fields"):
            _assert_npz_parity({"labels": np.array(["a"])}, {})

    def test_detects_ragged_object_rows(self):
        a = {"seq_to_sig_maps": np.array([np.arange(3)], dtype=object)}
        b = {"seq_to_sig_maps": np.array([np.arange(4)], dtype=object)}
        with pytest.raises(AssertionError):
            _assert_npz_parity(a, b)
