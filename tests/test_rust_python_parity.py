"""
Test that Rust and Python extraction paths produce identical signals.

Compares the _test_process_read Rust helper against the pure-Python pipeline
(normalize_signal, MoveTable.to_seq_to_sig_map, compute_dwell_features,
compute_signal_features) on synthetic data with realistic signal statistics.
"""

import numpy as np
import pytest

from leech.features import (
    MoveTable,
    compute_dwell_features,
    compute_signal_features,
    normalize_signal,
)

# Skip entire module if leech_core not built
pytest.importorskip("leech_core")

from leech._rust_accel import _rs_test_process_read  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_read(
    *,
    num_bases: int = 50,
    mean_dwell: int = 10,
    stride: int = 5,
    trim_offset: int = 10,
    seed: int = 42,
) -> dict:
    """Create a synthetic read with realistic signal characteristics."""
    rng = np.random.RandomState(seed)

    # Build move table: stride * num_total_positions entries, with `num_bases` moves
    total_positions = num_bases * mean_dwell
    moves = np.zeros(total_positions, dtype=np.uint8)
    # Place moves at roughly regular intervals
    move_positions = np.linspace(0, total_positions - 1, num_bases, dtype=int)
    moves[move_positions] = 1

    num_samples = trim_offset + total_positions * stride + 20  # extra padding
    # Generate raw signal: int16 DAC values with realistic nanopore characteristics
    # Base signal ~500-700 DAC with per-base level variation and noise
    base_levels = rng.uniform(400, 800, num_bases)
    raw_signal = np.zeros(num_samples, dtype=np.int16)
    for base_i, mpos in enumerate(move_positions):
        start = trim_offset + mpos * stride
        if base_i < num_bases - 1:
            end = trim_offset + move_positions[base_i + 1] * stride
        else:
            end = trim_offset + total_positions * stride
        end = min(end, num_samples)
        if start < end:
            raw_signal[start:end] = (base_levels[base_i] + rng.normal(0, 15, end - start)).astype(
                np.int16
            )

    return {
        "raw_signal": raw_signal,
        "moves": moves,
        "stride": stride,
        "trim_offset": trim_offset,
        "num_samples": num_samples,
        "num_bases": num_bases,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNormalizationParity:
    """Verify median-MAD normalization matches between Rust and Python."""

    @pytest.mark.parametrize("seed", [42, 123, 999])
    @pytest.mark.parametrize("reverse", [True, False])
    def test_normalized_signal_identical(self, seed, reverse):
        """Normalized signal values must be identical (within float32 eps)."""
        read = _make_synthetic_read(seed=seed)

        # --- Python path ---
        raw = read["raw_signal"]
        ts = read["trim_offset"]
        ns = read["num_samples"]
        trimmed = raw[ts:ns].astype(np.float32)
        if reverse:
            trimmed = trimmed[::-1].copy()
        py_norm, _ = normalize_signal(trimmed, method="median_mad")

        # --- Rust path ---
        rs_norm, rs_map, rs_dwells, rs_feats = _rs_test_process_read(
            read["raw_signal"].tolist(),
            read["moves"].tolist(),
            read["stride"],
            read["trim_offset"],
            read["num_samples"],
            reverse_signal=reverse,
        )

        np.testing.assert_allclose(
            rs_norm,
            py_norm,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Normalized signal mismatch",
        )


class TestSigMapParity:
    """Verify seq_to_sig_map matches between Rust and Python."""

    @pytest.mark.parametrize("reverse", [True, False])
    def test_sig_map_identical(self, reverse):
        read = _make_synthetic_read()

        # --- Python path ---
        mt = MoveTable(
            stride=read["stride"],
            moves=read["moves"],
            read_id="test",
            num_samples=read["num_samples"],
            trim_offset=read["trim_offset"],
        )
        py_map = mt.to_seq_to_sig_map() - read["trim_offset"]  # trimmed coords

        if reverse:
            sig_len = read["num_samples"] - read["trim_offset"]
            py_map = sig_len - py_map[::-1]

        # --- Rust path ---
        _, rs_map, _, _ = _rs_test_process_read(
            read["raw_signal"].tolist(),
            read["moves"].tolist(),
            read["stride"],
            read["trim_offset"],
            read["num_samples"],
            reverse_signal=reverse,
        )

        np.testing.assert_array_equal(
            rs_map,
            py_map,
            err_msg="seq_to_sig_map mismatch",
        )


class TestDwellParity:
    """Verify dwell time computation matches."""

    def test_dwells_identical(self):
        read = _make_synthetic_read()

        # Python path
        mt = MoveTable(
            stride=read["stride"],
            moves=read["moves"],
            read_id="test",
            num_samples=read["num_samples"],
            trim_offset=read["trim_offset"],
        )
        py_map = mt.to_seq_to_sig_map() - read["trim_offset"]
        sig_len = read["num_samples"] - read["trim_offset"]
        py_map = sig_len - py_map[::-1]  # reverse=True default
        py_dwells = np.diff(py_map).astype(np.float32)

        # Rust path
        _, _, rs_dwells, _ = _rs_test_process_read(
            read["raw_signal"].tolist(),
            read["moves"].tolist(),
            read["stride"],
            read["trim_offset"],
            read["num_samples"],
            reverse_signal=True,
        )

        np.testing.assert_array_equal(
            rs_dwells,
            py_dwells,
            err_msg="Dwell time mismatch",
        )


class TestFeatureParity:
    """Verify all 9 feature channels match between Rust and Python."""

    @pytest.mark.parametrize("seed", [42, 77])
    def test_features_identical(self, seed):
        read = _make_synthetic_read(seed=seed)

        # --- Python path ---
        raw = read["raw_signal"]
        ts = read["trim_offset"]
        ns = read["num_samples"]
        trimmed = raw[ts:ns].astype(np.float32)
        trimmed = trimmed[::-1].copy()  # reverse=True

        mt = MoveTable(
            stride=read["stride"],
            moves=read["moves"],
            read_id="test",
            num_samples=read["num_samples"],
            trim_offset=read["trim_offset"],
        )
        py_map = mt.to_seq_to_sig_map() - ts
        sig_len = ns - ts
        py_map = sig_len - py_map[::-1]

        py_norm, _ = normalize_signal(trimmed, method="median_mad")
        py_dwells = np.diff(py_map).astype(np.float32)

        # Dwell features (5 rows)
        py_dwell_feats = compute_dwell_features(py_dwells)
        # Signal features (4 rows)
        py_sig_feats = compute_signal_features(py_norm, py_map)

        # Stack in the same order as Rust: dwell(5) then signal(4)
        py_feat_rows = []
        for key in ["dwell", "dwell_log", "dwell_mean", "dwell_std", "dwell_ratio"]:
            py_feat_rows.append(py_dwell_feats[key])
        for key in ["level_mean", "level_median", "level_std", "level_range"]:
            py_feat_rows.append(py_sig_feats[key])
        py_features = np.stack(py_feat_rows, axis=0)

        # --- Rust path ---
        _, _, _, rs_features = _rs_test_process_read(
            read["raw_signal"].tolist(),
            read["moves"].tolist(),
            read["stride"],
            read["trim_offset"],
            read["num_samples"],
            reverse_signal=True,
        )

        assert rs_features.shape == py_features.shape, (
            f"Feature shape mismatch: Rust {rs_features.shape} vs Python {py_features.shape}"
        )

        feature_names = [
            "dwell",
            "dwell_log",
            "dwell_mean",
            "dwell_std",
            "dwell_ratio",
            "level_mean",
            "level_median",
            "level_std",
            "level_range",
        ]
        for row_idx, name in enumerate(feature_names):
            np.testing.assert_allclose(
                rs_features[row_idx],
                py_features[row_idx],
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Feature '{name}' (row {row_idx}) mismatch",
            )


class TestEndToEnd:
    """Full pipeline comparison with multiple signal sizes."""

    @pytest.mark.parametrize(
        "num_bases,mean_dwell",
        [(20, 8), (50, 12), (100, 6), (200, 15)],
    )
    def test_full_pipeline(self, num_bases, mean_dwell):
        """All outputs must match for various read lengths and dwell times."""
        read = _make_synthetic_read(num_bases=num_bases, mean_dwell=mean_dwell)

        # Python path
        raw = read["raw_signal"]
        ts = read["trim_offset"]
        ns = read["num_samples"]
        trimmed = raw[ts:ns].astype(np.float32)[::-1].copy()

        mt = MoveTable(
            stride=read["stride"],
            moves=read["moves"],
            read_id="test",
            num_samples=ns,
            trim_offset=ts,
        )
        py_map = mt.to_seq_to_sig_map() - ts
        py_map = (ns - ts) - py_map[::-1]
        py_norm, _ = normalize_signal(trimmed, method="median_mad")

        # Rust path
        rs_norm, rs_map, rs_dwells, rs_feats = _rs_test_process_read(
            raw.tolist(),
            read["moves"].tolist(),
            read["stride"],
            ts,
            ns,
            reverse_signal=True,
        )

        # Signal
        np.testing.assert_allclose(rs_norm, py_norm, rtol=1e-5, atol=1e-6)
        # Sig map
        np.testing.assert_array_equal(rs_map, py_map)
        # Dwells
        py_dwells = np.diff(py_map).astype(np.float32)
        np.testing.assert_array_equal(rs_dwells, py_dwells)
        # Features shape
        assert rs_feats.shape[0] == 9
        assert rs_feats.shape[1] == num_bases
