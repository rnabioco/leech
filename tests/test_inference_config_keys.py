"""Model-config keys must survive the trip to the extraction backend.

Two of them did not. `base_justify` was resolved from the model config,
threaded into the Python path's `ChunkConfig`, and then dropped before the Rust
call -- where the pyo3 signature's `"center"` default silently took over. And
the Rust extraction results were consumed without the feature-window narrowing
the Python path applies, so a model trained on a wide feature window was handed
the full window and `dwell_offset` did nothing.

Neither had a symptom: `validate_inference_shapes` checks channel count,
signal length and feature count, but not the feature *width*, and nothing at
all checks where in the base the signal window was centred.
"""

from __future__ import annotations

import numpy as np
import pytest

from leech.inference.helpers import (
    InferenceConfigError,
    InferenceSpec,
    build_rust_extraction_kwargs,
    is_multiclass,
    prepare_inference_features,
)


def _kwargs(**overrides):
    base = {
        "signal_context": (200, 200),
        "kmer_context": 5,
        "signal_len": 400,
        "compute_features": True,
        "reverse_signal": True,
        "feature_start": None,
        "feature_end": None,
        "anchor": "reference",
        "seq_encoding": "signal_kmer",
        "signal_kmer_context": (4, 4),
        "refine_signal_map": False,
        "signal_refiner": None,
        "refine_half_bandwidth": 5,
        "refine_scale_iters": 2,
        "signal_in_channels": 1,
        "base_justify": "center",
    }
    base.update(overrides)
    return build_rust_extraction_kwargs(**base)


class TestBaseJustifyReachesRust:
    @pytest.mark.parametrize("justify", ["center", "start", "end"])
    def test_base_justify_is_forwarded(self, justify):
        assert _kwargs(base_justify=justify)["base_justify"] == justify

    def test_base_justify_is_required(self):
        """Not defaulted here: the Rust signature already defaults it to
        "center", so an omission at this layer is invisible rather than loud."""
        with pytest.raises(TypeError):
            build_rust_extraction_kwargs(
                signal_context=(200, 200),
                kmer_context=5,
                signal_len=400,
                compute_features=True,
                reverse_signal=True,
                feature_start=None,
                feature_end=None,
                anchor="reference",
                seq_encoding="signal_kmer",
                signal_kmer_context=(4, 4),
                refine_signal_map=False,
                signal_refiner=None,
                refine_half_bandwidth=5,
                refine_scale_iters=2,
                signal_in_channels=1,
            )


class TestPrepareInferenceFeatures:
    """Must agree with ``ChunkDataset._prepare_features``, which is what ran
    at training time."""

    def _features(self, width: int, n_rows: int = 3) -> np.ndarray:
        # Row 0 counts columns so a slice is identifiable by its contents.
        return np.stack([np.arange(width, dtype=np.float32) for _ in range(n_rows)])

    def test_narrows_to_the_kmer_window(self):
        # Stored window [0, 20] (width 21), model kmer_len 11 -> kmer_context 5.
        # Column 0 sits at offset 0, the k-mer window starts at -5, so the
        # slice starts at -5 - 0 = -5 ... which is out of range. Use a window
        # that contains the k-mer window instead.
        feats = self._features(21)
        out = prepare_inference_features(
            feats, kmer_len=11, feature_start=-10, dwell_offset=0, wide_features=False
        )
        assert out.shape == (3, 11)
        # k-mer window starts 5 columns in from a window beginning at -10.
        np.testing.assert_array_equal(out[0], np.arange(5, 16, dtype=np.float32))

    def test_dwell_offset_shifts_the_window(self):
        feats = self._features(21)
        out = prepare_inference_features(
            feats, kmer_len=11, feature_start=-10, dwell_offset=2, wide_features=False
        )
        np.testing.assert_array_equal(out[0], np.arange(7, 18, dtype=np.float32))

    def test_matches_dataset_prepare_features(self):
        """The training-time slice, computed independently, must agree."""
        kmer_len, feat_start, dwell_offset = 11, -10, 3
        feats = self._features(21)
        kmer_context = kmer_len // 2
        expected_start = (-kmer_context) - feat_start + dwell_offset  # dataset.py
        expected = feats[:, expected_start : expected_start + kmer_len]
        out = prepare_inference_features(
            feats,
            kmer_len=kmer_len,
            feature_start=feat_start,
            dwell_offset=dwell_offset,
            wide_features=False,
        )
        np.testing.assert_array_equal(out, expected)

    def test_wide_feature_models_keep_the_full_window(self):
        feats = self._features(21)
        out = prepare_inference_features(
            feats, kmer_len=11, feature_start=-10, dwell_offset=0, wide_features=True
        )
        assert out.shape == (3, 21)

    def test_already_narrow_is_left_alone(self):
        feats = self._features(11)
        out = prepare_inference_features(
            feats, kmer_len=11, feature_start=-5, dwell_offset=0, wide_features=False
        )
        assert out.shape == (3, 11)

    def test_none_and_empty_pass_through(self):
        assert prepare_inference_features(None, kmer_len=11, feature_start=None) is None
        empty = np.zeros((0, 0), dtype=np.float32)
        assert prepare_inference_features(empty, kmer_len=11, feature_start=None).size == 0

    @pytest.mark.parametrize("dwell_offset", [-20, 20])
    def test_out_of_range_window_raises(self, dwell_offset):
        """Training raises for this; inference must not silently slide."""
        feats = self._features(21)
        with pytest.raises(InferenceConfigError, match="does not fit"):
            prepare_inference_features(
                feats,
                kmer_len=11,
                feature_start=-10,
                dwell_offset=dwell_offset,
                wide_features=False,
            )

    def test_templates_are_appended_before_narrowing(self):
        """Template channels are keyed to the *stored* window's column 0.

        Appending after a narrowing keys them to the shifted array, which is
        what the bundle's Python path used to do.
        """
        width, kmer_len, feat_start = 21, 11, -10
        # dwell (row 0) is 10 everywhere; one template row of expected dwell 5
        # over the whole stored window -> every ratio is 2.0.
        feats = np.full((3, width), 10.0, dtype=np.float32)
        templates = np.full((1, width), 5.0, dtype=np.float32)
        out = prepare_inference_features(
            feats,
            kmer_len=kmer_len,
            feature_start=feat_start,
            dwell_offset=0,
            wide_features=False,
            dwell_templates=templates,
            template_min_pos=feat_start,
        )
        assert out.shape == (4, kmer_len)
        # All template positions were covered, so no column fell back to 1.0.
        np.testing.assert_allclose(out[3], np.full(kmer_len, 2.0, dtype=np.float32))

    def test_template_alignment_is_window_relative(self):
        """A template covering only the window's left half must land there."""
        width, kmer_len, feat_start = 21, 11, -10
        feats = np.full((3, width), 10.0, dtype=np.float32)
        # Template covers stored positions [-10, -5) only; expected dwell 5.
        templates = np.full((1, 5), 5.0, dtype=np.float32)
        out = prepare_inference_features(
            feats,
            kmer_len=kmer_len,
            feature_start=feat_start,
            dwell_offset=0,
            wide_features=False,
            dwell_templates=templates,
            template_min_pos=feat_start,
        )
        # The k-mer window starts at stored column 5, i.e. position -5 -- just
        # past the template's coverage -- so every kept column is the 1.0
        # fallback. Appending after narrowing would have put the covered
        # columns here instead.
        np.testing.assert_allclose(out[3], np.ones(kmer_len, dtype=np.float32))


class TestExtractionSequence:
    """Motif positions index the sequence chunks are cut from.

    ``ReferenceMotifSearcher`` ignores its ``sequence`` argument and reads the
    alignment, which is why two of the three inference paths could pass the
    basecall under ``anchor="reference"`` unnoticed. But `predict` picks the
    searcher with ``mode="fasta" if reference_sequences else "bam"``, so a run
    without a reference FASTA gets the *basecalled* searcher while chunks are
    still cut in reference coordinates -- and there the argument decides the
    answer.
    """

    def test_reference_anchor_uses_the_reference_slice(self):
        from leech.chunking import extraction_sequence

        assert (
            extraction_sequence(
                anchor="reference",
                basecall="AAAA",
                reference_sequence="CCCC",
                cigar_tuples=[(0, 4)],
            )
            == "CCCC"
        )

    def test_basecall_anchor_uses_the_basecall(self):
        from leech.chunking import extraction_sequence

        assert (
            extraction_sequence(
                anchor="basecall",
                basecall="AAAA",
                reference_sequence="CCCC",
                cigar_tuples=[(0, 4)],
            )
            == "AAAA"
        )

    @pytest.mark.parametrize(
        ("ref", "cigar"),
        [(None, [(0, 4)]), ("CCCC", None), (None, None)],
    )
    def test_falls_back_to_basecall_without_both_inputs(self, ref, cigar):
        """Matches ``build_leech_read``: it needs a reference *and* a CIGAR to
        map through before it will anchor to reference coordinates."""
        from leech.chunking import extraction_sequence

        assert (
            extraction_sequence(
                anchor="reference", basecall="AAAA", reference_sequence=ref, cigar_tuples=cigar
            )
            == "AAAA"
        )

    def test_prepare_and_inference_agree(self):
        """The prepare adapter and the shared rule must not drift apart."""
        from types import SimpleNamespace

        from leech.chunking import extraction_sequence
        from leech.configs import PrepareConfig, SignalConfig
        from leech.preparation.parallel import _extraction_sequence

        read_info = SimpleNamespace(
            sequence="AAAA", reference_sequence="CCCC", cigar_tuples=[(0, 4)]
        )
        for anchor in ("reference", "basecall"):
            config = PrepareConfig(pod5_path="unused.pod5", signal=SignalConfig(anchor=anchor))
            assert _extraction_sequence(read_info, config) == extraction_sequence(
                anchor=anchor,
                basecall=read_info.sequence,
                reference_sequence=read_info.reference_sequence,
                cigar_tuples=read_info.cigar_tuples,
            )


class TestIsMulticlass:
    """One definition of "multiclass", not `> 1` in predict and `> 2` in
    eval/calibrate -- issue #269."""

    @pytest.mark.parametrize(("num_out", "expected"), [(1, False), (2, True), (3, True)])
    def test_matches_num_out_greater_than_one(self, num_out, expected):
        assert is_multiclass({"num_out": num_out}) is expected

    def test_missing_num_out_defaults_to_binary(self):
        assert is_multiclass({}) is False

    def test_two_output_model_is_multiclass(self):
        """The exact regression #269 names: a 2-output cross-entropy model
        used to get multiclass BAM tags from predict (`> 1`) but binary Platt
        calibration from `eval`/`calibrate` (`> 2`). One call, one answer."""
        config = {"num_out": 2}
        assert is_multiclass(config) is True


class TestInferenceSpecFromConfig:
    """`InferenceSpec.from_config` replaces the config-resolution block that
    used to be pasted into ``run_inference`` and ``run_bundle_inference``
    separately (issue #269)."""

    def _config(self, **overrides) -> dict:
        base = {
            "signal_len": 400,
            "kmer_len": 11,
            "motif": "CCAGGC",
            "motif_offset": 0,
            "num_out": 1,
        }
        base.update(overrides)
        return base

    def test_resolves_geometry_from_config(self):
        spec = InferenceSpec.from_config(self._config())
        assert spec.signal_len == 400
        assert spec.kmer_len == 11
        assert spec.kmer_context == 5
        assert spec.signal_context == (200, 200)

    def test_config_without_seq_encoding_defaults_to_base_onehot(self):
        """A config with no `seq_encoding` key at all predates the field, or
        was built directly via `get_model(name, **kwargs)` with no override --
        either way it is a base_onehot model, since that is the model
        classes' own [params] default (e.g. `conv_lstm.toml`), not
        `signal_kmer` (only the CLI's `model train` default, which always
        records the encoding it actually used)."""
        spec = InferenceSpec.from_config(self._config())
        assert spec.seq_encoding == "base_onehot"

    def test_explicit_seq_encoding_is_honored(self):
        spec = InferenceSpec.from_config(self._config(seq_encoding="signal_kmer"))
        assert spec.seq_encoding == "signal_kmer"

    def test_legacy_dwell_margin_corpus_resolves_feature_window(self):
        """Old configs recorded `dwell_margin_left`/`dwell_margin_right`
        instead of `feature_start`/`feature_end`; the resolver must still
        recover a usable window rather than falling back to the k-mer window."""
        spec = InferenceSpec.from_config(self._config(dwell_margin_left=3, dwell_margin_right=3))
        # kmer_context=5, so feature_start = -(5+3) = -8
        assert spec.feature_start == -8

    def test_two_output_model_is_flagged_multiclass_and_calibrated(self):
        """The same regression as TestIsMulticlass, seen through the spec
        every predict path now actually builds from."""
        spec = InferenceSpec.from_config(self._config(num_out=2, calibration={"pairs": {}}))
        assert spec.is_multiclass is True
        assert spec.calibration == {"pairs": {}}

    def test_binary_model_never_carries_calibration(self):
        spec = InferenceSpec.from_config(self._config(num_out=1, calibration={"a": 1}))
        assert spec.is_multiclass is False
        assert spec.calibration is None

    def test_missing_motif_raises(self):
        with pytest.raises(InferenceConfigError, match="motif is None"):
            InferenceSpec.from_config(self._config(motif=None))

    def test_cli_motif_offset_disagreeing_with_config_raises(self):
        with pytest.raises(InferenceConfigError, match="conflicts"):
            InferenceSpec.from_config(self._config(motif_offset=2), motif_offset=5, motif="CCAGGC")

    def test_explicit_cli_value_equal_to_default_is_still_checked(self):
        """The bug `parameter_sources` exists to fix: an explicit
        `--motif-offset 0` that disagrees with the config must raise even
        though 0 is also click's own default, so the naive
        `cli_value != cli_default` heuristic would wave it through."""
        with pytest.raises(InferenceConfigError, match="conflicts"):
            InferenceSpec.from_config(
                self._config(motif_offset=3),
                motif="CCAGGC",
                motif_offset=0,
                parameter_sources={"motif_offset": True},
            )

    def test_non_explicit_cli_value_equal_to_default_defers_to_config(self):
        """Without a parameter source, the same call is indistinguishable
        from "not passed" and the config wins -- the pre-existing, documented
        heuristic for callers with no click context."""
        spec = InferenceSpec.from_config(
            self._config(motif_offset=3), motif="CCAGGC", motif_offset=0
        )
        assert spec.motif_offset == 3

    def test_wide_features_model_falls_back_to_model_dwell_margin(self):
        spec = InferenceSpec.from_config(
            self._config(dwell_margin=4), model_type="TCNDwellResidualMotor"
        )
        assert spec.feature_start == -(5 + 4)
        assert spec.feature_end == 5 + 4

    def test_default_refine_scale_iters_differs_for_remora_vs_leech(self):
        """single.py passed a hardcoded `2` to the Rust kwargs builder
        regardless of is_remora even though its own refiner setup used `-1`
        for Remora models; the spec is now the one place this is decided."""
        leech_spec = InferenceSpec.from_config(self._config(), default_refine_scale_iters=2)
        remora_spec = InferenceSpec.from_config(self._config(), default_refine_scale_iters=-1)
        assert leech_spec.refine_scale_iters == 2
        assert remora_spec.refine_scale_iters == -1
