"""
Tests for config propagation hardening.

Verifies that motif, base_justify, motif_offset and all hyperparameters
are written to config.json by train_model() and run_grid_point(), that
motif=None raises ValueError, and that the inference shape validator
catches channel/feature mismatches.
"""

import json

import numpy as np
import pytest

from leech.inference import (
    InferenceConfigError,
    _check_config_consistency,
    validate_inference_shapes,
)
from leech.training import train_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


PROVENANCE_FIELDS = ("motif", "motif_offset", "base_justify")
TRAIN_DEFAULTS = {
    "model_name": "ConvLSTMDwell",
    "epochs": 1,
    "batch_size": 2,
    "device": "cpu",
    "motif": "CCAGGC",
    "motif_offset": 2,
    "base_justify": "center",
}


def _read_config(output_dir):
    with open(output_dir / "config.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# train_model provenance
# ---------------------------------------------------------------------------


class TestTrainModelProvenance:
    """train_model() must persist all provenance fields in config.json."""

    def test_train_model_writes_all_provenance(self, temp_chunks_file, tmp_path):
        output_dir = tmp_path / "train_prov"
        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=None,
            output_dir=output_dir,
            **TRAIN_DEFAULTS,
        )
        config = _read_config(output_dir)

        assert config["motif"] == "CCAGGC"
        assert config["motif_offset"] == 2
        assert config["base_justify"] == "center"
        # Also check training hyperparams are present
        assert "model_name" in config
        assert "signal_len" in config
        assert "kmer_len" in config
        assert "num_features" in config
        assert "signal_in_channels" in config

    def test_train_model_motif_none_raises(self, temp_chunks_file, tmp_path):
        output_dir = tmp_path / "train_none"
        with pytest.raises(ValueError, match="motif must not be None"):
            train_model(
                train_data_path=temp_chunks_file,
                val_data_path=None,
                output_dir=output_dir,
                model_name="ConvLSTMDwell",
                epochs=1,
                batch_size=2,
                device="cpu",
                motif=None,
            )


# ---------------------------------------------------------------------------
# run_grid_point provenance
# ---------------------------------------------------------------------------


class TestGridPointProvenance:
    """run_grid_point() must persist motif/base_justify in config.json."""

    def test_grid_point_writes_provenance(self, temp_chunks_file, tmp_path):
        from leech.gridsearch import run_grid_point

        output_dir = tmp_path / "grid_prov"
        run_grid_point(
            train_data_path=temp_chunks_file,
            val_data_path=None,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            left_context=200,
            right_context=200,
            kmer_len=11,
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            device="cpu",
            seed=42,
            motif="CCAGGC",
            motif_offset=2,
            base_justify="center",
        )
        config = _read_config(output_dir)

        assert config["motif"] == "CCAGGC"
        assert config["motif_offset"] == 2
        assert config["base_justify"] == "center"


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLIMotif:
    """CLI commands expose --motif as required."""

    def test_cli_train_passes_motif(self, temp_chunks_file, tmp_path):
        from click.testing import CliRunner

        from leech.cli import cli

        output_dir = tmp_path / "cli_train"
        result = CliRunner().invoke(
            cli,
            [
                "model",
                "train",
                "--train-data",
                str(temp_chunks_file),
                "--model",
                "ConvLSTMDwell",
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--device",
                "cpu",
                "--motif",
                "CCAGGC",
                "--motif-offset",
                "2",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        config = _read_config(output_dir)
        assert config["motif"] == "CCAGGC"
        assert config["motif_offset"] == 2

    def test_cli_train_missing_motif_errors(self, temp_chunks_file, tmp_path):
        from click.testing import CliRunner

        from leech.cli import cli

        output_dir = tmp_path / "cli_no_motif"
        result = CliRunner().invoke(
            cli,
            [
                "model",
                "train",
                "--train-data",
                str(temp_chunks_file),
                "--model",
                "ConvLSTMDwell",
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--device",
                "cpu",
                # intentionally omit --motif
            ],
        )
        assert result.exit_code != 0
        assert "Missing" in result.output or "required" in result.output.lower()

    def test_cli_optimize_missing_motif_errors(self, temp_chunks_file, tmp_path):
        from click.testing import CliRunner

        from leech.cli import cli

        output_dir = tmp_path / "cli_opt_no_motif"
        result = CliRunner().invoke(
            cli,
            [
                "model",
                "optimize",
                "--train-data",
                str(temp_chunks_file),
                "--model",
                "ConvLSTMDwell",
                "--output-dir",
                str(output_dir),
                "--context-grid",
                "200",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--device",
                "cpu",
                # intentionally omit --motif
            ],
        )
        assert result.exit_code != 0
        assert "Missing" in result.output or "required" in result.output.lower()


# ---------------------------------------------------------------------------
# Sequence encoding: the config records what was used, not what was asked for
# ---------------------------------------------------------------------------


@pytest.fixture
def chunks_file_without_maps(sample_chunks, tmp_path):
    """A corpus a signal_kmer run cannot be served from — the version-skew case."""
    from leech.chunking.serialization import save_chunks

    chunks = []
    for chunk in sample_chunks:
        stripped = dict(chunk)
        stripped["seq_to_sig_map"] = np.zeros(0, dtype=np.int64)
        stripped["sequence_with_kmer_context"] = ""
        chunks.append(stripped)
    path = tmp_path / "no_maps.npz"
    save_chunks(chunks, path)
    return path


class TestSeqEncodingProvenance:
    """A checkpoint that fell back has to say so (#230).

    ``--seq-encoding signal_kmer`` over a corpus with no base-to-signal maps
    trains on ``base_onehot`` — a different model input, ``(4, kmer_len)``
    against ``(36, signal_len)``. Writing the requested value into config.json
    left no way to audit the artifact afterwards, and #220's ONNX contract,
    derived from the same config, would publish an input the model does not
    have.
    """

    def test_config_records_the_effective_encoding(self, chunks_file_without_maps, tmp_path):
        output_dir = tmp_path / "fallback"
        train_model(
            train_data_path=chunks_file_without_maps,
            val_data_path=None,
            output_dir=output_dir,
            seq_encoding="signal_kmer",
            **TRAIN_DEFAULTS,
        )
        config = _read_config(output_dir)
        assert config["seq_encoding"] == "base_onehot"

    def test_the_model_matches_the_config_it_is_saved_with(
        self, chunks_file_without_maps, tmp_path
    ):
        """Config and weights have to describe the same sequence branch."""
        import torch

        from leech.models import get_model

        output_dir = tmp_path / "fallback_weights"
        train_model(
            train_data_path=chunks_file_without_maps,
            val_data_path=None,
            output_dir=output_dir,
            seq_encoding="signal_kmer",
            **TRAIN_DEFAULTS,
        )
        config = _read_config(output_dir)
        checkpoint = torch.load(
            output_dir / "model_best.pt", map_location="cpu", weights_only=False
        )
        state = checkpoint.get("model_state_dict", checkpoint)

        model = get_model(
            config["model_name"],
            signal_len=config["signal_len"],
            kmer_len=config["kmer_len"],
            seq_encoding=config["seq_encoding"],
            signal_kmer_context=tuple(config["signal_kmer_context"]),
            num_features=config["num_features"],
            signal_in_channels=config["signal_in_channels"],
            num_out=config["num_out"],
        )
        # Rebuilt from the saved config alone, the weights have to fit.
        model.load_state_dict(state)

    def test_encoding_stays_signal_kmer_when_the_corpus_can_supply_it(
        self, temp_chunks_file, tmp_path
    ):
        output_dir = tmp_path / "no_fallback"
        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=None,
            output_dir=output_dir,
            seq_encoding="signal_kmer",
            **TRAIN_DEFAULTS,
        )
        assert _read_config(output_dir)["seq_encoding"] == "signal_kmer"

    def test_refusing_the_fallback_stops_the_run(self, chunks_file_without_maps, tmp_path):
        with pytest.raises(ValueError, match="signal_kmer"):
            train_model(
                train_data_path=chunks_file_without_maps,
                val_data_path=None,
                output_dir=tmp_path / "refused",
                seq_encoding="signal_kmer",
                allow_encoding_fallback=False,
                **TRAIN_DEFAULTS,
            )

    def test_cli_explicit_seq_encoding_is_not_substituted(self, chunks_file_without_maps, tmp_path):
        """Naming the encoding on the command line makes the fallback an error."""
        from click.testing import CliRunner

        from leech.cli import cli

        result = CliRunner().invoke(
            cli,
            [
                "model",
                "train",
                "--train-data",
                str(chunks_file_without_maps),
                "--model",
                "ConvLSTMDwell",
                "--output-dir",
                str(tmp_path / "cli_explicit"),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--device",
                "cpu",
                "--motif",
                "CCAGGC",
                "--seq-encoding",
                "signal_kmer",
            ],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        assert "signal_kmer" in str(result.exception)

    def test_cli_default_seq_encoding_still_falls_back(self, chunks_file_without_maps, tmp_path):
        """Taking the default is the case the fallback exists for."""
        from click.testing import CliRunner

        from leech.cli import cli

        output_dir = tmp_path / "cli_default"
        result = CliRunner().invoke(
            cli,
            [
                "model",
                "train",
                "--train-data",
                str(chunks_file_without_maps),
                "--model",
                "ConvLSTMDwell",
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--device",
                "cpu",
                "--motif",
                "CCAGGC",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert _read_config(output_dir)["seq_encoding"] == "base_onehot"

    def test_cli_encoding_fallback_flag_overrides_the_explicit_request(
        self, chunks_file_without_maps, tmp_path
    ):
        from click.testing import CliRunner

        from leech.cli import cli

        output_dir = tmp_path / "cli_flag"
        result = CliRunner().invoke(
            cli,
            [
                "model",
                "train",
                "--train-data",
                str(chunks_file_without_maps),
                "--model",
                "ConvLSTMDwell",
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--device",
                "cpu",
                "--motif",
                "CCAGGC",
                "--seq-encoding",
                "signal_kmer",
                "--encoding-fallback",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert _read_config(output_dir)["seq_encoding"] == "base_onehot"


# ---------------------------------------------------------------------------
# Inference shape validator
# ---------------------------------------------------------------------------


class TestInferenceShapeValidator:
    """validate_inference_shapes catches mismatches early."""

    def test_catches_channel_mismatch(self):
        signal_1ch = np.zeros(400, dtype=np.float32)
        config_2ch = {"signal_in_channels": 2, "signal_len": 400, "num_features": 5}
        with pytest.raises(InferenceConfigError, match="channel"):
            validate_inference_shapes(signal_1ch, None, config_2ch)

    def test_catches_feature_mismatch(self):
        signal = np.zeros((2, 400), dtype=np.float32)
        features_wrong = np.zeros((9, 11), dtype=np.float32)
        config = {"signal_in_channels": 2, "signal_len": 400, "num_features": 12}
        with pytest.raises(InferenceConfigError, match="features"):
            validate_inference_shapes(signal, features_wrong, config)

    def test_catches_signal_len_mismatch(self):
        signal = np.zeros(300, dtype=np.float32)
        config = {"signal_in_channels": 1, "signal_len": 400}
        with pytest.raises(InferenceConfigError, match="Signal length"):
            validate_inference_shapes(signal, None, config)

    def test_passes_correct_shapes(self):
        signal = np.zeros((2, 400), dtype=np.float32)
        features = np.zeros((12, 11), dtype=np.float32)
        config = {"signal_in_channels": 2, "signal_len": 400, "num_features": 12}
        # Should not raise
        validate_inference_shapes(signal, features, config)

    def test_passes_no_features(self):
        signal = np.zeros(400, dtype=np.float32)
        config = {"signal_in_channels": 1, "signal_len": 400}
        # Should not raise
        validate_inference_shapes(signal, None, config)


# ---------------------------------------------------------------------------
# Config consistency guard
# ---------------------------------------------------------------------------


class TestConfigConsistency:
    """_check_config_consistency resolves params and catches conflicts."""

    def test_config_consistency_uses_config_value(self):
        """CLI default + config has value → returns config value."""
        result = _check_config_consistency("base-justify", "center", "end", "center")
        assert result == "end"

    def test_config_consistency_conflict_raises(self):
        """CLI='end' + config='center' → InferenceConfigError."""
        with pytest.raises(InferenceConfigError, match="conflicts with training config"):
            _check_config_consistency("base-justify", "end", "center", "center")

    def test_config_consistency_no_config_uses_cli(self):
        """config=None → returns CLI value (backward compat)."""
        result = _check_config_consistency("motif", "CCAGGC", None, None)
        assert result == "CCAGGC"

    def test_config_consistency_matching_ok(self):
        """CLI='center' + config='center' → no error, returns 'center'."""
        result = _check_config_consistency("base-justify", "center", "center", "center")
        assert result == "center"
