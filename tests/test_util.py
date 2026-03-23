"""
Tests for utility functions.

Tests metrics computation, model loading, and helper functions.
"""

import json

import numpy as np
import pytest
import torch

from leech.models import get_model
from leech.bundling import (
    create_bundle,
    create_torchscript_bundle,
    create_vmap_bundle,
    list_bundle_models,
    load_model_from_bundle,
)
from leech.metrics import compute_metrics, print_metrics, save_metrics
from leech.model_export import (
    deserialize_exported_model,
    deserialize_traced_model,
    export_model,
    export_single_model,
    serialize_exported_model,
    serialize_traced_model,
    trace_model,
)
from leech.model_loading import load_model_from_checkpoint


class TestComputeMetrics:
    """Test compute_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.95])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["auroc"] > 0.9  # Should be high for perfect predictions
        assert metrics["auprc"] > 0.9  # Should be high for perfect predictions

    def test_random_predictions(self, sample_predictions):
        """Test metrics with random predictions."""
        y_true, y_pred, y_prob = sample_predictions

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auroc" in metrics
        assert "auprc" in metrics
        assert "confusion_matrix" in metrics

        # All metrics should be between 0 and 1 (with small numerical tolerance)
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["auroc"] <= 1 + 1e-10  # Allow small numerical error
        assert 0 <= metrics["auprc"] <= 1 + 1e-10  # Allow small numerical error

    def test_confusion_matrix_structure(self, sample_predictions):
        """Test that confusion matrix has correct structure."""
        y_true, y_pred, y_prob = sample_predictions

        metrics = compute_metrics(y_true, y_pred, y_prob)
        cm = metrics["confusion_matrix"]

        assert isinstance(cm, list)
        assert len(cm) == 2  # 2x2 matrix
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2

        # All values should be non-negative integers
        for row in cm:
            for val in row:
                assert val >= 0
                assert isinstance(val, (int, np.integer))

    def test_confusion_matrix_sum(self, sample_predictions):
        """Test that confusion matrix sums to total predictions."""
        y_true, y_pred, y_prob = sample_predictions

        metrics = compute_metrics(y_true, y_pred, y_prob)
        cm = metrics["confusion_matrix"]

        total = sum(sum(row) for row in cm)
        assert total == len(y_true)

    def test_single_class_predictions(self):
        """Test metrics when only one class is present."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.15, 0.3])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        # AUROC and AUPRC should be 0 when only one class present
        assert metrics["auroc"] == 0.0
        assert metrics["auprc"] == 0.0
        assert metrics["accuracy"] == 1.0

    def test_all_wrong_predictions(self):
        """Test metrics with completely wrong predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.95, 0.1, 0.2, 0.15])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 0.0

    def test_imbalanced_classes(self):
        """Test metrics with imbalanced classes."""
        # 90 negatives, 10 positives
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.array([0] * 85 + [1] * 5 + [1] * 10)  # Some false positives
        y_prob = np.concatenate([np.random.rand(90) * 0.5, np.random.rand(10) * 0.5 + 0.5])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        # Metrics should still be computed correctly
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1


class TestSaveLoadMetrics:
    """Test saving and loading metrics."""

    def test_save_metrics(self, sample_predictions, tmp_path):
        """Test saving metrics to file."""
        y_true, y_pred, y_prob = sample_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob)

        output_path = tmp_path / "metrics.json"
        save_metrics(metrics, output_path)

        assert output_path.exists()

    def test_save_metrics_creates_directory(self, sample_predictions, tmp_path):
        """Test that save_metrics creates parent directories."""
        y_true, y_pred, y_prob = sample_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob)

        output_path = tmp_path / "subdir" / "metrics.json"
        save_metrics(metrics, output_path)

        assert output_path.exists()

    def test_saved_metrics_structure(self, sample_predictions, tmp_path):
        """Test that saved metrics have correct structure."""
        y_true, y_pred, y_prob = sample_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob)

        output_path = tmp_path / "metrics.json"
        save_metrics(metrics, output_path)

        # Load and verify
        with open(output_path) as f:
            loaded = json.load(f)

        assert "accuracy" in loaded
        assert "precision" in loaded
        assert "recall" in loaded
        assert "f1" in loaded
        assert "auroc" in loaded
        assert "auprc" in loaded
        assert "confusion_matrix" in loaded


class TestPrintMetrics:
    """Test print_metrics function."""

    def test_print_metrics_no_error(self, sample_predictions, capsys):
        """Test that print_metrics doesn't raise errors."""
        y_true, y_pred, y_prob = sample_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob)

        print_metrics(metrics)

        captured = capsys.readouterr()
        output_text = captured.out
        assert "Evaluation Metrics" in output_text
        assert "Accuracy" in output_text
        assert "Confusion Matrix" in output_text

    def test_print_metrics_without_confusion_matrix(self, capsys):
        """Test printing metrics without confusion matrix."""
        metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.89,
            "f1": 0.90,
            "auroc": 0.93,
            "auprc": 0.91,
        }

        print_metrics(metrics)

        captured = capsys.readouterr()
        output_text = captured.out
        assert "Evaluation Metrics" in output_text
        # Confusion matrix section should not appear if not in metrics
        assert "Confusion Matrix" not in output_text


class TestLoadModelFromCheckpoint:
    """Test load_model_from_checkpoint function."""

    def test_load_model_missing_config(self, tmp_path):
        """Test that loading fails when config.json is missing."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_model_from_checkpoint(model_dir)

    def test_load_model_missing_checkpoint(self, temp_model_dir):
        """Test that loading fails when checkpoint file is missing."""
        # temp_model_dir has config.json but no model_best.pt

        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            load_model_from_checkpoint(temp_model_dir)

    def test_load_model_success(self, temp_model_dir, model_config):
        """Test successful model loading."""
        # Create a model and save it
        model = get_model("ConvLSTMDwell", **model_config)

        checkpoint_path = temp_model_dir / "model_best.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
                "best_val_acc": 0.95,
                "best_epoch": 10,
            },
            checkpoint_path,
        )

        # Load the model
        loaded_model, config = load_model_from_checkpoint(temp_model_dir, device="cpu")

        assert loaded_model is not None
        assert config["model_name"] == "ConvLSTMDwell"
        assert config["signal_len"] == model_config["signal_len"]

    def test_load_model_correct_architecture(self, temp_model_dir, model_config):
        """Test that loaded model has correct architecture."""
        from leech.models import ConvLSTMDwell

        model = get_model("ConvLSTMDwell", **model_config)

        checkpoint_path = temp_model_dir / "model_best.pt"
        torch.save(
            {"model_state_dict": model.state_dict()},
            checkpoint_path,
        )

        loaded_model, _ = load_model_from_checkpoint(temp_model_dir, device="cpu")

        assert isinstance(loaded_model, ConvLSTMDwell)
        assert loaded_model.signal_len == model_config["signal_len"]
        assert loaded_model.kmer_len == model_config["kmer_len"]

    def test_load_model_eval_mode(self, temp_model_dir, model_config):
        """Test that loaded model is in eval mode."""
        model = get_model("ConvLSTMDwell", **model_config)

        checkpoint_path = temp_model_dir / "model_best.pt"
        torch.save(
            {"model_state_dict": model.state_dict()},
            checkpoint_path,
        )

        loaded_model, _ = load_model_from_checkpoint(temp_model_dir, device="cpu")

        assert not loaded_model.training  # Should be in eval mode

    def test_load_model_custom_checkpoint_name(self, temp_model_dir, model_config):
        """Test loading with custom checkpoint filename."""
        model = get_model("ConvLSTMDwell", **model_config)

        checkpoint_path = temp_model_dir / "model_custom.pt"
        torch.save(
            {"model_state_dict": model.state_dict()},
            checkpoint_path,
        )

        loaded_model, _ = load_model_from_checkpoint(
            temp_model_dir, device="cpu", checkpoint_name="model_custom.pt"
        )

        assert loaded_model is not None

    def test_load_model_state_dict_matches(self, temp_model_dir, model_config):
        """Test that loaded model has same weights as saved model."""
        model = get_model("ConvLSTMDwell", **model_config)
        original_state = model.state_dict()

        checkpoint_path = temp_model_dir / "model_best.pt"
        torch.save(
            {"model_state_dict": original_state},
            checkpoint_path,
        )

        loaded_model, _ = load_model_from_checkpoint(temp_model_dir, device="cpu")
        loaded_state = loaded_model.state_dict()

        # Check that all keys match
        assert set(original_state.keys()) == set(loaded_state.keys())

        # Check that weights match
        for key in original_state.keys():
            assert torch.allclose(original_state[key], loaded_state[key])


class TestModelBundle:
    """Test model bundling functions."""

    def _create_fake_model_dirs(self, tmp_path, model_config, pair_names):
        """Helper to create fake model directories for testing."""
        model_dirs = {}
        for pair in pair_names:
            pair_dir = tmp_path / pair
            pair_dir.mkdir(parents=True)

            # Save config.json
            config = {
                "model_name": "ConvLSTMDwell",
                **model_config,
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "device": "cpu",
                "seed": 42,
            }
            with open(pair_dir / "config.json", "w") as f:
                json.dump(config, f)

            # Save model_best.pt
            model = get_model("ConvLSTMDwell", **model_config)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": {"some_key": "some_value"},
                    "scheduler_state_dict": {"lr": 0.001},
                    "best_val_acc": 0.90 + 0.01 * len(pair),
                    "best_epoch": 15,
                },
                pair_dir / "model_best.pt",
            )
            model_dirs[pair] = pair_dir

        return model_dirs

    def _create_fake_kfold_dirs(self, tmp_path, model_config, pair_names, n_folds=3):
        """Helper to create fake k-fold model directories for testing."""
        for pair in pair_names:
            pair_dir = tmp_path / pair
            pair_dir.mkdir(parents=True)
            for fold_idx in range(n_folds):
                fold_dir = pair_dir / f"fold_{fold_idx}"
                fold_dir.mkdir()

                config = {
                    "model_name": "ConvLSTMDwell",
                    **model_config,
                    "epochs": 10,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "device": "cpu",
                    "seed": 42,
                }
                with open(fold_dir / "config.json", "w") as f:
                    json.dump(config, f)

                model = get_model("ConvLSTMDwell", **model_config)
                torch.save(
                    {"model_state_dict": model.state_dict()},
                    fold_dir / "model_best.pt",
                )

                # Write summary.json with different F1 scores and val losses per fold
                # fold_0 has lowest val loss (best generalization)
                # fold_2 has highest F1 but worst val loss (overfitting)
                f1_score = 0.80 + 0.05 * fold_idx
                val_loss = 0.10 + 0.10 * fold_idx
                with open(fold_dir / "summary.json", "w") as f:
                    json.dump(
                        {
                            "best_val_f1": f1_score,
                            "final_val_loss": val_loss,
                        },
                        f,
                    )

    def test_kfold_discovery_picks_best_fold(self, tmp_path, model_config):
        """Bundle CLI discovers fold_* subdirs and picks best fold by val F1."""
        from leech.commands.bundle import pick_best_fold

        pairs = ["Ala_notAla", "Gly_notGly"]
        root = tmp_path / "models"
        self._create_fake_kfold_dirs(root, model_config, pairs, n_folds=3)

        # Simulate the CLI discovery logic
        model_dirs = {}
        for subdir in sorted(root.iterdir()):
            if not subdir.is_dir():
                continue
            if (subdir / "model_best.pt").exists() and (subdir / "config.json").exists():
                model_dirs[subdir.name] = subdir
                continue
            fold_dirs = sorted(subdir.glob("fold_*/"))
            valid_folds = [
                f
                for f in fold_dirs
                if (f / "model_best.pt").exists() and (f / "config.json").exists()
            ]
            if valid_folds:
                best_fold = pick_best_fold(valid_folds)
                model_dirs[subdir.name] = best_fold

        assert len(model_dirs) == 2
        # fold_0 has lowest val loss (0.10) in our setup
        for pair in pairs:
            assert model_dirs[pair].name == "fold_0"

        # Verify the discovered dirs can be bundled
        bundle_path = tmp_path / "kfold_bundle.pt"
        create_bundle(model_dirs, bundle_path, "one_vs_all", "1.0.0")
        assert bundle_path.exists()

        metadata = list_bundle_models(bundle_path)
        assert metadata["num_models"] == 2
        assert set(metadata["pairs"]) == set(pairs)

    def test_kfold_discovery_fallback_no_summary(self, tmp_path, model_config):
        """When no summary.json exists, falls back to first fold."""
        from leech.commands.bundle import pick_best_fold

        pair_dir = tmp_path / "models" / "Ala_notAla"
        pair_dir.mkdir(parents=True)
        for fold_idx in range(3):
            fold_dir = pair_dir / f"fold_{fold_idx}"
            fold_dir.mkdir()
            config = {"model_name": "ConvLSTMDwell", **model_config}
            with open(fold_dir / "config.json", "w") as f:
                json.dump(config, f)
            model = get_model("ConvLSTMDwell", **model_config)
            torch.save(
                {"model_state_dict": model.state_dict()},
                fold_dir / "model_best.pt",
            )
            # No summary.json

        fold_dirs = sorted(pair_dir.glob("fold_*/"))
        valid_folds = [
            f for f in fold_dirs if (f / "model_best.pt").exists() and (f / "config.json").exists()
        ]
        best = pick_best_fold(valid_folds)
        assert best.name == "fold_0"  # fallback to first

    def test_kfold_overfitting_selects_lower_loss(self, tmp_path, model_config):
        """Fold with highest F1 but worst final_val_loss should NOT be selected."""
        from leech.commands.bundle import pick_best_fold

        pair_dir = tmp_path / "models" / "Asn_notAsn"
        pair_dir.mkdir(parents=True)

        # Simulate the real overfitting scenario:
        # fold_0: moderate F1 but low val loss (good generalization)
        # fold_2: highest F1 but high val loss (overfitting)
        fold_specs = [
            {"best_val_f1": 0.899, "final_val_loss": 0.105},  # fold_0
            {"best_val_f1": 0.870, "final_val_loss": 0.150},  # fold_1
            {"best_val_f1": 0.943, "final_val_loss": 0.344},  # fold_2 (overfitted)
        ]
        for fold_idx, spec in enumerate(fold_specs):
            fold_dir = pair_dir / f"fold_{fold_idx}"
            fold_dir.mkdir()
            config = {"model_name": "ConvLSTMDwell", **model_config}
            with open(fold_dir / "config.json", "w") as f:
                json.dump(config, f)
            model = get_model("ConvLSTMDwell", **model_config)
            torch.save(
                {"model_state_dict": model.state_dict()},
                fold_dir / "model_best.pt",
            )
            with open(fold_dir / "summary.json", "w") as f:
                json.dump(spec, f)

        fold_dirs = sorted(pair_dir.glob("fold_*/"))
        valid_folds = [
            f for f in fold_dirs if (f / "model_best.pt").exists() and (f / "config.json").exists()
        ]
        best = pick_best_fold(valid_folds)
        # Must select fold_0 (lowest val loss), NOT fold_2 (highest F1)
        assert best.name == "fold_0"

    def test_mixed_flat_and_kfold(self, tmp_path, model_config):
        """Bundle discovery handles mix of flat and k-fold pair dirs."""
        from leech.commands.bundle import pick_best_fold

        root = tmp_path / "models"

        # Flat pair (no folds)
        flat_pairs = ["Ser_notSer"]
        self._create_fake_model_dirs(root, model_config, flat_pairs)

        # K-fold pair
        kfold_pairs = ["Ala_notAla"]
        self._create_fake_kfold_dirs(root, model_config, kfold_pairs, n_folds=2)

        model_dirs = {}
        for subdir in sorted(root.iterdir()):
            if not subdir.is_dir():
                continue
            if (subdir / "model_best.pt").exists() and (subdir / "config.json").exists():
                model_dirs[subdir.name] = subdir
                continue
            fold_dirs = sorted(subdir.glob("fold_*/"))
            valid_folds = [
                f
                for f in fold_dirs
                if (f / "model_best.pt").exists() and (f / "config.json").exists()
            ]
            if valid_folds:
                best_fold = pick_best_fold(valid_folds)
                model_dirs[subdir.name] = best_fold

        assert len(model_dirs) == 2
        assert "Ser_notSer" in model_dirs
        assert "Ala_notAla" in model_dirs
        # Flat dir points to the pair dir itself
        assert model_dirs["Ser_notSer"].name == "Ser_notSer"
        # K-fold dir points to a fold subdir
        assert model_dirs["Ala_notAla"].name.startswith("fold_")

    def test_create_and_load_bundle(self, tmp_path, model_config):
        """Round-trip: create 3 fake models, bundle, load one, verify forward pass."""
        pairs = ["Ala_Gly", "Ala_Ser", "Gly_Ser"]
        model_dirs = self._create_fake_model_dirs(tmp_path / "models", model_config, pairs)

        bundle_path = tmp_path / "test_bundle.pt"
        create_bundle(model_dirs, bundle_path, "pairwise", "0.1.0-alpha.1")
        assert bundle_path.exists()

        # Load one model and verify forward pass
        loaded_model, config = load_model_from_bundle(bundle_path, "Ala_Gly", device="cpu")
        assert config["model_name"] == "ConvLSTMDwell"
        assert config["signal_len"] == model_config["signal_len"]

        # Forward pass
        signal = torch.randn(1, model_config["signal_len"])
        sequence = torch.randn(1, 4, model_config["kmer_len"])
        features = torch.randn(1, model_config["num_features"], model_config["kmer_len"])
        with torch.no_grad():
            output = loaded_model(signal, sequence, features)
        assert output.shape == (1, 1)

    def test_bundle_version(self, tmp_path, model_config):
        """Verify bundle_version in metadata matches what was passed."""
        model_dirs = self._create_fake_model_dirs(tmp_path / "models", model_config, ["Ala_Gly"])

        bundle_path = tmp_path / "test_bundle.pt"
        create_bundle(model_dirs, bundle_path, "pairwise", "1.2.3-beta.4")

        metadata = list_bundle_models(bundle_path)
        assert metadata["bundle_version"] == "1.2.3-beta.4"

    def test_bundle_strips_training_state(self, tmp_path, model_config):
        """Verify no optimizer/scheduler keys in bundle models."""
        model_dirs = self._create_fake_model_dirs(tmp_path / "models", model_config, ["Ala_Gly"])

        bundle_path = tmp_path / "test_bundle.pt"
        create_bundle(model_dirs, bundle_path, "pairwise", "0.1.0")

        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        model_entry = bundle["models"]["Ala_Gly"]
        assert "state_dict" in model_entry
        assert "optimizer_state_dict" not in model_entry
        assert "scheduler_state_dict" not in model_entry

        # Config should also be free of training params
        config = bundle["config"]
        assert "epochs" not in config
        assert "batch_size" not in config
        assert "learning_rate" not in config
        assert "device" not in config

    def test_load_missing_pair(self, tmp_path, model_config):
        """KeyError for nonexistent pair."""
        model_dirs = self._create_fake_model_dirs(tmp_path / "models", model_config, ["Ala_Gly"])

        bundle_path = tmp_path / "test_bundle.pt"
        create_bundle(model_dirs, bundle_path, "pairwise", "0.1.0")

        with pytest.raises(KeyError, match="nonexistent"):
            load_model_from_bundle(bundle_path, "nonexistent", device="cpu")

    def test_list_bundle_models(self, tmp_path, model_config):
        """Verify metadata includes version, pairs, architecture."""
        pairs = ["Asn_Gln", "Asn_Glu", "Gln_Glu"]
        model_dirs = self._create_fake_model_dirs(tmp_path / "models", model_config, pairs)

        bundle_path = tmp_path / "test_bundle.pt"
        create_bundle(model_dirs, bundle_path, "one_vs_all", "2.0.0")

        metadata = list_bundle_models(bundle_path)
        assert metadata["bundle_version"] == "2.0.0"
        assert metadata["architecture"] == "ConvLSTMDwell"
        assert metadata["comparison_type"] == "one_vs_all"
        assert metadata["num_models"] == 3
        assert set(metadata["pairs"]) == set(pairs)
        assert metadata["format_version"] == 1
        assert "created_at" in metadata

    def test_bundle_config_mismatch(self, tmp_path, model_config):
        """Error when models have incompatible architectures."""
        # Create two model dirs with different signal_len
        model_dirs = {}
        for pair, sig_len in [("Ala_Gly", 400), ("Ala_Ser", 200)]:
            pair_dir = tmp_path / "models" / pair
            pair_dir.mkdir(parents=True)

            config = {
                "model_name": "ConvLSTMDwell",
                **model_config,
                "signal_len": sig_len,
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
            }
            with open(pair_dir / "config.json", "w") as f:
                json.dump(config, f)

            mc = {**model_config, "signal_len": sig_len}
            model = get_model("ConvLSTMDwell", **mc)
            torch.save(
                {"model_state_dict": model.state_dict(), "best_val_acc": 0.9, "best_epoch": 5},
                pair_dir / "model_best.pt",
            )
            model_dirs[pair] = pair_dir

        bundle_path = tmp_path / "bad_bundle.pt"
        with pytest.raises(ValueError, match="Architecture config mismatch"):
            create_bundle(model_dirs, bundle_path, "pairwise", "0.1.0")


class TestAggregatePairwise:
    """Test pairwise voting aggregation."""

    def test_clear_winner(self):
        """When all models agree, the correct AA wins."""
        from leech.inference import aggregate_pairwise

        # Ala vs Gly: p=0.1 -> strong vote for Ala (label 0)
        # Ala vs Ser: p=0.2 -> strong vote for Ala (label 0)
        # Gly vs Ser: p=0.5 -> even split
        pairs = ["Ala_Gly", "Ala_Ser", "Gly_Ser"]
        probs = [0.1, 0.2, 0.5]

        predicted_aa, confidence, votes = aggregate_pairwise(pairs, probs)
        assert predicted_aa == "Ala"
        assert confidence > 0.0

    def test_symmetric_probabilities(self):
        """With perfectly even probabilities, votes distribute equally."""
        from leech.inference import aggregate_pairwise

        pairs = ["Ala_Gly", "Ala_Ser", "Gly_Ser"]
        probs = [0.5, 0.5, 0.5]

        _, confidence, votes = aggregate_pairwise(pairs, probs)
        # All amino acids should have equal votes
        assert abs(votes["Ala"] - votes["Gly"]) < 1e-10
        assert abs(votes["Gly"] - votes["Ser"]) < 1e-10

    def test_vote_strengths_sum(self):
        """Total vote strength equals number of pairs (each contributes 1.0)."""
        from leech.inference import aggregate_pairwise

        pairs = ["Ala_Gly", "Ala_Ser", "Gly_Ser"]
        probs = [0.3, 0.7, 0.9]

        _, _, votes = aggregate_pairwise(pairs, probs)
        total = sum(votes.values())
        assert abs(total - len(pairs)) < 1e-10

    def test_confidence_range(self):
        """Confidence is always between 0 and 1."""
        from leech.inference import aggregate_pairwise

        pairs = ["Ala_Gly", "Ala_Ser", "Gly_Ser"]
        for _ in range(20):
            probs = list(np.random.rand(3))
            _, confidence, _ = aggregate_pairwise(pairs, probs)
            assert 0.0 <= confidence <= 1.0

    def test_single_pair(self):
        """Works with a single pair."""
        from leech.inference import aggregate_pairwise

        pairs = ["Ala_Gly"]
        probs = [0.2]  # Favors Ala

        predicted_aa, confidence, votes = aggregate_pairwise(pairs, probs)
        assert predicted_aa == "Ala"
        assert abs(votes["Ala"] - 0.8) < 1e-10
        assert abs(votes["Gly"] - 0.2) < 1e-10


class TestAggregateOneVsAll:
    """Test one-vs-all aggregation."""

    def test_clear_winner(self):
        """The AA with lowest probability (most 'target-like') wins."""
        from leech.inference import aggregate_one_vs_all

        # Ala_notAla: p=0.1 -> score 0.9 (very likely Ala)
        # Gly_notGly: p=0.8 -> score 0.2 (unlikely Gly)
        # Ser_notSer: p=0.6 -> score 0.4 (unlikely Ser)
        pairs = ["Ala_notAla", "Gly_notGly", "Ser_notSer"]
        probs = [0.1, 0.8, 0.6]

        predicted_aa, confidence, scores = aggregate_one_vs_all(pairs, probs)
        assert predicted_aa == "Ala"
        assert abs(confidence - 0.9) < 1e-10

    def test_confidence_equals_max_score(self):
        """Confidence is the max score across all AAs."""
        from leech.inference import aggregate_one_vs_all

        pairs = ["Ala_notAla", "Gly_notGly"]
        probs = [0.3, 0.4]

        _, confidence, scores = aggregate_one_vs_all(pairs, probs)
        assert abs(confidence - max(scores.values())) < 1e-10

    def test_confidence_range(self):
        """Confidence is always between 0 and 1."""
        from leech.inference import aggregate_one_vs_all

        pairs = ["Ala_notAla", "Gly_notGly", "Ser_notSer"]
        for _ in range(20):
            probs = list(np.random.rand(3))
            _, confidence, _ = aggregate_one_vs_all(pairs, probs)
            assert 0.0 <= confidence <= 1.0


class TestMetricsEdgeCases:
    """Test edge cases in metrics computation."""

    def test_empty_predictions(self):
        """Test metrics with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        y_prob = np.array([])

        # Should either handle gracefully or raise informative error
        with pytest.raises((ValueError, IndexError)):
            compute_metrics(y_true, y_pred, y_prob)

    def test_mismatched_lengths(self):
        """Test metrics with mismatched array lengths."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])  # Wrong length
        y_prob = np.array([0.1, 0.9])

        # Should raise an error
        with pytest.raises((ValueError, IndexError)):
            compute_metrics(y_true, y_pred, y_prob)

    def test_probabilities_out_of_range(self):
        """Test metrics with probabilities outside [0,1]."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([-0.1, 1.5, 0.3, 0.8])  # Out of range

        # Should still compute metrics (sklearn doesn't validate prob range for AUC)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert "accuracy" in metrics


@pytest.mark.slow
class TestModelExport:
    """Test torch.export, TorchScript tracing, export, and bundle functions."""

    def test_export_round_trip(self, model_config):
        """Export a model, serialize, deserialize, and compare outputs."""
        model = get_model("ConvLSTMDwell", **model_config)
        model.eval()

        config = {"model_name": "ConvLSTMDwell", **model_config}
        ep = export_model(model, config)

        # Verify exported model produces same output
        signal = torch.randn(2, model_config["signal_len"])
        sequence = torch.randn(2, 4, model_config["kmer_len"])
        features = torch.randn(2, model_config["num_features"], model_config["kmer_len"])

        with torch.no_grad():
            original_out = model(signal, sequence, features)
            exported_out = ep.module()(signal, sequence, features)

        assert torch.allclose(original_out, exported_out, atol=1e-5)

    def test_serialize_deserialize_exported(self, model_config):
        """Test serialize/deserialize round-trip with torch.export."""
        model = get_model("ConvLSTMDwell", **model_config)
        config = {"model_name": "ConvLSTMDwell", **model_config}
        ep = export_model(model, config)

        data = serialize_exported_model(ep)
        assert isinstance(data, bytes)
        assert len(data) > 0

        restored = deserialize_exported_model(data, device="cpu")
        signal = torch.randn(1, model_config["signal_len"])
        sequence = torch.randn(1, 4, model_config["kmer_len"])
        features = torch.randn(1, model_config["num_features"], model_config["kmer_len"])

        with torch.no_grad():
            out1 = ep.module()(signal, sequence, features)
            out2 = restored(signal, sequence, features)

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_export_base_model(self):
        """Export a 2-arg model (no features)."""
        config = {
            "signal_len": 100,
            "kmer_len": 11,
            "conv_channels": [4, 16, 32],
            "lstm_hidden": 16,
        }
        model = get_model("ConvLSTMBase", **config)
        full_config = {"model_name": "ConvLSTMBase", **config}
        ep = export_model(model, full_config)

        signal = torch.randn(2, 100)
        sequence = torch.randn(2, 4, 11)
        with torch.no_grad():
            out = ep.module()(signal, sequence)
        assert out.shape == (2, 1)

    def test_legacy_trace_round_trip(self, model_config):
        """Legacy TorchScript trace still works."""
        model = get_model("ConvLSTMDwell", **model_config)
        model.eval()
        config = {"model_name": "ConvLSTMDwell", **model_config}
        traced = trace_model(model, config)

        signal = torch.randn(2, model_config["signal_len"])
        sequence = torch.randn(2, 4, model_config["kmer_len"])
        features = torch.randn(2, model_config["num_features"], model_config["kmer_len"])

        with torch.no_grad():
            original_out = model(signal, sequence, features)
            traced_out = traced(signal, sequence, features)

        assert torch.allclose(original_out, traced_out, atol=1e-5)

    def test_legacy_serialize_deserialize(self, model_config):
        """Legacy TorchScript serialize/deserialize still works."""
        model = get_model("ConvLSTMDwell", **model_config)
        config = {"model_name": "ConvLSTMDwell", **model_config}
        traced = trace_model(model, config)

        data = serialize_traced_model(traced)
        assert isinstance(data, bytes)
        restored = deserialize_traced_model(data, device="cpu")

        signal = torch.randn(1, model_config["signal_len"])
        sequence = torch.randn(1, 4, model_config["kmer_len"])
        features = torch.randn(1, model_config["num_features"], model_config["kmer_len"])

        with torch.no_grad():
            out1 = traced(signal, sequence, features)
            out2 = restored(signal, sequence, features)

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_export_single_model(self, tmp_path, model_config):
        """Test export_single_model creates a loadable file."""
        # Create a model directory
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        config = {
            "model_name": "ConvLSTMDwell",
            **model_config,
            "epochs": 10,
            "batch_size": 32,
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)

        model = get_model("ConvLSTMDwell", **model_config)
        torch.save({"model_state_dict": model.state_dict()}, model_dir / "model_best.pt")

        output_path = tmp_path / "exported.pt"
        result = export_single_model(model_dir, output_path)
        assert result.exists()

        # Load with bare torch.export.load — no leech needed
        extra = {"leech_meta.txt": ""}
        ep = torch.export.load(str(output_path), extra_files=extra)
        assert extra["leech_meta.txt"] != ""

        meta = json.loads(extra["leech_meta.txt"])
        assert meta["model_name"] == "ConvLSTMDwell"
        assert meta["signal_len"] == model_config["signal_len"]

        # Verify forward pass
        signal = torch.randn(1, model_config["signal_len"])
        sequence = torch.randn(1, 4, model_config["kmer_len"])
        features = torch.randn(1, model_config["num_features"], model_config["kmer_len"])
        with torch.no_grad():
            out = ep.module()(signal, sequence, features)
        assert out.shape == (1, 1)

    def _create_fake_model_dirs(self, tmp_path, model_config, pair_names):
        """Helper to create fake model directories for testing."""
        model_dirs = {}
        for pair in pair_names:
            pair_dir = tmp_path / pair
            pair_dir.mkdir(parents=True)

            config = {
                "model_name": "ConvLSTMDwell",
                **model_config,
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "device": "cpu",
                "seed": 42,
            }
            with open(pair_dir / "config.json", "w") as f:
                json.dump(config, f)

            model = get_model("ConvLSTMDwell", **model_config)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": 0.90,
                    "best_epoch": 15,
                },
                pair_dir / "model_best.pt",
            )
            model_dirs[pair] = pair_dir
        return model_dirs

    def test_torchscript_bundle_create_and_load(self, tmp_path, model_config):
        """Create an exported bundle, load a model from it, verify forward pass."""
        pairs = ["Ala_Gly", "Ala_Ser", "Gly_Ser"]
        model_dirs = self._create_fake_model_dirs(tmp_path / "models", model_config, pairs)

        bundle_path = tmp_path / "ts_bundle.pt"
        create_torchscript_bundle(model_dirs, bundle_path, "pairwise", "1.0.0")
        assert bundle_path.exists()

        # Verify metadata
        metadata = list_bundle_models(bundle_path)
        assert metadata["format_version"] == 3
        assert metadata["torchscript"] is True
        assert metadata["requires_features"] is True
        assert set(metadata["pairs"]) == set(pairs)

        # Load a model and run forward pass
        loaded_model, config = load_model_from_bundle(bundle_path, "Ala_Gly", device="cpu")
        signal = torch.randn(1, model_config["signal_len"])
        sequence = torch.randn(1, 4, model_config["kmer_len"])
        features = torch.randn(1, model_config["num_features"], model_config["kmer_len"])
        with torch.no_grad():
            output = loaded_model(signal, sequence, features)
        assert output.shape == (1, 1)

    def test_old_bundle_still_works(self, tmp_path, model_config):
        """Verify format_version=1 (state_dict) bundles are still loadable."""
        pairs = ["Ala_Gly"]
        model_dirs = self._create_fake_model_dirs(tmp_path / "models", model_config, pairs)

        bundle_path = tmp_path / "v1_bundle.pt"
        create_bundle(model_dirs, bundle_path, "pairwise", "0.1.0")

        metadata = list_bundle_models(bundle_path)
        assert metadata["format_version"] == 1
        assert metadata.get("torchscript", False) is False

        loaded_model, config = load_model_from_bundle(bundle_path, "Ala_Gly", device="cpu")
        signal = torch.randn(1, model_config["signal_len"])
        sequence = torch.randn(1, 4, model_config["kmer_len"])
        features = torch.randn(1, model_config["num_features"], model_config["kmer_len"])
        with torch.no_grad():
            output = loaded_model(signal, sequence, features)
        assert output.shape == (1, 1)

    def test_torchscript_bundle_config_mismatch(self, tmp_path, model_config):
        """Error when models in TorchScript bundle have incompatible configs."""
        model_dirs = {}
        for pair, sig_len in [("Ala_Gly", 400), ("Ala_Ser", 200)]:
            pair_dir = tmp_path / "models" / pair
            pair_dir.mkdir(parents=True)

            config = {
                "model_name": "ConvLSTMDwell",
                **model_config,
                "signal_len": sig_len,
                "epochs": 10,
                "batch_size": 32,
            }
            with open(pair_dir / "config.json", "w") as f:
                json.dump(config, f)

            mc = {**model_config, "signal_len": sig_len}
            m = get_model("ConvLSTMDwell", **mc)
            torch.save(
                {"model_state_dict": m.state_dict(), "best_val_acc": 0.9, "best_epoch": 5},
                pair_dir / "model_best.pt",
            )
            model_dirs[pair] = pair_dir

        bundle_path = tmp_path / "bad_ts_bundle.pt"
        with pytest.raises(ValueError, match="Architecture config mismatch"):
            create_torchscript_bundle(model_dirs, bundle_path, "pairwise", "0.1.0")


class TestVmapBundle:
    """Test vmap bundle creation and inference."""

    TCN_CONFIG = {
        "signal_len": 200,
        "kmer_len": 11,
        "num_features": 5,
        "hidden_channels": 16,
        "num_layers": 2,
        "kernel_size": 3,
        "num_attn_heads": 1,
        "dropout": 0.0,
    }

    def _create_tcn_model_dirs(self, tmp_path, pair_names, platt=False):
        """Create fake TCNDwellGN model directories for testing."""
        model_dirs = {}
        for pair in pair_names:
            pair_dir = tmp_path / pair
            pair_dir.mkdir(parents=True)

            config = {
                "model_name": "TCNDwellGN",
                **self.TCN_CONFIG,
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "device": "cpu",
                "seed": 42,
            }
            with open(pair_dir / "config.json", "w") as f:
                json.dump(config, f)

            model = get_model("TCNDwellGN", **self.TCN_CONFIG)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": 0.90,
                    "best_epoch": 5,
                },
                pair_dir / "model_best.pt",
            )

            if platt:
                with open(pair_dir / "platt.json", "w") as f:
                    json.dump({"platt_a": 1.05, "platt_b": -0.02}, f)

            model_dirs[pair] = pair_dir
        return model_dirs

    def test_create_vmap_bundle(self, tmp_path):
        """Create a vmap bundle from 3 TCNDwellGN models, verify format_version 4."""
        pairs = ["Ala_Gly", "Ala_Ser", "Gly_Ser"]
        model_dirs = self._create_tcn_model_dirs(tmp_path / "models", pairs)

        bundle_path = tmp_path / "vmap_bundle.pt"
        create_vmap_bundle(model_dirs, bundle_path, "pairwise", "1.0.0")
        assert bundle_path.exists()

        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        metadata = bundle["metadata"]
        assert metadata["format_version"] == 4
        assert metadata["vmap"] is True
        assert metadata["architecture"] == "TCNDwellGN"
        assert metadata["num_models"] == 3
        assert set(metadata["pairs"]) == set(pairs)

        # Verify stacked params have leading model dimension
        for key, val in bundle["stacked_params"].items():
            assert val.shape[0] == 3, f"stacked_params[{key}] leading dim should be 3"

        # Verify Platt scaling tensors
        assert bundle["platt_a"].shape == (3,)
        assert bundle["platt_b"].shape == (3,)

    def test_vmap_bundle_with_platt(self, tmp_path):
        """vmap bundle loads Platt scaling parameters correctly."""
        pairs = ["Ala_Gly"]
        model_dirs = self._create_tcn_model_dirs(tmp_path / "models", pairs, platt=True)

        bundle_path = tmp_path / "vmap_platt.pt"
        create_vmap_bundle(model_dirs, bundle_path, "pairwise", "1.0.0")

        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        assert abs(bundle["platt_a"][0].item() - 1.05) < 1e-5
        assert abs(bundle["platt_b"][0].item() - (-0.02)) < 1e-5

    def test_vmap_bundle_incompatible_architecture(self, tmp_path):
        """ValueError when architecture is vmap-incompatible (ConvLSTMDwell has LSTM)."""
        pair_dir = tmp_path / "models" / "Ala_Gly"
        pair_dir.mkdir(parents=True)

        config = {
            "model_name": "ConvLSTMDwell",
            "signal_len": 400,
            "kmer_len": 11,
            "num_features": 5,
            "conv_channels": [4, 16, 64],
            "lstm_hidden": 32,
            "dropout": 0.1,
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
        }
        with open(pair_dir / "config.json", "w") as f:
            json.dump(config, f)

        model = get_model(
            "ConvLSTMDwell",
            signal_len=400,
            kmer_len=11,
            num_features=5,
            conv_channels=[4, 16, 64],
            lstm_hidden=32,
            dropout=0.1,
        )
        torch.save(
            {"model_state_dict": model.state_dict()},
            pair_dir / "model_best.pt",
        )

        bundle_path = tmp_path / "bad_vmap.pt"
        with pytest.raises(ValueError, match="not vmap-compatible"):
            create_vmap_bundle({"Ala_Gly": pair_dir}, bundle_path, "pairwise", "1.0.0")

    def test_load_model_from_vmap_bundle(self, tmp_path):
        """Round-trip: create vmap bundle, load single model, verify forward pass."""
        pairs = ["Ala_Gly", "Ala_Ser"]
        model_dirs = self._create_tcn_model_dirs(tmp_path / "models", pairs)

        bundle_path = tmp_path / "vmap_bundle.pt"
        create_vmap_bundle(model_dirs, bundle_path, "pairwise", "1.0.0")

        model, config = load_model_from_bundle(bundle_path, "Ala_Gly", device="cpu")
        assert config["model_name"] == "TCNDwellGN"

        # Forward pass
        signal = torch.randn(1, self.TCN_CONFIG["signal_len"])
        sequence = torch.randn(1, 4, self.TCN_CONFIG["kmer_len"])
        feat_len = self.TCN_CONFIG["kmer_len"] + 2 * 15  # default dwell_margin=15
        features = torch.randn(1, self.TCN_CONFIG["num_features"], feat_len)
        with torch.no_grad():
            output = model(signal, sequence, features)
        assert output.shape == (1, 1)

    def test_load_missing_pair_from_vmap_bundle(self, tmp_path):
        """KeyError for nonexistent pair in vmap bundle."""
        pairs = ["Ala_Gly"]
        model_dirs = self._create_tcn_model_dirs(tmp_path / "models", pairs)

        bundle_path = tmp_path / "vmap_bundle.pt"
        create_vmap_bundle(model_dirs, bundle_path, "pairwise", "1.0.0")

        with pytest.raises(KeyError, match="nonexistent"):
            load_model_from_bundle(bundle_path, "nonexistent", device="cpu")

    def test_numerical_equivalence_v1_vs_v4(self, tmp_path):
        """Models loaded from v1 and v4 bundles produce identical outputs."""
        pairs = ["Ala_Gly", "Ala_Ser", "Gly_Ser"]
        model_dirs = self._create_tcn_model_dirs(tmp_path / "models", pairs)

        v1_path = tmp_path / "v1_bundle.pt"
        v4_path = tmp_path / "v4_bundle.pt"
        create_bundle(model_dirs, v1_path, "pairwise", "1.0.0")
        create_vmap_bundle(model_dirs, v4_path, "pairwise", "1.0.0")

        signal = torch.randn(2, self.TCN_CONFIG["signal_len"])
        sequence = torch.randn(2, 4, self.TCN_CONFIG["kmer_len"])
        feat_len = self.TCN_CONFIG["kmer_len"] + 2 * 15
        features = torch.randn(2, self.TCN_CONFIG["num_features"], feat_len)

        for pair in pairs:
            m_v1, _ = load_model_from_bundle(v1_path, pair, device="cpu")
            m_v4, _ = load_model_from_bundle(v4_path, pair, device="cpu")
            with torch.no_grad():
                out_v1 = m_v1(signal, sequence, features)
                out_v4 = m_v4(signal, sequence, features)
            np.testing.assert_allclose(
                out_v1.numpy(),
                out_v4.numpy(),
                atol=1e-6,
                err_msg=f"Numerical mismatch for pair {pair}",
            )

    def test_list_vmap_bundle_models(self, tmp_path):
        """Metadata from vmap bundle reports correct format."""
        pairs = ["Ala_Gly"]
        model_dirs = self._create_tcn_model_dirs(tmp_path / "models", pairs)

        bundle_path = tmp_path / "vmap_bundle.pt"
        create_vmap_bundle(model_dirs, bundle_path, "pairwise", "2.0.0")

        metadata = list_bundle_models(bundle_path)
        assert metadata["format_version"] == 4
        assert metadata["bundle_version"] == "2.0.0"
        assert metadata["vmap"] is True

    def test_vmap_bundle_config_mismatch(self, tmp_path):
        """Error when models have incompatible architectures in vmap bundle."""
        model_dirs = {}
        for pair, sig_len in [("Ala_Gly", 200), ("Ala_Ser", 100)]:
            pair_dir = tmp_path / "models" / pair
            pair_dir.mkdir(parents=True)

            config = {
                "model_name": "TCNDwellGN",
                **self.TCN_CONFIG,
                "signal_len": sig_len,
                "epochs": 10,
                "batch_size": 32,
            }
            with open(pair_dir / "config.json", "w") as f:
                json.dump(config, f)

            mc = {**self.TCN_CONFIG, "signal_len": sig_len}
            m = get_model("TCNDwellGN", **mc)
            torch.save(
                {"model_state_dict": m.state_dict()},
                pair_dir / "model_best.pt",
            )
            model_dirs[pair] = pair_dir

        bundle_path = tmp_path / "bad_vmap.pt"
        with pytest.raises(ValueError, match="Architecture config mismatch"):
            create_vmap_bundle(model_dirs, bundle_path, "pairwise", "1.0.0")


class TestVmapInferenceIntegration:
    """Integration tests for the vmap vectorized inference path.

    Exercises the full vmap inference pipeline: loading a vmap bundle,
    setting up vmapped_forward via torch.func, running batches of
    synthetic data through both vmap and sequential paths, and verifying
    numerical equivalence and correct Platt scaling.
    """

    TCN_CONFIG = {
        "signal_len": 200,
        "kmer_len": 11,
        "num_features": 5,
        "hidden_channels": 16,
        "num_layers": 2,
        "kernel_size": 3,
        "num_attn_heads": 1,
        "dropout": 0.0,
    }

    def _create_tcn_model_dirs(self, tmp_path, pair_names, platt_params=None):
        """Create TCNDwellGN model directories with optional per-pair Platt scaling."""
        model_dirs = {}
        for pair in pair_names:
            pair_dir = tmp_path / pair
            pair_dir.mkdir(parents=True)

            config = {
                "model_name": "TCNDwellGN",
                **self.TCN_CONFIG,
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "device": "cpu",
                "seed": 42,
            }
            with open(pair_dir / "config.json", "w") as f:
                json.dump(config, f)

            model = get_model("TCNDwellGN", **self.TCN_CONFIG)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": 0.90,
                    "best_epoch": 5,
                },
                pair_dir / "model_best.pt",
            )

            if platt_params and pair in platt_params:
                a, b = platt_params[pair]
                with open(pair_dir / "platt.json", "w") as f:
                    json.dump({"platt_a": a, "platt_b": b}, f)

            model_dirs[pair] = pair_dir
        return model_dirs

    def _make_synthetic_batch(self, batch_size):
        """Create a synthetic (signal, sequence, features) batch."""
        dwell_margin = 15  # TCNDwell default
        feat_len = self.TCN_CONFIG["kmer_len"] + 2 * dwell_margin
        signal = torch.randn(batch_size, self.TCN_CONFIG["signal_len"])
        sequence = torch.randn(batch_size, 4, self.TCN_CONFIG["kmer_len"])
        features = torch.randn(batch_size, self.TCN_CONFIG["num_features"], feat_len)
        return signal, sequence, features

    def _run_sequential_inference(self, bundle_path, pairs, signal, sequence, features):
        """Run inference the old way: load each model, run sequentially."""
        n_pairs = len(pairs)
        batch_size = signal.shape[0]
        probs = np.empty((n_pairs, batch_size), dtype=np.float32)

        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)

        for i, pair in enumerate(pairs):
            model, _ = load_model_from_bundle(bundle_path, pair, device="cpu")
            model.eval()
            with torch.no_grad():
                logits = model(signal, sequence, features)

            # Apply Platt scaling
            platt_a = bundle["platt_a"][i].item()
            platt_b = bundle["platt_b"][i].item()
            logits = platt_a * logits + platt_b

            probs[i] = torch.sigmoid(logits).numpy().flatten()
        return probs

    def _run_vmap_inference(self, bundle_path, signal, sequence, features):
        """Run inference the vmap way: stack_module_state + vmap."""
        from leech.model_loading import _instantiate_model

        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        config = bundle["config"]

        stacked_params = bundle["stacked_params"]
        stacked_buffers = bundle["stacked_buffers"]
        platt_a = bundle["platt_a"]
        platt_b = bundle["platt_b"]

        base_model = _instantiate_model(config).eval()

        def single_forward(params, buffers, sig, seq, feat):
            return torch.func.functional_call(base_model, (params, buffers), (sig, seq, feat))

        vmapped_forward = torch.vmap(single_forward, in_dims=(0, 0, None, None, None))

        with torch.inference_mode():
            all_logits = vmapped_forward(
                stacked_params, stacked_buffers, signal, sequence, features
            )
            all_logits = platt_a[:, None, None] * all_logits + platt_b[:, None, None]
            probs = torch.sigmoid(all_logits).squeeze(-1).numpy()

        return probs

    def test_vmap_matches_sequential_inference(self, tmp_path):
        """Core integration test: vmap and sequential paths produce identical results."""
        pairs = ["Ala_Gly", "Ala_Ser", "Gly_Ser"]
        model_dirs = self._create_tcn_model_dirs(tmp_path / "models", pairs)

        bundle_path = tmp_path / "vmap_bundle.pt"
        create_vmap_bundle(model_dirs, bundle_path, "pairwise", "1.0.0")

        signal, sequence, features = self._make_synthetic_batch(batch_size=8)

        probs_seq = self._run_sequential_inference(bundle_path, pairs, signal, sequence, features)
        probs_vmap = self._run_vmap_inference(bundle_path, signal, sequence, features)

        np.testing.assert_allclose(
            probs_seq,
            probs_vmap,
            atol=1e-6,
            err_msg="vmap inference does not match sequential inference",
        )

    def test_vmap_with_platt_scaling(self, tmp_path):
        """vmap path applies Platt scaling identically to sequential."""
        pairs = ["Ala_Gly", "Ala_Ser"]
        platt = {"Ala_Gly": (1.15, -0.3), "Ala_Ser": (0.9, 0.1)}
        model_dirs = self._create_tcn_model_dirs(tmp_path / "models", pairs, platt_params=platt)

        bundle_path = tmp_path / "vmap_platt.pt"
        create_vmap_bundle(model_dirs, bundle_path, "pairwise", "1.0.0")

        signal, sequence, features = self._make_synthetic_batch(batch_size=4)

        probs_seq = self._run_sequential_inference(bundle_path, pairs, signal, sequence, features)
        probs_vmap = self._run_vmap_inference(bundle_path, signal, sequence, features)

        np.testing.assert_allclose(
            probs_seq,
            probs_vmap,
            atol=1e-6,
            err_msg="Platt-scaled vmap inference does not match sequential",
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_vmap_various_batch_sizes(self, tmp_path, batch_size):
        """vmap produces correct results across different batch sizes."""
        pairs = ["Ala_Gly", "Gly_Ser"]
        model_dirs = self._create_tcn_model_dirs(tmp_path / "models", pairs)

        bundle_path = tmp_path / "vmap_batch.pt"
        create_vmap_bundle(model_dirs, bundle_path, "pairwise", "1.0.0")

        signal, sequence, features = self._make_synthetic_batch(batch_size=batch_size)

        probs_seq = self._run_sequential_inference(bundle_path, pairs, signal, sequence, features)
        probs_vmap = self._run_vmap_inference(bundle_path, signal, sequence, features)

        np.testing.assert_allclose(
            probs_seq,
            probs_vmap,
            atol=1e-6,
            err_msg=f"Mismatch at batch_size={batch_size}",
        )

    def test_vmap_output_shapes(self, tmp_path):
        """vmap output has correct shape (n_models, batch_size)."""
        pairs = ["Ala_Gly", "Ala_Ser", "Gly_Ser"]
        model_dirs = self._create_tcn_model_dirs(tmp_path / "models", pairs)

        bundle_path = tmp_path / "vmap_shape.pt"
        create_vmap_bundle(model_dirs, bundle_path, "pairwise", "1.0.0")

        signal, sequence, features = self._make_synthetic_batch(batch_size=5)
        probs = self._run_vmap_inference(bundle_path, signal, sequence, features)

        assert probs.shape == (3, 5), f"Expected (3, 5), got {probs.shape}"
        assert np.all((probs >= 0) & (probs <= 1)), "Probabilities must be in [0, 1]"

    def test_vmap_deterministic(self, tmp_path):
        """Two vmap runs on the same input produce identical results."""
        pairs = ["Ala_Gly"]
        model_dirs = self._create_tcn_model_dirs(tmp_path / "models", pairs)

        bundle_path = tmp_path / "vmap_det.pt"
        create_vmap_bundle(model_dirs, bundle_path, "pairwise", "1.0.0")

        signal, sequence, features = self._make_synthetic_batch(batch_size=4)

        probs1 = self._run_vmap_inference(bundle_path, signal, sequence, features)
        probs2 = self._run_vmap_inference(bundle_path, signal, sequence, features)

        np.testing.assert_array_equal(probs1, probs2, err_msg="vmap is not deterministic")

    def test_streaming_batch_scatter(self, tmp_path):
        """Simulate the streaming batch scatter pattern from run_bundle_inference.

        Verifies that splitting data into multiple flush_batch calls and scattering
        into read_probs produces the same result as a single batch.
        """
        pairs = ["Ala_Gly", "Ala_Ser", "Gly_Ser"]
        model_dirs = self._create_tcn_model_dirs(tmp_path / "models", pairs)

        bundle_path = tmp_path / "vmap_stream.pt"
        create_vmap_bundle(model_dirs, bundle_path, "pairwise", "1.0.0")

        from leech.model_loading import _instantiate_model

        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        config = bundle["config"]
        stacked_params = bundle["stacked_params"]
        stacked_buffers = bundle["stacked_buffers"]
        platt_a = bundle["platt_a"]
        platt_b = bundle["platt_b"]

        base_model = _instantiate_model(config).eval()

        def single_forward(params, buffers, sig, seq, feat):
            return torch.func.functional_call(base_model, (params, buffers), (sig, seq, feat))

        vmapped_forward = torch.vmap(single_forward, in_dims=(0, 0, None, None, None))

        # Create 10 chunks with read IDs
        total_chunks = 10
        batch_size = 3  # Will create batches of [3, 3, 3, 1]
        signal, sequence, features = self._make_synthetic_batch(batch_size=total_chunks)
        read_ids = [f"read_{i:04d}" for i in range(total_chunks)]

        # Simulate streaming: accumulate into small batches and scatter
        read_probs: dict[str, np.ndarray] = {}

        for start in range(0, total_chunks, batch_size):
            end = min(start + batch_size, total_chunks)
            sig_t = signal[start:end]
            seq_t = sequence[start:end]
            feat_t = features[start:end]

            with torch.inference_mode():
                all_logits = vmapped_forward(stacked_params, stacked_buffers, sig_t, seq_t, feat_t)
                all_logits = platt_a[:, None, None] * all_logits + platt_b[:, None, None]
                all_p = torch.sigmoid(all_logits).squeeze(-1).cpu().numpy()

            for i, rid in enumerate(read_ids[start:end]):
                read_probs[rid] = all_p[:, i]

        # Compare with single-batch reference
        with torch.inference_mode():
            ref_logits = vmapped_forward(
                stacked_params, stacked_buffers, signal, sequence, features
            )
            ref_logits = platt_a[:, None, None] * ref_logits + platt_b[:, None, None]
            ref_probs = torch.sigmoid(ref_logits).squeeze(-1).cpu().numpy()

        for i, rid in enumerate(read_ids):
            np.testing.assert_allclose(
                read_probs[rid],
                ref_probs[:, i],
                atol=1e-6,
                err_msg=f"Streaming batch scatter mismatch for {rid}",
            )

        assert len(read_probs) == total_chunks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
