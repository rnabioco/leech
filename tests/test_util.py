"""
Tests for utility functions.

Tests metrics computation, model loading, and helper functions.
"""

import json

import numpy as np
import pytest
import torch

from leech.models import get_model
from leech.util import (
    compute_metrics,
    load_model_from_checkpoint,
    print_metrics,
    save_metrics,
)


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
        assert metrics["auc"] > 0.9  # Should be high for perfect predictions

    def test_random_predictions(self, sample_predictions):
        """Test metrics with random predictions."""
        y_true, y_pred, y_prob = sample_predictions

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc" in metrics
        assert "confusion_matrix" in metrics

        # All metrics should be between 0 and 1
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["auc"] <= 1

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

        # AUC should be 0 when only one class present
        assert metrics["auc"] == 0.0
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
        assert "auc" in loaded
        assert "confusion_matrix" in loaded


class TestPrintMetrics:
    """Test print_metrics function."""

    def test_print_metrics_no_error(self, sample_predictions, capsys):
        """Test that print_metrics doesn't raise errors."""
        y_true, y_pred, y_prob = sample_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob)

        print_metrics(metrics)

        captured = capsys.readouterr()
        assert "EVALUATION METRICS" in captured.out
        assert "Accuracy" in captured.out
        assert "Confusion Matrix" in captured.out

    def test_print_metrics_without_confusion_matrix(self, capsys):
        """Test printing metrics without confusion matrix."""
        metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.89,
            "f1": 0.90,
            "auc": 0.93,
        }

        print_metrics(metrics)

        captured = capsys.readouterr()
        assert "EVALUATION METRICS" in captured.out
        # Confusion matrix section should not appear
        assert captured.out.count("Confusion Matrix") <= 1


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
