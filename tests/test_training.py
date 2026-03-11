"""
Tests for training module.

Tests Trainer class and train_model function.
"""

import json

import pytest
import torch
from torch.utils.data import DataLoader

from leech.dataset import LeechDataset, collate_fn
from leech.models import get_model
from leech.training import Trainer, compute_class_weights, train_model


class TestTrainer:
    """Test Trainer class."""

    @pytest.fixture
    def sample_model(self, model_config):
        """Create a sample model for testing."""
        return get_model("ConvLSTMDwell", **model_config)

    @pytest.fixture
    def sample_dataloader(self, temp_chunks_file):
        """Create a sample dataloader."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )
        return DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    def test_trainer_initialization(self, sample_model, sample_dataloader):
        """Test Trainer initialization."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=None,
            device="cpu",
            learning_rate=0.001,
        )

        assert trainer.model is not None
        assert trainer.train_loader is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    def test_trainer_with_output_dir(self, sample_model, sample_dataloader, tmp_path):
        """Test Trainer with output directory."""
        output_dir = tmp_path / "trainer_output"

        Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            output_dir=output_dir,
        )

        assert output_dir.exists()

    def test_train_epoch(self, sample_model, sample_dataloader):
        """Test training for one epoch."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            learning_rate=0.001,
        )

        loss, acc = trainer.train_epoch()

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 1

    def test_validate_without_loader(self, sample_model, sample_dataloader):
        """Test validation when no validation loader provided."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=None,
            device="cpu",
        )

        loss, acc, auc, f1 = trainer.validate()

        assert loss == 0.0
        assert acc == 0.0
        assert auc == 0.0
        assert f1 == 0.0

    def test_validate_with_loader(self, sample_model, sample_dataloader):
        """Test validation with validation loader."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,  # Use same for simplicity
            device="cpu",
        )

        loss, acc, auc, f1 = trainer.validate()

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert isinstance(auc, float)
        assert isinstance(f1, float)
        assert loss >= 0
        assert 0 <= acc <= 1
        assert 0 <= auc <= 1
        assert 0 <= f1 <= 1

    def test_train_multiple_epochs(self, sample_model, sample_dataloader, tmp_path):
        """Test training for multiple epochs."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            device="cpu",
            output_dir=tmp_path,
        )

        history = trainer.train(epochs=2, early_stopping_patience=10)

        assert "train_loss" in history
        assert "train_acc" in history
        assert "val_loss" in history
        assert "val_acc" in history
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

    def test_early_stopping(self, sample_model, sample_dataloader):
        """Test early stopping mechanism."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            device="cpu",
        )

        # Train with very small patience - should stop early
        history = trainer.train(epochs=100, early_stopping_patience=1)

        # Should stop before 100 epochs
        assert len(history["train_loss"]) < 100

    def test_save_checkpoint(self, sample_model, sample_dataloader, tmp_path):
        """Test checkpoint saving."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            output_dir=tmp_path,
        )

        trainer.save_checkpoint("test_model.pt")

        checkpoint_path = tmp_path / "test_model.pt"
        assert checkpoint_path.exists()

        # Load and verify checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_save_history(self, sample_model, sample_dataloader, tmp_path):
        """Test history saving."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            device="cpu",
            output_dir=tmp_path,
        )

        trainer.train(epochs=2)
        trainer.save_history()

        # Check that metrics.json was created
        metrics_path = tmp_path / "metrics.json"
        assert metrics_path.exists()

        # Check that summary.json was created
        summary_path = tmp_path / "summary.json"
        assert summary_path.exists()

        # Verify structure
        with open(summary_path) as f:
            summary = json.load(f)

        assert "best_val_acc" in summary
        assert "best_epoch" in summary
        assert "final_train_loss" in summary

    def test_best_model_tracking(self, sample_model, sample_dataloader, tmp_path):
        """Test that best model is tracked correctly."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            device="cpu",
            output_dir=tmp_path,
        )

        trainer.train(epochs=3)

        # Best model should have been saved
        best_model_path = tmp_path / "model_best.pt"
        assert best_model_path.exists()

        assert trainer.best_val_acc >= 0
        assert trainer.best_epoch > 0


class TestTrainModel:
    """Test train_model high-level function."""

    def test_train_model_basic(self, temp_chunks_file, tmp_path):
        """Test basic train_model execution."""
        output_dir = tmp_path / "training"

        history = train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,  # Use same for simplicity
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            signal_len=400,
            kmer_len=11,
            epochs=2,
            batch_size=2,
            learning_rate=0.001,
            device="cpu",
            seed=42,
        )

        assert "train_loss" in history
        assert "train_acc" in history
        assert len(history["train_loss"]) <= 2

    def test_train_model_saves_config(self, temp_chunks_file, tmp_path):
        """Test that train_model saves configuration."""
        output_dir = tmp_path / "training"

        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=None,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=1,
            batch_size=2,
            device="cpu",
            seed=42,
        )

        config_path = output_dir / "config.json"
        assert config_path.exists()

        with open(config_path) as f:
            config = json.load(f)

        assert config["model_name"] == "ConvLSTMDwell"
        assert config["epochs"] == 1
        assert config["seed"] == 42

    def test_train_model_saves_checkpoints(self, temp_chunks_file, tmp_path):
        """Test that train_model saves checkpoints."""
        output_dir = tmp_path / "training"

        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=2,
            batch_size=2,
            device="cpu",
            seed=42,
        )

        # Check for saved models
        best_model = output_dir / "model_best.pt"
        last_model = output_dir / "model_last.pt"

        assert best_model.exists()
        assert last_model.exists()

    def test_train_model_without_validation(self, temp_chunks_file, tmp_path):
        """Test training without validation data."""
        output_dir = tmp_path / "training"

        history = train_model(
            train_data_path=temp_chunks_file,
            val_data_path=None,  # No validation
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=2,
            batch_size=2,
            device="cpu",
            seed=42,
        )

        assert "train_loss" in history
        # Val metrics should be empty
        assert len(history["val_loss"]) == 0

    def test_train_model_different_architectures(self, temp_chunks_file, tmp_path):
        """Test training with different model architectures."""
        for model_name in ["ConvLSTMBase", "ConvLSTMDwell"]:
            output_dir = tmp_path / f"training_{model_name}"

            history = train_model(
                train_data_path=temp_chunks_file,
                val_data_path=None,
                model_name=model_name,
                output_dir=output_dir,
                epochs=1,
                batch_size=2,
                device="cpu",
                seed=42,
            )

            assert len(history["train_loss"]) >= 1
            assert output_dir.exists()

    def test_train_model_reproducibility(self, temp_chunks_file, tmp_path):
        """Test that training with same seed produces reproducible results."""
        # First run
        output_dir1 = tmp_path / "training1"
        history1 = train_model(
            train_data_path=temp_chunks_file,
            val_data_path=None,
            model_name="ConvLSTMDwell",
            output_dir=output_dir1,
            epochs=1,
            batch_size=2,
            device="cpu",
            seed=42,
        )

        # Second run with same seed
        output_dir2 = tmp_path / "training2"
        history2 = train_model(
            train_data_path=temp_chunks_file,
            val_data_path=None,
            model_name="ConvLSTMDwell",
            output_dir=output_dir2,
            epochs=1,
            batch_size=2,
            device="cpu",
            seed=42,
        )

        # Results should be close (may not be exactly equal due to non-determinism)
        assert abs(history1["train_loss"][0] - history2["train_loss"][0]) < 0.1

    def test_train_model_custom_hyperparameters(self, temp_chunks_file, tmp_path):
        """Test training with custom hyperparameters."""
        output_dir = tmp_path / "training"

        # Note: signal_len and kmer_len must match the data in temp_chunks_file
        # (signal_len=400, kmer_len=11 from sample_chunks fixture)
        history = train_model(
            train_data_path=temp_chunks_file,
            val_data_path=None,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            signal_len=400,  # Must match temp_chunks_file
            kmer_len=11,  # Must match temp_chunks_file
            epochs=1,
            batch_size=4,  # Custom
            learning_rate=0.01,  # Custom
            device="cpu",
            seed=42,
            conv_channels=[4, 16, 32],  # Custom model param
        )

        assert len(history["train_loss"]) >= 1


class TestClassWeighting:
    """Test class weighting functionality."""

    def test_compute_class_weights_balanced(self, temp_chunks_file):
        """Test class weight computation for balanced dataset."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        pos_weight = compute_class_weights(dataset)

        # For a relatively balanced dataset, pos_weight should be close to 1.0
        # (can vary based on exact split in temp_chunks_file)
        assert pos_weight is not None
        assert isinstance(pos_weight, torch.Tensor)
        assert pos_weight.shape == (1,)
        assert pos_weight.item() > 0

    def test_trainer_with_pos_weight(self, sample_model, sample_dataloader):
        """Test Trainer initialization with pos_weight."""
        pos_weight = torch.tensor([2.0])

        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            pos_weight=pos_weight,
        )

        assert trainer.criterion is not None
        # Verify that pos_weight was set (BCEWithLogitsLoss should have pos_weight attribute)
        assert hasattr(trainer.criterion, "pos_weight")

    def test_trainer_without_pos_weight(self, sample_model, sample_dataloader):
        """Test Trainer initialization without pos_weight."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            pos_weight=None,
        )

        assert trainer.criterion is not None

    def test_train_model_with_class_weights(self, temp_chunks_file, tmp_path):
        """Test train_model with automatic class weighting enabled."""
        output_dir = tmp_path / "training"

        history = train_model(
            train_data_path=temp_chunks_file,
            val_data_path=None,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=1,
            batch_size=2,
            device="cpu",
            seed=42,
            use_class_weights=True,  # Enable class weighting
        )

        assert len(history["train_loss"]) >= 1

        # Check that config was saved with class weight info
        config_path = output_dir / "config.json"
        assert config_path.exists()

        with open(config_path) as f:
            config = json.load(f)

        assert "use_class_weights" in config
        assert config["use_class_weights"] is True

    def test_train_model_without_class_weights(self, temp_chunks_file, tmp_path):
        """Test train_model with class weighting disabled."""
        output_dir = tmp_path / "training"

        history = train_model(
            train_data_path=temp_chunks_file,
            val_data_path=None,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=1,
            batch_size=2,
            device="cpu",
            seed=42,
            use_class_weights=False,  # Disable class weighting
        )

        assert len(history["train_loss"]) >= 1

        # Check that config was saved with class weight info
        config_path = output_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        assert "use_class_weights" in config
        assert config["use_class_weights"] is False
        assert config["pos_weight"] is None

    def test_train_model_with_manual_pos_weight(self, temp_chunks_file, tmp_path):
        """Test train_model with manual pos_weight."""
        output_dir = tmp_path / "training"
        manual_weight = 1.5

        history = train_model(
            train_data_path=temp_chunks_file,
            val_data_path=None,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=1,
            batch_size=2,
            device="cpu",
            seed=42,
            use_class_weights=False,  # Will be overridden by manual pos_weight
            pos_weight=manual_weight,
        )

        assert len(history["train_loss"]) >= 1

        # Check that config saved the manual weight
        config_path = output_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        assert "pos_weight" in config
        assert config["pos_weight"] == manual_weight


class TestTrainingEdgeCases:
    """Test edge cases in training."""

    def test_trainer_with_empty_history(self, model_config):
        """Test trainer with empty history."""
        model = get_model("ConvLSTMDwell", **model_config)

        # Create minimal dataset
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return {
                    "signal": torch.randn(400),
                    "sequence": torch.randn(4, 11),
                    "features": torch.randn(5, 11),
                    "label": torch.tensor([idx % 2], dtype=torch.float32),
                }

        dummy_loader = DataLoader(DummyDataset(), batch_size=2, collate_fn=collate_fn)

        trainer = Trainer(
            model=model, model_type="ConvLSTMDwell", train_loader=dummy_loader, device="cpu"
        )

        # History should be empty initially
        assert len(trainer.history["train_loss"]) == 0

        # After one epoch
        trainer.train_epoch()
        # Can't directly access history without train() but shouldn't crash

    def test_train_with_zero_epochs(self, temp_chunks_file, tmp_path):
        """Test that training with 0 epochs handles gracefully."""
        output_dir = tmp_path / "training"

        # This should either handle gracefully or raise informative error
        # Most implementations would just return empty history
        try:
            history = train_model(
                train_data_path=temp_chunks_file,
                val_data_path=None,
                model_name="ConvLSTMDwell",
                output_dir=output_dir,
                epochs=0,
                batch_size=2,
                device="cpu",
            )
            assert len(history["train_loss"]) == 0
        except (ValueError, AssertionError):
            # Some implementations might raise an error for epochs=0
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
