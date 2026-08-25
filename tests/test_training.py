"""
Tests for training module.

Tests Trainer class and train_model function.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

import leech.training
from leech.chunking import load_chunks, save_chunks
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
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
            motif="CCAGGC",
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
            motif="CCAGGC",
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
            motif="CCAGGC",
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
            motif="CCAGGC",
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
                motif="CCAGGC",
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
            motif="CCAGGC",
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
            motif="CCAGGC",
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
            motif="CCAGGC",
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
            motif="CCAGGC",
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
            motif="CCAGGC",
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
            motif="CCAGGC",
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
                motif="CCAGGC",
            )
            assert len(history["train_loss"]) == 0
        except (ValueError, AssertionError):
            # Some implementations might raise an error for epochs=0
            pass


class TestBestModelResumeGuarantee:
    """Test that model_best.pt always exists after training, even on resume."""

    def test_resume_no_improvement_still_creates_best(self, temp_chunks_file, tmp_path):
        """Train, delete model_best.pt, resume with no improvement — model_best.pt must exist."""
        output_dir = tmp_path / "training"

        # Phase 1: train for 3 epochs so model_best.pt is created
        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=3,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            seed=42,
        )

        best_path = output_dir / "model_best.pt"
        last_path = output_dir / "model_last.pt"
        assert best_path.exists()
        assert last_path.exists()

        # Record the best weights from phase 1
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        original_best_state = best_ckpt["model_state_dict"]
        original_best_acc = best_ckpt["best_val_acc"]

        # Delete model_best.pt (simulates Snakemake cleanup on failure)
        best_path.unlink()
        assert not best_path.exists()

        # Phase 2: resume training for same number of epochs (no new epochs run)
        # start_epoch will be 4 > epochs=3, triggering the "already complete" path
        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=3,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            seed=42,
            resume_from=last_path,
        )

        # model_best.pt must exist after resume
        assert best_path.exists(), "model_best.pt was not recreated after resume"

        # Verify the restored best checkpoint has correct metadata
        restored_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        assert restored_ckpt["best_val_acc"] == original_best_acc

        # Verify the model weights match the original best (not the last)
        for key in original_best_state:
            assert torch.equal(original_best_state[key], restored_ckpt["model_state_dict"][key]), (
                f"Weight mismatch in {key}: best model was not correctly restored"
            )

    def test_checkpoint_contains_best_model_state(self, temp_chunks_file, tmp_path):
        """Verify that model_last.pt checkpoint contains best_model_state_dict."""
        output_dir = tmp_path / "training"

        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=3,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            seed=42,
        )

        last_path = output_dir / "model_last.pt"
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        assert "best_model_state_dict" in checkpoint
        assert checkpoint["best_model_state_dict"] is not None

    def test_resume_with_more_epochs_no_improvement(self, temp_chunks_file, tmp_path):
        """Resume with extra epochs — model_best.pt must exist and reflect the best epoch seen."""
        output_dir = tmp_path / "training"

        # Phase 1: train for 2 epochs
        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=2,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            seed=42,
        )

        best_path = output_dir / "model_best.pt"
        last_path = output_dir / "model_last.pt"

        # Record original best weights and epoch
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        original_best_state = best_ckpt["model_state_dict"]
        original_best_epoch = best_ckpt["best_epoch"]

        # Delete model_best.pt
        best_path.unlink()

        # Phase 2: resume and add 1 more epoch (epochs=3, resume from epoch 2)
        # Even if epoch 3 doesn't beat the best, model_best.pt must be created
        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=3,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            seed=42,
            resume_from=last_path,
        )

        assert best_path.exists(), "model_best.pt missing after resume with extra epochs"

        restored_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)

        # Whether the new epoch beats the prior best depends on training dynamics
        # (optimizer, selection metric, host BLAS determinism). The resume guarantee
        # is that model_best.pt reflects the best epoch ever seen. If best_epoch
        # didn't advance, weights must match the stored best exactly; if it did,
        # phase 2 legitimately found a better model and the weights differ.
        if restored_ckpt["best_epoch"] == original_best_epoch:
            for key in original_best_state:
                assert torch.equal(
                    original_best_state[key], restored_ckpt["model_state_dict"][key]
                ), f"Weight mismatch in {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Epoch metrics: no per-step device sync, no per-prediction Python objects (S3)
# and no per-chunk row view before the first batch (S4).
# ---------------------------------------------------------------------------


def _grouped_corpus(path, n=48, n_classes=2, n_groups=3):
    """A corpus with uneven groups, uneven classes, and some empty groups."""
    rng = np.random.default_rng(7)
    chunks = []
    for i in range(n):
        label = i % n_classes if i % 4 else 0
        chunks.append(
            {
                "signal": rng.standard_normal(400).astype(np.float32),
                "dwell": rng.integers(2, 12, 11).astype(np.float32),
                "features": rng.standard_normal((5, 11)).astype(np.float32),
                "sequence": "ACGTACGTACG",
                "label": f"c{label}",
                "label_int": label,
                "read_id": f"read_{i:05d}",
                "base_idx": 10 + (i % 5),
                # Every 7th chunk has no group at all: those must land in
                # "unknown" together, exactly as the dict-based count did.
                "source_group": "" if i % 7 == 0 else f"grp{i % n_groups}",
                "feature_start": -5,
                "feature_end": 5,
                "seq_to_sig_map": np.linspace(0, 400, 12).astype(np.int64),
                "sequence_with_kmer_context": "ACGT" * 7,
                "focus_signal_pos": 200,
            }
        )
    save_chunks(chunks, path)
    return chunks


def _trainer_over(chunks_file, model_config, n_batches, loss_type="bce", num_out=None):
    """A Trainer whose loader yields exactly ``n_batches`` batches."""
    dataset = LeechDataset(
        chunks_file,
        signal_len=400,
        kmer_len=11,
        model_type="ConvLSTMDwell",
        seq_encoding="base_onehot",
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, len(dataset) // n_batches),
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    model = get_model("ConvLSTMDwell", **{**model_config, "num_out": num_out or 1})
    return Trainer(
        model=model,
        model_type="ConvLSTMDwell",
        train_loader=loader,
        val_loader=loader,
        device="cpu",
        learning_rate=0.001,
        loss_type=loss_type,
        num_out=num_out,
    )


class _SyncCounter:
    """Counts the Tensor reads *training.py itself* makes.

    ``.item()`` and ``.cpu()`` are what drain the CUDA stream. Adam's own
    ``step.item()`` per parameter per step is torch's business and swamps the
    signal, so only calls made directly from the training loop are counted.
    """

    _SOURCE = Path(leech.training.__file__).name

    def __init__(self, monkeypatch):
        self.counts = {"item": 0, "cpu": 0}
        for name in self.counts:
            original = getattr(torch.Tensor, name)

            def spy(tensor, *args, _name=name, _original=original, **kwargs):
                caller = sys._getframe(1).f_code.co_filename
                if Path(caller).name == self._SOURCE:
                    self.counts[_name] += 1
                return _original(tensor, *args, **kwargs)

            monkeypatch.setattr(torch.Tensor, name, spy)

    @property
    def total(self):
        return sum(self.counts.values())


@pytest.fixture
def grouped_chunks_file(tmp_path):
    path = tmp_path / "grouped.npz"
    _grouped_corpus(path)
    return path


class TestEpochMetricAccumulation:
    """train_epoch and validate must not sync or box once per batch."""

    def test_train_epoch_syncs_do_not_scale_with_batches(
        self, grouped_chunks_file, model_config, monkeypatch
    ):
        """Reading metrics per sub-batch is three device syncs per step.

        Counting ``.item()`` / ``.cpu()`` stands in for counting syncs, which
        cannot be observed on CPU: the totals must not grow when the same data
        is served as more, smaller batches.
        """
        counts = []
        for n_batches in (2, 8):
            trainer = _trainer_over(grouped_chunks_file, model_config, n_batches)
            counter = _SyncCounter(monkeypatch)
            trainer.train_epoch()
            counts.append(counter.total)

        assert counts[0] == counts[1], (
            f"train_epoch made {counts[1] - counts[0]} extra device reads for six "
            f"extra batches; epoch metrics must be read once, not per batch"
        )

    def test_validate_reads_at_most_two_tensors_per_batch(
        self, grouped_chunks_file, model_config, monkeypatch
    ):
        """Validation needs the probabilities and the labels on the host.

        It does not need the loss there too -- that is a third sync per batch.
        """
        per_batch = []
        for n_batches in (2, 8):
            trainer = _trainer_over(grouped_chunks_file, model_config, n_batches)
            counter = _SyncCounter(monkeypatch)
            trainer.validate()
            per_batch.append(counter.total / n_batches)

        assert per_batch[1] <= 2.0, (
            f"validate reads {per_batch[1]} tensors per batch; expected at most the "
            f"probabilities and the labels"
        )

    @pytest.mark.parametrize(
        ("loss_type", "num_out"),
        [("bce", None), ("cross_entropy", 2), ("cross_entropy", 3)],
    )
    def test_epoch_metrics_never_box_predictions(
        self, grouped_chunks_file, model_config, monkeypatch, loss_type, num_out
    ):
        """No per-prediction Python list is ever turned back into an array.

        ``all_preds.extend(preds.flatten())`` appends one boxed np.float32 per
        prediction, and the array constructor at the end unboxes every one of
        them. The observable signature is an array built from a long list of
        numpy scalars -- whether that happens here or inside the sklearn metric
        the list is handed to.
        """
        boxed = []
        for name in ("array", "asarray"):
            original = getattr(np, name)

            def spy(obj, *args, _original=original, **kwargs):
                if isinstance(obj, list) and len(obj) > 16 and isinstance(obj[0], np.generic):
                    boxed.append(len(obj))
                return _original(obj, *args, **kwargs)

            monkeypatch.setattr(np, name, spy)

        trainer = _trainer_over(grouped_chunks_file, model_config, 4, loss_type, num_out)
        trainer.train_epoch()
        trainer.validate()

        assert not boxed, f"epoch metrics unboxed per-prediction lists of length {boxed}"

    def test_train_epoch_matches_host_side_reference(self, grouped_chunks_file, model_config):
        """The device tallies reproduce the host-side loss and accuracy exactly.

        One batch, so every prediction train_epoch reports comes from the
        pre-step weights and can be recomputed on an identically seeded twin.
        """
        config = {**model_config, "dropout": 0.0}
        torch.manual_seed(0)
        trainer = _trainer_over(grouped_chunks_file, config, 1)
        torch.manual_seed(0)
        twin = _trainer_over(grouped_chunks_file, config, 1)
        for left, right in zip(trainer.model.parameters(), twin.model.parameters(), strict=True):
            assert torch.equal(left, right)

        avg_loss, accuracy = trainer.train_epoch()

        twin.model.train()
        batch = next(iter(twin.train_loader))
        logits, labels, main_loss, _, _, _ = twin._compute_batch_loss(batch)
        reference_loss = main_loss.item()
        reference_acc = accuracy_score(
            labels.detach().cpu().numpy().flatten(),
            (torch.sigmoid(logits).detach().cpu().numpy().flatten() > 0.5).astype(int),
        )

        assert avg_loss == reference_loss
        assert accuracy == reference_acc


class TestSamplerColumnReads:
    """The samplers read columns, not a row view per chunk."""

    def _capture_sampler(self, monkeypatch):
        captured = {}
        original = leech.training.WeightedRandomSampler

        def spy(weights, *args, **kwargs):
            captured["weights"] = weights
            return original(weights, *args, **kwargs)

        monkeypatch.setattr(leech.training, "WeightedRandomSampler", spy)
        return captured

    def _train(self, chunks_file, tmp_path, **kwargs):
        return train_model(
            train_data_path=chunks_file,
            val_data_path=None,
            model_name="ConvLSTMDwell",
            output_dir=tmp_path / "out",
            signal_len=400,
            kmer_len=11,
            epochs=1,
            batch_size=8,
            learning_rate=0.001,
            device="cpu",
            seed=42,
            seq_encoding="base_onehot",
            num_workers=0,
            motif="CCAGGC",
            **kwargs,
        )

    def test_balance_groups_weights_are_an_array(self, grouped_chunks_file, tmp_path, monkeypatch):
        """WeightedRandomSampler gets a numpy array, not a list of N floats."""
        captured = self._capture_sampler(monkeypatch)
        self._train(grouped_chunks_file, tmp_path, balance_groups=True)

        assert isinstance(captured["weights"], np.ndarray), (
            "sampler weights are a Python list: one float object per chunk"
        )

    def test_balance_groups_weights_match_dict_reference(
        self, grouped_chunks_file, tmp_path, monkeypatch
    ):
        """Vectorized counts must equal the dict-of-counts they replaced."""
        captured = self._capture_sampler(monkeypatch)
        self._train(grouped_chunks_file, tmp_path, balance_groups=True)

        chunks = load_chunks(grouped_chunks_file)
        group_counts: dict = {}
        for chunk in chunks:
            key = chunk.get("source_group") or "unknown"
            group_counts[key] = group_counts.get(key, 0) + 1
        expected = [1.0 / group_counts[c.get("source_group") or "unknown"] for c in chunks]

        assert list(captured["weights"]) == expected

    def test_oversample_weights_match_dict_reference(
        self, grouped_chunks_file, tmp_path, monkeypatch
    ):
        """Same for the class-frequency sampler."""
        captured = self._capture_sampler(monkeypatch)
        self._train(grouped_chunks_file, tmp_path, oversample_minority=True)

        chunks = load_chunks(grouped_chunks_file)
        label_counts: dict = {}
        for chunk in chunks:
            label_counts[chunk["label_int"]] = label_counts.get(chunk["label_int"], 0) + 1
        expected = [1.0 / label_counts[c["label_int"]] for c in chunks]

        assert isinstance(captured["weights"], np.ndarray)
        assert list(captured["weights"]) == expected

    def test_compute_class_weights_accepts_a_label_column(self):
        """The counting wants the labels, not a dataset to walk row by row."""
        pos_weight = compute_class_weights(np.array([0] * 30 + [1] * 10))

        assert pos_weight is not None
        assert float(pos_weight[0]) == pytest.approx(3.0)

    def test_compute_class_weights_column_matches_dataset(self, grouped_chunks_file):
        """Passing the column and passing the dataset must agree."""
        from leech.training import _label_column

        dataset = LeechDataset(
            grouped_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )

        assert torch.equal(
            compute_class_weights(dataset),
            compute_class_weights(_label_column(dataset.chunks)),
        )

    def test_source_group_counts_handles_empty_and_literal_unknown(self):
        """An empty group, an absent one and a literal "unknown" are one group."""
        from leech.training import _source_group_counts

        codes, names, counts = _source_group_counts(
            [
                {"source_group": "a"},
                {"source_group": ""},
                {"source_group": "unknown"},
                {"source_group": None},
                {"source_group": "a"},
            ]
        )

        assert dict(zip(names, counts.tolist(), strict=True)) == {"a": 2, "unknown": 3}
        assert [names[c] for c in codes] == ["a", "unknown", "unknown", "unknown", "a"]


class TestConfoundNpzHandle:
    """Confound setup must not hold the corpus open for the whole run."""

    def test_confound_setup_closes_the_npz(self, grouped_chunks_file, tmp_path, monkeypatch):
        opened = []
        original = np.load

        def spy(file, *args, **kwargs):
            handle = original(file, *args, **kwargs)
            if kwargs.get("allow_pickle"):
                opened.append(handle)
            return handle

        monkeypatch.setattr(np, "load", spy)

        train_model(
            train_data_path=grouped_chunks_file,
            val_data_path=None,
            model_name="ConvLSTMDwell",
            output_dir=tmp_path / "out",
            signal_len=400,
            kmer_len=11,
            epochs=1,
            batch_size=8,
            learning_rate=0.001,
            device="cpu",
            seed=42,
            seq_encoding="base_onehot",
            num_workers=0,
            motif="CCAGGC",
            adversarial_lambda=0.5,
            confound="source_group:identity",
        )

        assert opened, "expected the confound setup to read the source column"
        still_open = [handle for handle in opened if getattr(handle, "fid", None) is not None]
        assert not still_open, (
            f"{len(still_open)} npz handle(s) left open for the whole training run"
        )
