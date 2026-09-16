"""
Tests for training module.

Tests Trainer class and train_model function.
"""

import copy
import json
import logging
import os
import random
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

    def test_num_features_correct_when_dataset_degrades_to_list_storage(self, tmp_path):
        """num_features must read the true width even off LeechDataset's list fallback.

        `_TensorFill` degrades to a per-chunk list (`dataset.py`'s own
        documented behaviour) when a chunk's feature shape disagrees with the
        first one -- here, a ragged `feat_width` (11 vs 9) with a constant
        `num_features` (5) triggers exactly that, while leaving 5 as the one
        correct answer. #272 item 4 must not silently fall back to 1 in this
        state (issue: it reads `_needs_features and _features_tensor is not
        None`, which is false here even though the corpus has real features).
        """
        rng = np.random.default_rng(3)
        n = 24
        train_chunks = []
        for i in range(n):
            feat_width = 9 if i == n - 1 else 11  # one chunk ragged -> degrade
            train_chunks.append(
                {
                    "signal": rng.standard_normal(400).astype(np.float32),
                    "dwell": rng.integers(2, 12, 11).astype(np.float32),
                    "features": rng.standard_normal((5, feat_width)).astype(np.float32),
                    "sequence": "ACGTACGTACG",
                    "label": f"c{i % 2}",
                    "label_int": i % 2,
                    "read_id": f"read_{i:05d}",
                    "base_idx": 10 + (i % 5),
                    "source_group": "grp0",
                    "feature_start": -5,
                    "feature_end": 5,
                }
            )

        output_dir = tmp_path / "training"
        # A genuinely ragged feat_width can never complete a real forward
        # pass (Concat needs a uniform width across the whole run) -- that
        # is a separate, pre-existing limitation of the list-fallback path
        # at the *model* level, not what this test is checking. num_features
        # is derived (and config.json written) before the training loop
        # starts, so let the loop's later shape-mismatch RuntimeError
        # through and check config.json regardless.
        try:
            train_model(
                # train_chunks bypasses loading from this path, but
                # train_model still reads .parent off it for the
                # prepare_config.json sidecar check, so it must be a Path
                # even though nothing is read from it.
                train_data_path=tmp_path / "unused.npz",
                val_data_path=None,
                train_chunks=train_chunks,
                model_name="ConvLSTMDwell",
                output_dir=output_dir,
                signal_len=400,
                kmer_len=11,
                epochs=1,
                batch_size=1,
                device="cpu",
                motif="CCAGGC",
                seed=42,
            )
        except RuntimeError as e:
            assert "Sizes of tensors must match" in str(e), f"unexpected failure: {e}"

        with open(output_dir / "config.json") as f:
            config = json.load(f)
        assert config["num_features"] == 5, (
            f"num_features={config['num_features']!r}, expected 5 -- degraded list "
            f"storage must not silently fall back to 1"
        )


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


def _crash_after_n_epochs(n: int):
    """A ``Trainer.train_epoch`` replacement that raises on its n-th call.

    Patched in as ``monkeypatch.setattr(Trainer, "train_epoch", ...)`` to
    simulate an external kill (SLURM walltime, OOM, ``scancel``): the first
    ``n - 1`` epochs complete normally -- each getting its own rolling
    ``model_resume.pt`` -- and the n-th never finishes, so ``model_last.pt``
    is never written and ``model_resume.pt`` still holds epoch ``n - 1``.
    """
    original = Trainer.train_epoch
    calls = {"n": 0}

    def _wrapped(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == n:
            raise RuntimeError("simulated interruption")
        return original(self, *args, **kwargs)

    return _wrapped


class TestInterruptSafeResume:
    """``model_resume.pt``: recovering a run an external kill cut off mid-loop.

    Before this (#330), ``model_last.pt`` was written exactly once, after the
    epoch loop exited -- so a SLURM walltime kill left nothing for ``--resume``
    to find, and every retry started over from a fresh random seed. These
    tests exercise the rolling per-epoch checkpoint that fixes that, and the
    trainer state it has to carry for a resumed run to be a continuation
    rather than a warm start.
    """

    @pytest.fixture
    def sample_dataloader(self, temp_chunks_file):
        """A minimal loader for direct ``Trainer``-level checkpoint round trips.

        Mirrors ``TestTrainer.sample_dataloader`` -- duplicated rather than
        shared because a pytest fixture defined inside a class is only visible
        to that class's own tests, matching how ``TestTrainer`` already scopes
        its copy.
        """
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding="base_onehot",
        )
        return DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    def test_model_resume_removed_on_clean_exit(self, temp_chunks_file, tmp_path):
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

        assert not (output_dir / "model_resume.pt").exists()
        assert (output_dir / "model_last.pt").exists()

    def test_rolling_checkpoint_written_every_epoch(self, temp_chunks_file, tmp_path, monkeypatch):
        output_dir = tmp_path / "training"
        monkeypatch.setattr(Trainer, "train_epoch", _crash_after_n_epochs(3))

        with pytest.raises(RuntimeError, match="simulated interruption"):
            train_model(
                train_data_path=temp_chunks_file,
                val_data_path=temp_chunks_file,
                model_name="ConvLSTMDwell",
                output_dir=output_dir,
                epochs=4,
                batch_size=2,
                device="cpu",
                motif="CCAGGC",
                seed=42,
            )

        # Epochs 1-2 completed (the 3rd train_epoch call is the one that
        # raised), so the rolling checkpoint reflects epoch 2 and the
        # loop-exit-only model_last.pt was never written.
        resume_ckpt = torch.load(
            output_dir / "model_resume.pt", map_location="cpu", weights_only=False
        )
        assert resume_ckpt["epoch"] == 2
        assert not (output_dir / "model_last.pt").exists()

    def test_interrupted_run_resumes_to_identical_history(
        self, temp_chunks_file, tmp_path, monkeypatch
    ):
        """An interrupted-then-resumed run reproduces a straight run's history.

        This is a sequential SGD loop, not a permutation-invariant aggregate:
        forgetting the loader-generator position, the RNG state, or the
        optimizer's momentum changes which batches land in which order and
        what dropout draws, which moves every epoch after the resume point,
        not just the first one.
        """
        common = {
            "train_data_path": temp_chunks_file,
            "val_data_path": temp_chunks_file,
            "model_name": "ConvLSTMDwell",
            "epochs": 4,
            "batch_size": 2,
            "device": "cpu",
            "motif": "CCAGGC",
            "seed": 42,
            "num_workers": 0,
        }

        reference = train_model(output_dir=tmp_path / "reference", **common)

        interrupted_dir = tmp_path / "interrupted"
        original_train_epoch = Trainer.train_epoch
        monkeypatch.setattr(Trainer, "train_epoch", _crash_after_n_epochs(3))
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train_model(output_dir=interrupted_dir, **common)
        # Restore the real train_epoch for the resumed phase below, without
        # touching monkeypatch's own end-of-test teardown.
        monkeypatch.setattr(Trainer, "train_epoch", original_train_epoch)

        resumed = train_model(
            output_dir=interrupted_dir,
            resume_from=interrupted_dir / "model_resume.pt",
            **common,
        )

        assert len(resumed["train_loss"]) == 4
        np.testing.assert_allclose(
            resumed["train_loss"], reference["train_loss"], rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(resumed["val_loss"], reference["val_loss"], rtol=1e-4, atol=1e-4)
        assert not (interrupted_dir / "model_resume.pt").exists()

    def test_resume_continues_early_stopping(self, temp_chunks_file, tmp_path, monkeypatch):
        """``patience_counter`` is not reset on resume.

        val_auc is pinned flat after epoch 1 "improves" against the -inf
        floor: epoch 2 -> patience 1, epoch 3 -> patience 2. Interrupting
        right after epoch 2 and resuming with a fresh Trainer (patience reset
        to 0, the pre-#330 bug) would run two more epochs before stopping;
        carrying patience_counter over stops after exactly one.
        """

        def _flat_validate(self, progress=None, task_id=None):
            return 0.5, 0.6, 0.6, 0.6  # val_loss, val_acc, val_auc, val_f1

        output_dir = tmp_path / "training"
        original_train_epoch = Trainer.train_epoch
        monkeypatch.setattr(Trainer, "validate", _flat_validate)
        monkeypatch.setattr(Trainer, "train_epoch", _crash_after_n_epochs(3))

        with pytest.raises(RuntimeError, match="simulated interruption"):
            train_model(
                train_data_path=temp_chunks_file,
                val_data_path=temp_chunks_file,
                model_name="ConvLSTMDwell",
                output_dir=output_dir,
                epochs=10,
                batch_size=2,
                device="cpu",
                motif="CCAGGC",
                seed=42,
                early_stopping_patience=2,
                checkpoint_metric="val_auc",
            )
        monkeypatch.setattr(Trainer, "train_epoch", original_train_epoch)

        resumed = train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=10,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            seed=42,
            early_stopping_patience=2,
            checkpoint_metric="val_auc",
            resume_from=output_dir / "model_resume.pt",
        )

        # 2 epochs restored from the checkpoint + exactly 1 resumed epoch
        # before early stopping fires -- not 2, which is what a
        # patience_counter reset to 0 on resume would produce.
        assert len(resumed["train_loss"]) == 3
        assert not (output_dir / "model_resume.pt").exists()

    def test_resume_restores_adversarial_head(self, sample_dataloader, model_config, tmp_path):
        output_dir = tmp_path / "training"
        trainer1 = Trainer(
            model=get_model("ConvLSTMDwell", **model_config),
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            output_dir=output_dir,
            adversarial_lambda=0.1,
            adversarial_num_classes=3,
        )
        assert trainer1.adversarial_head is not None
        with torch.no_grad():
            for p in trainer1.adversarial_head.parameters():
                p.add_(1.0)
        expected_state = copy.deepcopy(trainer1.adversarial_head.state_dict())
        trainer1.save_checkpoint("model_resume.pt", epoch=1)

        trainer2 = Trainer(
            model=get_model("ConvLSTMDwell", **model_config),
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            output_dir=output_dir,
            adversarial_lambda=0.1,
            adversarial_num_classes=3,
            resume_checkpoint=output_dir / "model_resume.pt",
        )

        for key, value in expected_state.items():
            assert torch.equal(value, trainer2.adversarial_head.state_dict()[key]), key

    def test_resume_restores_clip_grad_buffer(self, sample_dataloader, model_config, tmp_path):
        output_dir = tmp_path / "training"
        trainer1 = Trainer(
            model=get_model("ConvLSTMDwell", **model_config),
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            output_dir=output_dir,
            quantile_grad_clip=True,
        )
        trainer1.clip_grad_fn.buffer[:] = np.linspace(0.1, 1.0, len(trainer1.clip_grad_fn.buffer))
        trainer1.clip_grad_fn.i = 37
        trainer1.save_checkpoint("model_resume.pt", epoch=1)

        trainer2 = Trainer(
            model=get_model("ConvLSTMDwell", **model_config),
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            output_dir=output_dir,
            quantile_grad_clip=True,
            resume_checkpoint=output_dir / "model_resume.pt",
        )

        np.testing.assert_array_equal(trainer2.clip_grad_fn.buffer, trainer1.clip_grad_fn.buffer)
        assert trainer2.clip_grad_fn.i == 37

    def test_atomic_write_leaves_previous_checkpoint_intact_on_failure(self, tmp_path, monkeypatch):
        path = tmp_path / "model_resume.pt"
        leech.training._atomic_torch_save({"epoch": 1, "marker": "first"}, path)
        assert path.exists()

        def _boom(obj, dest):
            # A real kill mid-torch.save leaves a truncated file at the
            # destination it was writing to -- which, because of the atomic
            # swap, is the sibling .tmp path, never `path` itself.
            Path(dest).write_bytes(b"not a valid checkpoint")
            raise RuntimeError("simulated crash mid-write")

        monkeypatch.setattr(leech.training.torch, "save", _boom)
        with pytest.raises(RuntimeError, match="simulated crash mid-write"):
            leech.training._atomic_torch_save({"epoch": 2, "marker": "second"}, path)
        monkeypatch.undo()

        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert loaded["marker"] == "first"
        leftover_tmp = path.with_name(path.name + ".tmp")
        assert leftover_tmp.exists()
        assert leftover_tmp.read_bytes() == b"not a valid checkpoint"

    def test_resume_refuses_recipe_mismatch(self, temp_chunks_file, tmp_path):
        output_dir = tmp_path / "training"
        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=1,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            seed=42,
            learning_rate=0.001,
        )
        last_path = output_dir / "model_last.pt"
        assert last_path.exists()

        with pytest.raises(RuntimeError, match="recipe changed"):
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
                learning_rate=0.01,  # changed from phase 1
                resume_from=last_path,
            )

    def test_resume_refuses_corpus_mismatch(self, temp_chunks_file, tmp_path):
        output_dir = tmp_path / "training"
        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=1,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            seed=42,
        )
        last_path = output_dir / "model_last.pt"

        # Simulate a Snakemake re-run regenerating the corpus: same path,
        # different mtime.
        stat = temp_chunks_file.stat()
        os.utime(temp_chunks_file, (stat.st_atime, stat.st_mtime + 100))

        with pytest.raises(RuntimeError, match="train corpus changed"):
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
                resume_from=last_path,
            )

    def test_resume_allows_longer_epochs_and_patience_without_raising(
        self, temp_chunks_file, tmp_path
    ):
        output_dir = tmp_path / "training"
        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=1,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            seed=42,
        )
        last_path = output_dir / "model_last.pt"

        # Must not raise despite epochs/early_stopping_patience differing --
        # both are excluded from the recipe guard on purpose.
        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=2,
            early_stopping_patience=3,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            seed=42,
            resume_from=last_path,
        )

    def test_resume_keeps_original_seed(self, temp_chunks_file, tmp_path):
        output_dir = tmp_path / "training"
        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=1,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            seed=123,
        )
        last_path = output_dir / "model_last.pt"

        # No --seed on the resume: train_model generates a fresh random one
        # internally, which must be immediately superseded by the
        # checkpoint's -- both training_seed.txt and config.json record what
        # the run actually ran under, not what it briefly considered.
        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=2,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            resume_from=last_path,
        )

        assert (output_dir / "training_seed.txt").read_text().strip() == "123"
        config = json.loads((output_dir / "config.json").read_text())
        assert config["seed"] == 123

    def test_resume_skips_rng_restore_on_world_size_mismatch(
        self, sample_dataloader, model_config, tmp_path, caplog
    ):
        output_dir = tmp_path / "training"
        trainer1 = Trainer(
            model=get_model("ConvLSTMDwell", **model_config),
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            output_dir=output_dir,
        )
        trainer1.save_checkpoint("model_resume.pt", epoch=1)

        # Hand-edit the checkpoint to look like it was written at world_size=2.
        path = output_dir / "model_resume.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert len(ckpt["rng_states"]) == 1
        ckpt["rng_states"] = ckpt["rng_states"] * 2
        ckpt["loader_generator_states"] = ckpt["loader_generator_states"] * 2
        torch.save(ckpt, path)

        with caplog.at_level(logging.WARNING, logger="leech.training"):
            trainer2 = Trainer(
                model=get_model("ConvLSTMDwell", **model_config),
                model_type="ConvLSTMDwell",
                train_loader=sample_dataloader,
                device="cpu",
                output_dir=output_dir,
                resume_checkpoint=path,
            )

        assert "world_size" in caplog.text
        # The rest of the resume still applies despite the RNG skip.
        assert trainer2.start_epoch == 2

    def test_capture_and_restore_rng_state_roundtrips(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        # Move off the fresh-seed state before capturing, so this isn't
        # trivially true of any two freshly-seeded RNGs.
        random.random()
        np.random.rand()
        torch.rand(1)

        state = leech.training._capture_rng_state()
        expected = (random.random(), float(np.random.rand()), torch.rand(3))

        # Simulate other work happening between save and resume.
        random.random()
        np.random.rand()
        torch.rand(5)

        leech.training._restore_rng_state(state)
        actual = (random.random(), float(np.random.rand()), torch.rand(3))

        assert actual[0] == expected[0]
        assert actual[1] == expected[1]
        assert torch.equal(actual[2], expected[2])


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

    def test_batch_tensors_move_non_blocking(self, grouped_chunks_file, model_config, monkeypatch):
        """Every batch tensor H2D copy must be async (issue #272 item 1).

        Both loaders set ``pin_memory=True``, so a blocking ``.to(device)``
        pays a stream sync it does not need to. Distinguishes a device
        transfer from a dtype-only cast (e.g. ``.to(torch.float64)``) by
        whether the first positional argument is a device-like value; a
        dtype cast on an already-placed tensor is unaffected by this rule.
        """
        # Built before the patch: one-time model/head placement at construction
        # (self.model.to(device)) is not a per-batch transfer and is exempt.
        trainer = _trainer_over(grouped_chunks_file, model_config, n_batches=2)

        offenders = []
        original_to = torch.Tensor.to
        sources = {Path(leech.training.__file__).name, "inference_wrapper.py"}

        def spy(tensor, *args, **kwargs):
            caller = sys._getframe(1).f_code.co_filename
            is_device_like = (args and isinstance(args[0], (str, torch.device))) or (
                "device" in kwargs
            )
            if Path(caller).name in sources and is_device_like:
                if not kwargs.get("non_blocking"):
                    offenders.append((Path(caller).name, sys._getframe(1).f_lineno))
            return original_to(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", spy)

        trainer.train_epoch()
        trainer.validate()

        assert not offenders, f"blocking device .to() calls found at: {sorted(set(offenders))}"

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

    def test_field_group_counts_treats_zero_as_a_real_value(self):
        """0 must not fall into "unknown" the way None/"" do (issue #282):
        junction_indel == 0 is the dominant, legitimate "exact match" value,
        not a missing one."""
        from leech.training import _field_group_counts

        codes, names, counts = _field_group_counts(
            [
                {"junction_indel": 0},
                {"junction_indel": 0},
                {"junction_indel": 1},
                {"junction_indel": None},  # genuinely absent -> "unknown"
            ],
            "junction_indel",
        )

        assert dict(zip(names, counts.tolist(), strict=True)) == {"0": 2, "1": 1, "unknown": 1}
        assert [names[c] for c in codes] == ["0", "0", "1", "unknown"]

    def test_field_group_counts_short_circuits_an_absent_chunktable_column(
        self, grouped_chunks_file
    ):
        """A ChunkTable whose column is entirely absent (a pre-#282 corpus'
        junction_indel, say) must not materialize a ChunkRow per chunk just
        to learn every value is "unknown" -- `skip` simulates the absence
        without needing a special hand-built .npz."""
        from leech.chunking import ChunkTable
        from leech.training import _field_group_counts

        table = ChunkTable.from_npz(grouped_chunks_file, skip={"source_group"})
        assert table.values("source_group") is None  # column genuinely absent

        codes, names, counts = _field_group_counts(table, "source_group")

        n = len(table)
        assert names == ["unknown"]
        assert counts.tolist() == [n]
        assert codes.tolist() == [0] * n

    def test_sample_weight_field_matches_balance_groups_on_source_group(
        self, grouped_chunks_file, tmp_path, monkeypatch
    ):
        """--sample-weight-field source_group is --balance-groups, generalized."""
        captured = self._capture_sampler(monkeypatch)
        self._train(grouped_chunks_file, tmp_path / "a", sample_weight_field="source_group")
        by_field = list(captured["weights"])

        captured = self._capture_sampler(monkeypatch)
        self._train(grouped_chunks_file, tmp_path / "b", balance_groups=True)
        by_flag = list(captured["weights"])

        assert by_field == by_flag

    def test_sampling_strategies_are_mutually_exclusive(self, grouped_chunks_file, tmp_path):
        with pytest.raises(ValueError, match="mutually exclusive"):
            self._train(
                grouped_chunks_file, tmp_path, balance_groups=True, sample_weight_field="label_int"
            )


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
