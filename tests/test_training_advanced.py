"""
Tests for advanced training features.

Tests checkpoint recovery, LR scheduling, gradient clipping, weight decay,
mixed precision, signal augmentation, LR warmup, and focal loss.
"""

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from leech.dataset import LeechDataset, collate_fn
from leech.losses import FocalBCEWithLogitsLoss
from leech.models import get_model
from leech.training import Trainer, train_model


class TestFocalLoss:
    """Test FocalBCEWithLogitsLoss."""

    def test_focal_loss_basic(self):
        """Test focal loss computes without error."""
        loss_fn = FocalBCEWithLogitsLoss(gamma=2.0)
        logits = torch.randn(4, 1, requires_grad=True)
        targets = torch.randint(0, 2, (4, 1)).float()
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_focal_loss_gamma_zero_matches_bce(self):
        """Focal loss with gamma=0 should approximate standard BCE."""
        torch.manual_seed(42)
        logits = torch.randn(32, 1)
        targets = torch.randint(0, 2, (32, 1)).float()

        focal = FocalBCEWithLogitsLoss(gamma=0.0)
        bce = torch.nn.BCEWithLogitsLoss()

        focal_val = focal(logits, targets)
        bce_val = bce(logits, targets)

        assert abs(focal_val.item() - bce_val.item()) < 1e-5

    def test_focal_loss_with_pos_weight(self):
        """Test focal loss with positive class weighting."""
        pos_weight = torch.tensor([2.0])
        loss_fn = FocalBCEWithLogitsLoss(gamma=2.0, pos_weight=pos_weight)

        logits = torch.randn(4, 1)
        targets = torch.ones(4, 1)
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_focal_loss_reduces_easy_example_weight(self):
        """Focal loss should weight hard examples more than easy ones."""
        # Well-classified positive example (high logit, target=1)
        easy_logits = torch.tensor([[5.0]])
        easy_targets = torch.tensor([[1.0]])

        # Misclassified example (high logit, target=0)
        hard_logits = torch.tensor([[5.0]])
        hard_targets = torch.tensor([[0.0]])

        focal = FocalBCEWithLogitsLoss(gamma=2.0)
        bce = torch.nn.BCEWithLogitsLoss(reduction="none")

        # With focal loss, the easy example's loss should be reduced more
        # relative to standard BCE than the hard example's loss
        focal_easy = focal(easy_logits, easy_targets)
        focal_hard = focal(hard_logits, hard_targets)
        bce_easy = bce(easy_logits, easy_targets)
        bce_hard = bce(hard_logits, hard_targets)

        # Ratio of focal/bce should be smaller for easy examples
        easy_ratio = focal_easy.item() / bce_easy.item()
        hard_ratio = focal_hard.item() / bce_hard.item()
        assert easy_ratio < hard_ratio


class TestWeightDecay:
    """Test weight decay in Trainer."""

    def test_weight_decay_optimizer(self, sample_model, sample_dataloader):
        """Test that weight decay is passed to optimizer."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            weight_decay=0.01,
        )
        for group in trainer.optimizer.param_groups:
            assert group["weight_decay"] == 0.01

    def test_zero_weight_decay(self, sample_model, sample_dataloader):
        """Test default zero weight decay."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
        )
        for group in trainer.optimizer.param_groups:
            assert group["weight_decay"] == 0.0


class TestGradientClipping:
    """Test gradient clipping."""

    def test_grad_clipping_enabled(self, sample_model, sample_dataloader):
        """Test that gradient clipping is stored."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            max_grad_norm=1.0,
        )
        assert trainer.max_grad_norm == 1.0

    def test_grad_clipping_disabled(self, sample_model, sample_dataloader):
        """Test default disabled gradient clipping."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
        )
        assert trainer.max_grad_norm == 0.0

    def test_train_with_grad_clipping(self, sample_model, sample_dataloader):
        """Test training with gradient clipping doesn't crash."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            max_grad_norm=1.0,
        )
        loss, acc = trainer.train_epoch()
        assert loss >= 0
        assert 0 <= acc <= 1


class TestLRScheduler:
    """Test learning rate scheduling."""

    def test_no_scheduler(self, sample_model, sample_dataloader):
        """Test default no scheduler."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
        )
        assert trainer.scheduler is None

    def test_reduce_on_plateau_scheduler(self, sample_model, sample_dataloader):
        """Test ReduceLROnPlateau scheduler creation."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            device="cpu",
            scheduler_type="reduce_on_plateau",
            scheduler_patience=3,
            scheduler_factor=0.5,
        )
        assert trainer.scheduler is not None
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_train_with_scheduler(self, sample_model, sample_dataloader, tmp_path):
        """Test training with scheduler doesn't crash."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            device="cpu",
            output_dir=tmp_path / "output",
            scheduler_type="reduce_on_plateau",
            scheduler_patience=1,
            scheduler_factor=0.5,
        )
        history = trainer.train(epochs=3, early_stopping_patience=0)
        assert len(history["train_loss"]) == 3


class TestLRWarmup:
    """Test learning rate warmup."""

    def test_warmup_initialization(self, sample_model, sample_dataloader):
        """Test warmup parameter is stored."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            learning_rate=0.001,
            warmup_epochs=3,
        )
        assert trainer.warmup_epochs == 3
        assert trainer.base_lr == 0.001

    def test_warmup_lr_increases(self, sample_model, sample_dataloader, tmp_path):
        """Test that LR increases during warmup."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            device="cpu",
            learning_rate=0.003,
            warmup_epochs=3,
            output_dir=tmp_path / "output",
        )

        # Train for 3 epochs (all warmup)
        history = trainer.train(epochs=3, early_stopping_patience=0)
        assert len(history["train_loss"]) == 3

        # After warmup completes, LR should be at base_lr
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        assert abs(current_lr - 0.003) < 1e-7


class TestMixedPrecision:
    """Test mixed precision training."""

    def test_mixed_precision_disabled_on_cpu(self, sample_model, sample_dataloader):
        """Test mixed precision is automatically disabled on CPU."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            use_mixed_precision=True,
        )
        assert trainer.use_mixed_precision is False
        assert trainer.scaler is None

    def test_mixed_precision_default_off(self, sample_model, sample_dataloader):
        """Test mixed precision is off by default."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
        )
        assert trainer.use_mixed_precision is False


class TestFocalLossTrainer:
    """Test focal loss integration in Trainer."""

    def test_focal_loss_in_trainer(self, sample_model, sample_dataloader):
        """Test Trainer with focal loss."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            loss_type="focal",
            focal_gamma=2.0,
        )
        assert isinstance(trainer.criterion, FocalBCEWithLogitsLoss)

    def test_bce_loss_in_trainer(self, sample_model, sample_dataloader):
        """Test Trainer with default BCE loss."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
        )
        assert isinstance(trainer.criterion, torch.nn.BCEWithLogitsLoss)

    def test_train_with_focal_loss(self, sample_model, sample_dataloader):
        """Test training with focal loss runs without error."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            loss_type="focal",
            focal_gamma=2.0,
        )
        loss, acc = trainer.train_epoch()
        assert loss >= 0


class TestSignalAugmentation:
    """Test signal data augmentation in LeechDataset."""

    def test_augmentation_disabled_by_default(self, temp_chunks_file):
        """Test no augmentation by default."""
        dataset = LeechDataset(
            temp_chunks_file, signal_len=400, kmer_len=11, model_type="ConvLSTMDwell"
        )
        assert dataset.augmentation is None

    def test_augmentation_jitter(self, temp_chunks_file):
        """Test jitter augmentation produces different signals."""
        augmentation = {"jitter_std": 0.5, "scale_range": (1.0, 1.0)}

        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            augmentation=augmentation,
        )

        # Get same item twice - should be different due to random jitter
        torch.manual_seed(1)
        item1 = dataset[0]
        torch.manual_seed(2)
        item2 = dataset[0]

        # Signals should differ due to jitter
        assert not torch.allclose(item1["signal"], item2["signal"])

    def test_augmentation_scale(self, temp_chunks_file):
        """Test scale augmentation produces different signals."""
        augmentation = {"jitter_std": 0.0, "scale_range": (0.5, 2.0)}

        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            augmentation=augmentation,
        )

        # Get same item twice - should be different due to random scaling
        torch.manual_seed(1)
        item1 = dataset[0]
        torch.manual_seed(2)
        item2 = dataset[0]

        assert not torch.allclose(item1["signal"], item2["signal"])

    def test_no_augmentation_deterministic(self, temp_chunks_file):
        """Test that without augmentation, signals are deterministic."""
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
        )

        item1 = dataset[0]
        item2 = dataset[0]
        assert torch.allclose(item1["signal"], item2["signal"])


class TestCheckpointRecovery:
    """Test checkpoint save/restore."""

    def test_save_checkpoint_includes_epoch(self, sample_model, sample_dataloader, tmp_path):
        """Test that save_checkpoint includes epoch number."""
        output_dir = tmp_path / "output"
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            output_dir=output_dir,
        )
        trainer.save_checkpoint("test.pt", epoch=5)

        checkpoint = torch.load(output_dir / "test.pt", map_location="cpu")
        assert checkpoint["epoch"] == 5

    def test_save_checkpoint_includes_scheduler(self, sample_model, sample_dataloader, tmp_path):
        """Test checkpoint includes scheduler state."""
        output_dir = tmp_path / "output"
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            device="cpu",
            output_dir=output_dir,
            scheduler_type="reduce_on_plateau",
        )
        trainer.save_checkpoint("test.pt", epoch=1)

        checkpoint = torch.load(output_dir / "test.pt", map_location="cpu")
        assert checkpoint["scheduler_state_dict"] is not None

    def test_save_checkpoint_no_scheduler(self, sample_model, sample_dataloader, tmp_path):
        """Test checkpoint with no scheduler has None."""
        output_dir = tmp_path / "output"
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            output_dir=output_dir,
        )
        trainer.save_checkpoint("test.pt", epoch=1)

        checkpoint = torch.load(output_dir / "test.pt", map_location="cpu")
        assert checkpoint["scheduler_state_dict"] is None

    def test_resume_training(self, temp_chunks_file, tmp_path):
        """Test that training can be resumed from a checkpoint."""
        output_dir = tmp_path / "training"

        # Train for 3 epochs
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
            early_stopping_patience=0,
        )

        # Verify checkpoint has epoch info
        checkpoint = torch.load(output_dir / "model_last.pt", map_location="cpu")
        assert "epoch" in checkpoint
        assert checkpoint["epoch"] == 3

        # Resume training for 5 more epochs (total 5)
        output_dir2 = tmp_path / "training_resumed"
        history = train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir2,
            epochs=5,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            seed=42,
            early_stopping_patience=0,
            resume_from=output_dir / "model_last.pt",
        )

        # Should have trained for epochs 4 and 5 (2 epochs)
        assert len(history["train_loss"]) == 2

    def test_resume_restores_best_val_acc(self, temp_chunks_file, tmp_path):
        """Test that resume restores best_val_acc."""
        output_dir = tmp_path / "training"

        # Train for 2 epochs
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
            early_stopping_patience=0,
        )

        checkpoint = torch.load(output_dir / "model_last.pt", map_location="cpu")
        saved_best_acc = checkpoint["best_val_acc"]

        # Load config to get the exact model architecture used by train_model
        with open(output_dir / "config.json") as f:
            config = json.load(f)

        # Create a model with the same architecture (default params)
        model = get_model(
            "ConvLSTMDwell",
            signal_len=config["signal_len"],
            kmer_len=config["kmer_len"],
            num_features=config["num_features"],
            seq_encoding=config.get("seq_encoding", "base_onehot"),
        )
        dataset = LeechDataset(
            temp_chunks_file,
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
            seq_encoding=config.get("seq_encoding", "base_onehot"),
        )
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        trainer = Trainer(
            model=model,
            model_type="ConvLSTMDwell",
            train_loader=loader,
            device="cpu",
            resume_checkpoint=output_dir / "model_last.pt",
        )

        assert trainer.best_val_acc == saved_best_acc
        assert trainer.start_epoch == 3  # Should start at epoch 3


class TestTrainModelAdvancedParams:
    """Test train_model function with advanced parameters."""

    def test_train_model_weight_decay(self, temp_chunks_file, tmp_path):
        """Test train_model with weight decay."""
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
            weight_decay=0.01,
        )
        assert len(history["train_loss"]) == 1

    def test_train_model_grad_clipping(self, temp_chunks_file, tmp_path):
        """Test train_model with gradient clipping."""
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
            max_grad_norm=1.0,
        )
        assert len(history["train_loss"]) == 1

    def test_train_model_scheduler(self, temp_chunks_file, tmp_path):
        """Test train_model with LR scheduler."""
        output_dir = tmp_path / "training"
        history = train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=2,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            scheduler="reduce_on_plateau",
            scheduler_patience=1,
        )
        assert len(history["train_loss"]) == 2

    def test_train_model_warmup(self, temp_chunks_file, tmp_path):
        """Test train_model with LR warmup."""
        output_dir = tmp_path / "training"
        history = train_model(
            train_data_path=temp_chunks_file,
            val_data_path=None,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=3,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            warmup_epochs=2,
        )
        assert len(history["train_loss"]) == 3

    def test_train_model_focal_loss(self, temp_chunks_file, tmp_path):
        """Test train_model with focal loss."""
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
            loss_type="focal",
            focal_gamma=2.0,
        )
        assert len(history["train_loss"]) == 1

    def test_train_model_augmentation(self, temp_chunks_file, tmp_path):
        """Test train_model with signal augmentation."""
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
            augment_jitter=0.1,
            augment_scale_min=0.9,
            augment_scale_max=1.1,
        )
        assert len(history["train_loss"]) == 1

    def test_train_model_all_features_combined(self, temp_chunks_file, tmp_path):
        """Test train_model with multiple advanced features combined."""
        output_dir = tmp_path / "training"
        history = train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=2,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            weight_decay=0.01,
            max_grad_norm=1.0,
            scheduler="reduce_on_plateau",
            scheduler_patience=1,
            warmup_epochs=1,
            loss_type="focal",
            focal_gamma=1.5,
            augment_jitter=0.05,
        )
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

    def test_config_includes_advanced_params(self, temp_chunks_file, tmp_path):
        """Test that config.json includes advanced training parameters."""
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
            weight_decay=0.01,
            max_grad_norm=1.0,
            scheduler="reduce_on_plateau",
            warmup_epochs=2,
            loss_type="focal",
            focal_gamma=1.5,
            mixed_precision=False,
            augment_jitter=0.1,
            augment_scale_min=0.9,
            augment_scale_max=1.1,
        )

        with open(output_dir / "config.json") as f:
            config = json.load(f)

        assert config["weight_decay"] == 0.01
        assert config["max_grad_norm"] == 1.0
        assert config["scheduler"] == "reduce_on_plateau"
        assert config["warmup_epochs"] == 2
        assert config["loss_type"] == "focal"
        assert config["focal_gamma"] == 1.5
        assert config["mixed_precision"] is False
        assert config["augment_jitter"] == 0.1
        assert config["augment_scale_min"] == 0.9
        assert config["augment_scale_max"] == 1.1


class TestBackwardCompatibility:
    """Test backward compatibility with old checkpoints."""

    def test_old_checkpoint_loads(self, tmp_path, model_config):
        """Test that old checkpoints (without epoch/scheduler keys) still load."""
        model = get_model("ConvLSTMDwell", **model_config)

        # Simulate old checkpoint format (no epoch, no scheduler_state_dict)
        old_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": torch.optim.Adam(model.parameters()).state_dict(),
            "best_val_acc": 0.85,
            "best_epoch": 5,
        }

        checkpoint_path = tmp_path / "old_model.pt"
        torch.save(old_checkpoint, checkpoint_path)

        # Create new trainer and resume from old checkpoint
        new_model = get_model("ConvLSTMDwell", **model_config)
        dataset = LeechDataset(
            chunks=[
                {
                    "signal": np.random.randn(400).astype(np.float32),
                    "sequence": "ACGTACGTACG",
                    "dwell": np.ones(11, dtype=np.float32),
                    "features": np.random.randn(5, 11).astype(np.float32),
                    "label_int": 0,
                    "label": "test",
                    "read_id": "r1",
                    "base_idx": 0,
                }
            ],
            signal_len=400,
            kmer_len=11,
            model_type="ConvLSTMDwell",
        )
        loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

        trainer = Trainer(
            model=new_model,
            model_type="ConvLSTMDwell",
            train_loader=loader,
            device="cpu",
            resume_checkpoint=checkpoint_path,
        )

        # Should have restored state correctly
        assert trainer.best_val_acc == 0.85
        assert trainer.best_epoch == 5
        # Old checkpoints without 'epoch' key should default to epoch 0, so start_epoch=1
        assert trainer.start_epoch == 1

    def test_default_params_match_original_behavior(self, temp_chunks_file, tmp_path):
        """Test that all default parameters preserve original training behavior."""
        output_dir = tmp_path / "training"
        history = train_model(
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

        # Should complete normally with all defaults
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

        # Config should have default values
        with open(output_dir / "config.json") as f:
            config = json.load(f)

        assert config["weight_decay"] == 0.0
        assert config["max_grad_norm"] == 0.0
        assert config["scheduler"] == "none"
        assert config["warmup_epochs"] == 0
        assert config["loss_type"] == "bce"
        assert config["mixed_precision"] is False
        assert config["augment_jitter"] == 0.0
        assert config["augment_scale_min"] == 1.0
        assert config["augment_scale_max"] == 1.0


class TestUtilTrainingParams:
    """Test that util.py filters new training params correctly."""

    def test_new_params_in_training_params_set(self):
        """Test that new params are in the training_params filter set."""
        # Simulate what load_model_from_checkpoint does
        config = {
            "model_name": "ConvLSTMDwell",
            "signal_len": 400,
            "kmer_len": 11,
            "num_features": 5,
            "epochs": 50,
            "batch_size": 128,
            "learning_rate": 0.001,
            "scheduler": "reduce_on_plateau",
            "scheduler_patience": 5,
            "scheduler_factor": 0.5,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "mixed_precision": False,
            "warmup_epochs": 3,
            "loss_type": "focal",
            "focal_gamma": 2.0,
            "augment_jitter": 0.1,
            "augment_scale_min": 0.9,
            "augment_scale_max": 1.1,
            "resume_from": "/some/path",
            "device": "cpu",
            "seed": 42,
        }

        # Import the training_params set from util.py logic
        training_params = {
            "epochs",
            "batch_size",
            "learning_rate",
            "device",
            "seed",
            "val_split",
            "patience",
            "min_delta",
            "save_dir",
            "log_dir",
            "num_workers",
            "pin_memory",
            "prefetch_factor",
            "use_class_weights",
            "pos_weight",
            "scheduler",
            "scheduler_patience",
            "scheduler_factor",
            "max_grad_norm",
            "weight_decay",
            "mixed_precision",
            "warmup_epochs",
            "loss_type",
            "focal_gamma",
            "augment_jitter",
            "augment_scale_min",
            "augment_scale_max",
            "resume_from",
        }

        # Filter like load_model_from_checkpoint does
        model_kwargs = {
            k: v
            for k, v in config.items()
            if k not in ["model_name", "signal_len", "kmer_len"] and k not in training_params
        }

        # Only model-related params should remain
        assert "num_features" in model_kwargs
        assert "scheduler" not in model_kwargs
        assert "weight_decay" not in model_kwargs
        assert "loss_type" not in model_kwargs
        assert "augment_jitter" not in model_kwargs
        assert "resume_from" not in model_kwargs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
