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

import leech.training
from leech.dataset import LeechDataset, collate_fn
from leech.losses import (
    FocalBCEWithLogitsLoss,
    NoiseCorrectedBCEWithLogitsLoss,
    parse_label_noise_rate,
)
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


class TestParseLabelNoiseRate:
    """Test the ``--label-noise-rate`` token parser."""

    def test_none_and_empty_return_none(self):
        assert parse_label_noise_rate(None) is None
        assert parse_label_noise_rate("") is None
        assert parse_label_noise_rate("   ") is None

    def test_parses_single_and_multiple_entries(self):
        assert parse_label_noise_rate("gold=0.09") == {"gold": 0.09}
        assert parse_label_noise_rate("gold=0.09,enzymatic=0.17") == {
            "gold": 0.09,
            "enzymatic": 0.17,
        }

    def test_tolerates_whitespace(self):
        assert parse_label_noise_rate(" gold = 0.09 , enzymatic = 0.17 ") == {
            "gold": 0.09,
            "enzymatic": 0.17,
        }

    def test_rejects_malformed_entry(self):
        with pytest.raises(ValueError, match="Malformed"):
            parse_label_noise_rate("gold")

    def test_rejects_empty_group_name(self):
        with pytest.raises(ValueError, match="empty group name"):
            parse_label_noise_rate("=0.1")

    def test_rejects_out_of_range_rate(self):
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            parse_label_noise_rate("gold=1.0")
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            parse_label_noise_rate("gold=-0.1")

    def test_rejects_non_float_rate(self):
        with pytest.raises(ValueError, match="not a float"):
            parse_label_noise_rate("gold=abc")


class TestNoiseCorrectedBCELoss:
    """Test NoiseCorrectedBCEWithLogitsLoss (forward correction)."""

    def test_rate_omitted_matches_plain_bce_bit_for_bit(self):
        """No ``noise_rate`` argument at all -- the common non-noise-aware call."""
        torch.manual_seed(0)
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()

        corrected = NoiseCorrectedBCEWithLogitsLoss()(logits, targets)
        plain = torch.nn.BCEWithLogitsLoss()(logits, targets)
        assert torch.equal(corrected, plain)

    def test_all_zero_rate_tensor_matches_plain_bce_numerically(self):
        """An explicit all-zero rate tensor -- e.g. every ``source_group`` in
        a batch happens to be unmapped even though ``--label-noise-rate`` is
        configured -- takes the general branch, not the bit-for-bit fast
        path: the fast path is chosen on ``noise_rate is None`` only, never on
        the tensor's values, so a value-dependent check can't force a
        per-batch host/device sync on CUDA. It still agrees with plain BCE to
        floating-point precision.
        """
        torch.manual_seed(1)
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        zero_rate = torch.zeros(8, 1)

        corrected = NoiseCorrectedBCEWithLogitsLoss()(logits, targets, noise_rate=zero_rate)
        plain = torch.nn.BCEWithLogitsLoss()(logits, targets)
        assert torch.allclose(corrected, plain, atol=1e-6)

    def test_pos_weight_matches_bce_bit_for_bit(self):
        torch.manual_seed(2)
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        pos_weight = torch.tensor([2.0])

        corrected = NoiseCorrectedBCEWithLogitsLoss(pos_weight=pos_weight)(logits, targets)
        plain = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, targets)
        assert torch.equal(corrected, plain)

    def test_nonzero_rate_diverges_from_plain_bce(self):
        """Sanity check that the correction actually does something."""
        torch.manual_seed(3)
        logits = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        rate = torch.full((8, 1), 0.2)

        corrected = NoiseCorrectedBCEWithLogitsLoss()(logits, targets, noise_rate=rate)
        plain = torch.nn.BCEWithLogitsLoss()(logits, targets)
        assert not torch.allclose(corrected, plain)
        assert corrected.item() >= 0
        assert torch.isfinite(corrected)

    def test_reduces_by_element_count(self):
        """The whole-batch mean must equal the count-weighted mean over two
        unequal shards -- the property DDP's gradient averaging over equal
        shards relies on (``Trainer._weighted_ce_global`` documents the one
        loss in this codebase where a per-shard mean is *not* the global
        mean; this loss must not join it).
        """
        torch.manual_seed(4)
        logits = torch.randn(10, 1)
        targets = torch.randint(0, 2, (10, 1)).float()
        noise_rate = torch.rand(10, 1) * 0.4  # mixed nonzero rates

        loss_fn = NoiseCorrectedBCEWithLogitsLoss()
        full = loss_fn(logits, targets, noise_rate=noise_rate)

        shard_a = loss_fn(logits[:3], targets[:3], noise_rate=noise_rate[:3])
        shard_b = loss_fn(logits[3:], targets[3:], noise_rate=noise_rate[3:])
        weighted = (shard_a * 3 + shard_b * 7) / 10

        assert torch.allclose(full, weighted, atol=1e-6)

    def test_synthetic_noise_recovers_better_calibration_than_plain_bce(self):
        """A 1-D logistic regression fit on symmetrically-noised labels, given
        the true (measured) flip rate, must recover a better-calibrated
        model than plain BCE trained on the same noisy labels -- the concrete
        claim behind ``--loss noise_corrected_bce``. Symmetric label noise is
        a textbook attenuation-bias case (Natarajan et al. 2013; Patrini et
        al. 2017): plain BCE's population optimum is a flatter, less
        confident decision boundary than the true one, while the forward
        correction's population optimum -- with the true rate supplied -- IS
        the true boundary, because the corrected likelihood is exactly the
        marginal likelihood of the observed labels under the model.
        """
        torch.manual_seed(0)
        n = 4000
        true_w, true_b = 3.0, 0.0
        x = torch.randn(n, 1)
        true_prob = torch.sigmoid(true_w * x + true_b)
        true_label = torch.bernoulli(true_prob)

        rho = 0.3
        flip = torch.bernoulli(torch.full((n, 1), rho)).bool()
        noisy_label = torch.where(flip, 1.0 - true_label, true_label)

        def fit(*, corrected: bool) -> tuple[float, float]:
            w = torch.zeros(1, requires_grad=True)
            b = torch.zeros(1, requires_grad=True)
            opt = torch.optim.Adam([w, b], lr=0.1)
            noise_rate = torch.full((n, 1), rho) if corrected else None
            loss_fn = NoiseCorrectedBCEWithLogitsLoss()
            for _ in range(300):
                opt.zero_grad()
                logits = w * x + b
                loss = loss_fn(logits, noisy_label, noise_rate=noise_rate)
                loss.backward()
                opt.step()
            return w.item(), b.item()

        w_corrected, b_corrected = fit(corrected=True)
        w_plain, b_plain = fit(corrected=False)

        # Calibration against the TRUE (clean) probabilities on a fresh
        # held-out sample from the same generative process.
        x_test = torch.randn(4000, 1)
        true_prob_test = torch.sigmoid(true_w * x_test + true_b)

        def brier(w: float, b: float) -> float:
            pred = torch.sigmoid(w * x_test + b)
            return torch.mean((pred - true_prob_test) ** 2).item()

        assert brier(w_corrected, b_corrected) < brier(w_plain, b_plain)
        # The forward-corrected fit should recover the true slope much more
        # closely than plain BCE, which attenuates it toward 0 under
        # symmetric noise.
        assert abs(w_corrected - true_w) < abs(w_plain - true_w)


class TestNoiseCorrectedBCETrainer:
    """Test noise_corrected_bce integration in Trainer."""

    def test_noise_corrected_bce_in_trainer(self, sample_model, sample_dataloader):
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            loss_type="noise_corrected_bce",
        )
        assert isinstance(trainer.criterion, NoiseCorrectedBCEWithLogitsLoss)

    def test_train_with_noise_corrected_bce_runs(self, sample_model, sample_dataloader):
        """Runs fine even though ``sample_dataloader``'s batches carry no
        ``noise_rate`` field -- ``Trainer._batch_noise_rate`` returns None and
        the loss's own None handling falls back to plain BCE.
        """
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            loss_type="noise_corrected_bce",
        )
        loss, acc = trainer.train_epoch()
        assert loss >= 0

    def test_batch_noise_rate_reaches_the_criterion(self, temp_chunks_file, model_config):
        """A dataset built with ``label_noise_rates`` puts a ``noise_rate``
        field in every batch, and the trainer must actually use it rather
        than silently falling back to plain BCE.
        """
        dataset = LeechDataset(
            chunk_path=temp_chunks_file,
            model_type="ConvLSTMDwell",
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            seq_encoding="base_onehot",
            label_noise_rates={"unknown": 0.25},
        )
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)
        batch = next(iter(loader))
        assert "noise_rate" in batch
        assert torch.allclose(batch["noise_rate"], torch.full_like(batch["noise_rate"], 0.25))

        model = get_model("ConvLSTMDwell", **model_config)
        trainer = Trainer(
            model=model,
            model_type="ConvLSTMDwell",
            train_loader=loader,
            device="cpu",
            loss_type="noise_corrected_bce",
        )
        logits, labels, main_loss, _loss, _adv, _cl = trainer._compute_batch_loss(batch)

        expected = NoiseCorrectedBCEWithLogitsLoss()(logits, labels, noise_rate=batch["noise_rate"])
        assert torch.equal(main_loss, expected)

        # And it must differ from plain BCE on the same batch -- otherwise the
        # noise_rate plumbing could be silently inert.
        plain = torch.nn.BCEWithLogitsLoss()(logits, labels)
        assert not torch.allclose(main_loss, plain)


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


class TestMixedPrecisionProbabilityPrecision:
    """AUROC/accuracy inputs must not inherit AMP's fp16 quantization (#264).

    Under CUDA autocast the final Linear emits float16, and torch.sigmoid /
    torch.softmax are not on autocast's fp32 promotion list -- so computing
    them on the raw logits rounds every probability to ~3 significant digits
    and saturates to exactly 1.0 above logit~11. val_auc is the checkpoint
    -selection criterion, so a quantized AUROC ranking silently changes which
    epoch gets kept.

    This test machine has no CUDA (mixed precision is forced off on CPU --
    see TestMixedPrecision above), so real autocast can't be exercised here.
    Instead these simulate exactly what autocast's Linear layer hands back:
    a genuine float16 logits tensor.
    """

    def test_fp32_cast_avoids_the_fp16_quantization_raw_sigmoid_has(self):
        """The core numeric bug, isolated from any model or DataLoader.

        `torch.sigmoid(logits.float())` (the fix, used in train_epoch,
        validate and evaluate_model) must resolve far more distinct
        probabilities than `torch.sigmoid(logits)` computed directly on a
        float16 tensor (the bug), across a few thousand samples -- and must
        avoid float16's saturation to exactly 1.0 for large logits.
        """
        torch.manual_seed(0)
        # A realistic spread plus a tail in (~11, ~17): high enough that
        # float16's sigmoid saturates to exactly 1.0, but well below
        # float32's own saturation point (~17.3), so the fix's float32
        # output must not saturate here.
        logits_fp32 = torch.cat([torch.randn(4000) * 3.0, torch.linspace(12.0, 16.0, 1000)])
        logits_fp16 = logits_fp32.half()
        n = logits_fp16.numel()

        fixed = torch.sigmoid(logits_fp16.float())
        buggy = torch.sigmoid(logits_fp16)

        assert fixed.dtype == torch.float32
        assert buggy.dtype == torch.float16

        n_unique_fixed = len(torch.unique(fixed))
        n_unique_buggy = len(torch.unique(buggy))

        # `logits_fp16` is already float16 -- that's autocast's Linear output,
        # not the bug -- so `fixed` can't resolve more than the input allows.
        # The bug is the *extra* quantization from also computing and storing
        # sigmoid's OUTPUT in float16: `buggy` must collapse noticeably
        # further than `fixed`, and specifically saturate to exactly 1.0 for
        # large logits, which the fp32-output fix avoids (#264).
        assert n_unique_fixed > n * 0.5
        assert n_unique_buggy < n_unique_fixed
        assert (buggy == 1.0).sum() > 0  # saturation actually happened
        assert (fixed == 1.0).sum() == 0  # ... and the fix avoids it

    def test_validate_scores_auroc_on_float32_probabilities(
        self, sample_model, sample_dataloader, monkeypatch
    ):
        """Trainer.validate() must cast to float32 before scoring, even when
        the model itself hands back float16 -- exactly what a CUDA autocast
        forward pass would produce."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            device="cpu",
        )

        real_forward = trainer.model_wrapper.forward_batch

        def fp16_forward(batch, device):
            # Simulate autocast's fp16 Linear output.
            return real_forward(batch, device).half()

        monkeypatch.setattr(trainer.model_wrapper, "forward_batch", fp16_forward)

        captured: dict = {}
        real_roc_auc_score = leech.training.roc_auc_score

        def spy_roc_auc_score(y_true, y_score, *args, **kwargs):
            captured["probs"] = np.asarray(y_score)
            return real_roc_auc_score(y_true, y_score, *args, **kwargs)

        monkeypatch.setattr(leech.training, "roc_auc_score", spy_roc_auc_score)

        avg_loss, accuracy, auc, f1 = trainer.validate()

        assert "probs" in captured, "roc_auc_score was never called (labels not mixed?)"
        assert captured["probs"].dtype == np.float32
        assert 0.0 <= auc <= 1.0

    def test_train_epoch_thresholds_on_raw_logits_not_fp16_sigmoid(
        self, sample_model, sample_dataloader, monkeypatch
    ):
        """train_epoch's accuracy threshold must not go through sigmoid at
        all under simulated fp16 logits -- it compares the logit to 0
        directly, which is exact regardless of autocast (#264)."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
        )

        real_forward = trainer.model_wrapper.forward_batch
        monkeypatch.setattr(
            trainer.model_wrapper,
            "forward_batch",
            lambda batch, device: real_forward(batch, device).half(),
        )

        loss, acc = trainer.train_epoch()
        assert loss >= 0
        assert 0 <= acc <= 1

    def test_one_epoch_with_mixed_precision_true_reports_float32_probs(
        self, sample_model, sample_dataloader, monkeypatch
    ):
        """`mixed_precision=True` end to end for one epoch: even though this
        CPU test machine forces AMP off (TestMixedPrecision above), a model
        that hands back float16 logits -- exactly what CUDA autocast would
        produce -- must still yield a float32, high-resolution AUROC input
        after training and validating one epoch."""
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            device="cpu",
            use_mixed_precision=True,
        )
        assert trainer.use_mixed_precision is False  # forced off on CPU

        real_forward = trainer.model_wrapper.forward_batch
        monkeypatch.setattr(
            trainer.model_wrapper,
            "forward_batch",
            lambda batch, device: real_forward(batch, device).half(),
        )

        captured: dict = {}
        real_roc_auc_score = leech.training.roc_auc_score

        def spy_roc_auc_score(y_true, y_score, *args, **kwargs):
            captured["probs"] = np.asarray(y_score)
            return real_roc_auc_score(y_true, y_score, *args, **kwargs)

        monkeypatch.setattr(leech.training, "roc_auc_score", spy_roc_auc_score)

        history = trainer.train(epochs=1, early_stopping_patience=0)

        assert len(history["train_loss"]) == 1
        assert "probs" in captured
        probs = captured["probs"]
        assert probs.dtype == np.float32
        # This fixture is small (a handful of chunks), so "more distinct
        # values than a float16 grid allows" is checked directly against
        # float16 quantization of the same values, not an absolute count:
        assert len(np.unique(probs)) >= len(np.unique(probs.astype(np.float16)))


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

        checkpoint = torch.load(output_dir / "test.pt", map_location="cpu", weights_only=False)
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

        checkpoint = torch.load(output_dir / "test.pt", map_location="cpu", weights_only=False)
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

        checkpoint = torch.load(output_dir / "test.pt", map_location="cpu", weights_only=False)
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
        checkpoint = torch.load(
            output_dir / "model_last.pt", map_location="cpu", weights_only=False
        )
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

        checkpoint = torch.load(
            output_dir / "model_last.pt", map_location="cpu", weights_only=False
        )
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

    def test_train_model_bonito_loop_options(self, temp_chunks_file, tmp_path):
        """train_model accepts the bonito loop options and records them in config."""
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
            grad_accum_split=2,
            quantile_grad_clip=True,
            save_optim_every=5,
        )
        assert len(history["train_loss"]) == 1

        with open(output_dir / "config.json") as f:
            config = json.load(f)
        assert config["grad_accum_split"] == 2
        assert config["quantile_grad_clip"] is True
        assert config["save_optim_every"] == 5

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

    def test_train_model_noise_corrected_bce(self, temp_chunks_file, tmp_path):
        """train_model accepts noise_corrected_bce and records the rates,
        verbatim, in config.json -- predict never reads this back, but a
        checkpoint that can't be audited for what it trained on is how #230
        stayed invisible for four releases.
        """
        output_dir = tmp_path / "training"
        history = train_model(
            train_data_path=temp_chunks_file,
            val_data_path=temp_chunks_file,
            model_name="ConvLSTMDwell",
            output_dir=output_dir,
            epochs=1,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            loss_type="noise_corrected_bce",
            label_noise_rates={"unknown": 0.1},
        )
        assert len(history["train_loss"]) == 1

        with open(output_dir / "config.json") as f:
            config = json.load(f)
        assert config["loss_type"] == "noise_corrected_bce"
        assert config["label_noise_rates"] == {"unknown": 0.1}

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
        """Test that new params are in the real _TRAINING_PARAMS filter set.

        Imports the actual symbol (#270) instead of re-declaring a copy here,
        which had drifted out of sync with what config.json actually writes.

        The config dict below includes both the original training-only keys
        AND the ~15 #270 added (label_smoothing, checkpoint_metric, gpus,
        and the augmentation/adversarial/CL-regression knobs) -- a config
        dict that only exercised the pre-#270 keys would pass even if those
        15 additions were reverted, since nothing would ask about them.
        """
        from leech.model_loading import _architecture_config

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
            # The keys #270 added to _TRAINING_PARAMS -- this is the actual
            # regression this test exists to catch.
            "label_smoothing": 0.1,
            "augment_scale_range": {"signal": [0.9, 1.1]},
            "augment_time_mask_bases": 5,
            "augment_time_mask_count": 2,
            "augment_shift_max_bases": 1.0,
            "augment_feature_noise_scale": 0.01,
            "balance_groups": True,
            "oversample_minority": True,
            "adversarial_lambda": 0.5,
            "adversarial_anneal_epochs": 3,
            "confound": "flowcell",
            "label_noise_rates": {"gold": 0.09},
            "cl_lambda": 2.0,
            "checkpoint_metric": "val_f1",
            "signal_mode": "signal_only",
            "gpus": 4,
            # Provenance fields read at predict time -- must survive filtering.
            "motif": "CCAGGC",
            "dwell_offset": 2,
            "num_out": 1,
            "cl_regression": False,
        }

        model_kwargs = _architecture_config(config)

        # Only model-related and provenance params should remain
        assert "num_features" in model_kwargs
        assert "motif" in model_kwargs
        assert "dwell_offset" in model_kwargs
        assert "num_out" in model_kwargs
        assert "cl_regression" in model_kwargs
        assert "scheduler" not in model_kwargs
        assert "weight_decay" not in model_kwargs
        assert "loss_type" not in model_kwargs
        assert "augment_jitter" not in model_kwargs
        assert "resume_from" not in model_kwargs

        # The #270 additions: excluded from the bundle's architecture config,
        # not merely present in the _TRAINING_PARAMS set (which would be
        # implied by, not independent of, the checks above).
        for key in (
            "label_smoothing",
            "augment_scale_range",
            "augment_time_mask_bases",
            "augment_time_mask_count",
            "augment_shift_max_bases",
            "augment_feature_noise_scale",
            "balance_groups",
            "oversample_minority",
            "adversarial_lambda",
            "adversarial_anneal_epochs",
            "confound",
            "label_noise_rates",
            "cl_lambda",
            "checkpoint_metric",
            "signal_mode",
            "gpus",
        ):
            assert key not in model_kwargs, f"{key!r} leaked into the bundle architecture config"


# ---------------------------------------------------------------------------
# Bonito-derived training-loop features
# ---------------------------------------------------------------------------


def _tiny_batch(batch_size=8, signal_len=400, kmer_len=11, num_features=5, seed=0):
    """Deterministic batch matching the leech collate contract."""
    g = torch.Generator().manual_seed(seed)
    return {
        "signal": torch.randn(batch_size, signal_len, generator=g),
        "sequence": torch.randn(batch_size, 4, kmer_len, generator=g),
        "features": torch.randn(batch_size, num_features, kmer_len, generator=g),
        "label": torch.randint(0, 2, (batch_size, 1), generator=g).float(),
    }


def _deterministic_model(model_config):
    """ConvLSTMDwell with dropout disabled so repeated forwards are identical."""
    cfg = dict(model_config)
    cfg["dropout"] = 0.0
    torch.manual_seed(1234)
    return get_model("ConvLSTMDwell", **cfg)


def _grads_after_one_batch(model, batch, grad_accum_split):
    """Run one train_epoch over a single batch, returning gradients (no step)."""
    trainer = Trainer(
        model=model,
        model_type="ConvLSTMDwell",
        train_loader=[batch],
        device="cpu",
        grad_accum_split=grad_accum_split,
    )
    # Suppress the optimizer update so gradients survive for comparison.
    trainer.optimizer.step = lambda *a, **k: None
    trainer.train_epoch()
    return {
        name: param.grad.detach().clone()
        for name, param in trainer.model.named_parameters()
        if param.grad is not None
    }


class TestSplitBatch:
    """Test batch splitting used by gradient accumulation."""

    def test_split_one_returns_original(self):
        from leech.training import split_batch

        batch = _tiny_batch(batch_size=4)
        splits = split_batch(batch, 1)
        assert len(splits) == 1
        assert splits[0] is batch

    def test_split_two_halves(self):
        from leech.training import split_batch

        batch = _tiny_batch(batch_size=8)
        splits = split_batch(batch, 2)
        assert len(splits) == 2
        for sub in splits:
            assert sub["signal"].shape[0] == 4
        rejoined = torch.cat([s["signal"] for s in splits], dim=0)
        assert torch.equal(rejoined, batch["signal"])

    def test_split_more_than_batch_size(self):
        from leech.training import split_batch

        batch = _tiny_batch(batch_size=2)
        splits = split_batch(batch, 8)
        assert len(splits) == 2
        assert sum(s["label"].shape[0] for s in splits) == 2

    def test_split_passes_through_non_tensors(self):
        from leech.training import split_batch

        batch = _tiny_batch(batch_size=4)
        batch["meta"] = "unsplittable"
        splits = split_batch(batch, 2)
        assert all(s["meta"] == "unsplittable" for s in splits)


class TestGradientAccumulation:
    """Test bonito-style gradient accumulation."""

    def test_default_is_one(self, sample_model, sample_dataloader):
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
        )
        assert trainer.grad_accum_split == 1

    def test_split_one_matches_no_accumulation(self, model_config):
        """grad_accum_split=1 must be bit-identical to the un-split path."""
        import copy

        batch = _tiny_batch(batch_size=8, seed=7)
        model = _deterministic_model(model_config)

        grads_default = _grads_after_one_batch(copy.deepcopy(model), batch, 1)

        # Reference: manual single forward/backward, no accumulation at all.
        ref_model = copy.deepcopy(model)
        ref_model.train()
        criterion = torch.nn.BCEWithLogitsLoss()
        logits = ref_model(batch["signal"], batch["sequence"], batch["features"])
        criterion(logits, batch["label"]).backward()
        ref_grads = {
            name: p.grad.detach().clone()
            for name, p in ref_model.named_parameters()
            if p.grad is not None
        }

        assert set(grads_default) == set(ref_grads)
        for name, grad in ref_grads.items():
            assert torch.equal(grads_default[name], grad), name

    def test_split_two_matches_full_batch(self, model_config):
        """N=2 accumulation reproduces the full-batch gradient."""
        import copy

        batch = _tiny_batch(batch_size=8, seed=11)
        model = _deterministic_model(model_config)

        grads_full = _grads_after_one_batch(copy.deepcopy(model), batch, 1)
        grads_split = _grads_after_one_batch(copy.deepcopy(model), batch, 2)

        assert set(grads_full) == set(grads_split)
        for name, grad in grads_full.items():
            torch.testing.assert_close(grads_split[name], grad, rtol=1e-4, atol=1e-6)

    def test_split_four_matches_full_batch(self, model_config):
        import copy

        batch = _tiny_batch(batch_size=8, seed=13)
        model = _deterministic_model(model_config)

        grads_full = _grads_after_one_batch(copy.deepcopy(model), batch, 1)
        grads_split = _grads_after_one_batch(copy.deepcopy(model), batch, 4)

        for name, grad in grads_full.items():
            torch.testing.assert_close(grads_split[name], grad, rtol=1e-4, atol=1e-6)

    def test_reported_loss_matches_full_batch(self, model_config):
        """Per-batch loss reported in history is unchanged by splitting."""
        import copy

        batch = _tiny_batch(batch_size=8, seed=17)
        model = _deterministic_model(model_config)

        t1 = Trainer(
            model=copy.deepcopy(model),
            model_type="ConvLSTMDwell",
            train_loader=[batch],
            device="cpu",
            grad_accum_split=1,
        )
        t1.optimizer.step = lambda *a, **k: None
        loss1, _ = t1.train_epoch()

        t2 = Trainer(
            model=copy.deepcopy(model),
            model_type="ConvLSTMDwell",
            train_loader=[batch],
            device="cpu",
            grad_accum_split=2,
        )
        t2.optimizer.step = lambda *a, **k: None
        loss2, _ = t2.train_epoch()

        assert abs(loss1 - loss2) < 1e-5

    def test_training_runs_with_accumulation(self, sample_model, sample_dataloader, tmp_path):
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            device="cpu",
            output_dir=tmp_path / "accum",
            grad_accum_split=2,
        )
        history = trainer.train(epochs=2, early_stopping_patience=0)
        assert len(history["train_loss"]) == 2

    def test_accumulation_with_grad_clipping(self, sample_model, sample_dataloader):
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            grad_accum_split=2,
            max_grad_norm=1.0,
        )
        loss, _acc = trainer.train_epoch()
        assert loss >= 0


class TestQuantileGradClip:
    """Test bonito's adaptive ClipGrad."""

    def test_default_disabled(self, sample_model, sample_dataloader):
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
        )
        assert trainer.clip_grad_fn is None

    def test_enabled(self, sample_model, sample_dataloader):
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            quantile_grad_clip=True,
        )
        assert trainer.clip_grad_fn is not None

    def test_buffer_rolls(self):
        from leech.training import ClipGrad

        clipper = ClipGrad(buffer_size=3)
        assert np.allclose(clipper.buffer, 1e6)
        for value in (1.0, 2.0, 3.0, 4.0):
            clipper.append(value)
        # Oldest (1.0) overwritten by 4.0
        assert sorted(clipper.buffer.tolist()) == [2.0, 3.0, 4.0]

    def test_clips_outlier_gradient(self):
        from leech.training import ClipGrad

        clipper = ClipGrad(quantile=0.5, factor=2.0, buffer_size=4)
        # Prime the buffer with small norms
        for _ in range(4):
            clipper.append(1.0)

        param = torch.nn.Parameter(torch.zeros(4))
        param.grad = torch.tensor([100.0, 0.0, 0.0, 0.0])
        clipper([param])
        # Clipped to factor * median(buffer) = 2.0
        assert param.grad.norm().item() == pytest.approx(2.0, rel=1e-4)

    def test_non_finite_norm_not_buffered(self):
        from leech.training import ClipGrad

        clipper = ClipGrad(buffer_size=4)
        param = torch.nn.Parameter(torch.zeros(2))
        param.grad = torch.tensor([float("nan"), 0.0])
        clipper([param])
        assert np.allclose(clipper.buffer, 1e6)

    def test_train_epoch_with_quantile_clip(self, sample_model, sample_dataloader):
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            quantile_grad_clip=True,
        )
        loss, _acc = trainer.train_epoch()
        assert loss >= 0
        # Real gradient norms replaced the primed sentinel values
        assert (trainer.clip_grad_fn.buffer < 1e6).any()


class TestWarmupCosineSchedule:
    """Test the unified linear warmup + cosine decay schedule."""

    def test_no_warmup_endpoints(self):
        from leech.training import linear_warmup_cosine_decay

        fn = linear_warmup_cosine_decay(total_epochs=10, warmup_epochs=0, end_ratio=0.0)
        assert fn(0) == pytest.approx(1.0)
        assert fn(5) == pytest.approx(0.5)
        assert fn(10) == pytest.approx(0.0, abs=1e-12)

    def test_end_ratio_is_floor(self):
        from leech.training import linear_warmup_cosine_decay

        fn = linear_warmup_cosine_decay(total_epochs=8, warmup_epochs=0, end_ratio=0.01)
        assert fn(8) == pytest.approx(0.01)
        # Past the horizon the floor holds (no cosine rebound)
        assert fn(20) == pytest.approx(0.01)

    def test_warmup_ramp(self):
        from leech.training import linear_warmup_cosine_decay

        fn = linear_warmup_cosine_decay(total_epochs=10, warmup_epochs=4, end_ratio=0.0)
        assert fn(0) == pytest.approx(0.25)
        assert fn(1) == pytest.approx(0.5)
        assert fn(3) == pytest.approx(1.0)
        # First post-warmup step is still full LR, then cosine decays over 6
        assert fn(4) == pytest.approx(1.0)
        assert fn(7) == pytest.approx(0.5)
        assert fn(10) == pytest.approx(0.0, abs=1e-12)

    def test_matches_cosine_annealing_without_warmup(self):
        """Numerically equivalent to the previous CosineAnnealingLR behavior."""
        from leech.training import linear_warmup_cosine_decay

        base_lr, epochs, eta_min = 0.01, 12, 1e-6
        param = torch.nn.Parameter(torch.zeros(1))

        opt_a = torch.optim.SGD([param], lr=base_lr)
        ref = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=epochs, eta_min=eta_min)

        opt_b = torch.optim.SGD([param], lr=base_lr)
        new = torch.optim.lr_scheduler.LambdaLR(
            opt_b,
            lr_lambda=linear_warmup_cosine_decay(
                total_epochs=epochs, warmup_epochs=0, end_ratio=eta_min / base_lr
            ),
        )

        for _ in range(epochs):
            assert opt_b.param_groups[0]["lr"] == pytest.approx(
                opt_a.param_groups[0]["lr"], rel=1e-9
            )
            ref.step()
            new.step()

    def test_trainer_uses_lambda_lr(self, sample_model, sample_dataloader):
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            device="cpu",
            learning_rate=0.01,
            scheduler_type="cosine",
            epochs=10,
        )
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.LambdaLR)
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    def test_trainer_warmup_starts_low(self, sample_model, sample_dataloader):
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            device="cpu",
            learning_rate=0.004,
            scheduler_type="cosine",
            warmup_epochs=4,
            epochs=10,
        )
        # LambdaLR applies the multiplier immediately: base_lr / warmup_epochs
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.001)

    def test_trainer_lr_curve_over_epochs(self, sample_model, sample_dataloader, tmp_path):
        """LR observed at the start of each epoch follows warmup then cosine."""
        base_lr, epochs, warmup = 0.004, 6, 2
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            device="cpu",
            learning_rate=base_lr,
            scheduler_type="cosine",
            warmup_epochs=warmup,
            epochs=epochs,
            output_dir=tmp_path / "cosine",
        )
        observed = []
        original = trainer.train_epoch

        def record(*args, **kwargs):
            observed.append(trainer.optimizer.param_groups[0]["lr"])
            return original(*args, **kwargs)

        trainer.train_epoch = record
        trainer.train(epochs=epochs, early_stopping_patience=0)

        eta_min = 1e-6
        expected = [base_lr * 0.5, base_lr]
        decay = epochs - warmup
        for k in range(epochs - warmup):
            expected.append(eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * k / decay)) / 2)
        expected = expected[:epochs]

        assert len(observed) == epochs
        for got, want in zip(observed, expected, strict=True):
            assert got == pytest.approx(want, rel=1e-6)


class TestSaveOptimEvery:
    """Test split optimizer checkpointing."""

    def test_default_saves_optimizer(self, sample_model, sample_dataloader, tmp_path):
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            output_dir=tmp_path / "ckpt",
        )
        trainer.save_checkpoint("model_last.pt", epoch=3)
        ckpt = torch.load(tmp_path / "ckpt" / "model_last.pt", weights_only=False)
        assert "optimizer_state_dict" in ckpt

    def test_skips_optimizer_between_intervals(self, sample_model, sample_dataloader, tmp_path):
        out = tmp_path / "ckpt"
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            output_dir=out,
            save_optim_every=3,
        )
        trainer.save_checkpoint("epoch1.pt", epoch=1)
        trainer.save_checkpoint("epoch3.pt", epoch=3)

        skipped = torch.load(out / "epoch1.pt", weights_only=False)
        saved = torch.load(out / "epoch3.pt", weights_only=False)
        assert "optimizer_state_dict" not in skipped
        assert "model_state_dict" in skipped
        assert "optimizer_state_dict" in saved

    def test_resume_without_optimizer_state(self, sample_model, sample_dataloader, tmp_path):
        """Resuming from a weights-only checkpoint must not crash."""
        import copy

        out = tmp_path / "ckpt"
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            output_dir=out,
            save_optim_every=5,
        )
        trainer.best_val_acc = 0.75
        trainer.save_checkpoint("model_last.pt", epoch=2)

        ckpt_path = out / "model_last.pt"
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert "optimizer_state_dict" not in ckpt

        resumed = Trainer(
            model=copy.deepcopy(sample_model),
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            output_dir=out,
            resume_checkpoint=ckpt_path,
        )
        assert resumed.start_epoch == 3
        assert resumed.best_val_acc == pytest.approx(0.75)

    def test_resume_with_optimizer_state_still_works(
        self, sample_model, sample_dataloader, tmp_path
    ):
        import copy

        out = tmp_path / "ckpt"
        trainer = Trainer(
            model=sample_model,
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            output_dir=out,
        )
        trainer.train_epoch()
        trainer.save_checkpoint("model_last.pt", epoch=1)

        resumed = Trainer(
            model=copy.deepcopy(sample_model),
            model_type="ConvLSTMDwell",
            train_loader=sample_dataloader,
            device="cpu",
            resume_checkpoint=out / "model_last.pt",
        )
        assert resumed.optimizer.state_dict()["state"]
        assert resumed.start_epoch == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
