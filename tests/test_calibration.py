"""Tests for multiclass calibration methods (temperature, matrix, dirichlet)."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from leech.calibration import (
    apply_calibration,
    learn_dirichlet_scaling,
    learn_matrix_scaling,
    learn_temperature,
    load_calibration,
)


def _make_overconfident_logits(
    n: int = 500, k: int = 5, scale: float = 3.0, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate K-class logits with controllable miscalibration.

    Predictions are correct but overconfident by `scale` factor.
    """
    torch.manual_seed(seed)
    labels = torch.randint(0, k, (n,))
    # Start with moderate logits that have correct argmax
    base_logits = torch.randn(n, k) * 0.5
    # Boost the correct class
    base_logits[torch.arange(n), labels] += 2.0
    # Scale to make overconfident
    logits = base_logits * scale
    return logits, labels


def _make_well_calibrated_logits(
    n: int = 500, k: int = 5, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate well-calibrated K-class logits (no overconfidence)."""
    torch.manual_seed(seed)
    labels = torch.randint(0, k, (n,))
    logits = torch.randn(n, k) * 0.5
    logits[torch.arange(n), labels] += 2.0
    return logits, labels


def _compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Compute top-1 confidence ECE."""
    probs = torch.softmax(logits, dim=-1)
    confidences, preds = probs.max(dim=-1)
    accuracies = (preds == labels).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean().item()
        bin_acc = accuracies[mask].mean().item()
        ece += mask.sum().item() / n * abs(bin_conf - bin_acc)
    return ece


# ============================================================================
# Temperature scaling tests
# ============================================================================


def test_learn_temperature_recovers_scale():
    """Overconfident logits → T > 1 (softens distribution)."""
    # Create logits that are overconfident: moderate accuracy but extreme confidence
    torch.manual_seed(42)
    n, k = 2000, 10
    labels = torch.randint(0, k, (n,))
    # Noisy logits where ~60% are correct but scaled to be very confident
    logits = torch.randn(n, k)
    logits[torch.arange(n), labels] += 1.0  # slight boost to correct class
    logits = logits * 5.0  # scale up → overconfident
    t = learn_temperature(logits, labels)
    assert t > 1.0, f"Expected T > 1.0 for overconfident logits, got {t:.4f}"


# ============================================================================
# Matrix scaling tests
# ============================================================================


def test_learn_matrix_scaling_identity():
    """Well-calibrated logits → W should be near identity, b near zero."""
    logits, labels = _make_well_calibrated_logits(n=1000, k=5)
    W, b = learn_matrix_scaling(logits, labels, reg_lambda=0.1)
    # W should be close to identity
    identity = torch.eye(5)
    assert (W - identity).abs().max() < 0.5, "W deviates too much from identity"
    assert b.abs().max() < 0.5, f"b should be near zero, got {b}"


def test_learn_matrix_scaling_reduces_ece():
    """Overconfident logits → matrix scaling should reduce ECE."""
    logits, labels = _make_overconfident_logits(n=1000, k=5, scale=3.0)
    ece_before = _compute_ece(logits, labels)

    W, b = learn_matrix_scaling(logits, labels)
    cal_logits = apply_calibration(logits, {"method": "matrix", "W": W.tolist(), "b": b.tolist()})
    ece_after = _compute_ece(cal_logits, labels)

    assert ece_after < ece_before, f"ECE should decrease: {ece_before:.4f} -> {ece_after:.4f}"


# ============================================================================
# Dirichlet scaling tests
# ============================================================================


def test_learn_dirichlet_scaling_identity():
    """Well-calibrated logits → alpha should be near 1, beta near zero."""
    logits, labels = _make_well_calibrated_logits(n=1000, k=5)
    alpha, beta = learn_dirichlet_scaling(logits, labels, reg_lambda=0.1)
    assert (alpha - 1.0).abs().max() < 0.5, f"alpha should be near 1, got {alpha}"
    assert beta.abs().max() < 0.5, f"beta should be near 0, got {beta}"


def test_learn_dirichlet_scaling_reduces_ece():
    """Overconfident logits → dirichlet scaling should reduce ECE."""
    logits, labels = _make_overconfident_logits(n=1000, k=5, scale=3.0)
    ece_before = _compute_ece(logits, labels)

    alpha, beta = learn_dirichlet_scaling(logits, labels)
    cal_logits = apply_calibration(
        logits, {"method": "dirichlet", "alpha": alpha.tolist(), "beta": beta.tolist()}
    )
    ece_after = _compute_ece(cal_logits, labels)

    assert ece_after < ece_before, f"ECE should decrease: {ece_before:.4f} -> {ece_after:.4f}"


# ============================================================================
# apply_calibration tests
# ============================================================================


def test_apply_calibration_temperature():
    """Verify apply_calibration temperature produces logits / T."""
    logits = torch.randn(10, 5)
    T = 2.5
    result = apply_calibration(logits, {"method": "temperature", "temperature": T})
    expected = logits / T
    assert torch.allclose(result, expected, atol=1e-6)


def test_apply_calibration_matrix():
    """Verify apply_calibration matrix produces logits @ W^T + b."""
    logits = torch.randn(10, 5)
    W = torch.eye(5) * 1.5
    b = torch.ones(5) * 0.1
    result = apply_calibration(logits, {"method": "matrix", "W": W.tolist(), "b": b.tolist()})
    expected = logits @ W.T + b
    assert torch.allclose(result, expected, atol=1e-6)


def test_apply_calibration_dirichlet():
    """Verify apply_calibration dirichlet produces alpha * log_softmax(logits) + beta."""
    logits = torch.randn(10, 5)
    alpha = torch.ones(5) * 0.8
    beta = torch.ones(5) * 0.1
    result = apply_calibration(
        logits, {"method": "dirichlet", "alpha": alpha.tolist(), "beta": beta.tolist()}
    )
    expected = alpha * torch.log_softmax(logits, dim=-1) + beta
    assert torch.allclose(result, expected, atol=1e-6)


def test_apply_calibration_unknown_method():
    """Unknown method should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown calibration method"):
        apply_calibration(torch.randn(10, 5), {"method": "bogus"})


# ============================================================================
# load_calibration tests
# ============================================================================


def test_load_calibration_unified_roundtrip():
    """Write calibration.json, load it back — should round-trip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cal_data = {
            "method": "dirichlet",
            "alpha": [1.0, 1.1, 0.9],
            "beta": [0.01, -0.02, 0.03],
            "reg_lambda": 0.01,
            "ece_before": 0.08,
            "ece_after": 0.03,
            "n_val_samples": 1000,
            "num_classes": 3,
        }
        cal_path = Path(tmpdir) / "calibration.json"
        with open(cal_path, "w") as f:
            json.dump(cal_data, f)

        loaded = load_calibration(Path(tmpdir))
        assert loaded is not None
        assert loaded["method"] == "dirichlet"
        assert loaded["alpha"] == cal_data["alpha"]
        assert loaded["beta"] == cal_data["beta"]


def test_load_calibration_temperature_roundtrip():
    """calibration.json with method=temperature round-trips correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cal_data = {
            "method": "temperature",
            "temperature": 1.42,
            "ece_before": 0.08,
            "ece_after": 0.03,
            "n_val_samples": 4500,
            "num_classes": 12,
        }
        cal_path = Path(tmpdir) / "calibration.json"
        with open(cal_path, "w") as f:
            json.dump(cal_data, f)

        loaded = load_calibration(Path(tmpdir))
        assert loaded is not None
        assert loaded["method"] == "temperature"
        assert loaded["temperature"] == 1.42
        assert loaded["ece_before"] == 0.08


def test_load_calibration_ignores_temperature_json():
    """temperature.json alone is NOT loaded (no legacy fallback)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(Path(tmpdir) / "temperature.json", "w") as f:
            json.dump({"temperature": 1.5}, f)

        assert load_calibration(Path(tmpdir)) is None


def test_load_calibration_none_when_missing():
    """No calibration files → return None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        assert load_calibration(Path(tmpdir)) is None


class TestCalibrateModelReconstructsFullConfig:
    """calibrate_model()/calibrate_model_multiclass() must build the SAME
    model predict/eval/bundle would, via model_loading._instantiate_model --
    not a hand-picked model_kwargs dict that silently drops new constructor
    params such as --standardize-features' feature_mean/feature_std or
    --model-config's causal (issue #283 code review finding: this function
    used to reimplement the constructor-kwargs resolution and fell out of
    sync the moment those two params were added elsewhere)."""

    def test_standardized_model_calibrates_without_a_state_dict_mismatch(
        self, temp_chunks_file, tmp_path
    ):
        from leech.calibration import calibrate_model
        from leech.models import get_model

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        kwargs = {
            "signal_len": 400,
            "kmer_len": 11,
            "num_features": 5,
            "conv_channels": [4, 8, 16],
            "lstm_hidden": 8,
            "seq_encoding": "base_onehot",
            "feature_mean": [0.0] * 5,
            "feature_std": [1.0] * 5,
        }
        model = get_model("ConvLSTMDwell", **kwargs)
        torch.save({"model_state_dict": model.state_dict()}, model_dir / "model_best.pt")
        (model_dir / "config.json").write_text(
            json.dumps({"model_name": "ConvLSTMDwell", **kwargs})
        )

        # Before the fix: calibrate_model rebuilt the model from a hand-picked
        # kwargs dict that dropped feature_mean/feature_std, so the freshly
        # built model had no AffineStandardize buffers and the checkpoint's
        # state_dict (which has them) failed to load under strict=True.
        a, b = calibrate_model(model_dir, temp_chunks_file, device="cpu", batch_size=4)
        assert isinstance(a, float)
        assert isinstance(b, float)


def _build_ablation_corpus(tmp_path, channel_names, n=24, feat_width=11):
    """Write a small npz corpus (+ matching ``prepare_config.json`` sidecar)
    whose ``features`` rows are named ``channel_names``, each a distinct,
    recoverable per-base constant (``channel_index * 1000 + position``), so a
    selected subset can be checked by tensor VALUE, not just shape.
    """
    import numpy as np

    from leech.chunking import save_chunks

    rng = np.random.default_rng(23)
    chunks = []
    for i in range(n):
        rows = [
            np.arange(feat_width, dtype=np.float32) + ch_idx * 1000.0
            for ch_idx in range(len(channel_names))
        ]
        chunks.append(
            {
                "signal": rng.standard_normal(400).astype(np.float32),
                "dwell": rng.integers(2, 12, feat_width).astype(np.float32),
                "features": np.stack(rows, axis=0),
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
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    path = corpus_dir / "train.npz"
    save_chunks(chunks, path)
    (corpus_dir / "prepare_config.json").write_text(json.dumps({"feature_channels": channel_names}))
    return path


class TestCalibrateModelFeatureChannelSelection:
    """``leech model calibrate`` (:func:`~leech.calibration.calibrate_model`)
    must select the checkpoint's own channel subset out of a wider corpus
    (channel ablation, issue #358 follow-up) -- the same gap ``eval test``
    had: an ablation sweep's checkpoints are typically calibrated against
    the SAME all-channel corpus.
    """

    FULL = [
        "dwell",
        "dwell_log",
        "dwell_mean",
        "dwell_std",
        "dwell_ratio",
        "level_mean",
        "level_median",
    ]

    @staticmethod
    def _write_checkpoint(model_dir, num_features, feature_channels=None):
        from leech.models import get_model

        model_dir.mkdir()
        kwargs = {
            "signal_len": 400,
            "kmer_len": 11,
            "num_features": num_features,
            "conv_channels": [4, 8, 16],
            "lstm_hidden": 8,
            "seq_encoding": "base_onehot",
        }
        model = get_model("ConvLSTMDwell", **kwargs)
        torch.save({"model_state_dict": model.state_dict()}, model_dir / "model_best.pt")
        config = {"model_name": "ConvLSTMDwell", **kwargs}
        if feature_channels is not None:
            config["feature_channels"] = feature_channels
        (model_dir / "config.json").write_text(json.dumps(config))

    def test_subset_checkpoint_receives_selected_rows_by_value(self, tmp_path, monkeypatch):
        from leech.calibration import calibrate_model
        from leech.models.inference_wrapper import ModelInferenceWrapper

        corpus = _build_ablation_corpus(tmp_path, self.FULL)
        subset = ["level_mean", "dwell", "dwell_std"]

        model_dir = tmp_path / "model"
        self._write_checkpoint(model_dir, num_features=len(subset), feature_channels=subset)

        captured: list[torch.Tensor] = []
        original_forward = ModelInferenceWrapper.forward_batch

        def spy(self, batch, device):
            captured.append(batch["features"].clone())
            return original_forward(self, batch, device)

        monkeypatch.setattr(ModelInferenceWrapper, "forward_batch", spy)

        calibrate_model(model_dir, corpus, device="cpu", batch_size=8)

        assert captured
        features = torch.cat(captured, dim=0)
        assert features.shape[1] == len(subset)
        for row_idx, name in enumerate(subset):
            expected_row = torch.arange(11, dtype=torch.float32) + self.FULL.index(name) * 1000.0
            torch.testing.assert_close(
                features[:, row_idx, :], expected_row.unsqueeze(0).expand(features.shape[0], -1)
            )

    def test_old_style_checkpoint_takes_the_identity_path(self, temp_chunks_file, tmp_path):
        """A checkpoint with no recorded ``feature_channels`` key, scored
        against a corpus that agrees with its own fallback, must pass
        ``selected_feature_channels=None`` -- the pre-existing, byte-identical
        path -- rather than a spurious selection.
        """
        from unittest import mock

        from leech.calibration import calibrate_model
        from leech.dataset import LeechDataset

        model_dir = tmp_path / "model"
        self._write_checkpoint(model_dir, num_features=5)  # no feature_channels key at all

        seen_selection: list = []
        original_init = LeechDataset.__init__

        def spy_init(self, *args, **kwargs):
            seen_selection.append(kwargs.get("selected_feature_channels"))
            return original_init(self, *args, **kwargs)

        with mock.patch.object(LeechDataset, "__init__", spy_init):
            calibrate_model(model_dir, temp_chunks_file, device="cpu", batch_size=4)

        assert seen_selection == [None]
