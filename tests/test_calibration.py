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
