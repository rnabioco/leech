"""Tests for kmer residual features and signal residual channel."""

import numpy as np
import pytest
import torch

from leech.features import compute_kmer_residual_features, compute_signal_residual


class TestComputeKmerResidualFeatures:
    """Tests for compute_kmer_residual_features()."""

    def _make_kmer_table(self, kmer_len: int = 3) -> dict[str, float]:
        """Build a small synthetic kmer table for testing."""
        bases = "ACGT"
        table = {}
        level = 50.0
        for a in bases:
            for b in bases:
                for c in bases:
                    kmer = a + b + c
                    table[kmer] = level
                    level += 0.5
        return table

    def test_basic_output_shapes(self):
        """Output arrays have correct shape (num_bases,)."""
        seq = "ACGTACGTAC"
        num_bases = len(seq)
        # Signal: 10 bases, each gets 10 samples
        sig = np.random.randn(100).astype(np.float32)
        s2s = np.arange(0, 101, 10, dtype=np.int64)

        table = self._make_kmer_table()
        result = compute_kmer_residual_features(sig, s2s, seq, table, kmer_len=3)

        assert "kmer_expected" in result
        assert "kmer_residual" in result
        assert "kmer_residual_abs" in result
        for key in result:
            assert result[key].shape == (num_bases,)
            assert result[key].dtype == np.float32

    def test_residual_is_signed(self):
        """kmer_residual = level_mean - kmer_expected (can be negative)."""
        seq = "ACGTAC"
        # 6 bases, each 10 samples
        sig = np.zeros(60, dtype=np.float32)  # all zero signal
        s2s = np.arange(0, 61, 10, dtype=np.int64)

        table = self._make_kmer_table()
        result = compute_kmer_residual_features(sig, s2s, seq, table, kmer_len=3)

        # With zero signal and positive expected levels, residual should be negative
        # (except edge positions where expected is 0)
        middle = result["kmer_residual"][1:-1]  # skip edges where kmer can't be formed
        assert np.all(middle <= 0)

    def test_residual_abs_is_unsigned(self):
        """kmer_residual_abs = |kmer_residual|."""
        seq = "ACGTACGTAC"
        sig = np.random.randn(100).astype(np.float32)
        s2s = np.arange(0, 101, 10, dtype=np.int64)

        table = self._make_kmer_table()
        result = compute_kmer_residual_features(sig, s2s, seq, table, kmer_len=3)

        np.testing.assert_allclose(result["kmer_residual_abs"], np.abs(result["kmer_residual"]))

    def test_matching_signal_gives_zero_residual(self):
        """When signal matches expected levels, residual ≈ 0."""
        seq = "ACGTAC"
        table = self._make_kmer_table()

        # Build signal where each base's mean matches its expected level
        from leech.signal_refine import extract_levels

        expected = extract_levels(seq, table, 3)
        sig = np.repeat(expected.astype(np.float32), 10)
        s2s = np.arange(0, len(sig) + 1, 10, dtype=np.int64)

        result = compute_kmer_residual_features(sig, s2s, seq, table, kmer_len=3)

        # Where expected != 0, residual should be ~0
        valid = expected != 0.0
        np.testing.assert_allclose(result["kmer_residual"][valid], 0.0, atol=1e-5)


class TestComputeSignalResidual:
    """Tests for compute_signal_residual()."""

    def test_output_shape(self):
        """Output has same shape as input signal."""
        sig = np.random.randn(100).astype(np.float32)
        s2s = np.array([0, 20, 40, 60, 80, 100], dtype=np.int64)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        residual = compute_signal_residual(sig, s2s, expected)
        assert residual.shape == sig.shape
        assert residual.dtype == np.float32

    def test_constant_signal_minus_expected(self):
        """With constant signal, residual = signal_val - expected per base."""
        sig = np.full(50, 10.0, dtype=np.float32)
        s2s = np.array([0, 10, 20, 30, 40, 50], dtype=np.int64)
        expected = np.array([8.0, 10.0, 12.0, 0.0, 9.0], dtype=np.float32)

        residual = compute_signal_residual(sig, s2s, expected)

        # Base 0: 10 - 8 = 2
        np.testing.assert_allclose(residual[0:10], 2.0, atol=1e-6)
        # Base 1: 10 - 10 = 0
        np.testing.assert_allclose(residual[10:20], 0.0, atol=1e-6)
        # Base 2: 10 - 12 = -2
        np.testing.assert_allclose(residual[20:30], -2.0, atol=1e-6)
        # Base 3: expected=0 → residual=0
        np.testing.assert_allclose(residual[30:40], 0.0, atol=1e-6)
        # Base 4: 10 - 9 = 1
        np.testing.assert_allclose(residual[40:50], 1.0, atol=1e-6)

    def test_zero_expected_gives_zero_residual(self):
        """Positions with expected_level=0 should have residual=0."""
        sig = np.ones(30, dtype=np.float32) * 5.0
        s2s = np.array([0, 10, 20, 30], dtype=np.int64)
        expected = np.array([0.0, 3.0, 0.0], dtype=np.float32)

        residual = compute_signal_residual(sig, s2s, expected)

        np.testing.assert_allclose(residual[0:10], 0.0)
        np.testing.assert_allclose(residual[10:20], 2.0, atol=1e-6)
        np.testing.assert_allclose(residual[20:30], 0.0)

    def test_empty_bases(self):
        """Handles bases with zero-length signal spans."""
        sig = np.ones(20, dtype=np.float32)
        # Base 1 has zero length (s2s[1]==s2s[2])
        s2s = np.array([0, 10, 10, 20], dtype=np.int64)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        residual = compute_signal_residual(sig, s2s, expected)
        # Should not crash, base 0: 1-1=0, base 2: 1-3=-2
        np.testing.assert_allclose(residual[0:10], 0.0, atol=1e-6)
        np.testing.assert_allclose(residual[10:20], -2.0, atol=1e-6)


class TestTCNDwellResidualModel:
    """Test that TCNDwellResidual handles 2-channel signal input."""

    def test_forward_2channel(self):
        """Model accepts (batch, 2, signal_len) signal input."""
        from leech.models import get_model

        model = get_model(
            "TCNDwellResidual",
            signal_len=100,
            kmer_len=11,
            num_features=12,
            dwell_margin=3,
            signal_in_channels=2,
            hidden_channels=16,
            num_layers=2,
            kernel_size=3,
        )
        model.eval()

        batch_size = 4
        signal = torch.randn(batch_size, 2, 100)
        sequence = torch.randn(batch_size, 4, 11)
        features = torch.randn(batch_size, 12, 17)  # 11 + 3 + 3

        with torch.no_grad():
            logits = model(signal, sequence, features)

        assert logits.shape == (batch_size, 1)

    def test_forward_1channel_fallback(self):
        """Model handles (batch, signal_len) input via unsqueeze."""
        from leech.models import get_model

        model = get_model(
            "TCNDwellResidual",
            signal_len=100,
            kmer_len=11,
            num_features=9,
            dwell_margin=3,
            signal_in_channels=1,
            hidden_channels=16,
            num_layers=2,
            kernel_size=3,
        )
        model.eval()

        batch_size = 4
        signal = torch.randn(batch_size, 100)  # 2D input
        sequence = torch.randn(batch_size, 4, 11)
        features = torch.randn(batch_size, 9, 17)

        with torch.no_grad():
            logits = model(signal, sequence, features)

        assert logits.shape == (batch_size, 1)
