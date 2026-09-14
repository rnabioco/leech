"""Tests for leech.models.components."""

import pytest
import torch

from leech.models.components import (
    TCN,
    AdaptiveAvgPool1d,
    AffineStandardize,
    FeatureBranch,
    TemporalBlock,
    _cached_segment_mean_tensor,
    segment_mean_matrix,
)


class TestAdaptiveAvgPool1dCache:
    """The eager cache (#272 item 6) must never leak an inference tensor.

    ``torch.inference_mode()`` marks every tensor it creates as an inference
    tensor; using one later in a grad-enabled computation raises at
    ``backward()``. The cache is keyed by ``(length, output_size, dtype,
    device)`` and is process-global (a module-level ``lru_cache``), so if its
    first population for a given key happened inside ``validate()``, ``eval
    test``, ``predict`` or ``bundle`` (all run under ``inference_mode``), a
    later training forward at that same shape would reuse the tainted tensor.
    """

    def setup_method(self):
        _cached_segment_mean_tensor.cache_clear()

    def test_cache_populated_under_inference_mode_stays_usable_for_backward(self):
        pool = AdaptiveAvgPool1d(7)
        with torch.inference_mode():
            _ = pool(torch.randn(2, 3, 100))

        x = torch.randn(2, 3, 100, requires_grad=True)
        out = pool(x)
        out.sum().backward()
        assert x.grad is not None

    def test_cached_weight_is_never_an_inference_tensor(self):
        with torch.inference_mode():
            AdaptiveAvgPool1d(7)(torch.randn(2, 3, 100))
        w = _cached_segment_mean_tensor(100, 7, torch.float32, torch.device("cpu"))
        assert not w.is_inference()

    def test_cache_hit_matches_a_fresh_tensor(self):
        """The cache must not change the value, only avoid rebuilding it."""
        cached = _cached_segment_mean_tensor(100, 7, torch.float32, torch.device("cpu"))
        fresh = segment_mean_matrix(100, 7, dtype=torch.float32, device=torch.device("cpu"))
        torch.testing.assert_close(cached, fresh)

    def test_forward_output_matches_reference_adaptive_avg_pool(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 100)
        got = AdaptiveAvgPool1d(7)(x)
        want = torch.nn.functional.adaptive_avg_pool1d(x, 7)
        torch.testing.assert_close(got, want)

    def test_repeated_forward_uses_the_cache_not_a_fresh_tensor_each_time(self):
        pool = AdaptiveAvgPool1d(7)
        x = torch.randn(2, 3, 100)
        pool(x)
        hits_before = _cached_segment_mean_tensor.cache_info().hits
        pool(x)
        assert _cached_segment_mean_tensor.cache_info().hits == hits_before + 1


class TestTemporalBlockCausal:
    """``causal`` toggle on TemporalBlock/TCN (issue #283)."""

    def test_causal_is_the_default(self):
        block = TemporalBlock(in_channels=2, out_channels=2, kernel_size=3, dilation=1)
        assert block.causal is True

    def test_output_length_preserved_either_way(self):
        x = torch.randn(1, 2, 37)
        for causal in (True, False):
            block = TemporalBlock(
                in_channels=2, out_channels=2, kernel_size=3, dilation=2, causal=causal
            )
            out = block(x)
            assert out.shape == x.shape

    def test_causal_output_position_ignores_future_input(self):
        """Position 0 of a causal block's output must not depend on x[1].

        Two cascaded left-only-padded convs: conv1's output at 0 depends only
        on x[0] (everything else is the zero pad), so conv2's output at 0
        depends only on conv1's output at 0 -- i.e. only on x[0].
        """
        torch.manual_seed(0)
        block = TemporalBlock(in_channels=2, out_channels=2, kernel_size=3, dilation=1, causal=True)
        block.eval()
        x1 = torch.randn(1, 2, 10)
        x2 = x1.clone()
        x2[:, :, 1] = x2[:, :, 1] + 5.0  # perturb only a "future" position
        out1 = block(x1)
        out2 = block(x2)
        torch.testing.assert_close(out1[:, :, 0], out2[:, :, 0])

    def test_noncausal_output_position_does_depend_on_future_input(self):
        """The same perturbation as above DOES move position 0 when
        causal=False, confirming the two modes are genuinely different
        (not just an unused flag)."""
        torch.manual_seed(0)
        block = TemporalBlock(
            in_channels=2, out_channels=2, kernel_size=3, dilation=1, causal=False
        )
        block.eval()
        x1 = torch.randn(1, 2, 10)
        x2 = x1.clone()
        x2[:, :, 1] = x2[:, :, 1] + 5.0
        out1 = block(x1)
        out2 = block(x2)
        assert not torch.allclose(out1[:, :, 0], out2[:, :, 0])

    def test_state_dict_keys_and_shapes_identical_regardless_of_causal(self):
        """causal only changes how padding is applied, never a parameter
        shape -- a checkpoint trained one way loads into the other (the
        values just aren't a meaningful initialization for it)."""
        causal_block = TemporalBlock(in_channels=3, out_channels=5, kernel_size=3, dilation=4)
        noncausal_block = TemporalBlock(
            in_channels=3, out_channels=5, kernel_size=3, dilation=4, causal=False
        )
        sd1 = causal_block.state_dict()
        sd2 = noncausal_block.state_dict()
        assert set(sd1) == set(sd2)
        for key in sd1:
            assert sd1[key].shape == sd2[key].shape

        # And it actually loads: no shape/key mismatch at load time either.
        noncausal_block.load_state_dict(sd1)

    def test_tcn_threads_causal_to_every_block(self):
        tcn = TCN(in_channels=1, hidden_channels=4, num_layers=3, kernel_size=3, causal=False)
        for layer in tcn.network:
            assert layer.causal is False

    def test_tcn_causal_default_matches_temporal_block_default(self):
        tcn = TCN(in_channels=1, hidden_channels=4, num_layers=2, kernel_size=3)
        assert all(layer.causal is True for layer in tcn.network)


class TestAffineStandardize:
    """Frozen per-channel affine standardization (issue #283)."""

    def test_forward_matches_manual_computation(self):
        mean = [1.0, -2.0, 0.5]
        std = [2.0, 0.5, 1.0]
        layer = AffineStandardize(mean, std)
        x = torch.randn(4, 3, 11)
        got = layer(x)
        mean_t = torch.tensor(mean).view(1, -1, 1)
        std_t = torch.tensor(std).view(1, -1, 1)
        torch.testing.assert_close(got, (x - mean_t) / std_t)

    def test_buffers_not_parameters(self):
        layer = AffineStandardize([0.0, 0.0], [1.0, 1.0])
        assert list(layer.parameters()) == []
        assert {name for name, _ in layer.named_buffers()} == {"mean", "std"}

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="same length"):
            AffineStandardize([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_zero_std_is_floored_not_a_divide_by_zero(self):
        layer = AffineStandardize([0.0], [0.0])
        out = layer(torch.tensor([[[5.0]]]))
        assert torch.isfinite(out).all()

    def test_nan_std_is_rescued_not_left_nan(self):
        """torch.std's unbiased estimator gives 0/0 = NaN for a channel with
        a single total sample (a one-chunk corpus, or kmer_len=1) -- clamp_min
        alone does not rescue that (clamp_min(NaN, eps) is still NaN)."""
        layer = AffineStandardize([1.0], [float("nan")])
        assert torch.isfinite(layer.std).all()
        out = layer(torch.tensor([[[5.0]]]))
        assert torch.isfinite(out).all()


class TestFeatureBranchStandardize:
    """``FeatureBranch``'s optional frozen-affine first layer (issue #283)."""

    def test_default_has_no_standardize_layer(self):
        branch = FeatureBranch(num_features=5, conv_channels=[4, 8])
        assert not isinstance(branch.conv_layers[0], AffineStandardize)
        assert isinstance(branch.conv_layers[0], torch.nn.Conv1d)

    def test_feature_mean_and_std_prepend_standardize_layer(self):
        branch = FeatureBranch(
            num_features=5,
            conv_channels=[4, 8],
            feature_mean=[0.0] * 5,
            feature_std=[1.0] * 5,
        )
        assert isinstance(branch.conv_layers[0], AffineStandardize)
        out = branch(torch.randn(2, 5, 11))
        assert out.shape == (2, 8, 11)

    def test_mean_without_std_raises(self):
        with pytest.raises(ValueError, match="both be given or both omitted"):
            FeatureBranch(num_features=5, feature_mean=[0.0] * 5)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="must equal num_features"):
            FeatureBranch(num_features=5, feature_mean=[0.0] * 3, feature_std=[1.0] * 3)
