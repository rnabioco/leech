"""Tests for leech.models.components."""

import torch

from leech.models.components import (
    AdaptiveAvgPool1d,
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
