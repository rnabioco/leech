"""Smoke tests for the training-step benchmark harness."""

from __future__ import annotations

import json

import torch

from leech.profiling import GpuUtilSampler, benchmark_training


class _FakeWrapper:
    """Minimal stand-in for ModelInferenceWrapper used by benchmark_training."""

    def __init__(self, model: torch.nn.Module, requires_features: bool) -> None:
        self.model = model
        self.requires_features = requires_features

    def forward_batch(self, batch: dict, device: str) -> torch.Tensor:
        signal = batch["signal"].to(device)
        sequence = batch["sequence"].to(device)
        if self.requires_features:
            features = batch["features"].to(device)
            return self.model(signal, sequence, features)
        return self.model(signal, sequence)


class _TinyModel(torch.nn.Module):
    """Flattens inputs and projects to 1 logit — fast enough for CPU test."""

    def __init__(self, signal_len: int, kmer_len: int, num_features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(signal_len + 4 * kmer_len + num_features * kmer_len, 1)

    def forward(self, signal, sequence, features):
        b = signal.shape[0]
        sig = signal.reshape(b, -1)
        seq = sequence.reshape(b, -1)
        feat = features.reshape(b, -1)
        x = torch.cat([sig, seq, feat], dim=1)
        return self.linear(x)


def test_benchmark_training_cpu_smoke(sample_dataloader):
    """Benchmark runs end-to-end on CPU and produces a sensible report."""
    loader = sample_dataloader

    batch = next(iter(loader))
    signal_len = batch["signal"].shape[-1]
    kmer_len = batch["features"].shape[-1]
    num_features = batch["features"].shape[1]

    model = _TinyModel(signal_len, kmer_len, num_features)
    wrapper = _FakeWrapper(model, requires_features=True)

    report = benchmark_training(
        model=model,
        model_wrapper=wrapper,
        train_loader=loader,
        device="cpu",
        num_steps=5,
        warmup_steps=2,
        mixed_precision=False,
        num_out=1,
    )

    assert report.num_steps == 5
    assert report.device == "cpu"
    assert report.step_ms_median > 0
    # Phase fractions should sum to ~1.0
    total_frac = (
        report.data_wait_frac
        + report.h2d_frac
        + report.forward_frac
        + report.backward_frac
        + report.optimizer_frac
    )
    assert 0.99 <= total_frac <= 1.01
    # On CPU we fold everything into forward_frac; h2d/bwd/opt are 0
    assert report.h2d_frac == 0.0

    # to_dict serializable
    json.dumps(report.to_dict())


def test_gpu_util_sampler_no_cuda_is_noop():
    """Without CUDA, sampler start/stop should be safe no-op with no samples."""
    sampler = GpuUtilSampler(interval_ms=10)
    sampler.start()
    sampler.stop()
    assert sampler.util_samples == []
