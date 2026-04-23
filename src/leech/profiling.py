"""
Training-loop benchmark primitives.

Answers: where does a training step spend its time, and is the GPU being kept fed?

Two tools:
- ``benchmark_training``: runs a short loop and times each phase (data wait,
  H2D, forward, backward, optimizer) via CUDA events. Samples ``nvidia-smi``
  GPU utilization in the background.
- ``run_torch_profiler``: wraps N steps in ``torch.profiler.profile`` and
  writes a Chrome trace for kernel-level inspection.
"""

from __future__ import annotations

import logging
import os
import statistics
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("leech.profiling")


@dataclass
class StepBreakdown:
    """Per-step timing (milliseconds)."""

    data_wait_ms: float
    h2d_ms: float
    forward_ms: float
    backward_ms: float
    optimizer_ms: float
    total_ms: float


@dataclass
class BenchmarkReport:
    num_steps: int
    batch_size: int
    device: str
    mixed_precision: bool
    num_workers: int
    pin_memory: bool
    prefetch_factor: int | None
    non_blocking: bool
    # Summary percentiles (milliseconds)
    step_ms_median: float
    step_ms_p95: float
    step_ms_mean: float
    samples_per_sec: float
    # Per-phase mean share (fraction of total wall time)
    data_wait_frac: float
    h2d_frac: float
    forward_frac: float
    backward_frac: float
    optimizer_frac: float
    # GPU util / memory
    gpu_util_mean: float | None = None
    gpu_util_p95: float | None = None
    gpu_util_samples: int = 0
    gpu_mem_max_gb: float | None = None
    steps: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_steps": self.num_steps,
            "batch_size": self.batch_size,
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
            "non_blocking": self.non_blocking,
            "step_ms_median": self.step_ms_median,
            "step_ms_p95": self.step_ms_p95,
            "step_ms_mean": self.step_ms_mean,
            "samples_per_sec": self.samples_per_sec,
            "phase_fraction": {
                "data_wait": self.data_wait_frac,
                "h2d": self.h2d_frac,
                "forward": self.forward_frac,
                "backward": self.backward_frac,
                "optimizer": self.optimizer_frac,
            },
            "gpu_util_mean": self.gpu_util_mean,
            "gpu_util_p95": self.gpu_util_p95,
            "gpu_util_samples": self.gpu_util_samples,
            "gpu_mem_max_gb": self.gpu_mem_max_gb,
            "steps": self.steps,
        }


class GpuUtilSampler:
    """Background thread that polls `nvidia-smi` for GPU utilization.

    Sampling via subprocess has ~2-3 ms overhead per call; at 50 ms interval
    that's <6% of one CPU core and completely outside the training thread.
    """

    def __init__(self, interval_ms: float = 50.0, device_index: int = 0) -> None:
        self.interval = interval_ms / 1000.0
        self.device_index = device_index
        self.util_samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not torch.cuda.is_available():
            return
        if self._thread is not None:
            return
        self._stop.clear()
        self.util_samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            f"--id={self.device_index}",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(cmd, timeout=1.0, text=True)
                self.util_samples.append(float(out.strip()))
            except (subprocess.SubprocessError, ValueError, OSError):
                # nvidia-smi missing or transient failure — just skip
                pass
            self._stop.wait(self.interval)


def _cuda_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def benchmark_training(
    *,
    model: torch.nn.Module,
    model_wrapper: Any,
    train_loader: Any,
    device: torch.device | str,
    num_steps: int = 100,
    warmup_steps: int = 10,
    mixed_precision: bool = True,
    num_out: int = 1,
    non_blocking: bool = False,
) -> BenchmarkReport:
    """Benchmark the current training step.

    Matches the forward/backward pattern used by ``Trainer.train_epoch`` but
    strips out scheduler/checkpoint/metric bookkeeping. Uses CUDA events for
    on-device phase timing (synchronizes once per step) and ``perf_counter``
    for data-wait (time spent blocked on the DataLoader iterator).
    """

    dev = torch.device(device) if isinstance(device, str) else device
    on_cuda = dev.type == "cuda"

    if num_out == 1:
        criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(enabled=mixed_precision and on_cuda)

    model.train()

    def _labels_from_batch(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        labels = batch["label"].to(dev, non_blocking=non_blocking)
        if num_out == 1:
            return labels
        return labels.squeeze(-1).long()

    def _forward_backward(batch: dict[str, torch.Tensor]) -> None:
        optimizer.zero_grad(set_to_none=True)
        if mixed_precision and on_cuda:
            with torch.amp.autocast("cuda"):
                logits = model_wrapper.forward_batch(batch, str(dev))
                labels = _labels_from_batch(batch)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model_wrapper.forward_batch(batch, str(dev))
            labels = _labels_from_batch(batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    # --- warmup: let cuDNN benchmark, torch.compile, and prefetch queue settle
    it = iter(train_loader)
    for _ in range(warmup_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        _forward_backward(batch)
    if on_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)

    # --- timed loop
    sampler = GpuUtilSampler()
    sampler.start()

    steps: list[StepBreakdown] = []
    total_start = time.perf_counter()

    try:
        for _ in range(num_steps):
            step_start = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            data_ready = time.perf_counter()
            data_wait_ms = (data_ready - step_start) * 1000.0

            if on_cuda:
                ev_h2d_start = _cuda_event()
                ev_h2d_end = _cuda_event()
                ev_fwd_end = _cuda_event()
                ev_bwd_end = _cuda_event()
                ev_opt_end = _cuda_event()

                ev_h2d_start.record()
                # H2D transfers
                signal = batch["signal"].to(dev, non_blocking=non_blocking)
                sequence = batch["sequence"].to(dev, non_blocking=non_blocking)
                features = (
                    batch["features"].to(dev, non_blocking=non_blocking)
                    if "features" in batch and model_wrapper.requires_features
                    else None
                )
                labels = _labels_from_batch(batch)
                ev_h2d_end.record()

                # Forward
                optimizer.zero_grad(set_to_none=True)
                if mixed_precision:
                    with torch.amp.autocast("cuda"):
                        if features is not None:
                            logits = model_wrapper.model(signal, sequence, features)
                        else:
                            logits = model_wrapper.model(signal, sequence)
                        loss = criterion(logits, labels)
                else:
                    if features is not None:
                        logits = model_wrapper.model(signal, sequence, features)
                    else:
                        logits = model_wrapper.model(signal, sequence)
                    loss = criterion(logits, labels)
                ev_fwd_end.record()

                # Backward
                if mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                ev_bwd_end.record()

                # Optimizer
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                ev_opt_end.record()

                torch.cuda.synchronize()

                h2d_ms = ev_h2d_start.elapsed_time(ev_h2d_end)
                fwd_ms = ev_h2d_end.elapsed_time(ev_fwd_end)
                bwd_ms = ev_fwd_end.elapsed_time(ev_bwd_end)
                opt_ms = ev_bwd_end.elapsed_time(ev_opt_end)
                total_ms = data_wait_ms + h2d_ms + fwd_ms + bwd_ms + opt_ms
            else:
                # CPU path — use perf_counter for everything
                t0 = time.perf_counter()
                _forward_backward(batch)
                t1 = time.perf_counter()
                h2d_ms = 0.0
                fwd_ms = (t1 - t0) * 1000.0
                bwd_ms = 0.0
                opt_ms = 0.0
                total_ms = data_wait_ms + fwd_ms

            steps.append(
                StepBreakdown(
                    data_wait_ms=data_wait_ms,
                    h2d_ms=h2d_ms,
                    forward_ms=fwd_ms,
                    backward_ms=bwd_ms,
                    optimizer_ms=opt_ms,
                    total_ms=total_ms,
                )
            )
    finally:
        sampler.stop()

    total_elapsed = time.perf_counter() - total_start

    # --- summarize
    batch_size = steps[0].total_ms and (
        train_loader.batch_size if hasattr(train_loader, "batch_size") else 0
    )
    # Wall-clock samples/sec (real throughput, ignores our sync overhead bias)
    total_samples = num_steps * (batch_size or 0)
    samples_per_sec = total_samples / total_elapsed if total_elapsed > 0 else 0.0

    totals = [s.total_ms for s in steps]
    totals_sorted = sorted(totals)
    p95_idx = max(0, int(0.95 * (len(totals_sorted) - 1)))
    step_ms_median = statistics.median(totals)
    step_ms_mean = statistics.mean(totals)
    step_ms_p95 = totals_sorted[p95_idx]

    phase_sum = sum(totals)
    data_wait_sum = sum(s.data_wait_ms for s in steps)
    h2d_sum = sum(s.h2d_ms for s in steps)
    fwd_sum = sum(s.forward_ms for s in steps)
    bwd_sum = sum(s.backward_ms for s in steps)
    opt_sum = sum(s.optimizer_ms for s in steps)

    def frac(x: float) -> float:
        return x / phase_sum if phase_sum > 0 else 0.0

    gpu_util_mean: float | None = None
    gpu_util_p95: float | None = None
    if sampler.util_samples:
        gpu_util_mean = statistics.mean(sampler.util_samples)
        gpu_util_p95 = sorted(sampler.util_samples)[
            max(0, int(0.95 * (len(sampler.util_samples) - 1)))
        ]

    gpu_mem_max_gb: float | None = None
    if on_cuda:
        gpu_mem_max_gb = torch.cuda.max_memory_allocated(dev) / (1024**3)

    loader_kwargs = {
        "num_workers": getattr(train_loader, "num_workers", 0),
        "pin_memory": getattr(train_loader, "pin_memory", False),
        "prefetch_factor": getattr(train_loader, "prefetch_factor", None),
    }

    return BenchmarkReport(
        num_steps=num_steps,
        batch_size=batch_size or 0,
        device=str(dev),
        mixed_precision=mixed_precision,
        num_workers=loader_kwargs["num_workers"],
        pin_memory=loader_kwargs["pin_memory"],
        prefetch_factor=loader_kwargs["prefetch_factor"],
        non_blocking=non_blocking,
        step_ms_median=step_ms_median,
        step_ms_p95=step_ms_p95,
        step_ms_mean=step_ms_mean,
        samples_per_sec=samples_per_sec,
        data_wait_frac=frac(data_wait_sum),
        h2d_frac=frac(h2d_sum),
        forward_frac=frac(fwd_sum),
        backward_frac=frac(bwd_sum),
        optimizer_frac=frac(opt_sum),
        gpu_util_mean=gpu_util_mean,
        gpu_util_p95=gpu_util_p95,
        gpu_util_samples=len(sampler.util_samples),
        gpu_mem_max_gb=gpu_mem_max_gb,
        steps=[s.__dict__ for s in steps],
    )


@contextmanager
def run_torch_profiler(
    trace_path: Path,
    *,
    wait: int = 2,
    warmup: int = 3,
    active: int = 10,
    record_shapes: bool = False,
):
    """Context manager wrapping ``torch.profiler.profile`` for a Chrome trace.

    Call ``profiler.step()`` once per training step inside the context.

    The trace is saved to ``trace_path`` and can be opened in Chrome at
    ``chrome://tracing`` or Perfetto (https://ui.perfetto.dev).
    """
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def _on_trace_ready(prof: torch.profiler.profile) -> None:
        # Export single trace (last cycle wins if multiple)
        prof.export_chrome_trace(str(trace_path))
        logger.info(f"Chrome trace written to {trace_path}")
        # Also print a summary sorted by CUDA time
        try:
            table = prof.key_averages().table(
                sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
                row_limit=20,
            )
            logger.info("Profiler top-20 ops:\n%s", table)
        except Exception as e:  # pragma: no cover
            logger.warning(f"key_averages() failed: {e}")

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=_on_trace_ready,
        record_shapes=record_shapes,
        with_stack=False,
    ) as prof:
        yield prof


def set_nvidia_smi_env_if_missing() -> None:
    """Ensure nvidia-smi is on PATH (common cluster quirk)."""
    if "nvidia-smi" in os.environ.get("PATH", ""):
        return
    for candidate in ("/usr/bin", "/usr/local/nvidia/bin", "/usr/local/cuda/bin"):
        if os.path.exists(os.path.join(candidate, "nvidia-smi")):
            os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + candidate
            return
