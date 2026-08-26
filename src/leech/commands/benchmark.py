"""Handler for ``leech model benchmark`` — profile one training step."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from rich.table import Table
from torch.utils.data import DataLoader

from leech.cli_config import make_console
from leech.dataset import LeechDataset, collate_fn
from leech.models import get_model
from leech.models.inference_wrapper import ModelInferenceWrapper
from leech.profiling import (
    benchmark_training,
    run_torch_profiler,
    set_nvidia_smi_env_if_missing,
)

logger = logging.getLogger("leech.commands.benchmark")
console = make_console()


def _build_loader_and_model(
    *,
    train_data: Path,
    model_name: str,
    batch_size: int,
    device: str,
    num_workers: int,
    prefetch_factor: int,
    signal_len: int,
    kmer_len: int,
    seq_encoding: str,
    signal_mode: str,
) -> tuple[DataLoader, torch.nn.Module, ModelInferenceWrapper, int]:
    """Minimal mirror of ``train_model`` construction — loader, model, wrapper."""
    dataset = LeechDataset(
        chunk_path=train_data,
        signal_len=signal_len,
        kmer_len=kmer_len,
        model_type=model_name,
        seq_encoding=seq_encoding,
        signal_mode=signal_mode,
    )
    logger.info(f"Loaded {len(dataset)} chunks from {train_data}")

    loader_kwargs: dict[str, Any] = {
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "drop_last": True,
    }
    if device != "cpu":
        loader_kwargs["pin_memory"] = True
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    # dataloader-workers: unresolved -- the worker count is the independent
    # variable being benchmarked here, so resolving it defeats the measurement.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)

    first = next(iter(loader))
    num_features = first.get("features", torch.zeros(1, 1, kmer_len)).shape[1]
    signal_shape = first["signal"].shape
    signal_in_channels = signal_shape[1] if len(signal_shape) == 3 else 1

    max_label = max(int(c["label_int"]) for c in dataset.chunks)
    num_out = max_label + 1 if max_label > 1 else 1

    model = get_model(
        model_name,
        signal_len=signal_len,
        kmer_len=kmer_len,
        # What the dataset yields, not what was asked for — a signal_kmer
        # request over a corpus with no base-to-signal maps degrades, and the
        # sequence branch has to be built for the input it will actually get
        # rather than die on a channel count at the first step (#230).
        seq_encoding=dataset.effective_seq_encoding,
        num_features=num_features,
        signal_in_channels=signal_in_channels,
        num_out=num_out,
    )
    model.to(device)

    if device != "cpu":
        torch.backends.cudnn.benchmark = True
    if device != "cpu" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"torch.compile failed, falling back to eager: {e}")

    wrapper = ModelInferenceWrapper(model, model_type=model_name)
    return loader, model, wrapper, num_out  # ty: ignore[invalid-return-type]


def _print_report(report) -> None:
    table = Table(title="Training-step benchmark", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="green")

    table.add_row("device", report.device)
    table.add_row("batch_size", str(report.batch_size))
    table.add_row("num_steps", str(report.num_steps))
    table.add_row("num_workers", str(report.num_workers))
    table.add_row("pin_memory", str(report.pin_memory))
    table.add_row("prefetch_factor", str(report.prefetch_factor))
    table.add_row("mixed_precision", str(report.mixed_precision))
    table.add_row("non_blocking H2D", str(report.non_blocking))
    table.add_row("", "")
    table.add_row("step ms (median)", f"{report.step_ms_median:.2f}")
    table.add_row("step ms (mean)", f"{report.step_ms_mean:.2f}")
    table.add_row("step ms (p95)", f"{report.step_ms_p95:.2f}")
    table.add_row("samples / sec", f"{report.samples_per_sec:,.0f}")
    table.add_row("", "")
    table.add_row("data-wait fraction", f"{report.data_wait_frac:.1%}")
    table.add_row("H2D fraction", f"{report.h2d_frac:.1%}")
    table.add_row("forward fraction", f"{report.forward_frac:.1%}")
    table.add_row("backward fraction", f"{report.backward_frac:.1%}")
    table.add_row("optimizer fraction", f"{report.optimizer_frac:.1%}")
    table.add_row("", "")
    if report.gpu_util_mean is not None:
        table.add_row("GPU util mean (%)", f"{report.gpu_util_mean:.1f}")
        table.add_row("GPU util p95 (%)", f"{report.gpu_util_p95:.1f}")
        table.add_row("GPU util samples", str(report.gpu_util_samples))
    if report.gpu_mem_max_gb is not None:
        table.add_row("GPU mem peak (GB)", f"{report.gpu_mem_max_gb:.2f}")
    console.print(table)


def handle_benchmark(
    *,
    train_data: Path,
    model_name: str,
    output_dir: Path,
    batch_size: int,
    device: str,
    num_steps: int,
    warmup_steps: int,
    num_workers: int,
    prefetch_factor: int,
    mixed_precision: bool,
    non_blocking: bool,
    signal_len: int,
    kmer_len: int,
    seq_encoding: str,
    signal_mode: str,
    trace: bool,
    trace_active_steps: int,
) -> dict[str, Any]:
    set_nvidia_smi_env_if_missing()
    output_dir.mkdir(parents=True, exist_ok=True)

    loader, model, wrapper, num_out = _build_loader_and_model(
        train_data=train_data,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        signal_len=signal_len,
        kmer_len=kmer_len,
        seq_encoding=seq_encoding,
        signal_mode=signal_mode,
    )

    report = benchmark_training(
        model=model,
        model_wrapper=wrapper,
        train_loader=loader,
        device=device,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        mixed_precision=mixed_precision,
        num_out=num_out,
        non_blocking=non_blocking,
    )

    _print_report(report)

    report_path = output_dir / "benchmark.json"
    with open(report_path, "w") as f:
        # steps list can be huge; keep summary + first 20 step records
        payload = report.to_dict()
        payload["steps"] = payload["steps"][:20]
        json.dump(payload, f, indent=2)
    console.print(f"[green]Report written to {report_path}[/green]")

    if trace:
        trace_path = output_dir / "trace.json"
        console.print(
            f"[cyan]Collecting torch.profiler trace ({trace_active_steps} steps)...[/cyan]"
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        if num_out == 1:
            criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler(enabled=mixed_precision and device != "cpu")
        model.train()

        wait, warmup = 2, 3
        total_prof_steps = wait + warmup + trace_active_steps
        it = iter(loader)
        with run_torch_profiler(
            trace_path,
            wait=wait,
            warmup=warmup,
            active=trace_active_steps,
        ) as prof:
            for _ in range(total_prof_steps):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)

                optimizer.zero_grad(set_to_none=True)
                if mixed_precision and device != "cpu":
                    with torch.amp.autocast("cuda"):
                        logits = wrapper.forward_batch(batch, device)
                        labels = batch["label"].to(device, non_blocking=non_blocking)
                        if num_out > 1:
                            labels = labels.squeeze(-1).long()
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = wrapper.forward_batch(batch, device)
                    labels = batch["label"].to(device, non_blocking=non_blocking)
                    if num_out > 1:
                        labels = labels.squeeze(-1).long()
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                prof.step()

    return report.to_dict()
