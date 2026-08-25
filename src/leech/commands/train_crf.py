"""Handler for the 'model train-crf' command.

Thin: everything that decides anything lives in :mod:`leech.crf.training`, so
this marshals options and reports. The reporting is not incidental — an epoch
mean cannot tell one catastrophic batch from a thousand mediocre ones, so the
table carries the numbers that separate those cases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rich.table import Table

from leech.cli_config import make_console

logger = logging.getLogger("leech.commands.train_crf")
console = make_console()


def handle_train_crf(
    corpus: Path,
    output_dir: Path | None,
    *,
    arch_config: Path | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Train a CTC-CRF model on a prepared corpus.

    Args:
        corpus: corpus stem — ``<corpus>_X.npy`` and ``<corpus>_meta.npz``.
        output_dir: where ``model.pt`` and ``model.json`` are written.
        arch_config: architecture TOML; defaults to the packaged geometry.
        **options: forwarded to :class:`~leech.crf.training.CrfTrainConfig`.

    Returns:
        The training result, without the model object.
    """
    from leech.crf import CrfTrainConfig, CrfTrainer
    from leech.crf.config import load_config

    config = CrfTrainConfig(**options)
    trainer = CrfTrainer(
        corpus,
        config=config,
        arch_config=load_config(arch_config) if arch_config else None,
        output_dir=output_dir,
    )

    logger.info("Training CTC-CRF on %s", corpus)
    result = trainer.train()
    result.pop("model", None)

    console.print("\n[bold green]Training complete![/bold green]")

    summary = Table(title="CRF Training Summary", show_header=True, header_style="bold magenta")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Corpus", str(corpus))
    summary.add_row("Window", f"{result['chunk']} samples")
    # Emission is target_len - state_len at any window width, so showing both is
    # what stops someone widening the window to get a longer decode.
    summary.add_row(
        "Target",
        f"{result['target_len']} nt -> emits {result['emits']} (state_len {result['state_len']})",
    )
    summary.add_row("Standardisation", f"mean={result['mean']:.3f} std={result['std']:.3f}")
    summary.add_row(
        "Split",
        f"{result['split_source']} — train {result['n_train']:,}, test {result['n_test']:,}",
    )
    summary.add_row("Label quality", f"coverage {result['quality_coverage']:.3f}")
    summary.add_row("Final loss", f"{result['final_loss']:.4f}")
    summary.add_row(
        "Shipped weights",
        f"epoch {result['selected_epoch']} "
        f"(loss {result['selected_loss']:.4f}) — {result['selected_because']}",
    )
    console.print(summary)

    # Only worth a table when something actually happened: a skipped step is
    # invisible otherwise, since the scaler silently discards it.
    anomalies = [e for e in result["history"] if e["n_skipped"] or e["n_nonfinite"]]
    if anomalies:
        table = Table(
            title="Epochs with discarded steps or non-finite gradients",
            show_header=True,
            header_style="bold yellow",
        )
        for column in ("Epoch", "Loss", "Worst batch", "|g| max", "Skipped", "Non-finite"):
            table.add_column(column)
        for e in anomalies:
            table.add_row(
                str(e["epoch"]),
                f"{e['loss']:.4f}",
                f"{e['worst_batch']:.4f}",
                f"{e['grad_max']:.2f}",
                str(e["n_skipped"]),
                str(e["n_nonfinite"]),
            )
        console.print(table)
        console.print(
            "[dim]A few discarded steps early on are normal GradScaler warm-up: it "
            "starts at a high scale and halves until nothing overflows.[/dim]"
        )

    if output_dir is not None:
        console.print(f"\nWrote [bold]{output_dir}/model.pt[/bold] and model.json")
        console.print(
            "[dim]model.json carries the standardisation constants. They are in "
            "neither the architecture config nor the checkpoint, so an exporter or "
            "a runtime that does not read them decodes silently worse.[/dim]"
        )
    return result
