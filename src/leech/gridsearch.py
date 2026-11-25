"""
Grid search functionality for chunk context optimization.

Implements systematic grid search over signal window sizes (left/right context)
to find optimal model performance, as described in the leech training strategy.
"""

import csv
import dataclasses
import itertools
import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.table import Table

from leech.chunking import load_chunks
from leech.cli_config import make_console
from leech.configs import TrainConfig
from leech.context_grid import parse_context_grid, parse_values  # noqa: F401  (re-export)
from leech.metrics import (
    PARAMETRIC_SELECTION_METRICS,
    PLAIN_SELECTION_METRICS,
    parse_selection_metric,
)
from leech.models import requires_features
from leech.training import train_model

logger = logging.getLogger("leech.gridsearch")
console = make_console()


_VALID_SELECTION_METRICS = PLAIN_SELECTION_METRICS


def _resolve_selection_metric(metric: str, *, n_classes: int) -> str:
    """Resolve selection_metric "auto" to a concrete history key.

    Multiclass (>=3 classes) -> val_f1 (macro-F1, unaffected by class imbalance).
    Binary -> val_auc (threshold-free, robust to one-vs-all imbalance).

    Also accepts the two parametric metrics Trainer does ("tpr_at_fpr:<f>" /
    "callable_at_precision:<p>", issue #280) -- binary-only, same restriction
    Trainer's own ``_resolve_checkpoint_metric`` enforces, since each grid
    point's checkpoint_metric is set to this function's return value
    (run_grid_point) and would otherwise fail deep inside training instead of
    before the sweep starts.
    """
    if metric in _VALID_SELECTION_METRICS:
        if metric != "auto":
            return metric
        return "val_f1" if n_classes > 2 else "val_auc"
    kind, _param = parse_selection_metric(metric)
    if kind not in PARAMETRIC_SELECTION_METRICS:
        raise ValueError(
            f"selection_metric must be one of {_VALID_SELECTION_METRICS} or a "
            f"parametric metric ({PARAMETRIC_SELECTION_METRICS}, each with a "
            f"':<float>' suffix, e.g. 'tpr_at_fpr:0.0034'), got {metric!r}"
        )
    if n_classes > 2:
        raise ValueError(
            f"selection_metric={metric!r} is binary-only -- this training "
            f"corpus has {n_classes} classes. Use 'val_f1' or 'val_auc' "
            "instead."
        )
    return metric


@dataclass
class GridSearchConfig:
    """
    Configuration for chunk context grid search.

    The training recipe (epochs, learning_rate, motif, augmentation, aux
    heads, ...) lives in ``cfg: TrainConfig`` (#270) instead of being
    redeclared here as ~30 individual fields -- this class only adds what a
    grid search needs beyond one training recipe: which geometry to sweep and
    how to run the sweep.

    Attributes:
        train_data_path: Path to training chunks or BAM/POD5 for preparation
        val_data_path: Path to validation chunks or BAM/POD5
        model_name: Model architecture to use
        output_dir: Base output directory for all grid results
        left_contexts: List of left context sizes to test
        right_contexts: List of right context sizes to test
        cfg: Training recipe shared by every grid point, except where
            run_grid_point derives a point-specific override (left_context,
            right_context, the resolved selection_metric/pos_weight) via
            dataclasses.replace
        kmer_context: K-mer context for sequence encoding
        device: Device for training
        seed: Random seed
        dwell_offsets: Dwell offset values to sweep (None = [0])
        n_parallel: Number of grid points to run concurrently
        num_workers: DataLoader workers per grid point
        selection_metric: Grid-point ranking criterion ("auto" mirrors the
            training checkpoint criterion: val_f1 for multiclass, val_auc for
            binary)
    """

    train_data_path: Path
    val_data_path: Path | None
    model_name: str
    output_dir: Path
    left_contexts: list[int]
    right_contexts: list[int]
    cfg: TrainConfig = field(default_factory=TrainConfig)
    kmer_context: int = 5
    device: str = "cuda"
    seed: int | None = None  # None = generate random seed
    dwell_offsets: list[int] | None = None
    n_parallel: int = 1
    num_workers: int = 0
    selection_metric: str = "auto"


def _training_label_column(path: Path) -> np.ndarray:
    """Every valid ``label_int`` in a chunk corpus, without loading its arrays.

    One npz member, so the signals and features -- the whole reason a corpus is
    tens of gigabytes -- never enter the process. A negative label is the
    missing-value sentinel, the same one ``ChunkTable`` and ``LeechDataset``
    filter on. Corpora that predate the flat array format need the full read.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            labels = data["labels_int"]
    except (KeyError, ValueError):
        return np.array([c["label_int"] for c in load_chunks(path) if c["label_int"] is not None])
    return labels[labels >= 0]


def run_grid_point(
    train_data_path: Path,
    val_data_path: Path | None,
    model_name: str,
    output_dir: Path,
    left_context: int,
    right_context: int,
    kmer_len: int,
    device: str,
    seed: int,
    cfg: TrainConfig,
    dwell_offset: int = 0,
    pos_weight: float | None = None,
    num_workers: int = 0,
    selection_metric: str = "auto",
) -> dict:
    """
    Train model for a single grid point.

    Args:
        train_data_path: Path to training data
        val_data_path: Path to validation data
        model_name: Model architecture
        output_dir: Output directory for this grid point
        left_context: Left signal context
        right_context: Right signal context
        kmer_len: K-mer length
        device: Device
        seed: Random seed
        cfg: Training recipe shared across the grid (#270); this point's
            left_context/right_context and the resolved selection_metric/
            pos_weight are applied to it via dataclasses.replace below, so
            train_model receives one complete, point-specific cfg rather
            than ~30 individually forwarded fields.
        dwell_offset: Dwell feature offset (bases toward 3' end)
        pos_weight: Pre-computed positive class weight (avoids redundant computation)

    Returns:
        Dictionary with grid point results
    """
    signal_len = left_context + right_context

    logger.info(f"\n{'=' * 80}")
    logger.info(
        f"Training grid point: left={left_context}, right={right_context}, dwell_offset={dwell_offset}"
    )
    logger.info(f"Signal length: {signal_len}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'=' * 80}\n")

    start_time = time.time()

    # checkpoint_metric/pos_weight are resolved once in run_grid_search from
    # data/config that doesn't vary per point; left_context/right_context are
    # this point's actual swept values. All four fold into one point-specific
    # cfg rather than reaching train_model as loose kwargs.
    point_cfg = dataclasses.replace(
        cfg,
        left_context=left_context,
        right_context=right_context,
        checkpoint_metric=selection_metric,
        pos_weight=pos_weight,
    )

    try:
        history = train_model(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            model_name=model_name,
            output_dir=output_dir,
            signal_len=signal_len,
            kmer_len=kmer_len,
            device=device,
            seed=seed,
            dwell_offset=dwell_offset,
            num_workers=num_workers,
            cfg=point_cfg,
        )

        train_time = time.time() - start_time

        # Extract best metrics. best_epoch is keyed off the selection metric so
        # the recorded epoch matches the model that gets checkpointed in training.
        # `selection_metric` is the resolved key (run_grid_search collapses "auto"
        # to val_f1 / val_auc / val_acc, or a parametric metric, before this point).
        best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
        best_val_auc = max(history["val_auc"]) if history["val_auc"] else 0.0
        best_val_f1 = max(history["val_f1"]) if history["val_f1"] else 0.0
        # Trainer's history carries "val_selection" -- whatever
        # selection_metric resolved to, every epoch -- which is what makes a
        # parametric metric's best_epoch derivable at all: history has no key
        # literally named "tpr_at_fpr:0.0034". Falls back to the old
        # dict-key-by-name lookup for a history that predates val_selection
        # (a mocked train_model in tests), where it still works for the
        # three plain metrics.
        epoch_series = (
            history.get("val_selection") or history.get(selection_metric) or history.get("val_acc")
        )
        best_epoch = int(np.argmax(epoch_series) + 1) if epoch_series else cfg.epochs
        best_val_selection = max(history["val_selection"]) if history.get("val_selection") else 0.0

        result = {
            "left_context": left_context,
            "right_context": right_context,
            "dwell_offset": dwell_offset,
            "signal_len": signal_len,
            "best_val_acc": best_val_acc,
            "best_val_auc": best_val_auc,
            "best_val_f1": best_val_f1,
            "best_val_selection": best_val_selection,
            "best_val_loss": min(history["val_loss"]) if history["val_loss"] else 0.0,
            "best_epoch": best_epoch,
            "final_train_acc": history["train_acc"][-1] if history["train_acc"] else 0.0,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0.0,
            "train_time_sec": train_time,
            "model_path": str(output_dir / "model_best.pt"),
            "selection_metric": selection_metric,
            "status": "success",
        }

    except Exception as e:
        logger.exception(f"Training failed for grid point: {e}")
        result = {
            "left_context": left_context,
            "right_context": right_context,
            "dwell_offset": dwell_offset,
            "signal_len": left_context + right_context,
            "status": "failed",
            "error": str(e),
        }

    return result


_WORKER_DEVICE: str | None = None


def assign_worker_devices(device: str, n_workers: int) -> list[str]:
    """Return one device string per pool worker.

    ``"cuda"`` is spread round-robin over the visible GPUs; an explicit
    ``cuda:K`` or ``cpu`` is passed through unchanged for every worker.
    """
    if device != "cuda":
        return [device] * n_workers
    import torch

    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        return [device] * n_workers
    return [f"cuda:{i % n_gpus}" for i in range(n_workers)]


def _init_grid_worker(device_queue) -> None:
    """Pool initializer: claim this worker's fixed device."""
    global _WORKER_DEVICE
    _WORKER_DEVICE = device_queue.get()


def _grid_point_worker(args: dict) -> dict:
    """Run a single grid point.

    Each grid point re-reads the corpus rather than sharing a pre-loaded copy.
    The cache this replaces held the whole decompressed corpus in every pool
    process, which forced ``LeechDataset`` down its eager ``chunks=`` branch --
    numpy arrays resident alongside the tensors built from them, times
    ``--parallel N``. The streaming loader reads a row block at a time and this
    storage does 724 MB/s sequentially (ADR 0006), so the re-read is cheap
    where the resident copy was not.

    Forwards ``args`` wholesale rather than re-listing each key: hand-copying
    this call used to silently drop ``oversample_minority`` under
    ``--parallel > 1`` (#270) while the sequential path (``run_grid_point(**args)``
    in ``run_grid_search``) forwarded everything -- the same ``grid_args`` dict
    reaches both dispatchers, so a key present there can no longer go missing
    in just one of them.
    """
    if _WORKER_DEVICE is not None:
        args = {**args, "device": _WORKER_DEVICE}
    result = run_grid_point(**args)
    # Free CUDA memory between grid points to prevent accumulation
    if args.get("device", "cpu") != "cpu":
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def run_grid_search(config: GridSearchConfig) -> Path:
    """
    Run grid search over chunk contexts.

    Args:
        config: Grid search configuration

    Returns:
        Path to grid search summary CSV
    """
    from leech.constants import generate_random_seed

    # Generate random seed if not provided
    if config.seed is None:
        seed = generate_random_seed()
        logger.info(f"Generated random seed: {seed}")
        config.seed = seed
    else:
        seed = config.seed
        logger.info(f"Using provided seed: {seed}")

    # Default dwell_offsets to [0] if not provided
    dwell_offsets = config.dwell_offsets if config.dwell_offsets is not None else [0]

    # Skip dwell_offset grid for models without a feature branch
    if not requires_features(config.model_name):
        if dwell_offsets != [0]:
            logger.info(
                f"Model {config.model_name} has no feature branch; collapsing dwell_offsets to [0]"
            )
            dwell_offsets = [0]

    logger.info("=" * 80)
    logger.info("Starting Grid Search")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Left contexts: {config.left_contexts}")
    logger.info(f"Right contexts: {config.right_contexts}")
    logger.info(f"Dwell offsets: {dwell_offsets}")
    logger.info(
        f"Total grid points: {len(config.left_contexts) * len(config.right_contexts) * len(dwell_offsets)}"
    )
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Random seed: {seed}")
    logger.info("=" * 80)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save seed to file
    seed_file = config.output_dir / "grid_search_seed.txt"
    with open(seed_file, "w") as f:
        f.write(f"{seed}\n")

    # Save grid search config
    config_dict = {
        "model_name": config.model_name,
        "left_contexts": config.left_contexts,
        "right_contexts": config.right_contexts,
        "dwell_offsets": dwell_offsets,
        "kmer_context": config.kmer_context,
        "epochs": config.cfg.epochs,
        "batch_size": config.cfg.batch_size,
        "learning_rate": config.cfg.optim.learning_rate,
        "device": config.device,
        "seed": seed,
    }

    with open(config.output_dir / "grid_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Pre-compute class weights from the training labels (they don't change
    # across grid points). Only the label column is needed, so read that --
    # not the corpus.
    labels_array = _training_label_column(config.train_data_path)
    logger.info(f"Read {len(labels_array)} training labels")
    unique, counts = np.unique(labels_array, return_counts=True)
    pos_weight = None
    if len(unique) == 2:
        label_counts = dict(zip(unique, counts, strict=True))
        neg_count = label_counts.get(0, 0)
        pos_count = label_counts.get(1, 0)
        if pos_count > 0:
            pos_weight = neg_count / pos_count
            logger.info(
                f"Pre-computed class weights: negative={neg_count}, "
                f"positive={pos_count}, pos_weight={pos_weight:.4f}"
            )

    # Resolve selection_metric "auto" once so every grid point uses the same
    # criterion: macro-F1 for multiclass, AUROC for binary (AUROC is threshold-
    # free and not gamed by majority-class prediction on imbalanced heads).
    resolved_metric = _resolve_selection_metric(config.selection_metric, n_classes=len(unique))
    logger.info(f"Selection metric: {config.selection_metric!r} -> {resolved_metric!r}")

    # Generate grid
    grid_points = list(
        itertools.product(config.left_contexts, config.right_contexts, dwell_offsets)
    )
    summary_path = config.output_dir / "grid_summary.csv"

    # Build grid point argument dicts
    grid_args = []
    for left, right, dwoff in grid_points:
        if len(dwell_offsets) > 1 or dwell_offsets != [0]:
            grid_output_dir = config.output_dir / f"left_{left}_right_{right}_dwoff_{dwoff}"
        else:
            grid_output_dir = config.output_dir / f"left_{left}_right_{right}"

        grid_args.append(
            {
                "train_data_path": config.train_data_path,
                "val_data_path": config.val_data_path,
                "model_name": config.model_name,
                "output_dir": grid_output_dir,
                "left_context": left,
                "right_context": right,
                "kmer_len": 2 * config.kmer_context + 1,
                "device": config.device,
                "seed": config.seed,
                "cfg": config.cfg,
                "dwell_offset": dwoff,
                "pos_weight": pos_weight,
                "num_workers": config.num_workers,
                "selection_metric": resolved_metric,
            }
        )

    results: list[dict] = []

    if config.n_parallel > 1:
        # Parallel execution: each worker streams the corpus per grid point
        logger.info(
            f"Running {len(grid_points)} grid points with {config.n_parallel} parallel workers"
        )
        # CUDA does not support fork; use spawn to avoid hangs with multiple GPU workers
        ctx = multiprocessing.get_context("spawn" if config.device != "cpu" else "fork")
        worker_devices = assign_worker_devices(config.device, config.n_parallel)
        logger.info(
            "Worker -> device mapping: "
            + ", ".join(f"worker {i}: {d}" for i, d in enumerate(worker_devices))
        )
        device_queue = ctx.Queue()
        for d in worker_devices:
            device_queue.put(d)
        # ProcessPoolExecutor workers are not daemonic, so a grid point may
        # start DataLoader workers (and --gpus > 1).
        with ProcessPoolExecutor(
            max_workers=config.n_parallel,
            mp_context=ctx,
            initializer=_init_grid_worker,
            initargs=(device_queue,),
        ) as pool:
            futures = {pool.submit(_grid_point_worker, a): a for a in grid_args}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    result = fut.result()
                except Exception as e:
                    a = futures[fut]
                    logger.error(f"Grid point failed: {e}")
                    result = {
                        "left_context": a["left_context"],
                        "right_context": a["right_context"],
                        "dwell_offset": a["dwell_offset"],
                        "status": "failed",
                        "error": str(e),
                    }
                results.append(result)
                logger.info(f"Completed grid point {i}/{len(grid_points)}")
                save_grid_summary(results, summary_path)
    else:
        # Sequential execution. Grid points must not share a chunk list:
        # LeechDataset drops each chunk's arrays once it has tensorized them,
        # so the second point got chunks whose `signal` was None and died with
        # "'NoneType' object has no attribute 'dtype'".
        for i, args in enumerate(grid_args, 1):
            logger.info(f"\n\nGrid point {i}/{len(grid_points)}")
            result = run_grid_point(**args)
            results.append(result)
            save_grid_summary(results, summary_path)

    # Print summary with Rich tables
    console.print("\n[bold green]Grid Search Complete![/bold green]\n")

    successful_results = [r for r in results if r.get("status") == "success"]

    if not successful_results:
        logger.error(f"All {len(results)} grid points failed")
        raise RuntimeError(f"Grid search failed: all {len(results)} grid points failed")

    if successful_results:
        # Map history-key metric -> result-dict key produced by run_grid_point.
        # A parametric metric (tpr_at_fpr:<f> / callable_at_precision:<p>) has
        # no dedicated best_val_* column of its own -- it ranks on the generic
        # best_val_selection field instead (populated for every metric kind).
        sort_key = {
            "val_acc": "best_val_acc",
            "val_f1": "best_val_f1",
            "val_auc": "best_val_auc",
        }.get(resolved_metric, "best_val_selection")

        # Create results table; tag the column we ranked on with [*]. A
        # parametric metric marks none of the three plain columns -- its
        # value shows in the always-present "Selection" column instead.
        col_marker = {
            "best_val_acc": ("Val Accuracy[*]", "Val F1", "Val AUC"),
            "best_val_f1": ("Val Accuracy", "Val F1[*]", "Val AUC"),
            "best_val_auc": ("Val Accuracy", "Val F1", "Val AUC[*]"),
        }.get(sort_key, ("Val Accuracy", "Val F1", "Val AUC"))
        selection_col = f"{resolved_metric}[*]" if sort_key == "best_val_selection" else "Selection"
        table = Table(title="Grid Search Results", show_header=True, header_style="bold magenta")
        table.add_column("Left Context", justify="right", style="cyan")
        table.add_column("Right Context", justify="right", style="cyan")
        table.add_column("Dwell Offset", justify="right", style="cyan")
        table.add_column(col_marker[0], justify="right", style="green")
        table.add_column(col_marker[1], justify="right", style="green")
        table.add_column(col_marker[2], justify="right", style="yellow")
        table.add_column(selection_col, justify="right", style="magenta")
        table.add_column("Best Epoch", justify="right", style="blue")
        table.add_column("Training Time", justify="right", style="white")

        sorted_results = sorted(successful_results, key=lambda x: x.get(sort_key, 0), reverse=True)

        for r in sorted_results[:10]:  # Show top 10
            table.add_row(
                str(r["left_context"]),
                str(r["right_context"]),
                str(r.get("dwell_offset", 0)),
                f"{r.get('best_val_acc', 0):.4f}",
                f"{r.get('best_val_f1', 0):.4f}",
                f"{r.get('best_val_auc', 0):.4f}",
                f"{r.get('best_val_selection', 0):.4f}",
                str(r.get("best_epoch", 0)),
                f"{r.get('train_time_sec', 0):.1f}s",
                style="bold" if r == sorted_results[0] else None,
            )

        console.print(table)

        # Best configuration summary
        best_result = sorted_results[0]
        summary_table = Table(
            title="Best Configuration", show_header=True, header_style="bold magenta"
        )
        summary_table.add_column("Parameter", style="cyan")
        summary_table.add_column("Value", justify="right", style="green")

        summary_table.add_row("Selection Metric", f"{config.selection_metric} -> {resolved_metric}")
        summary_table.add_row("Left Context", str(best_result["left_context"]))
        summary_table.add_row("Right Context", str(best_result["right_context"]))
        summary_table.add_row("Dwell Offset", str(best_result.get("dwell_offset", 0)))
        summary_table.add_row("Validation Accuracy", f"{best_result['best_val_acc']:.4f}")
        summary_table.add_row("Validation F1", f"{best_result.get('best_val_f1', 0):.4f}")
        summary_table.add_row("Validation AUC", f"{best_result.get('best_val_auc', 0):.4f}")
        if sort_key == "best_val_selection":
            summary_table.add_row(
                f"Validation {resolved_metric}", f"{best_result.get('best_val_selection', 0):.4f}"
            )
        summary_table.add_row("Best Epoch", str(best_result.get("best_epoch", 0)))
        summary_table.add_row("Model Path", str(best_result["model_path"]))

        console.print(summary_table)

        best_params_path = config.output_dir / "best_params.json"
        with open(best_params_path, "w") as f:
            json.dump(
                {
                    "left_context": best_result["left_context"],
                    "right_context": best_result["right_context"],
                    "dwell_offset": best_result.get("dwell_offset", 0),
                    "selection_metric": resolved_metric,
                },
                f,
                indent=2,
            )
        console.print(f"[bold]Best params saved to:[/bold] {best_params_path}")

    console.print(f"\n[bold]Results saved to:[/bold] {summary_path}")

    return summary_path


def save_grid_summary(results: list[dict], output_path: Path) -> None:
    """
    Save grid search results to CSV.

    Args:
        results: List of result dictionaries
        output_path: Output CSV path
    """
    if not results:
        return

    # Define columns
    columns = [
        "left_context",
        "right_context",
        "dwell_offset",
        "signal_len",
        "best_val_acc",
        "best_val_f1",
        "best_val_auc",
        "best_val_selection",
        "best_val_loss",
        "best_epoch",
        "final_train_acc",
        "final_train_loss",
        "train_time_sec",
        "model_path",
        "selection_metric",
        "status",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
