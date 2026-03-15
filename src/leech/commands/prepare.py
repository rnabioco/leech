"""
Handler for the 'prepare' command.

This module contains the business logic for preparing training data from
POD5 and BAM files, including parallel processing, splitting, and result display.
"""

import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from leech.constants import DEFAULT_SEED

logger = logging.getLogger("leech.commands.prepare")
console = Console()


def handle_prepare(
    pod5: Path,
    bam: Path,
    output_dir: Path,
    motif: str | None = None,
    motif_offset: int = 0,
    motif_reference: str = "fasta",
    reference_fasta: Path | None = None,
    skip_motif_indels: bool = False,
    label: str | None = None,
    min_mapq: int = 0,
    feature_set: str = "signal+dwell+levels",
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int | None = DEFAULT_SEED,
    no_split: bool = False,
    workers: int = 8,
    chunk_size: int = 100,
    base_justify: str = "center",
    feature_start: int | None = None,
    feature_end: int | None = None,
    reverse_signal: bool = True,
    anchor: str = "reference",
    signal_norm: str = "median_mad",
    pa_mean: float | None = None,
    pa_stdev: float | None = None,
    refine_signal_map: bool = True,
    kmer_table: Path | None = None,
    scale_iters: int = 0,
) -> dict[str, Any]:
    """
    Handle the prepare command logic.

    Args:
        pod5: Path to POD5 file with raw signal
        bam: Path to BAM file with alignments and mv tags
        output_dir: Output directory for training chunks
        motif: Sequence motif to extract (e.g., "CCAGGC")
        motif_offset: Offset within motif for focus base
        motif_reference: Where to search for motif ("fasta" or "bam")
        reference_fasta: External reference FASTA file
        skip_motif_indels: Skip reads with indels in motif region
        label: Label identifier for this sample
        min_mapq: Minimum mapping quality
        feature_set: Feature set to extract
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        seed: Random seed for reproducibility
        no_split: Extract chunks without splitting
        workers: Number of parallel workers
        chunk_size: Number of reads to process per worker batch

    Returns:
        Dictionary with extraction statistics
    """
    from leech.chunking import save_chunks
    from leech.configs import ChunkConfig, LabelConfig, MotifConfig, PrepareConfig, SignalConfig
    from leech.io import get_reference_sequences
    from leech.preparation import prepare_training_data_parallel, prepare_training_data_with_split
    from leech.splitting import split_chunks_by_read
    from leech.util import setup_random_seed

    logger.info(f"Preparing data from {pod5} and {bam}")
    logger.info(f"Motif reference mode: {motif_reference}")
    logger.info(f"Anchor mode: {anchor}, Signal norm: {signal_norm}")

    # Setup signal refiner if requested
    signal_refiner = None
    if refine_signal_map:
        from leech.signal_refine import SigMapRefiner

        if kmer_table is None:
            from leech.data import get_kmer_table

            kmer_table = get_kmer_table()
        signal_refiner = SigMapRefiner.from_table(kmer_table, scale_iters=scale_iters)
        logger.info(f"Signal map refinement enabled with kmer table: {kmer_table}")
    if workers > 1:
        logger.info(f"Parallel mode: {workers} workers, {chunk_size} reads per batch")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference sequences if using reference-based motif search
    reference_sequences = None
    if motif_reference == "fasta":
        logger.info("Loading reference sequences for reference-based motif search")
        reference_sequences = get_reference_sequences(bam, reference_fasta)

    # Construct PrepareConfig from CLI kwargs
    config = PrepareConfig(
        pod5_path=pod5,
        signal=SignalConfig(
            reverse_signal=reverse_signal,
            anchor=anchor,
            norm_method=signal_norm,
            pa_mean=pa_mean,
            pa_stdev=pa_stdev,
            refine_signal_map=refine_signal_map,
            signal_refiner=signal_refiner,
        ),
        motif=MotifConfig(
            motif=motif,
            motif_offset=motif_offset,
            motif_reference=motif_reference,
            reference_sequences=reference_sequences,
            skip_motif_indels=skip_motif_indels,
        ),
        chunk=ChunkConfig(
            base_justify=base_justify,
            feature_start=feature_start,
            feature_end=feature_end,
        ),
        labeling=LabelConfig(
            label=label,
            label_int=None,  # Will be assigned during merge-and-split
        ),
    )

    # Write preparation config sidecar for downstream provenance
    import json as _json

    prepare_config_path = output_dir / "prepare_config.json"
    with open(prepare_config_path, "w") as f:
        _json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Saved preparation config to {prepare_config_path}")

    # Extract chunks (parallel or sequential)
    if workers > 1:
        # Parallel processing
        logger.info("Extracting chunks in parallel...")
        chunks, stats = prepare_training_data_parallel(
            bam_path=bam,
            config=config,
            num_workers=workers,
            chunk_size=chunk_size,
            min_mapq=min_mapq,
        )

        if len(chunks) == 0:
            reads_processed = stats.get("total_reads", 0)
            logger.warning(
                f"0 chunks extracted from {reads_processed} reads. "
                f"Common causes: indels at motif site, insufficient context, "
                f"or MAPQ filtering (--min-mapq={min_mapq})."
            )
            console.print(
                f"[bold red]Warning: 0 chunks extracted from {reads_processed} reads.[/bold red]\n"
                f"[yellow]Stats: {stats}[/yellow]"
            )
            return {
                "n_chunks": 0,
                "n_train": 0,
                "n_val": 0,
                "n_test": 0,
            }

        # Setup seed and handle splitting/saving
        setup_random_seed(seed, output_dir)

        if no_split:
            all_file = output_dir / "all.npz"
            save_chunks(chunks, all_file)
            logger.info(f"Saved all chunks to {all_file}")
            result = {
                "n_chunks": len(chunks),
                "n_train": 0,
                "n_val": 0,
                "n_test": 0,
            }
        else:
            train_chunks, val_chunks, test_chunks = split_chunks_by_read(
                chunks, train_frac=train_split, val_frac=val_split, seed=seed
            )

            if train_chunks:
                train_file = output_dir / "train.npz"
                save_chunks(train_chunks, train_file)
                logger.info(f"Saved {len(train_chunks)} train chunks to {train_file}")

            if val_chunks:
                val_file = output_dir / "val.npz"
                save_chunks(val_chunks, val_file)
                logger.info(f"Saved {len(val_chunks)} val chunks to {val_file}")

            if test_chunks:
                test_file = output_dir / "test.npz"
                save_chunks(test_chunks, test_file)
                logger.info(f"Saved {len(test_chunks)} test chunks to {test_file}")

            result = {
                "n_chunks": len(chunks),
                "n_train": len(train_chunks),
                "n_val": len(val_chunks),
                "n_test": len(test_chunks),
            }
    else:
        # Sequential processing with refactored function
        from rich.progress import Progress, TaskID

        progress_container: dict[str, Progress | TaskID | None] = {"progress": None, "task": None}

        def update_progress(n_chunks):
            prog = progress_container["progress"]
            task = progress_container["task"]
            if prog is not None and task is not None and isinstance(prog, Progress):
                prog.update(
                    task,
                    advance=1,
                    description=f"[cyan]Extracted {n_chunks} chunks...",
                )

        with Progress(console=console) as progress:
            progress_container["progress"] = progress
            progress_container["task"] = progress.add_task("[cyan]Extracting chunks...", total=None)

            # Ensure seed is int for the function call
            from leech.constants import generate_random_seed

            actual_seed: int
            if seed is not None:
                actual_seed = seed
            elif DEFAULT_SEED is not None:
                actual_seed = DEFAULT_SEED
            else:
                actual_seed = generate_random_seed()

            result = prepare_training_data_with_split(
                bam_path=bam,
                config=config,
                output_dir=output_dir,
                reference_fasta=reference_fasta,
                min_mapq=min_mapq,
                feature_set=feature_set,
                train_split=train_split,
                val_split=val_split,
                seed=actual_seed,
                no_split=no_split,
                progress_callback=update_progress,
            )

            task_id = progress_container["task"]
            if task_id is not None:
                progress.update(task_id, completed=True)

    # Display results
    _display_prepare_results(result, no_split)

    return result


def _display_prepare_results(result: dict[str, Any], no_split: bool) -> None:
    """Display results of the prepare command."""
    console.print(f"[green]Extracted {result['n_chunks']} training chunks[/green]")

    if no_split:
        console.print(
            "[yellow]Skipped splitting (--no-split). All chunks saved to all.npz[/yellow]"
        )
    else:
        table = Table(
            title="Data Split (Read-Level)", show_header=True, header_style="bold magenta"
        )
        table.add_column("Split", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="yellow")

        n_total = result["n_chunks"]
        table.add_row("Train", str(result["n_train"]), f"{result['n_train'] / n_total * 100:.1f}%")
        table.add_row("Validation", str(result["n_val"]), f"{result['n_val'] / n_total * 100:.1f}%")
        table.add_row("Test", str(result["n_test"]), f"{result['n_test'] / n_total * 100:.1f}%")
        table.add_row("Total", str(n_total), "100.0%", style="bold")

        console.print(table)

    console.print("[bold green]Data preparation complete![/bold green]")
