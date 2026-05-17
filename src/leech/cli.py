"""
Command-line interface for leech.

Designed for Snakemake integration with clear input/output paths.
"""

import logging
from importlib.metadata import version as pkg_version
from pathlib import Path

import rich_click as click

from leech.cli_config import configure_rich_click, console
from leech.cli_options import LazyChoice, get_model_choices, model_provenance, training_hyperparams
from leech.constants import DEFAULT_DEVICE, DEFAULT_SEED
from leech.logging_config import setup_logging

# Apply rich-click styling
configure_rich_click()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=pkg_version("leech"), prog_name="leech")
def cli():
    """LEECH - Learning Enhanced Electrical Classifiers from Hanopore signals

    A workflow-based CLI for nanopore signal classification:

    • leech data     - Prepare and process training data
    • leech model    - Train and optimize models
    • leech eval     - Evaluate and analyze models
    • leech predict  - Run inference on new data
    """
    setup_logging(level=logging.INFO)


# ============================================================================
# DATA PREPARATION COMMANDS
# ============================================================================


@cli.group()
def data():
    """Prepare and process training data from POD5/BAM files."""
    pass


@data.command()
@click.option(
    "--pod5",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="POD5 file with raw signal",
)
@click.option(
    "--bam",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="BAM file with alignments and mv tags",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for training chunks",
)
@click.option(
    "--motif",
    type=str,
    default=None,
    help='Sequence motif to extract (e.g., "CCAGGC" for tRNA 3\' end)',
)
@click.option(
    "--motif-offset",
    type=int,
    default=0,
    help="Offset within motif for focus base (0-indexed)",
)
@click.option(
    "--motif-reference",
    type=click.Choice(["fasta", "bam"]),
    default="fasta",
    help='Where to search for motif: "fasta" (reference sequence, recommended) or "bam" (basecalled sequence)',
)
@click.option(
    "--reference-fasta",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="External reference FASTA file (if not embedded in BAM @SQ header)",
)
@click.option(
    "--skip-motif-indels/--no-skip-motif-indels",
    default=False,
    help="Skip reads with indels in motif region (for motif-reference=fasta)",
)
@click.option(
    "--label",
    type=str,
    default=None,
    help="Label identifier for this sample (e.g., 'Ala', 'Gly', 'charged', 'uncharged'). Numeric labels (0/1) are assigned during merge-and-split for pairwise comparisons.",
)
@click.option(
    "--min-mapq",
    type=int,
    default=0,
    help="Minimum mapping quality",
)
@click.option(
    "--feature-set",
    type=click.Choice(["signal", "signal+dwell", "signal+levels", "signal+dwell+levels"]),
    default="signal+dwell+levels",
    help="Feature set to extract",
)
@click.option(
    "--train-split",
    type=float,
    default=0.7,
    help="Fraction of data for training",
)
@click.option(
    "--val-split",
    type=float,
    default=0.15,
    help="Fraction of data for validation",
)
@click.option(
    "--seed",
    type=int,
    default=DEFAULT_SEED,
    help="Random seed for reproducibility",
)
@click.option(
    "--no-split",
    is_flag=True,
    default=False,
    help="Extract chunks without splitting (for later merge-then-split workflow)",
)
@click.option(
    "--no-compress",
    is_flag=True,
    default=False,
    help="Save uncompressed NPZ (faster writes, larger files). Useful for intermediate outputs that will be merged later.",
)
@click.option(
    "--workers",
    type=int,
    default=8,
    help="Number of parallel workers for data processing (1=sequential, >1=parallel)",
)
@click.option(
    "--chunk-size",
    type=int,
    default=100,
    help="Number of reads to process per worker batch (for parallel processing)",
)
@click.option(
    "--base-justify",
    type=click.Choice(["start", "center", "end"]),
    default="center",
    help='Where to center signal chunk within the focus base: "start" (first sample), "center" (midpoint, default), or "end" (last sample, useful for 3\' modifications)',
)
@click.option(
    "--feature-start",
    type=int,
    default=None,
    help="Feature window start as signed offset from focus base. Negative=left, 0=at focus. Default: -kmer_context (-5).",
)
@click.option(
    "--feature-end",
    type=int,
    default=None,
    help="Feature window end as signed offset from focus base (inclusive). Default: kmer_context (5). E.g., --feature-start 0 --feature-end 20 for right-only.",
)
@click.option(
    "--no-reverse-signal",
    is_flag=True,
    default=False,
    help="Do NOT reverse the raw signal. By default, signal is reversed for direct RNA sequencing (POD5 stores 3'→5', basecaller expects 5'→3'). Use this flag for DNA data.",
)
@click.option(
    "--anchor",
    type=click.Choice(["basecall", "reference"]),
    default="reference",
    help='Anchor mode: "reference" (default) uses ref sequence + ref->signal mapping via CIGAR, "basecall" uses basecalled sequence',
)
@click.option(
    "--signal-norm",
    type=click.Choice(["median_mad", "zscore", "quantile", "pa_scaling"]),
    default="median_mad",
    help="Signal normalization method",
)
@click.option(
    "--pa-mean",
    type=float,
    default=None,
    help="Global shift for pa_scaling normalization (from basecaller model)",
)
@click.option(
    "--pa-stdev",
    type=float,
    default=None,
    help="Global scale for pa_scaling normalization (from basecaller model)",
)
@click.option(
    "--refine-signal-map/--no-refine-signal-map",
    default=True,
    help="Apply signal map refinement using kmer level tables (default: enabled). Use --no-refine-signal-map for DNA data.",
)
@click.option(
    "--kmer-table",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to kmer level table for signal map refinement (e.g., rna004_9mer_levels.txt)",
)
@click.option(
    "--scale-iters",
    type=int,
    default=2,
    help="Signal map refinement iterations: -1=rescale only (no DP), 0=one round of banded DP, >0=N rounds with rescaling",
)
@click.option(
    "--rough-rescale/--no-rough-rescale",
    default=True,
    help="Rough-rescale signal to match kmer table before refinement. "
    "Disable when using an experiment-specific table already in native scale.",
)
@click.option(
    "--signal-context",
    nargs=2,
    type=int,
    default=None,
    help="Asymmetric signal context window as LEFT RIGHT (raw samples). E.g., --signal-context 90 450. Default: symmetric (200, 200).",
)
@click.option(
    "--focus-tsv",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=(
        "TSV with columns `read_id`, `label_int`, `anchor_sample` for per-read "
        "labeling + externally-anchored chunk extraction. When set, only reads "
        "listed in the TSV are kept; each gets its own numeric label and exactly "
        "one chunk centered at the given signal-sample offset (via the read's "
        "move-table). Skips motif search entirely. Intended for pipelines that "
        "pre-detect a region of interest (e.g. adapter midpoint) per read."
    ),
)
@click.option(
    "--recover-softclip-signal/--no-recover-softclip-signal",
    default=False,
    help=(
        "In ref-anchored mode, fill chunk samples that extend past the aligned "
        "region with real soft-clipped signal instead of zeros. Default off "
        "preserves Remora-compatible behavior. Only effective on the Python "
        "(sequential / workers=1) prep path — the Rust path ignores this flag."
    ),
)
def prepare(
    pod5,
    bam,
    output_dir,
    motif,
    motif_offset,
    motif_reference,
    reference_fasta,
    skip_motif_indels,
    label,
    min_mapq,
    feature_set,
    train_split,
    val_split,
    seed,
    no_split,
    no_compress,
    workers,
    chunk_size,
    base_justify,
    feature_start,
    feature_end,
    no_reverse_signal,
    anchor,
    signal_norm,
    pa_mean,
    pa_stdev,
    refine_signal_map,
    kmer_table,
    scale_iters,
    rough_rescale,
    signal_context,
    focus_tsv,
    recover_softclip_signal,
):
    """Prepare training data from POD5 and BAM files."""
    from leech.commands import handle_prepare

    # Validate pa_scaling params
    if signal_norm == "pa_scaling" and (pa_mean is None or pa_stdev is None):
        raise click.UsageError("--signal-norm pa_scaling requires --pa-mean and --pa-stdev")
    if refine_signal_map and kmer_table is None:
        from leech.data import get_kmer_table

        kmer_table = get_kmer_table()
        if not kmer_table.exists():
            raise click.UsageError(
                "--refine-signal-map requires --kmer-table (bundled table not found)"
            )

    handle_prepare(
        pod5=pod5,
        bam=bam,
        output_dir=output_dir,
        motif=motif,
        motif_offset=motif_offset,
        motif_reference=motif_reference,
        reference_fasta=reference_fasta,
        skip_motif_indels=skip_motif_indels,
        label=label,
        min_mapq=min_mapq,
        feature_set=feature_set,
        train_split=train_split,
        val_split=val_split,
        seed=seed,
        no_split=no_split,
        compress=not no_compress,
        workers=workers,
        chunk_size=chunk_size,
        base_justify=base_justify,
        feature_start=feature_start,
        feature_end=feature_end,
        reverse_signal=not no_reverse_signal,
        anchor=anchor,
        signal_norm=signal_norm,
        pa_mean=pa_mean,
        pa_stdev=pa_stdev,
        refine_signal_map=refine_signal_map,
        kmer_table=kmer_table,
        scale_iters=scale_iters,
        rough_rescale=rough_rescale,
        signal_context=signal_context,
        focus_tsv=focus_tsv,
        recover_softclip_signal=recover_softclip_signal,
    )


# ============================================================================
# MODEL TRAINING COMMANDS
# ============================================================================


@cli.group()
def model():
    """Train and optimize models."""
    pass


@data.command()
@click.option(
    "--input-chunks",
    "-i",
    required=True,
    multiple=True,
    type=str,
    help="Input chunk files with labels. Format: label=file.npz (e.g., -i Ala=ala.npz -i Gly=gly.npz or -i basic=lys.npz -i basic=arg.npz -i acidic=asp.npz)",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for split chunks",
)
@click.option(
    "--train-split",
    type=float,
    default=0.7,
    help="Fraction of reads for training",
)
@click.option(
    "--val-split",
    type=float,
    default=0.15,
    help="Fraction of reads for validation",
)
@click.option(
    "--seed",
    type=int,
    default=DEFAULT_SEED,
    help="Random seed for reproducibility",
)
@click.option(
    "--k-fold",
    type=int,
    default=1,
    help="Number of cross-validation folds. When > 1, creates k-fold splits instead of a single train/val/test split. Must be >= 3 when used.",
)
@click.option(
    "--comparison-spec",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="TSV file with comparison specifications (4 columns: meta_label1, label_set1, meta_label2, label_set2). When provided, input-chunks should be directories.",
)
@click.option(
    "--split-by",
    type=str,
    default=None,
    help="NPZ field name to split by group instead of by read (e.g., 'reference_names'). All reads sharing a group value are assigned to the same split.",
)
def merge(
    input_chunks, output_dir, train_split, val_split, seed, k_fold, comparison_spec, split_by
):
    """Merge multiple chunk files and split at read level to prevent data leakage.

    This command implements the correct workflow for multi-sample datasets:
    1. Merge all chunks from different samples
    2. Split merged data at the READ level into train/val/test

    This prevents data leakage that can occur when splitting each sample
    independently and then merging the splits.

    Examples:
        # Pairwise comparison with single labels
        leech data merge -i Ala=ala.npz -i Gly=gly.npz -o merged/

        # Multi-label comparison (chemical properties)
        leech data merge -i basic=lys.npz -i basic=arg.npz -i acidic=asp.npz -o merged/

        # Batch processing from TSV spec
        leech data merge -i chunks/dir1 -i chunks/dir2 --comparison-spec spec.tsv -o merged/

        # 5-fold cross-validation
        leech data merge -i Ala=ala.npz -i Gly=gly.npz -o kfold/ --k-fold 5

        # Split by isodecoder (group-level split)
        leech data merge -i Ala=ala.npz -i Gly=gly.npz -o merged/ --split-by reference_names
    """
    if k_fold > 1:
        from leech.commands import handle_merge_and_split_kfold

        handle_merge_and_split_kfold(
            input_chunks=input_chunks,
            output_dir=output_dir,
            k_fold=k_fold,
            seed=seed,
        )
    else:
        from leech.commands import handle_merge_and_split

        handle_merge_and_split(
            input_chunks=input_chunks,
            output_dir=output_dir,
            train_split=train_split,
            val_split=val_split,
            seed=seed,
            comparison_spec=comparison_spec,
            split_by=split_by,
        )


@model.command()
@click.option(
    "--train-data",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Training dataset config (JSON)",
)
@click.option(
    "--val-data",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Validation dataset config (JSON)",
)
@click.option(
    "--model",
    type=LazyChoice(get_model_choices),
    default="ConvLSTMDwell",
    help="Model architecture",
)
@click.option(
    "--model-config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Model hyperparameters (JSON)",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for model and logs",
)
@training_hyperparams
@model_provenance
@click.option(
    "--use-class-weights/--no-class-weights",
    default=True,
    help="Auto-compute class weights from training data to handle class imbalance (default: enabled)",
)
@click.option(
    "--pos-weight",
    type=float,
    default=None,
    help="Manual positive class weight for BCEWithLogitsLoss (overrides --use-class-weights)",
)
@click.option(
    "--resume",
    type=click.Path(path_type=Path),
    default=None,
    help="Resume training from a checkpoint file (e.g., model_last.pt). Ignored if file doesn't exist.",
)
@click.option(
    "--seq-encoding",
    type=click.Choice(["base_onehot", "signal_kmer"]),
    default="signal_kmer",
    help="Sequence encoding type: signal_kmer (36, signal_len) or base_onehot (4, kmer_len)",
)
@click.option(
    "--num-workers",
    type=int,
    default=0,
    help="DataLoader workers (0=auto: 8 for GPU, 0 for CPU)",
)
@click.option(
    "--balance-groups/--no-balance-groups",
    default=False,
    help="Balance sampling across source groups (e.g., per-AA) so each group contributes equally per epoch",
)
@click.option(
    "--oversample-minority/--no-oversample-minority",
    default=False,
    help="Oversample minority classes so each class contributes equally per epoch "
    "(mutually exclusive with --balance-groups)",
)
@click.option(
    "--num-out",
    type=int,
    default=1,
    help="Number of output classes. 1 = binary (BCE), >1 = multi-class (CrossEntropy). Default: 1.",
)
@click.option(
    "--adversarial-lambda",
    type=float,
    default=0.0,
    help="Gradient reversal strength for adversarial confound removal (0 = disabled).",
)
@click.option(
    "--adversarial-anneal-epochs",
    type=int,
    default=0,
    help="Linearly ramp adversarial lambda from 0 to target over this many epochs (0 = constant).",
)
@click.option(
    "--confound",
    type=click.Choice(["disc_base", "trna_id"]),
    default=None,
    help="Confound to decorrelate via gradient reversal. 'disc_base' = discriminator base at position 73 (4 classes). 'trna_id' = full tRNA isoacceptor identity (N classes, one per unique reference tRNA).",
)
@click.option(
    "--cl-regression/--no-cl-regression",
    default=False,
    help="Enable continuous charging-level (CL) regression head. Uses CL tag from BAM as regression target.",
)
@click.option(
    "--cl-lambda",
    type=float,
    default=1.0,
    help="Weight for CL regression loss (default: 1.0). Combined loss = main + cl_lambda * cl_loss.",
)
@click.option(
    "--signal-mode",
    type=click.Choice(["both", "residual", "signal"]),
    default="both",
    help="Signal input: 'both' (raw+residual, 2ch), 'residual' (1ch), 'signal' (1ch).",
)
@click.option(
    "--dwell-template-table",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="TSV with per-AA per-position expected dwell for 20-channel template features.",
)
def train(
    train_data,
    val_data,
    model,
    model_config,
    output_dir,
    epochs,
    batch_size,
    learning_rate,
    device,
    seed,
    early_stopping,
    use_class_weights,
    pos_weight,
    resume,
    weight_decay,
    max_grad_norm,
    scheduler,
    scheduler_patience,
    scheduler_factor,
    warmup_epochs,
    loss_type,
    focal_gamma,
    label_smoothing,
    mixed_precision,
    augment_jitter,
    augment_scale_min,
    augment_scale_max,
    augment_time_mask_bases,
    augment_time_mask_count,
    augment_shift_max_bases,
    augment_feature_noise_scale,
    motif,
    motif_offset,
    base_justify,
    seq_encoding,
    num_workers,
    balance_groups,
    oversample_minority,
    num_out,
    adversarial_lambda,
    adversarial_anneal_epochs,
    confound,
    cl_regression,
    cl_lambda,
    signal_mode,
    dwell_template_table,
):
    """Train a model on prepared data."""
    from leech.commands.train import handle_train

    handle_train(
        train_data=train_data,
        val_data=val_data,
        model_name=model,
        model_config=model_config,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
        early_stopping=early_stopping,
        use_class_weights=use_class_weights,
        pos_weight=pos_weight,
        resume=resume,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        scheduler=scheduler,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        warmup_epochs=warmup_epochs,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
        mixed_precision=mixed_precision,
        augment_jitter=augment_jitter,
        augment_scale_min=augment_scale_min,
        augment_scale_max=augment_scale_max,
        augment_time_mask_bases=augment_time_mask_bases,
        augment_time_mask_count=augment_time_mask_count,
        augment_shift_max_bases=augment_shift_max_bases,
        augment_feature_noise_scale=augment_feature_noise_scale,
        num_workers=num_workers,
        motif=motif,
        motif_offset=motif_offset,
        base_justify=base_justify,
        seq_encoding=seq_encoding,
        balance_groups=balance_groups,
        oversample_minority=oversample_minority,
        num_out=num_out,
        adversarial_lambda=adversarial_lambda,
        adversarial_anneal_epochs=adversarial_anneal_epochs,
        confound=confound,
        cl_regression=cl_regression,
        cl_lambda=cl_lambda,
        signal_mode=signal_mode,
        dwell_template_table=dwell_template_table,
    )


@model.command()
@click.option(
    "--train-data",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Training dataset (.npz or JSON chunks config) to benchmark against",
)
@click.option(
    "--model",
    "model_name",
    type=LazyChoice(get_model_choices),
    default="ConvLSTMDwell",
    help="Model architecture to benchmark",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory to write benchmark.json (and trace.json if --trace)",
)
@click.option("--batch-size", type=int, default=128, help="Batch size")
@click.option("--device", type=click.Choice(["cuda", "cpu"]), default=DEFAULT_DEVICE, help="Device")
@click.option("--num-steps", type=int, default=100, help="Timed training steps")
@click.option(
    "--warmup-steps",
    type=int,
    default=10,
    help="Warmup steps (cuDNN benchmark, torch.compile, prefetch queue)",
)
@click.option(
    "--num-workers",
    type=int,
    default=8,
    help="DataLoader workers (0 disables multiprocessing)",
)
@click.option("--prefetch-factor", type=int, default=4, help="DataLoader prefetch_factor")
@click.option(
    "--mixed-precision/--no-mixed-precision",
    default=True,
    help="Run with torch.amp autocast + GradScaler",
)
@click.option(
    "--non-blocking/--blocking",
    default=False,
    help="Use .to(device, non_blocking=True) for H2D transfers",
)
@click.option("--signal-len", type=int, default=400, help="Signal length")
@click.option("--kmer-len", type=int, default=11, help="K-mer length")
@click.option(
    "--seq-encoding",
    type=click.Choice(["base_onehot", "signal_kmer"]),
    default="signal_kmer",
    help="Sequence encoding",
)
@click.option(
    "--signal-mode",
    type=click.Choice(["both", "residual", "signal"]),
    default="both",
    help="Signal input channels",
)
@click.option(
    "--trace/--no-trace",
    default=False,
    help="Also collect a torch.profiler Chrome trace (saved to output-dir/trace.json)",
)
@click.option(
    "--trace-active-steps",
    type=int,
    default=10,
    help="Number of active steps captured in the torch.profiler trace",
)
def benchmark(
    train_data,
    model_name,
    output_dir,
    batch_size,
    device,
    num_steps,
    warmup_steps,
    num_workers,
    prefetch_factor,
    mixed_precision,
    non_blocking,
    signal_len,
    kmer_len,
    seq_encoding,
    signal_mode,
    trace,
    trace_active_steps,
):
    """Benchmark one training step: per-phase timing + GPU utilization."""
    from leech.commands.benchmark import handle_benchmark

    handle_benchmark(
        train_data=train_data,
        model_name=model_name,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        mixed_precision=mixed_precision,
        non_blocking=non_blocking,
        signal_len=signal_len,
        kmer_len=kmer_len,
        seq_encoding=seq_encoding,
        signal_mode=signal_mode,
        trace=trace,
        trace_active_steps=trace_active_steps,
    )


@model.command()
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Root dir containing pair subdirectories (each with model_best.pt + config.json)",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output bundle .pt file path",
)
@click.option(
    "--version",
    "-v",
    "bundle_version",
    required=True,
    type=str,
    help='Semantic version string (e.g., "0.1.0-alpha.1")',
)
@click.option(
    "--comparison-type",
    type=click.Choice(["pairwise", "one_vs_all", "group", "multiclass"]),
    default="pairwise",
    help="Comparison type (default: pairwise)",
)
@click.option(
    "--torchscript/--no-torchscript",
    default=False,
    help="Bundle as TorchScript (standalone, no leech needed to load). Default: False.",
)
@click.option(
    "--vmap/--no-vmap",
    default=False,
    help="Bundle with pre-stacked parameters for vectorized inference (requires vmap-compatible architecture, e.g. TCN with GroupNorm/LayerNorm). Default: False.",
)
def bundle(model_dir, output, bundle_version, comparison_type, torchscript, vmap):
    """Bundle trained models into a single versioned file."""
    from leech.commands.bundle import handle_bundle

    if vmap and torchscript:
        raise click.UsageError("--vmap and --torchscript are mutually exclusive")

    handle_bundle(
        model_dir=model_dir,
        output=output,
        bundle_version=bundle_version,
        comparison_type=comparison_type,
        torchscript=torchscript,
        vmap=vmap,
    )


@model.command(name="bundle-info")
@click.option(
    "--bundle",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to bundle .pt file",
)
def bundle_info(bundle):
    """Display metadata from a model bundle."""
    from leech.commands.bundle import handle_bundle_info

    handle_bundle_info(bundle)


@model.command()
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Model directory (with config.json and model_best.pt), or parent with pair subdirs",
)
@click.option(
    "--val-data",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Validation data (.npz) used to learn Platt scaling parameters",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help="Device for inference (default: cpu)",
)
@click.option(
    "--batch-size",
    type=int,
    default=1024,
    help="Batch size for validation pass (default: 1024)",
)
@click.option(
    "--num-workers",
    type=int,
    default=0,
    help="DataLoader workers (default: 0)",
)
@click.option(
    "--method",
    type=click.Choice(["temperature", "matrix", "dirichlet"]),
    default="temperature",
    help="Multiclass calibration method (default: temperature). Binary models always use Platt scaling.",
)
@click.option(
    "--reg-lambda",
    type=float,
    default=0.01,
    help="L2 regularization toward identity for matrix/dirichlet methods (default: 0.01)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for calibration JSON (default: model_dir/calibration.json). Single model only.",
)
def calibrate(model_dir, val_data, device, batch_size, num_workers, method, reg_lambda, output):
    """Learn post-hoc calibration on the validation set.

    Binary models: Platt scaling — fits a, b so sigmoid(a*logit + b) is
    better calibrated. Writes platt.json.

    Multiclass models: Temperature (default), matrix, or Dirichlet scaling.
    Writes calibration.json.

    For a parent directory with pair subdirs (e.g., one_vs_all/Ala_notAla/),
    calibrates each pair independently.
    """
    from leech.commands.calibrate import handle_calibrate

    handle_calibrate(
        model_dir=model_dir,
        val_data=val_data,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        method=method,
        reg_lambda=reg_lambda,
        output=output,
    )


@model.command()
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Model checkpoint directory (with config.json and model_best.pt)",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output TorchScript .pt file path",
)
def export(model_dir, output):
    """Export a trained model as a standalone .pt file.

    The exported file is loadable with just torch.export.load() — no leech
    codebase required. Model config is embedded in the file.
    """
    from leech.commands.bundle import handle_export

    handle_export(model_dir=model_dir, output=output)


@model.command()
@click.option(
    "--train-data",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Training dataset (.npz)",
)
@click.option(
    "--val-data",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Validation dataset (.npz)",
)
@click.option(
    "--model",
    type=LazyChoice(get_model_choices),
    default="ConvLSTMDwell",
    help="Model architecture",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for grid results",
)
@click.option(
    "--context-grid",
    type=str,
    required=False,
    default=None,
    help="Fallback context values when --left-contexts or --right-contexts are not provided. Comma-separated (e.g., '200,500,1000') or range as start:stop:step (e.g., '200:1000:200')",
)
@click.option(
    "--left-contexts",
    type=str,
    default=None,
    help="Override left contexts (comma-separated or start:stop:step). If not provided, uses --context-grid",
)
@click.option(
    "--right-contexts",
    type=str,
    default=None,
    help="Override right contexts (comma-separated or start:stop:step). If not provided, uses --context-grid",
)
@click.option(
    "--kmer-context",
    type=int,
    default=5,
    help="K-mer context for sequence encoding",
)
@training_hyperparams
@model_provenance
@click.option(
    "--dwell-offsets",
    type=str,
    default="0",
    help='Dwell offset values to search (comma-separated or start:stop:step). Shifts dwell/feature window toward 3\' end. Default: "0" (no offset). Requires feature_left >= kmer_context + max_offset.',
)
@click.option(
    "--parallel",
    type=int,
    default=1,
    help="Number of grid points to run concurrently (default: 1, sequential). Each worker loads data independently.",
)
@click.option(
    "--num-workers",
    type=int,
    default=0,
    help="DataLoader workers (0=auto: 8 for GPU, 0 for CPU)",
)
@click.option(
    "--balance-groups/--no-balance-groups",
    default=False,
    help="Balance sampling across source groups (e.g., per-AA) so each group contributes equally per epoch",
)
@click.option(
    "--oversample-minority/--no-oversample-minority",
    default=False,
    help="Oversample minority classes so each class contributes equally per epoch "
    "(mutually exclusive with --balance-groups)",
)
@click.option(
    "--adversarial-lambda",
    type=float,
    default=0.0,
    help="Gradient reversal strength for adversarial confound removal (0 = disabled).",
)
@click.option(
    "--adversarial-anneal-epochs",
    type=int,
    default=0,
    help="Linearly ramp adversarial lambda from 0 to target over this many epochs (0 = constant).",
)
@click.option(
    "--confound",
    type=click.Choice(["disc_base", "trna_id"]),
    default=None,
    help="Confound to decorrelate via gradient reversal. 'disc_base' = discriminator base at position 73 (4 classes). 'trna_id' = full tRNA isoacceptor identity (N classes, one per unique reference tRNA).",
)
@click.option(
    "--cl-regression/--no-cl-regression",
    default=False,
    help="Enable continuous charging-level (CL) regression head.",
)
@click.option(
    "--cl-lambda",
    type=float,
    default=1.0,
    help="Weight for CL regression loss (default: 1.0).",
)
@click.option(
    "--signal-mode",
    type=click.Choice(["both", "residual", "signal"]),
    default="both",
    help="Signal input: 'both' (raw+residual, 2ch), 'residual' (1ch), 'signal' (1ch).",
)
def optimize(
    train_data,
    val_data,
    model,
    output_dir,
    context_grid,
    left_contexts,
    right_contexts,
    kmer_context,
    epochs,
    batch_size,
    learning_rate,
    device,
    seed,
    early_stopping,
    base_justify,
    dwell_offsets,
    parallel,
    weight_decay,
    max_grad_norm,
    scheduler,
    scheduler_patience,
    scheduler_factor,
    warmup_epochs,
    loss_type,
    focal_gamma,
    label_smoothing,
    mixed_precision,
    augment_jitter,
    augment_scale_min,
    augment_scale_max,
    augment_time_mask_bases,
    augment_time_mask_count,
    augment_shift_max_bases,
    augment_feature_noise_scale,
    num_workers,
    balance_groups,
    oversample_minority,
    motif,
    motif_offset,
    adversarial_lambda,
    adversarial_anneal_epochs,
    confound,
    cl_regression,
    cl_lambda,
    signal_mode,
):
    """Optimize model hyperparameters using grid search over chunk contexts."""
    from leech.commands.optimize import handle_optimize

    handle_optimize(
        train_data=train_data,
        val_data=val_data,
        model_name=model,
        output_dir=output_dir,
        context_grid=context_grid,
        left_contexts=left_contexts,
        right_contexts=right_contexts,
        kmer_context=kmer_context,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
        early_stopping=early_stopping,
        base_justify=base_justify,
        dwell_offsets=dwell_offsets,
        parallel=parallel,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        scheduler=scheduler,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        warmup_epochs=warmup_epochs,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
        mixed_precision=mixed_precision,
        augment_jitter=augment_jitter,
        augment_scale_min=augment_scale_min,
        augment_scale_max=augment_scale_max,
        augment_time_mask_bases=augment_time_mask_bases,
        augment_time_mask_count=augment_time_mask_count,
        augment_shift_max_bases=augment_shift_max_bases,
        augment_feature_noise_scale=augment_feature_noise_scale,
        num_workers=num_workers,
        balance_groups=balance_groups,
        oversample_minority=oversample_minority,
        motif=motif,
        motif_offset=motif_offset,
        adversarial_lambda=adversarial_lambda,
        adversarial_anneal_epochs=adversarial_anneal_epochs,
        confound=confound,
        cl_regression=cl_regression,
        cl_lambda=cl_lambda,
        signal_mode=signal_mode,
    )


# ============================================================================
# MODEL EVALUATION COMMANDS
# ============================================================================


@cli.group()
def eval():
    """Evaluate and analyze trained models."""
    pass


@eval.command()
@click.option(
    "--model",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Trained model file (.pt)",
)
@click.option(
    "--test-data",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Test dataset config (JSON)",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output metrics file (JSON)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default=DEFAULT_DEVICE,
    help="Device for inference",
)
@click.option(
    "--batch-size",
    type=int,
    default=512,
    help="Batch size for evaluation (default: 512)",
)
def test(model, test_data, output, device, batch_size):
    """Test a trained model on a holdout test set."""
    from leech.commands.eval import handle_test

    handle_test(
        model=model, test_data=test_data, output=output, device=device, batch_size=batch_size
    )


@eval.command()
@click.option(
    "--model-dirs",
    "-m",
    required=True,
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Model directories to compare (can specify multiple)",
)
@click.option(
    "--test-data",
    "-t",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Test dataset for evaluation",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for comparison results",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default=DEFAULT_DEVICE,
    help="Device for evaluation",
)
@click.option(
    "--no-plot",
    is_flag=True,
    help="Skip generating plots",
)
def compare(model_dirs, test_data, output_dir, device, no_plot):
    """Compare multiple trained models on the same test set."""
    from leech.commands.analyze import handle_compare

    handle_compare(
        model_dirs=list(model_dirs),
        test_data=test_data,
        output_dir=output_dir,
        device=device,
        plot=not no_plot,
    )

    console.print("[bold green]Model comparison complete![/bold green]")


@eval.command()
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to trained model checkpoint",
)
@click.option(
    "--test-data",
    "-t",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Test dataset for analysis",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for results",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default=DEFAULT_DEVICE,
    help="Device for computation",
)
@click.option(
    "--method",
    type=click.Choice(["gradient", "integrated_gradients"]),
    default="gradient",
    help="Feature importance method",
)
@click.option(
    "--no-plot",
    is_flag=True,
    help="Skip generating plots",
)
def importance(model, test_data, output_dir, device, method, no_plot):
    """Compute feature importance scores for a trained model."""
    from leech.commands.analyze import handle_feature_importance

    handle_feature_importance(
        model_path=model,
        test_data=test_data,
        output_dir=output_dir,
        device=device,
        method=method,
        plot=not no_plot,
    )

    console.print("[bold green]Feature importance analysis complete![/bold green]")


@eval.command()
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to trained model checkpoint",
)
@click.option(
    "--test-data",
    "-t",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Test dataset for analysis",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for results",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default=DEFAULT_DEVICE,
    help="Device for computation",
)
@click.option(
    "--no-plot",
    is_flag=True,
    help="Skip generating plots",
)
def ablation(model, test_data, output_dir, device, no_plot):
    """Test model performance with sequence ablation."""
    from leech.commands.analyze import handle_sequence_ablation

    handle_sequence_ablation(
        model_path=model,
        test_data=test_data,
        output_dir=output_dir,
        device=device,
        plot=not no_plot,
    )

    console.print("[bold green]Sequence ablation test complete![/bold green]")


# ============================================================================
# INFERENCE COMMAND
# ============================================================================


@cli.command()
@click.option(
    "--model",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Trained model checkpoint directory",
)
@click.option(
    "--bundle",
    "bundle_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Model bundle .pt file (mutually exclusive with --model)",
)
@click.option(
    "--pair",
    type=str,
    default=None,
    help="Run a single model from the bundle (requires --bundle)",
)
@click.option(
    "--all",
    "run_all",
    is_flag=True,
    default=False,
    help="Run every model in the bundle on each read (requires --bundle)",
)
@click.option(
    "--raw",
    is_flag=True,
    default=False,
    help="Write full float probabilities for all tags (default: compact uint8 encoding for ac/pp)",
)
@click.option(
    "--min-confidence",
    type=click.IntRange(0, 255),
    default=0,
    help="Confidence threshold in uint8 space (0-255). Reads below threshold are called 'unc' (uncharged). Default: 0 (all reads called).",
)
@click.option(
    "--min-margin",
    type=click.IntRange(0, 255),
    default=0,
    help="Margin threshold in uint8 space (0-255). Margin = max_prob - 2nd_prob. Reads below threshold are called 'unc'. Default: 0 (no margin filter).",
)
@click.option(
    "--pod5",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="POD5 file with raw signal",
)
@click.option(
    "--bam",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="BAM file with alignments",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output BAM with predictions",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default=DEFAULT_DEVICE,
    help="Device for inference",
)
@click.option(
    "--base-justify",
    type=click.Choice(["start", "center", "end"]),
    default="center",
    help='Where to center signal chunk within the focus base: "start" (first sample), "center" (midpoint, default), or "end" (last sample, useful for 3\' modifications)',
)
@click.option(
    "--no-reverse-signal",
    is_flag=True,
    default=False,
    help="Do NOT reverse the raw signal. By default, signal is reversed for direct RNA sequencing (POD5 stores 3'→5', basecaller expects 5'→3'). Use this flag for DNA data.",
)
@click.option(
    "--anchor",
    type=click.Choice(["basecall", "reference"]),
    default="reference",
    help='Anchor mode: "reference" (default) uses ref sequence + ref->signal mapping via CIGAR, "basecall" uses basecalled sequence',
)
@click.option(
    "--reference-anchored",
    is_flag=True,
    default=False,
    hidden=True,
    help="Deprecated: use --anchor reference instead.",
)
@click.option(
    "--reference-fasta",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Reference FASTA file for --anchor reference mode (required if BAM header lacks embedded sequences).",
)
@click.option(
    "--motif",
    type=str,
    default=None,
    help="Motif to search for in reads (auto-read from model config if not provided; required for Remora models)",
)
@click.option(
    "--motif-offset",
    type=int,
    default=0,
    help="Offset within motif for the prediction position (0-indexed)",
)
@click.option(
    "--batch-size",
    type=int,
    default=256,
    help="Chunks per forward pass (default: 256)",
)
@click.option(
    "--min-mapq",
    type=int,
    default=0,
    help="Minimum mapping quality (default: 0)",
)
@click.option(
    "--workers",
    type=int,
    default=0,
    help="Parallel chunk extraction workers (0=sequential). Only useful with GPU inference; CPU-only is faster with 0.",
)
@click.option(
    "--aggregation",
    type=click.Choice(["naive", "weighted", "tournament"]),
    default="naive",
    help='Pairwise aggregation method: "naive" (sum votes), "weighted" (confidence-weighted), "tournament" (two-round elimination). Default: naive.',
)
@click.option(
    "--read-batch-size",
    type=int,
    default=10_000,
    help="Reads per mega-batch for memory-bounded streaming (default: 10000). Smaller values reduce peak memory and GPU idle time during POD5 preload.",
)
@click.option(
    "--backend",
    type=click.Choice(["auto", "rust", "python"]),
    default="auto",
    help='Extraction backend: "auto" (default, Rust if available), "rust" (force Rust, error if unavailable), "python" (force Python). Useful for comparing outputs between paths.',
)
@click.option(
    "--no-compile",
    is_flag=True,
    default=False,
    help="Disable torch.compile (skip CUDA graph compilation overhead). Faster startup for small inference runs.",
)
@click.option(
    "--copy-tags",
    type=str,
    default=None,
    help='Comma-separated BAM tags to copy to TSV output (e.g., "CL,RG"). Ignored for BAM output.',
)
def predict(
    model,
    bundle_path,
    pair,
    run_all,
    raw,
    min_confidence,
    min_margin,
    pod5,
    bam,
    output,
    device,
    base_justify,
    no_reverse_signal,
    anchor,
    reference_anchored,
    reference_fasta,
    motif,
    motif_offset,
    batch_size,
    min_mapq,
    workers,
    aggregation,
    read_batch_size,
    backend,
    no_compile,
    copy_tags,
):
    """Run inference on new data to generate predictions."""
    import warnings

    from leech.commands.predict import handle_predict

    if reference_anchored:
        warnings.warn(
            "--reference-anchored is deprecated, use --anchor reference instead",
            FutureWarning,
            stacklevel=2,
        )
        anchor = "reference"

    # Parse --copy-tags into a list
    parsed_copy_tags = [t.strip() for t in copy_tags.split(",") if t.strip()] if copy_tags else None

    # Detect output format from file extension
    output_name = str(output).lower()
    if output_name.endswith(".tsv.gz") or output_name.endswith(".tsv"):
        output_format = "tsv"
    elif output_name.endswith(".bam"):
        output_format = "bam"
    else:
        import rich_click as click

        raise click.UsageError(
            f"Cannot determine output format from extension: {output}. "
            "Use .bam for BAM output or .tsv / .tsv.gz for TSV output."
        )

    handle_predict(
        model=model,
        bundle_path=bundle_path,
        pair=pair,
        run_all=run_all,
        raw=raw,
        min_confidence=min_confidence,
        min_margin=min_margin,
        pod5=pod5,
        bam=bam,
        output=output,
        device=device,
        base_justify=base_justify,
        no_reverse_signal=no_reverse_signal,
        anchor=anchor,
        reference_fasta=reference_fasta,
        motif=motif,
        motif_offset=motif_offset,
        batch_size=batch_size,
        min_mapq=min_mapq,
        workers=workers,
        aggregation=aggregation,
        read_batch_size=read_batch_size,
        backend=backend,
        no_compile=no_compile,
        output_format=output_format,
        copy_tags=parsed_copy_tags,
    )


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
