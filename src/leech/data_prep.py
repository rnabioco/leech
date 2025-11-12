"""
Data preparation: high-level orchestration for reading POD5 and BAM files.

This module provides high-level functions that orchestrate the data preparation
pipeline by composing functionality from io/, chunking/, and splitting/ modules.

For detailed implementations, see:
- leech.io: BAM/POD5 reading, reference sequences, motif search
- leech.chunking: Chunk extraction and serialization
- leech.splitting: Read-level splitting
"""

import logging
import multiprocessing as mp
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from pod5 import DatasetReader

from leech.chunking import LeechRead, extract_training_chunks, load_chunks, save_chunks
from leech.features import (
    compute_dwell_features,
    compute_dwell_times,
    compute_signal_features,
    extract_move_table,
    normalize_signal,
)
from leech.io import (
    POD5Reader,
    ReadInfo,
    collect_read_infos,
    get_motif_searcher,
    get_reference_sequences,
)
from leech.splitting import (
    merge_and_split_chunks,
    parse_comparison_spec,
    process_comparison_spec,
    split_chunks_by_read,
)

logger = logging.getLogger("leech.data_prep")


# Re-export public API from submodules for backward compatibility
__all__ = [
    # From chunking
    "LeechRead",
    "extract_training_chunks",
    "save_chunks",
    "load_chunks",
    # From splitting
    "split_chunks_by_read",
    "merge_and_split_chunks",
    "parse_comparison_spec",
    "process_comparison_spec",
    # From io
    "get_reference_sequences",
    # This module
    "iter_bam_with_pod5",
    "prepare_training_data",
    "prepare_training_data_parallel",
    "prepare_training_data_with_split",
    # Encoding utilities (kept in this module)
    "encode_kmer",
    "seq_to_int",
    "int_to_seq",
    "one_hot_encode_sequence",
]


def read_pod5_signal(pod5_path: Path, read_id: str) -> tuple[np.ndarray, dict]:
    """
    Read raw signal from POD5 file for a specific read.

    Args:
        pod5_path: Path to POD5 file
        read_id: Read identifier

    Returns:
        Tuple of (signal_array, metadata_dict)

    Note:
        This is kept for backward compatibility. For new code, use:
        `from leech.io import read_pod5_signal`
    """
    from leech.io import read_pod5_signal as _read_pod5_signal

    return _read_pod5_signal(pod5_path, read_id)


def iter_bam_with_pod5(
    bam_path: Path,
    pod5_path: Path,
    reference_fasta: Path | None = None,
    min_mapq: int = 0,
    require_tags: list[str] | None = None,
) -> Iterator[LeechRead]:
    """
    Iterate over aligned reads, loading signal from POD5.

    Args:
        bam_path: Path to BAM file with mv tags
        pod5_path: Path to POD5 file
        reference_fasta: Optional reference for MD tag parsing
        min_mapq: Minimum mapping quality
        require_tags: BAM tags that must be present

    Yields:
        LeechRead objects with full feature extraction

    Example:
        >>> for read in iter_bam_with_pod5(Path("alignments.bam"), Path("reads.pod5")):
        ...     print(f"{read.read_id}: {read.num_bases} bases")
    """
    from leech.io import iter_bam_alignments

    if require_tags is None:
        require_tags = ["mv", "ns"]

    with POD5Reader(pod5_path) as pod5_reader:
        for aln in iter_bam_alignments(bam_path, min_mapq=min_mapq, require_tags=require_tags):
            try:
                # Extract move table
                move_table = extract_move_table(aln)

                # Check for required query fields
                read_id = aln.query_name
                read_seq = aln.query_sequence
                if read_id is None or read_seq is None:
                    continue

                # Read signal from POD5
                raw_signal, pod5_metadata = pod5_reader.get_signal(read_id)

                # Normalize signal
                norm_signal, norm_params = normalize_signal(raw_signal, method="median_mad")

                # Compute seq-to-signal mapping
                seq_to_sig_map = move_table.to_seq_to_sig_map()

                # Compute features
                dwells = compute_dwell_times(move_table)
                dwell_feats = compute_dwell_features(dwells)
                signal_feats = compute_signal_features(norm_signal, seq_to_sig_map)

                # Build metadata
                metadata = {
                    **pod5_metadata,
                    "normalization": norm_params,
                    "mapping_quality": aln.mapping_quality,
                    "reference_name": aln.reference_name,
                    "reference_start": aln.reference_start,
                    "reference_end": aln.reference_end,
                    "is_reverse": aln.is_reverse,
                    "alignment": aln,  # Store alignment for reference-based motif search
                }

                yield LeechRead(
                    read_id=read_id,
                    sequence=read_seq,
                    signal=norm_signal,
                    seq_to_sig_map=seq_to_sig_map,
                    dwells=dwells,
                    dwell_features=dwell_feats,
                    signal_features=signal_feats,
                    metadata=metadata,
                )

            except Exception as e:
                logger.warning(f"Skipping read {aln.query_name}: {e}")
                continue


def _process_read_chunk_worker(
    args: tuple[
        list[ReadInfo],
        Path,
        str | None,
        int,
        str | None,
        int | None,
        str,
        dict[str, str] | None,
        bool,
    ],
) -> list[dict[str, np.ndarray | str | int | None]]:
    """
    Worker function to process a chunk of reads in parallel.

    Args:
        args: Tuple of (read_infos, pod5_path, motif, motif_offset, label, label_int,
                        motif_reference, reference_sequences, skip_motif_indels)

    Returns:
        List of extracted chunks from all reads in this chunk
    """
    (
        read_infos,
        pod5_path,
        motif,
        motif_offset,
        label,
        label_int,
        motif_reference,
        reference_sequences,
        skip_motif_indels,
    ) = args

    # Get motif searcher
    if motif is not None:
        motif_searcher = get_motif_searcher(
            mode=motif_reference,
            reference_sequences=reference_sequences,
            skip_indels=skip_motif_indels,
        )
    else:
        motif_searcher = None

    all_chunks = []

    # Open POD5 once for this worker
    with DatasetReader(pod5_path) as pod5_reader:
        for read_info in read_infos:
            try:
                # Read signal from POD5
                signal_found = False
                for read in pod5_reader.reads([read_info.read_id]):
                    raw_signal = read.signal
                    pod5_metadata = {
                        "read_id": str(read.read_id),
                        "channel": read.pore.channel,
                        "well": read.pore.well,
                        "pore_type": read.pore.pore_type,
                        "calibration_offset": read.calibration.offset,
                        "calibration_scale": read.calibration.scale,
                        "sample_rate": read.run_info.sample_rate,
                    }
                    signal_found = True
                    break

                if not signal_found:
                    continue

                # Reconstruct move table
                move_table = read_info.to_move_table()

                # Normalize signal
                norm_signal, norm_params = normalize_signal(raw_signal, method="median_mad")

                # Compute seq-to-signal mapping
                seq_to_sig_map = move_table.to_seq_to_sig_map()

                # Compute features
                dwells = compute_dwell_times(move_table)
                dwell_feats = compute_dwell_features(dwells)
                signal_feats = compute_signal_features(norm_signal, seq_to_sig_map)

                # Build metadata (create mock alignment for reference-based search)
                metadata = {
                    **pod5_metadata,
                    "normalization": norm_params,
                    "mapping_quality": read_info.mapping_quality,
                    "reference_name": read_info.reference_name,
                    "reference_start": read_info.reference_start,
                    "reference_end": read_info.reference_end,
                    "is_reverse": read_info.is_reverse,
                }

                # For reference-based motif search, create a mock alignment object
                if motif_reference == "fasta" and reference_sequences is not None:
                    # Create minimal mock alignment for CIGAR parsing
                    class MockAlignment:
                        def __init__(self, read_info: ReadInfo):
                            self.reference_name = read_info.reference_name
                            self.reference_start = read_info.reference_start
                            self.reference_end = read_info.reference_end
                            self.cigartuples = read_info.cigar_tuples
                            self.is_reverse = read_info.is_reverse

                    metadata["alignment"] = MockAlignment(read_info)

                # Create LeechRead
                leech_read = LeechRead(
                    read_id=read_info.read_id,
                    sequence=read_info.sequence,
                    signal=norm_signal,
                    seq_to_sig_map=seq_to_sig_map,
                    dwells=dwells,
                    dwell_features=dwell_feats,
                    signal_features=signal_feats,
                    metadata=metadata,
                )

                # Extract training chunks
                read_chunks = extract_training_chunks(
                    leech_read,
                    motif=motif,
                    motif_offset=motif_offset,
                    label=label,
                    label_int=label_int,
                    motif_searcher=motif_searcher,
                )

                all_chunks.extend(read_chunks)

            except Exception as e:
                logger.warning(f"Worker failed to process read {read_info.read_id}: {e}")
                continue

    return all_chunks


def prepare_training_data_parallel(
    bam_path: Path,
    pod5_path: Path,
    motif: str | None = None,
    motif_offset: int = 0,
    label: str | None = None,
    label_int: int | None = None,
    min_mapq: int = 0,
    motif_reference: str = "bam",
    reference_sequences: dict[str, str] | None = None,
    skip_motif_indels: bool = True,
    num_workers: int = 8,
    chunk_size: int = 100,
) -> tuple[list[dict[str, np.ndarray | str | int | None]], dict[str, int]]:
    """
    Prepare training data from BAM and POD5 files using multiprocessing.

    Args:
        bam_path: Path to BAM file with alignments
        pod5_path: Path to POD5 file with signal
        motif: Optional sequence motif to filter
        motif_offset: Offset within motif for focus base
        label: String label identifier (e.g., "Ala", "Gly", "charged", "uncharged")
        label_int: Optional numeric label (0, 1) - assigned during merge for pairwise comparisons
        min_mapq: Minimum mapping quality
        motif_reference: Where to search for motif ("bam" or "fasta")
        reference_sequences: Dict of reference sequences (for motif_reference="fasta")
        skip_motif_indels: Skip reads with indels in motif region
        num_workers: Number of parallel workers
        chunk_size: Number of reads to process per worker batch

    Returns:
        Tuple of (chunks, statistics)

    Example:
        >>> chunks, stats = prepare_training_data_parallel(
        ...     bam_path=Path("alignments.bam"),
        ...     pod5_path=Path("reads.pod5"),
        ...     motif="CCAGGC",
        ...     num_workers=8
        ... )
        >>> print(f"Extracted {stats['total_chunks']} chunks")
    """
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    logger.info(f"Starting parallel data preparation with {num_workers} workers")

    # First pass: collect read info from BAM (lightweight, sequential)
    logger.info("Pass 1: Collecting read info from BAM...")
    read_infos = collect_read_infos(bam_path, min_mapq=min_mapq)
    total_reads = len(read_infos)
    logger.info(f"Found {total_reads} reads to process")

    if total_reads == 0:
        return [], {
            "total_reads": 0,
            "reads_with_motif": 0,
            "reads_without_motif": 0,
            "total_chunks": 0,
        }

    # Split read_infos into chunks for workers
    read_chunks = [read_infos[i : i + chunk_size] for i in range(0, len(read_infos), chunk_size)]
    logger.info(f"Split into {len(read_chunks)} chunks of up to {chunk_size} reads each")

    # Prepare worker arguments
    worker_args = [
        (
            chunk,
            pod5_path,
            motif,
            motif_offset,
            label,
            label_int,
            motif_reference,
            reference_sequences,
            skip_motif_indels,
        )
        for chunk in read_chunks
    ]

    # Second pass: parallel processing with progress bar
    logger.info("Pass 2: Processing reads in parallel...")
    all_chunks = []

    # Check if we're in a TTY (interactive terminal) or redirected (e.g., log file)
    use_progress_bar = sys.stdout.isatty()

    if use_progress_bar:
        # Interactive terminal - use rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.fields[chunks_extracted]} chunks extracted"),
        ) as progress:
            task = progress.add_task(
                "Processing chunks", total=len(read_chunks), chunks_extracted=0
            )

            with mp.Pool(processes=num_workers) as pool:
                # Use imap_unordered for progress tracking
                for chunk_results in pool.imap_unordered(_process_read_chunk_worker, worker_args):
                    all_chunks.extend(chunk_results)
                    progress.update(task, advance=1, chunks_extracted=len(all_chunks))
    else:
        # Redirected output (log file) - use periodic logging
        log_interval = max(1, len(read_chunks) // 20)  # Log every 5%
        with mp.Pool(processes=num_workers) as pool:
            for i, chunk_results in enumerate(
                pool.imap_unordered(_process_read_chunk_worker, worker_args), 1
            ):
                all_chunks.extend(chunk_results)
                # Log progress periodically
                if i % log_interval == 0 or i == len(read_chunks):
                    pct = (i / len(read_chunks)) * 100
                    logger.info(
                        f"Progress: {i}/{len(read_chunks)} batches ({pct:.1f}%) | "
                        f"{len(all_chunks)} chunks extracted"
                    )

    # Compile statistics (approximate - we don't track individual read success)
    stats = {
        "total_reads": total_reads,
        "reads_with_motif": len(all_chunks),  # Approximate
        "reads_without_motif": total_reads - len(all_chunks),  # Approximate
        "total_chunks": len(all_chunks),
    }

    logger.info(
        f"Parallel processing complete: extracted {len(all_chunks)} chunks from {total_reads} reads"
    )

    return all_chunks, stats


def prepare_training_data(
    bam_path: Path,
    pod5_path: Path,
    motif: str | None = None,
    motif_offset: int = 0,
    label: str | None = None,
    label_int: int | None = None,
    min_mapq: int = 0,
) -> tuple[list[dict[str, np.ndarray | str | int | None]], dict[str, int]]:
    """
    Prepare training data from BAM and POD5 files with statistics tracking.

    Args:
        bam_path: Path to BAM file with alignments
        pod5_path: Path to POD5 file with signal
        motif: Optional sequence motif to filter
        motif_offset: Offset within motif for focus base
        label: String label identifier (e.g., "Ala", "Gly", "charged", "uncharged")
        label_int: Optional numeric label (0, 1) - assigned during merge for pairwise comparisons
        min_mapq: Minimum mapping quality

    Returns:
        Tuple of (chunks, statistics) where statistics is a dict with:
        - total_reads: Total number of reads processed
        - reads_with_motif: Number of reads containing motif
        - reads_without_motif: Number of reads without motif
        - total_chunks: Total number of chunks extracted
    """
    total_reads = 0
    reads_with_motif = 0
    reads_without_motif = 0
    chunks = []

    # Get motif searcher if motif is provided
    motif_searcher = None
    if motif is not None:
        motif_searcher = get_motif_searcher(mode="bam")

    for read in iter_bam_with_pod5(bam_path, pod5_path, min_mapq=min_mapq):
        total_reads += 1
        read_chunks = extract_training_chunks(
            read,
            motif=motif,
            motif_offset=motif_offset,
            label=label,
            label_int=label_int,
            motif_searcher=motif_searcher,
        )

        # Track whether this read had motif matches
        if len(read_chunks) > 0:
            reads_with_motif += 1
        else:
            reads_without_motif += 1

        chunks.extend(read_chunks)

    # Compile statistics
    stats = {
        "total_reads": total_reads,
        "reads_with_motif": reads_with_motif,
        "reads_without_motif": reads_without_motif,
        "total_chunks": len(chunks),
    }

    # Log statistics
    logger.info(f"Processed {total_reads} reads")
    if motif is not None:
        logger.info(
            f"Reads with motif '{motif}': {reads_with_motif} "
            f"({100.0 * reads_with_motif / total_reads:.1f}%)"
        )
        logger.info(
            f"Reads without motif: {reads_without_motif} "
            f"({100.0 * reads_without_motif / total_reads:.1f}%)"
        )
    logger.info(f"Extracted {len(chunks)} training chunks")

    return chunks, stats


def prepare_training_data_with_split(
    pod5_path: Path,
    bam_path: Path,
    output_dir: Path,
    motif: str | None = None,
    motif_offset: int = 0,
    motif_reference: str = "fasta",
    reference_fasta: Path | None = None,
    skip_motif_indels: bool = True,
    label: str | None = None,
    label_int: int | None = None,
    min_mapq: int = 10,
    feature_set: str = "signal+dwell+levels",
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42,
    no_split: bool = False,
    progress_callback: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """
    Prepare training data from POD5/BAM files with optional splitting.

    This function handles the full pipeline:
    1. Setup random seed for reproducibility
    2. Load reference sequences if needed
    3. Extract training chunks from reads
    4. Split at read level (if requested)
    5. Save to disk

    Args:
        pod5_path: Path to POD5 file
        bam_path: Path to BAM file
        output_dir: Directory for output files
        motif: Sequence motif to extract
        motif_offset: Offset within motif for focus base
        motif_reference: Where to search for motif ("fasta" or "bam")
        reference_fasta: External reference FASTA file
        skip_motif_indels: Skip reads with indels in motif region
        label: String label identifier (e.g., "Ala", "Gly", "charged", "uncharged")
        label_int: Optional numeric label (0, 1) - assigned during merge for pairwise comparisons
        min_mapq: Minimum mapping quality
        feature_set: Feature set to extract (not currently used, reserved for future)
        train_split: Fraction for training
        val_split: Fraction for validation
        seed: Random seed
        no_split: If True, save all chunks without splitting
        progress_callback: Optional callback(n_chunks) for progress updates

    Returns:
        Dictionary with statistics
    """
    from leech.util import setup_random_seed

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup seed
    setup_random_seed(seed, output_dir)

    # Load reference sequences if needed
    reference_sequences = None
    if motif_reference == "fasta":
        logger.info("Loading reference sequences for reference-based motif search")
        reference_sequences = get_reference_sequences(bam_path, reference_fasta)

    # Get motif searcher
    motif_searcher = None
    if motif is not None:
        motif_searcher = get_motif_searcher(
            mode=motif_reference,
            reference_sequences=reference_sequences,
            skip_indels=skip_motif_indels,
        )

    # Extract chunks
    logger.info("Extracting chunks...")
    chunks = []
    for read in iter_bam_with_pod5(bam_path, pod5_path, min_mapq=min_mapq):
        read_chunks = extract_training_chunks(
            read,
            motif=motif,
            motif_offset=motif_offset,
            label=label,
            label_int=label_int,
            motif_searcher=motif_searcher,
        )
        chunks.extend(read_chunks)

        # Progress callback
        if progress_callback:
            progress_callback(len(chunks))

    logger.info(f"Extracted {len(chunks)} chunks")

    # Save or split
    if no_split:
        all_file = output_dir / "all.npz"
        save_chunks(chunks, all_file)
        logger.info(f"Saved all chunks to {all_file}")
        return {
            "n_chunks": len(chunks),
            "n_train": 0,
            "n_val": 0,
            "n_test": 0,
            "output_files": {"all": all_file},
        }
    else:
        # Split at read level
        train_chunks, val_chunks, test_chunks = split_chunks_by_read(
            chunks, train_frac=train_split, val_frac=val_split, seed=seed
        )

        # Save splits
        output_files = {}
        if train_chunks:
            train_file = output_dir / "train.npz"
            save_chunks(train_chunks, train_file)
            output_files["train"] = train_file
            logger.info(f"Saved {len(train_chunks)} train chunks to {train_file}")

        if val_chunks:
            val_file = output_dir / "val.npz"
            save_chunks(val_chunks, val_file)
            output_files["val"] = val_file
            logger.info(f"Saved {len(val_chunks)} val chunks to {val_file}")

        if test_chunks:
            test_file = output_dir / "test.npz"
            save_chunks(test_chunks, test_file)
            output_files["test"] = test_file
            logger.info(f"Saved {len(test_chunks)} test chunks to {test_file}")

        return {
            "n_chunks": len(chunks),
            "n_train": len(train_chunks),
            "n_val": len(val_chunks),
            "n_test": len(test_chunks),
            "output_files": output_files,
        }


# Sequence encoding utilities (kept in this module for convenience)
def seq_to_int(seq: str) -> np.ndarray:
    """
    Convert DNA sequence to integer encoding.

    A=0, C=1, G=2, T=3, N=4, U=3 (treat as T)

    Args:
        seq: DNA/RNA sequence string

    Returns:
        Integer array
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3, "N": 4}
    return np.array([mapping.get(b.upper(), 4) for b in seq], dtype=np.int64)


def int_to_seq(int_seq: np.ndarray) -> str:
    """Convert integer encoding back to sequence."""
    bases = ["A", "C", "G", "T", "N"]
    return "".join(bases[i] if i < 5 else "N" for i in int_seq)


def encode_kmer(sequence: str) -> Any:  # Returns torch.Tensor
    """
    One-hot encode a DNA sequence for model input.

    This is the canonical sequence encoding function used throughout leech.
    Returns a PyTorch tensor suitable for direct model input.

    Args:
        sequence: DNA sequence string (A, C, G, T, N)

    Returns:
        One-hot encoded tensor of shape (4, len(sequence))
        Bases are encoded as: A=0, C=1, G=2, T=3
        Unknown bases (e.g., N) are encoded as all zeros

    Example:
        >>> seq = "ACGT"
        >>> encoded = encode_kmer(seq)
        >>> encoded.shape
        torch.Size([4, 4])
    """
    import torch

    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq_len = len(sequence)
    encoded = torch.zeros(4, seq_len, dtype=torch.float32)

    for i, base in enumerate(sequence.upper()):
        if base in base_to_idx:
            encoded[base_to_idx[base], i] = 1.0
        # If base not in dict (e.g., N), leave as zeros

    return encoded


def one_hot_encode_sequence(seq: str, kmer_len: int = 1) -> np.ndarray:
    """
    One-hot encode a sequence with k-mer context (advanced version).

    NOTE: For standard sequence encoding, use encode_kmer() instead.
    This function is for specialized k-mer context encoding where each
    position includes information from neighboring bases.

    Args:
        seq: DNA sequence
        kmer_len: K-mer length for encoding (context window)

    Returns:
        Array of shape (kmer_len * 4, seq_len) for model input
        Each position encodes kmer_len neighboring bases

    See Also:
        encode_kmer: Standard one-hot encoding function (recommended)
    """
    int_seq = seq_to_int(seq)
    seq_len = len(int_seq)

    # Create one-hot encoding for each k-mer position
    encoding = np.zeros((kmer_len * 4, seq_len), dtype=np.float32)

    for pos in range(seq_len):
        for k in range(kmer_len):
            offset = pos - kmer_len // 2 + k
            if 0 <= offset < seq_len:
                base = int_seq[offset]
                if base < 4:  # Valid base (not N)
                    encoding[k * 4 + base, pos] = 1.0

    return encoding
