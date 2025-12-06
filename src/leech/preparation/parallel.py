"""
Parallel data preparation using multiprocessing.

This module provides parallel processing capabilities for extracting
training chunks from large datasets using multiple worker processes.
"""

import logging
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from pod5 import DatasetReader
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from leech.chunking import LeechRead, extract_training_chunks
from leech.features import (
    compute_dwell_features,
    compute_dwell_times,
    compute_signal_features,
    normalize_signal,
)
from leech.io import ReadInfo, collect_read_infos, get_motif_searcher
from leech.io.bed_reader import BedIndex, BedRegionSearcher

logger = logging.getLogger("leech.preparation.parallel")


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
        dict[str, list] | None,  # BED index regions (serialized)
    ],
) -> list[dict[str, np.ndarray | str | int | None]]:
    """
    Worker function to process a chunk of reads in parallel.

    Args:
        args: Tuple of (read_infos, pod5_path, motif, motif_offset, label, label_int,
                        motif_reference, reference_sequences, skip_motif_indels, bed_regions)

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
        bed_regions,
    ) = args

    # Get position finder (either motif or BED-based)
    bed_searcher = None
    motif_searcher = None

    if bed_regions is not None:
        # Reconstruct BedIndex from serialized regions
        from leech.io.bed_reader import BedRegion

        bed_index = BedIndex()
        for chrom, region_dicts in bed_regions.items():
            for rd in region_dicts:
                region = BedRegion(
                    chrom=rd["chrom"],
                    start=rd["start"],
                    end=rd["end"],
                    name=rd["name"],
                )
                bed_index.add_region(region)

        bed_searcher = BedRegionSearcher(
            bed_index=bed_index,
            skip_indels=skip_motif_indels,
            default_label=label,
        )
    elif motif is not None:
        motif_searcher = get_motif_searcher(
            mode=motif_reference,
            reference_sequences=reference_sequences,
            skip_indels=skip_motif_indels,
        )

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

                # For reference-based motif search or BED-based search, create a mock alignment object
                if (motif_reference == "fasta" and reference_sequences is not None) or bed_searcher is not None:
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
                if bed_searcher is not None:
                    # BED-based extraction
                    alignment = leech_read.metadata.get("alignment")
                    focus_positions = bed_searcher.find_positions_with_labels(
                        read_id=read_info.read_id,
                        sequence=read_info.sequence,
                        alignment=alignment,
                    )
                    read_chunks = extract_training_chunks(
                        leech_read,
                        label_int=label_int,
                        focus_positions=focus_positions,
                    )
                else:
                    # Motif-based extraction (or all bases if no motif)
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
    bed_path: Path | None = None,
    num_workers: int = 8,
    chunk_size: int = 100,
) -> tuple[list[dict[str, np.ndarray | str | int | None]], dict[str, int]]:
    """
    Prepare training data from BAM and POD5 files using multiprocessing.

    This function provides significant speedup for large datasets by processing
    reads in parallel across multiple CPU cores. Expected speedup is 3-6x on
    typical multi-core machines for CPU-bound feature extraction tasks.

    Args:
        bam_path: Path to BAM file with alignments
        pod5_path: Path to POD5 file with signal
        motif: Optional sequence motif to filter (mutually exclusive with bed_path)
        motif_offset: Offset within motif for focus base
        label: String label identifier (e.g., "Ala", "Gly", "charged", "uncharged")
        label_int: Optional numeric label (0, 1) - assigned during merge for pairwise comparisons
        min_mapq: Minimum mapping quality
        motif_reference: Where to search for motif ("bam" or "fasta")
        reference_sequences: Dict of reference sequences (for motif_reference="fasta")
        skip_motif_indels: Skip reads with indels in motif region
        bed_path: Path to BED file for region-based extraction (mutually exclusive with motif)
        num_workers: Number of parallel workers
        chunk_size: Number of reads to process per worker batch

    Returns:
        Tuple of (chunks, statistics) where statistics contains:
        - total_reads: Total number of reads found in BAM
        - reads_with_motif: Approximate count of reads with motif matches
        - reads_without_motif: Approximate count without matches
        - total_chunks: Total number of chunks extracted

    Raises:
        ValueError: If both motif and bed_path are provided

    Examples:
        >>> chunks, stats = prepare_training_data_parallel(
        ...     bam_path=Path("alignments.bam"),
        ...     pod5_path=Path("reads.pod5"),
        ...     motif="CCAGGC",
        ...     num_workers=8,
        ...     chunk_size=100
        ... )
        >>> print(f"Extracted {stats['total_chunks']} chunks from {stats['total_reads']} reads")
    """
    # Validate mutual exclusivity
    if motif is not None and bed_path is not None:
        raise ValueError("Cannot specify both --motif and --bed. Choose one selection method.")

    logger.info(f"Starting parallel data preparation with {num_workers} workers")

    # Load and serialize BED regions if provided
    bed_regions_serialized: dict[str, list] | None = None
    if bed_path is not None:
        from leech.io.bed_reader import load_bed_regions

        logger.info(f"Loading BED regions from {bed_path}")
        bed_index = load_bed_regions(bed_path)

        # Serialize BedIndex to a picklable format for workers
        bed_regions_serialized = {}
        for chrom, regions in bed_index.regions.items():
            bed_regions_serialized[chrom] = [
                {"chrom": r.chrom, "start": r.start, "end": r.end, "name": r.name}
                for r in regions
            ]

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
            bed_regions_serialized,
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
