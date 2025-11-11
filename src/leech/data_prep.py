"""
Data preparation: reading POD5 and BAM files, extracting chunks for training.

Adapted from Remora but modernized with NumPy arrays and type hints.
"""

import logging
import multiprocessing as mp
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pysam
import torch
from pod5 import DatasetReader

from leech.constants import DEFAULT_KMER_CONTEXT, DEFAULT_SIGNAL_CONTEXT
from leech.features import (
    compute_dwell_features,
    compute_dwell_times,
    compute_signal_features,
    extract_move_table,
    normalize_signal,
)

logger = logging.getLogger("leech.data_prep")


@dataclass
class LeechRead:
    """
    Container for a single read's data with all features.

    Attributes:
        read_id: Unique read identifier
        sequence: Basecalled sequence
        signal: Normalized signal array
        seq_to_sig_map: Mapping from base indices to signal indices
        dwells: Per-base dwell times
        dwell_features: Dict of dwell-derived features
        signal_features: Dict of signal-level features
        labels: Optional labels for training (e.g., 0=uncharged, 1=charged)
        metadata: Additional metadata (alignment info, etc.)
    """

    read_id: str
    sequence: str
    signal: np.ndarray
    seq_to_sig_map: np.ndarray
    dwells: np.ndarray
    dwell_features: dict[str, np.ndarray] = field(default_factory=dict)
    signal_features: dict[str, np.ndarray] = field(default_factory=dict)
    labels: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def num_bases(self) -> int:
        """Number of bases in the read."""
        return len(self.sequence)

    @property
    def num_samples(self) -> int:
        """Number of signal samples."""
        return len(self.signal)

    def get_chunk(
        self,
        base_idx: int,
        signal_context: tuple[int, int] = DEFAULT_SIGNAL_CONTEXT,
        kmer_context: int = DEFAULT_KMER_CONTEXT,
    ) -> dict[str, np.ndarray | str | int | None] | None:
        """
        Extract a training chunk centered on a specific base.

        Args:
            base_idx: Index of the focus base
            signal_context: (left, right) signal padding around focus base
            kmer_context: Number of bases on each side for k-mer encoding

        Returns:
            Dictionary with 'signal', 'kmer', 'dwell', 'features' arrays,
            or None if chunk cannot be extracted
        """
        # Check boundaries
        if base_idx < kmer_context or base_idx >= self.num_bases - kmer_context:
            return None

        # Extract signal chunk
        focus_sig_pos = self.seq_to_sig_map[base_idx]
        sig_start = max(0, focus_sig_pos - signal_context[0])
        sig_end = min(self.num_samples, focus_sig_pos + signal_context[1])

        if sig_end - sig_start < signal_context[0] + signal_context[1]:
            return None  # Not enough signal context

        signal_chunk = self.signal[sig_start:sig_end]

        # Extract k-mer sequence context
        kmer_start = base_idx - kmer_context
        kmer_end = base_idx + kmer_context + 1
        kmer_seq = self.sequence[kmer_start:kmer_end]

        # Extract dwell features
        dwell_start = base_idx - kmer_context
        dwell_end = base_idx + kmer_context + 1
        dwell_chunk = self.dwells[dwell_start:dwell_end]

        # Compile additional features
        features = []
        for _feat_name, feat_array in {**self.dwell_features, **self.signal_features}.items():
            features.append(feat_array[dwell_start:dwell_end])

        return {
            "signal": signal_chunk,
            "sequence": kmer_seq,
            "dwell": dwell_chunk,
            "features": np.stack(features, axis=0) if features else np.array([]),
            "base_idx": base_idx,
            "label": self.labels[base_idx] if self.labels is not None else None,
        }


def read_pod5_signal(pod5_path: Path, read_id: str) -> tuple[np.ndarray, dict]:
    """
    Read raw signal from POD5 file for a specific read.

    Args:
        pod5_path: Path to POD5 file
        read_id: Read identifier

    Returns:
        Tuple of (signal_array, metadata_dict)
    """
    with DatasetReader(pod5_path) as reader:
        for read in reader.reads([read_id]):
            signal = read.signal
            metadata = {
                "read_id": str(read.read_id),
                "channel": read.pore.channel,
                "well": read.pore.well,
                "pore_type": read.pore.pore_type,
                "calibration_offset": read.calibration.offset,
                "calibration_scale": read.calibration.scale,
                "sample_rate": read.run_info.sample_rate,
            }
            return signal, metadata

    raise ValueError(f"Read {read_id} not found in {pod5_path}")


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
    """
    if require_tags is None:
        require_tags = ["mv", "ns"]
    bam = pysam.AlignmentFile(str(bam_path), "rb")

    for aln in bam:
        # Filter alignments
        if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
            continue
        if aln.mapping_quality < min_mapq:
            continue

        # Check required tags
        if not all(aln.has_tag(tag) for tag in require_tags):
            continue

        try:
            # Extract move table
            move_table = extract_move_table(aln)

            # Check for required query fields
            read_id = aln.query_name
            read_seq = aln.query_sequence
            if read_id is None or read_seq is None:
                continue

            # Read signal from POD5
            raw_signal, pod5_metadata = read_pod5_signal(pod5_path, read_id)

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

    bam.close()


def extract_reference_from_bam(bam_path: Path) -> dict[str, str]:
    """
    Extract reference sequences from BAM header @SQ records.

    Args:
        bam_path: Path to BAM file

    Returns:
        Dictionary mapping reference name to sequence
        Empty dict if no sequences in header
    """
    references = {}

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        # Check if reference sequences are in header
        if hasattr(bam.header, "to_dict"):
            header_dict = bam.header.to_dict()
            for sq in header_dict.get("SQ", []):
                ref_name = sq.get("SN")
                ref_seq = sq.get("SQ", None)  # SQ tag contains sequence
                if ref_name and ref_seq:
                    references[ref_name] = ref_seq

    if references:
        logger.info(f"Extracted {len(references)} reference sequences from BAM header")
    else:
        logger.warning("No reference sequences found in BAM @SQ header")

    return references


def load_reference_fasta(fasta_path: Path) -> dict[str, str]:
    """
    Load reference sequences from FASTA file.

    Args:
        fasta_path: Path to FASTA file

    Returns:
        Dictionary mapping reference name to sequence
    """
    references = {}

    with pysam.FastaFile(str(fasta_path)) as fasta:
        for ref_name in fasta.references:
            references[ref_name] = fasta.fetch(ref_name)

    logger.info(f"Loaded {len(references)} reference sequences from {fasta_path}")
    return references


def get_reference_sequences(bam_path: Path, fasta_path: Path | None = None) -> dict[str, str]:
    """
    Get reference sequences, trying BAM header first, then FASTA.

    Args:
        bam_path: Path to BAM file
        fasta_path: Optional path to FASTA file

    Returns:
        Dictionary mapping reference name to sequence

    Raises:
        ValueError: If no reference sequences available
    """
    # Try BAM header first
    references = extract_reference_from_bam(bam_path)

    # Fall back to FASTA if provided and BAM had no sequences
    if not references and fasta_path:
        logger.info("Falling back to external FASTA file")
        references = load_reference_fasta(fasta_path)

    # Error if still no references
    if not references:
        raise ValueError(
            "No reference sequences found. BAM file must contain @SQ sequences "
            "or provide --reference-fasta path."
        )

    return references


def map_reference_to_query_coords(
    aln: pysam.AlignedSegment, ref_start: int, ref_end: int, skip_indels: bool = True
) -> tuple[int, int] | None:
    """
    Map reference coordinates to query coordinates using CIGAR string.

    Args:
        aln: Aligned segment from BAM
        ref_start: Start position in reference (0-based)
        ref_end: End position in reference (0-based, exclusive)
        skip_indels: If True, return None if indels found in region

    Returns:
        Tuple of (query_start, query_end) or None if mapping fails
        or indels detected (when skip_indels=True)
    """
    # Check if region is within aligned portion
    if aln.reference_end is None or ref_start < aln.reference_start or ref_end > aln.reference_end:
        return None

    # Parse CIGAR to build mapping
    if aln.cigartuples is None:
        return None

    ref_pos = aln.reference_start
    query_pos = 0  # Start from beginning of query sequence

    query_start = None
    query_end = None
    has_indel_in_region = False

    for op, length in aln.cigartuples:
        # Check if we've passed the region
        if query_start is not None and query_end is not None:
            break

        # M/=/X: match/mismatch (consumes both)
        if op in (0, 7, 8):  # BAM_CMATCH, BAM_CEQUAL, BAM_CDIFF
            for _ in range(length):
                if ref_pos == ref_start:
                    query_start = query_pos
                if ref_pos == ref_end - 1:
                    query_end = query_pos + 1
                ref_pos += 1
                query_pos += 1

        # I: insertion (consumes query only)
        elif op == 1:  # BAM_CINS
            if ref_pos >= ref_start and ref_pos < ref_end:
                has_indel_in_region = True
            query_pos += length

        # D: deletion (consumes reference only)
        elif op == 2:  # BAM_CDEL
            if ref_pos >= ref_start and ref_pos + length > ref_start:
                has_indel_in_region = True
            ref_pos += length

        # S: soft clip (consumes query only, not aligned)
        elif op == 4:  # BAM_CSOFT_CLIP
            query_pos += length

        # H: hard clip (not in sequence)
        # N: ref skip (e.g., intron)
        # P: padding
        # These don't affect our mapping

    # Check if we found valid coordinates
    if query_start is None or query_end is None:
        return None

    # Check indels if requested
    if skip_indels and has_indel_in_region:
        return None

    return (query_start, query_end)


def find_motif_in_reference(ref_seq: str, motif: str, ref_start: int, ref_end: int) -> list[int]:
    """
    Find all occurrences of motif in reference sequence region.

    Args:
        ref_seq: Full reference sequence
        motif: Motif to search for
        ref_start: Start of region to search (0-based)
        ref_end: End of region to search (0-based, exclusive)

    Returns:
        List of reference positions where motif starts
    """
    positions = []
    motif_len = len(motif)
    search_region = ref_seq[ref_start:ref_end]

    for i in range(len(search_region) - motif_len + 1):
        if search_region[i : i + motif_len] == motif:
            positions.append(ref_start + i)

    return positions


def extract_training_chunks(
    leech_read: LeechRead,
    motif: str | None = None,
    motif_offset: int = 0,
    label: str | None = None,
    label_int: int | None = None,
    motif_reference: str = "bam",
    reference_sequences: dict[str, str] | None = None,
    skip_motif_indels: bool = True,
) -> list[dict[str, np.ndarray | str | int | None]]:
    """
    Extract all training chunks from a read, optionally filtered by motif.

    Args:
        leech_read: LeechRead object
        motif: Optional sequence motif to filter (e.g., "CCAGGC")
        motif_offset: Offset within motif for focus base
        label: String label identifier (e.g., "Ala", "Gly", "charged", "uncharged")
        label_int: Optional numeric label (0, 1) - assigned during merge for pairwise comparisons
        motif_reference: Where to search for motif: "bam" (basecalled) or "fasta" (reference)
        reference_sequences: Dict of reference name -> sequence (required if motif_reference="fasta")
        skip_motif_indels: If True, skip reads with indels in motif region (only for motif_reference="fasta")

    Returns:
        List of chunk dictionaries
    """
    chunks: list[dict] = []

    # Set numeric labels for all bases if provided
    if label_int is not None:
        leech_read.labels = np.full(leech_read.num_bases, label_int, dtype=np.int64)

    # Find focus bases (either all or motif matches)
    if motif is None:
        focus_bases = list(range(5, leech_read.num_bases - 5))  # Avoid edges
    elif motif_reference == "bam":
        # Original behavior: search in basecalled sequence
        focus_bases = []
        motif_len = len(motif)
        for i in range(len(leech_read.sequence) - motif_len + 1):
            if leech_read.sequence[i : i + motif_len] == motif:
                focus_bases.append(i + motif_offset)
    elif motif_reference == "fasta":
        # New behavior: search in reference sequence, map to query
        if reference_sequences is None:
            raise ValueError("reference_sequences required when motif_reference='fasta'")

        # Get alignment from metadata
        aln = leech_read.metadata.get("alignment")
        if aln is None:
            raise ValueError("Alignment object not found in LeechRead.metadata")

        # Get reference sequence
        ref_name = aln.reference_name
        if ref_name not in reference_sequences:
            logger.warning(f"Reference {ref_name} not found in reference sequences, skipping read")
            return chunks

        ref_seq = reference_sequences[ref_name]

        # Find motif in reference (within aligned region)
        ref_start = aln.reference_start
        ref_end = aln.reference_end
        motif_positions = find_motif_in_reference(ref_seq, motif, ref_start, ref_end)

        # Map each motif position to query coordinates
        focus_bases = []
        motif_len = len(motif)
        for ref_motif_start in motif_positions:
            # Map the motif region to query
            ref_motif_end = ref_motif_start + motif_len
            query_coords = map_reference_to_query_coords(
                aln, ref_motif_start, ref_motif_end, skip_indels=skip_motif_indels
            )

            if query_coords is None:
                continue  # Skip if indels or mapping failed

            query_start, query_end = query_coords

            # Calculate query position of focus base
            # The focus base is offset from the start of the motif
            query_focus_pos = query_start + motif_offset

            # Sanity check: ensure focus position is within the motif region
            if query_start <= query_focus_pos < query_end:
                focus_bases.append(query_focus_pos)
    else:
        raise ValueError(f"Invalid motif_reference: {motif_reference}. Must be 'bam' or 'fasta'")

    # Extract chunks
    for base_idx in focus_bases:
        chunk = leech_read.get_chunk(base_idx)
        if chunk is not None:
            chunk["read_id"] = leech_read.read_id
            # Rename numeric "label" from get_chunk() to "label_int"
            chunk["label_int"] = chunk.pop("label", None)
            # Add string label
            chunk["label"] = label
            chunks.append(chunk)

    return chunks


@dataclass
class ReadInfo:
    """Lightweight container for BAM read information to pass to workers."""

    read_id: str
    sequence: str
    move_table_data: tuple[int, np.ndarray]  # (stride, moves)
    num_samples: int
    trim_offset: int
    mapping_quality: int
    reference_name: str | None
    reference_start: int | None
    reference_end: int | None
    is_reverse: bool
    cigar_tuples: list[tuple[int, int]] | None


def collect_read_infos_from_bam(
    bam_path: Path,
    min_mapq: int = 0,
    require_tags: list[str] | None = None,
) -> list[ReadInfo]:
    """
    First pass: collect lightweight read info from BAM (no POD5 access).

    Args:
        bam_path: Path to BAM file
        min_mapq: Minimum mapping quality
        require_tags: BAM tags that must be present

    Returns:
        List of ReadInfo objects
    """
    if require_tags is None:
        require_tags = ["mv", "ns"]

    read_infos = []

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for aln in bam:
            # Filter alignments
            if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                continue
            if aln.mapping_quality < min_mapq:
                continue

            # Check required tags
            if not all(aln.has_tag(tag) for tag in require_tags):
                continue

            # Check for required query fields
            read_id = aln.query_name
            read_seq = aln.query_sequence
            if read_id is None or read_seq is None:
                continue

            # Extract move table data (lightweight)
            try:
                mv_tag = aln.get_tag("mv")
                if isinstance(mv_tag, np.ndarray):
                    moves = mv_tag
                else:
                    moves = np.array(mv_tag, dtype=np.int16)

                stride = int(moves[0])
                move_table_data = (stride, moves)

                # Extract signal metadata
                num_samples = int(aln.get_tag("ns"))
                trim_offset = int(aln.get_tag("ts")) if aln.has_tag("ts") else 0

                # Create ReadInfo
                read_info = ReadInfo(
                    read_id=read_id,
                    sequence=read_seq,
                    move_table_data=move_table_data,
                    num_samples=num_samples,
                    trim_offset=trim_offset,
                    mapping_quality=aln.mapping_quality,
                    reference_name=aln.reference_name,
                    reference_start=aln.reference_start,
                    reference_end=aln.reference_end,
                    is_reverse=aln.is_reverse,
                    cigar_tuples=aln.cigartuples,
                )
                read_infos.append(read_info)

            except Exception as e:
                logger.warning(f"Skipping read {read_id}: {e}")
                continue

    return read_infos


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
    from leech.features import MoveTable

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
                stride, moves = read_info.move_table_data
                move_table = MoveTable(
                    stride=stride,
                    moves=moves,
                    read_id=read_info.read_id,
                    num_samples=read_info.num_samples,
                    trim_offset=read_info.trim_offset,
                )

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
                    motif_reference=motif_reference,
                    reference_sequences=reference_sequences,
                    skip_motif_indels=skip_motif_indels,
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
    """
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    logger.info(f"Starting parallel data preparation with {num_workers} workers")

    # First pass: collect read info from BAM (lightweight, sequential)
    logger.info("Pass 1: Collecting read info from BAM...")
    read_infos = collect_read_infos_from_bam(bam_path, min_mapq=min_mapq)
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

    for read in iter_bam_with_pod5(bam_path, pod5_path, min_mapq=min_mapq):
        total_reads += 1
        read_chunks = extract_training_chunks(
            read, motif=motif, motif_offset=motif_offset, label=label, label_int=label_int
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


def encode_kmer(sequence: str) -> torch.Tensor:
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


def split_chunks_by_read(
    chunks: list[dict],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split chunks into train/val/test sets at the READ level to prevent data leakage.

    Groups chunks by read_id, then splits the read IDs into train/val/test sets.
    This ensures that no read appears in multiple splits, preventing the model
    from seeing similar signals from the same molecule during training and validation.

    Args:
        chunks: List of chunk dictionaries (must have 'read_id' key)
        train_frac: Fraction of reads for training
        val_frac: Fraction of reads for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_chunks, val_chunks, test_chunks)

    Raises:
        ValueError: If fractions don't sum to <= 1.0
    """
    import random

    if train_frac + val_frac > 1.0:
        raise ValueError(f"train_frac ({train_frac}) + val_frac ({val_frac}) must be <= 1.0")

    # Set seed if provided
    if seed is not None:
        random.seed(seed)

    # Group chunks by read_id
    read_to_chunks: dict[str, list[dict]] = {}
    for chunk in chunks:
        read_id = chunk["read_id"]
        if read_id not in read_to_chunks:
            read_to_chunks[read_id] = []
        read_to_chunks[read_id].append(chunk)

    # Get list of unique read IDs and shuffle
    read_ids = list(read_to_chunks.keys())
    random.shuffle(read_ids)

    # Split read IDs
    n_reads = len(read_ids)
    n_train = int(n_reads * train_frac)
    n_val = int(n_reads * val_frac)

    train_read_ids = set(read_ids[:n_train])
    val_read_ids = set(read_ids[n_train : n_train + n_val])
    test_read_ids = set(read_ids[n_train + n_val :])

    # Assign chunks to splits based on read ID
    train_chunks = []
    val_chunks = []
    test_chunks = []

    for read_id, read_chunks in read_to_chunks.items():
        if read_id in train_read_ids:
            train_chunks.extend(read_chunks)
        elif read_id in val_read_ids:
            val_chunks.extend(read_chunks)
        elif read_id in test_read_ids:
            test_chunks.extend(read_chunks)

    logger.info(f"Split {n_reads} reads into train/val/test")
    logger.info(
        f"  Train: {len(train_read_ids)} reads ({len(train_chunks)} chunks, "
        f"{len(train_chunks) / len(chunks) * 100:.1f}%)"
    )
    logger.info(
        f"  Val: {len(val_read_ids)} reads ({len(val_chunks)} chunks, "
        f"{len(val_chunks) / len(chunks) * 100:.1f}%)"
    )
    logger.info(
        f"  Test: {len(test_read_ids)} reads ({len(test_chunks)} chunks, "
        f"{len(test_chunks) / len(chunks) * 100:.1f}%)"
    )

    return train_chunks, val_chunks, test_chunks


def merge_and_split_chunks(
    input_paths: list[Path],
    output_dir: Path | None = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int | None = None,
    relabel_pairwise: tuple[str, str] | None = None,
) -> tuple[list[dict], list[dict], list[dict]] | dict[str, Any]:
    """
    Merge multiple chunk files and split at read level to prevent data leakage.

    This implements the correct workflow for multi-sample datasets:
    1. Load and merge all chunks from different samples
    2. Split merged data at the READ level into train/val/test
    3. Optionally relabel chunks for pairwise classification
    4. Optionally save splits to disk

    This prevents data leakage that occurs when splitting each sample independently
    and then merging the splits, which can allow reads from the same molecule to
    appear in both training and validation sets.

    Memory-efficient implementation: First pass collects only read IDs and metadata
    to determine splits, second pass loads and assigns chunks to appropriate splits.

    Args:
        input_paths: List of paths to .npz chunk files to merge
        output_dir: Directory to save splits (if None, only return chunks without saving)
        train_frac: Fraction of reads for training
        val_frac: Fraction of reads for validation
        seed: Random seed for reproducibility
        relabel_pairwise: Optional tuple of (label_type1, label_type2) for pairwise comparison.
            If provided, chunks with label_type1 get label=0, label_type2 get label=1.
            Example: ("Ala", "Gly") will assign label=0 to Ala chunks, label=1 to Gly chunks.

    Returns:
        If output_dir is None: Tuple of (train_chunks, val_chunks, test_chunks)
        If output_dir provided: Dictionary with statistics:
        {
            'n_total': int,
            'n_train': int,
            'n_val': int,
            'n_test': int,
            'output_files': {'train': Path, 'val': Path, 'test': Path}
        }

    Example:
        >>> # Merge charged and uncharged samples, then split
        >>> result = merge_and_split_chunks(
        ...     [Path("charged_all.npz"), Path("uncharged_all.npz")],
        ...     output_dir=Path("merged"),
        ...     train_frac=0.7,
        ...     val_frac=0.15,
        ...     seed=42
        ... )

        >>> # Pairwise amino acid comparison with relabeling
        >>> result = merge_and_split_chunks(
        ...     [Path("ala_all.npz"), Path("gly_all.npz")],
        ...     output_dir=Path("merged/Ala_vs_Gly"),
        ...     relabel_pairwise=("Ala", "Gly"),
        ...     seed=42
        ... )
    """
    import random

    from leech.util import setup_random_seed

    if train_frac + val_frac > 1.0:
        raise ValueError(f"train_frac ({train_frac}) + val_frac ({val_frac}) must be <= 1.0")

    # Setup seed and output directory if provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        setup_random_seed(seed, output_dir)
    elif seed is not None:
        # If no output_dir but seed provided, just set the seed
        random.seed(seed)
        logger.info(f"Using provided seed: {seed}")

    logger.info(f"Merging {len(input_paths)} chunk files (memory-efficient mode)")

    # First pass: collect only read IDs from all files (minimal memory)
    logger.info("Pass 1: Collecting read IDs for split assignment")
    all_read_ids: set[str] = set()
    file_chunk_counts = []

    for chunk_path in input_paths:
        logger.info(f"  Scanning {chunk_path}")
        # Load only read_ids array (much smaller than full data)
        with np.load(chunk_path, allow_pickle=True) as data:
            read_ids = data["read_ids"]
            all_read_ids.update(str(rid) for rid in read_ids)
            file_chunk_counts.append(len(read_ids))
            logger.info(f"    Found {len(read_ids)} chunks")

    logger.info(f"Total unique reads across all files: {len(all_read_ids)}")

    # Determine read-level splits
    if seed is not None:
        random.seed(seed)

    read_ids_list = list(all_read_ids)
    random.shuffle(read_ids_list)

    n_reads = len(read_ids_list)
    n_train = int(n_reads * train_frac)
    n_val = int(n_reads * val_frac)

    train_read_ids = set(read_ids_list[:n_train])
    val_read_ids = set(read_ids_list[n_train : n_train + n_val])
    test_read_ids = set(read_ids_list[n_train + n_val :])

    logger.info(f"Split {n_reads} unique reads:")
    logger.info(
        f"  Train: {len(train_read_ids)} reads ({len(train_read_ids) / n_reads * 100:.1f}%)"
    )
    logger.info(f"  Val: {len(val_read_ids)} reads ({len(val_read_ids) / n_reads * 100:.1f}%)")
    logger.info(f"  Test: {len(test_read_ids)} reads ({len(test_read_ids) / n_reads * 100:.1f}%)")

    # Second pass: load chunks and assign to splits (one file at a time)
    logger.info("Pass 2: Loading chunks and assigning to splits")
    train_chunks = []
    val_chunks = []
    test_chunks = []

    for chunk_path in input_paths:
        logger.info(f"  Processing {chunk_path}")
        # Load chunks from this file using the standard loader
        chunks = load_chunks(chunk_path)

        # Relabel chunks if pairwise comparison is requested
        if relabel_pairwise is not None:
            label_type1, label_type2 = relabel_pairwise
            relabeled_count = 0
            skipped_count = 0
            for chunk in chunks:
                chunk_label_type = chunk.get("label_type")
                if chunk_label_type == label_type1:
                    chunk["label"] = 0
                    relabeled_count += 1
                elif chunk_label_type == label_type2:
                    chunk["label"] = 1
                    relabeled_count += 1
                else:
                    # Skip chunks that don't match either label type
                    logger.warning(
                        f"Chunk with label_type='{chunk_label_type}' does not match "
                        f"pairwise comparison ({label_type1}, {label_type2}), keeping original label"
                    )
                    skipped_count += 1
            if relabeled_count > 0:
                logger.info(
                    f"    Relabeled {relabeled_count} chunks for pairwise comparison "
                    f"({label_type1}=0, {label_type2}=1)"
                )
            if skipped_count > 0:
                logger.warning(f"    Skipped {skipped_count} chunks with mismatched label_types")

        # Assign to appropriate split
        for chunk in chunks:
            read_id = chunk["read_id"]
            if read_id in train_read_ids:
                train_chunks.append(chunk)
            elif read_id in val_read_ids:
                val_chunks.append(chunk)
            elif read_id in test_read_ids:
                test_chunks.append(chunk)

        logger.info(f"    Assigned {len(chunks)} chunks")
        # Clear chunks to free memory before loading next file
        del chunks

    n_total = len(train_chunks) + len(val_chunks) + len(test_chunks)
    logger.info("Final chunk distribution:")
    logger.info(f"  Train: {len(train_chunks)} chunks")
    logger.info(f"  Val: {len(val_chunks)} chunks")
    logger.info(f"  Test: {len(test_chunks)} chunks")

    # Check label distribution and warn if all labels are the same
    all_chunks_combined = train_chunks + val_chunks + test_chunks
    unique_labels = {chunk["label"] for chunk in all_chunks_combined if chunk["label"] is not None}
    if len(unique_labels) == 1:
        logger.warning(
            f"⚠️  WARNING: All chunks have the same label ({list(unique_labels)[0]})! "
            "This suggests pairwise relabeling may not have worked correctly. "
            "Check that label_type values match the relabel_pairwise argument."
        )
    elif len(unique_labels) > 0:
        label_counts = {}
        for chunk in all_chunks_combined:
            label = chunk["label"]
            if label is not None:
                label_counts[label] = label_counts.get(label, 0) + 1
        logger.info(f"Label distribution: {label_counts}")

    # If output_dir provided, save splits and return statistics
    if output_dir is not None:
        train_file = output_dir / "train.npz"
        val_file = output_dir / "val.npz"
        test_file = output_dir / "test.npz"

        save_chunks(train_chunks, train_file)
        logger.info(f"Saved {len(train_chunks)} train chunks to {train_file}")

        save_chunks(val_chunks, val_file)
        logger.info(f"Saved {len(val_chunks)} val chunks to {val_file}")

        save_chunks(test_chunks, test_file)
        logger.info(f"Saved {len(test_chunks)} test chunks to {test_file}")

        return {
            "n_total": n_total,
            "n_train": len(train_chunks),
            "n_val": len(val_chunks),
            "n_test": len(test_chunks),
            "output_files": {
                "train": train_file,
                "val": val_file,
                "test": test_file,
            },
        }
    else:
        # Legacy behavior: return tuple of chunks without saving
        return train_chunks, val_chunks, test_chunks


def save_chunks(chunks: list[dict], output_path: Path) -> None:
    """
    Save training chunks to compressed numpy format.

    Args:
        chunks: List of chunk dictionaries from extract_training_chunks
        output_path: Output file path (.npz)

    Format:
        Saves as .npz with arrays:
        - signals: (N, signal_len) raw signal chunks
        - sequences: (N,) string array of k-mer sequences
        - dwells: (N, kmer_len) dwell times
        - features: (N, num_features, kmer_len) feature arrays
        - labels: (N,) integer labels
        - read_ids: (N,) string array of read IDs
        - base_indices: (N,) base indices
    """
    if not chunks:
        raise ValueError("No chunks to save")

    # Collect arrays
    signals = []
    sequences = []
    dwells = []
    features = []
    labels = []
    labels_int = []
    read_ids = []
    base_indices = []

    for chunk in chunks:
        signals.append(chunk["signal"])
        sequences.append(chunk["sequence"])
        dwells.append(chunk["dwell"])
        features.append(chunk["features"])
        labels.append(chunk.get("label", ""))  # String label (e.g., "Ala", "Gly")
        labels_int.append(
            chunk.get("label_int", -1) if chunk.get("label_int") is not None else -1
        )  # Numeric label or -1
        read_ids.append(chunk["read_id"])
        base_indices.append(chunk["base_idx"])

    # Convert to arrays
    # Signals may have variable length, so we'll save them as object array
    signals_arr = np.array(signals, dtype=object)
    sequences_arr = np.array(sequences, dtype=str)
    dwells_arr = np.array(dwells, dtype=object)
    features_arr = np.array(features, dtype=object)
    labels_arr = np.array(labels, dtype=str)  # String labels
    labels_int_arr = np.array(labels_int, dtype=np.int64)  # Numeric labels
    read_ids_arr = np.array(read_ids, dtype=str)
    base_indices_arr = np.array(base_indices, dtype=np.int64)

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    np.savez_compressed(
        output_path,
        signals=signals_arr,
        sequences=sequences_arr,
        dwells=dwells_arr,
        features=features_arr,
        labels=labels_arr,
        labels_int=labels_int_arr,
        read_ids=read_ids_arr,
        base_indices=base_indices_arr,
    )

    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(input_path: Path) -> list[dict]:
    """
    Load training chunks from compressed numpy format.

    Args:
        input_path: Path to .npz file

    Returns:
        List of chunk dictionaries compatible with extract_training_chunks output

    Note:
        This function loads all arrays into memory at once. The arrays are stored
        as numpy object arrays (dtype=object) to handle variable-length signals.
        The loaded data is kept in memory-mapped form when possible, but converting
        to individual dictionaries will create copies in memory.
    """
    # Load all arrays at once (keeps data memory-mapped when possible)
    with np.load(input_path, allow_pickle=True) as data:
        # Extract all arrays first (this creates copies but only once)
        signals = data["signals"]
        sequences = data["sequences"]
        dwells = data["dwells"]
        features = data["features"]
        labels_arr = data["labels"]  # String labels
        labels_int_arr = data["labels_int"]  # Numeric labels
        read_ids = data["read_ids"]
        base_indices = data["base_indices"]

        n_chunks = len(labels_arr)
        chunks = []

        # Create dictionaries with references to array elements
        # This is more memory efficient than accessing data[key][i] each time
        for i in range(n_chunks):
            chunk = {
                "signal": signals[i],
                "sequence": str(sequences[i]),
                "dwell": dwells[i],
                "features": features[i],
                "read_id": str(read_ids[i]),
                "base_idx": int(base_indices[i]),
                "label": str(labels_arr[i]) if labels_arr[i] != "" else None,
                "label_int": int(labels_int_arr[i]) if labels_int_arr[i] >= 0 else None,
            }

            chunks.append(chunk)

    return chunks


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
    """Prepare training data from POD5/BAM files with optional splitting.

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
        Dictionary with statistics:
        {
            'n_chunks': int,
            'n_train': int,
            'n_val': int,
            'n_test': int,
            'output_files': {'train': Path, 'val': Path, 'test': Path} or {'all': Path}
        }
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
            motif_reference=motif_reference,
            reference_sequences=reference_sequences,
            skip_motif_indels=skip_motif_indels,
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
