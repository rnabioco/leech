"""
BAM file reading utilities with alignment filtering and move table extraction.

Provides efficient reading and filtering of BAM alignments with ONT-specific
tags (mv, ns, ts) required for nanopore signal analysis.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import pysam

from leech.features import MoveTable, extract_move_table

logger = logging.getLogger("leech.io.bam_reader")


def iter_bam_alignments(
    bam_path: Path,
    min_mapq: int = 0,
    require_tags: list[str] | None = None,
    include_unmapped: bool = False,
    include_secondary: bool = False,
    include_supplementary: bool = False,
) -> Iterator[pysam.AlignedSegment]:
    """
    Iterate over BAM alignments with filtering.

    Args:
        bam_path: Path to BAM file
        min_mapq: Minimum mapping quality
        require_tags: List of required BAM tags (default: ["mv", "ns"])
        include_unmapped: Include unmapped reads
        include_secondary: Include secondary alignments
        include_supplementary: Include supplementary alignments

    Yields:
        Filtered BAM alignments

    Example:
        >>> for aln in iter_bam_alignments(Path("alignments.bam"), min_mapq=10):
        ...     print(f"{aln.query_name}: {aln.mapping_quality}")
    """
    if require_tags is None:
        require_tags = ["mv", "ns"]

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for aln in bam:
            # Filter by alignment flags
            if not include_unmapped and aln.is_unmapped:
                continue
            if not include_secondary and aln.is_secondary:
                continue
            if not include_supplementary and aln.is_supplementary:
                continue

            # Filter by mapping quality
            if aln.mapping_quality < min_mapq:
                continue

            # Check required tags
            if not all(aln.has_tag(tag) for tag in require_tags):
                continue

            # Check for required fields
            if aln.query_name is None or aln.query_sequence is None:
                continue

            yield aln


class ReadInfo:
    """
    Lightweight container for BAM read information.

    Used for passing read metadata to workers without heavy alignment objects.
    """

    read_id: str
    sequence: str

    def __init__(self, aln: pysam.AlignedSegment):
        """
        Extract read info from alignment.

        Args:
            aln: BAM alignment
        """
        # These should not be None for properly aligned reads
        assert aln.query_name is not None, "Read ID is None"
        assert aln.query_sequence is not None, "Sequence is None"

        self.read_id = aln.query_name
        self.sequence = aln.query_sequence
        self.mapping_quality = aln.mapping_quality
        self.reference_name = aln.reference_name
        self.reference_start = aln.reference_start
        self.reference_end = aln.reference_end
        self.is_reverse = aln.is_reverse
        self.cigar_tuples = aln.cigartuples

        # Extract move table data
        # mv_tag is an array: [stride, move1, move2, ...]
        mv_tag = aln.get_tag("mv")
        # Type narrowing: we know mv_tag is array-like
        assert hasattr(mv_tag, "__getitem__"), "mv tag must be indexable"
        self.stride = int(mv_tag[0])  # type: ignore[index]
        self.moves = mv_tag[1:]  # type: ignore[index]
        self.num_samples = int(aln.get_tag("ns"))
        self.trim_offset = int(aln.get_tag("ts")) if aln.has_tag("ts") else 0

    def to_move_table(self) -> MoveTable:
        """
        Reconstruct MoveTable from stored data.

        Returns:
            MoveTable object
        """
        import numpy as np

        return MoveTable(
            stride=self.stride,
            moves=np.array(self.moves, dtype=np.int8),
            read_id=self.read_id,
            num_samples=self.num_samples,
            trim_offset=self.trim_offset,
        )


def collect_read_infos(
    bam_path: Path,
    min_mapq: int = 0,
    require_tags: list[str] | None = None,
) -> list[ReadInfo]:
    """
    Collect lightweight read information from BAM file.

    This is useful for two-pass processing where you first collect metadata,
    then process reads in parallel.

    Args:
        bam_path: Path to BAM file
        min_mapq: Minimum mapping quality
        require_tags: List of required BAM tags (default: ["mv", "ns"])

    Returns:
        List of ReadInfo objects

    Example:
        >>> read_infos = collect_read_infos(Path("alignments.bam"))
        >>> print(f"Found {len(read_infos)} reads")
        >>> for info in read_infos[:5]:
        ...     print(f"{info.read_id}: {len(info.sequence)} bases")
    """
    if require_tags is None:
        require_tags = ["mv", "ns"]

    read_infos = []

    for aln in iter_bam_alignments(bam_path, min_mapq=min_mapq, require_tags=require_tags):
        try:
            read_info = ReadInfo(aln)
            read_infos.append(read_info)
        except Exception as e:
            logger.warning(f"Skipping read {aln.query_name}: {e}")
            continue

    logger.info(f"Collected {len(read_infos)} read infos from {bam_path}")
    return read_infos


class BAMReader:
    """
    Context manager for efficient BAM reading.

    Provides high-level interface for reading BAM alignments with filtering.

    Example:
        >>> with BAMReader(Path("alignments.bam"), min_mapq=10) as reader:
        ...     for aln in reader.iter_alignments():
        ...         move_table = reader.extract_move_table(aln)
        ...         print(f"{aln.query_name}: {move_table.num_bases} bases")
    """

    def __init__(
        self,
        bam_path: Path,
        min_mapq: int = 0,
        require_tags: list[str] | None = None,
    ):
        """
        Initialize BAM reader.

        Args:
            bam_path: Path to BAM file
            min_mapq: Minimum mapping quality
            require_tags: List of required BAM tags (default: ["mv", "ns"])
        """
        self.bam_path = bam_path
        self.min_mapq = min_mapq
        self.require_tags = require_tags if require_tags is not None else ["mv", "ns"]
        self._bam = None

    def __enter__(self):
        """Open BAM file."""
        self._bam = pysam.AlignmentFile(str(self.bam_path), "rb")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close BAM file."""
        if self._bam is not None:
            self._bam.close()

    def iter_alignments(self) -> Iterator[pysam.AlignedSegment]:
        """
        Iterate over filtered alignments.

        Yields:
            Filtered BAM alignments

        Raises:
            RuntimeError: If reader not opened (use as context manager)
        """
        if self._bam is None:
            raise RuntimeError("BAMReader must be used as a context manager")

        for aln in self._bam:
            # Apply filters
            if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                continue
            if aln.mapping_quality < self.min_mapq:
                continue
            if not all(aln.has_tag(tag) for tag in self.require_tags):
                continue
            if aln.query_name is None or aln.query_sequence is None:
                continue

            yield aln

    @staticmethod
    def extract_move_table(aln: pysam.AlignedSegment) -> MoveTable:
        """
        Extract move table from alignment.

        Args:
            aln: BAM alignment

        Returns:
            MoveTable object

        Raises:
            ValueError: If required tags missing
        """
        return extract_move_table(aln)

    def get_header(self) -> pysam.AlignmentHeader:
        """
        Get BAM header.

        Returns:
            BAM header

        Raises:
            RuntimeError: If reader not opened (use as context manager)
        """
        if self._bam is None:
            raise RuntimeError("BAMReader must be used as a context manager")

        return self._bam.header

    def count_alignments(self) -> int:
        """
        Count filtered alignments in BAM.

        Returns:
            Number of alignments passing filters

        Raises:
            RuntimeError: If reader not opened (use as context manager)
        """
        if self._bam is None:
            raise RuntimeError("BAMReader must be used as a context manager")

        count = sum(1 for _ in self.iter_alignments())
        # Re-open to reset iterator
        self._bam.close()
        self._bam = pysam.AlignmentFile(str(self.bam_path), "rb")
        return count
