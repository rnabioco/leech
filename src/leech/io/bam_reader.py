"""
BAM file reading utilities with alignment filtering and move table extraction.

Provides efficient reading and filtering of BAM alignments with ONT-specific
tags (mv, ns, ts) required for nanopore signal analysis.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import pysam

from leech.constants import REQUIRED_BAM_TAGS
from leech.features import MoveTable, extract_move_table

logger = logging.getLogger("leech.io.bam_reader")


def count_bam_reads(bam_path: Path) -> int:
    """Count mapped reads from BAM index (O(1), no iteration)."""
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        return bam.mapped


def iter_bam_batches(
    bam_path: Path,
    batch_size: int = 200_000,
    min_mapq: int = 0,
    require_tags: list[str] | None = None,
) -> Iterator[list[pysam.AlignedSegment]]:
    """
    Yield lists of pysam.AlignedSegment in mega-batches.

    Used for streaming inference to bound memory usage. Each batch contains
    up to ``batch_size`` alignments in file order.

    Args:
        bam_path: Path to BAM file
        batch_size: Maximum alignments per batch
        min_mapq: Minimum mapping quality
        require_tags: List of required BAM tags (default: ["mv", "ns"])

    Yields:
        Lists of filtered BAM alignments, each up to batch_size long
    """
    batch: list[pysam.AlignedSegment] = []
    for aln in iter_bam_alignments(bam_path, min_mapq=min_mapq, require_tags=require_tags):
        batch.append(aln)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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
        require_tags = REQUIRED_BAM_TAGS

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

        # Reference sequence: reconstructed on first read, not here. See the
        # `reference_sequence` property -- `get_reference_sequence()` walks the
        # MD tag and the CIGAR and is most of the cost of building a ReadInfo
        # (74% of it on 6.8 kb reads), while only reference-anchored runs read
        # the result.
        self._aln: pysam.AlignedSegment | None = aln
        self._reference_sequence: str | None = None

        # Extract move table data
        # mv_tag is an array: [stride, move1, move2, ...]
        mv_tag = aln.get_tag("mv")
        # Type narrowing: we know mv_tag is array-like
        assert hasattr(mv_tag, "__getitem__"), "mv tag must be indexable"
        self.stride = int(mv_tag[0])
        self.moves = mv_tag[1:]
        self.num_samples = int(aln.get_tag("ns"))
        self.trim_offset = int(aln.get_tag("ts")) if aln.has_tag("ts") else 0

        # Optional charging level tag (CL)
        try:
            cl_tag = aln.get_tag("CL")
            self.cl_value: int | None = (
                int(cl_tag[0]) if hasattr(cl_tag, "__getitem__") else int(cl_tag)
            )
        except (KeyError, TypeError):
            self.cl_value = None

    @property
    def reference_sequence(self) -> str | None:
        """Reference sequence over the aligned region, or None if unavailable.

        Reconstructed from the alignment on first access and cached. The
        alignment reference is dropped at the same moment, for two reasons: a
        ``pysam.AlignedSegment`` is several KB per read (1.3-1.7x a ReadInfo,
        measured) and this class exists to be the *lightweight* stand-in, and
        it is not picklable, while ReadInfo is sent to multiprocessing workers
        by both prepare and inference.

        Which is also why :meth:`__getstate__` forces the value out before
        pickling. Returning None to a worker instead would not raise anywhere:
        ``build_leech_read`` would quietly fall back to the basecalled
        sequence, and an ``anchor="reference"`` run would cut every chunk in
        the wrong coordinate frame.
        """
        aln = self._aln
        if aln is None:
            return self._reference_sequence
        # One attempt, cached either way; drop the alignment before anything
        # can raise so a failure cannot leave it pinned.
        self._aln = None
        try:
            self._reference_sequence = aln.get_reference_sequence()
        except Exception as e:
            logger.debug("Could not get reference sequence for %s: %s", self.read_id, e)
            self._reference_sequence = None
        return self._reference_sequence

    def materialize_reference_sequence(self) -> None:
        """Resolve :attr:`reference_sequence` now and release the alignment.

        For callers that hold many ReadInfos at once and have no alignment list
        of their own keeping those objects alive anyway (:func:`collect_read_infos`).
        """
        _ = self.reference_sequence

    def __getstate__(self) -> dict:
        """Pickle support: resolve the reference sequence, drop the alignment."""
        _ = self.reference_sequence  # also clears self._aln
        return self.__dict__

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

    def to_mock_alignment(self) -> "MockAlignment":
        """
        Create a lightweight mock alignment for reference-based motif search.

        Returns a picklable object with the same attributes that
        ``ReferenceMotifSearcher`` reads from ``pysam.AlignedSegment``.

        Returns:
            MockAlignment instance
        """
        return MockAlignment(self)


class MockAlignment:
    """Lightweight, picklable stand-in for ``pysam.AlignedSegment``.

    Only exposes the attributes used by ``ReferenceMotifSearcher``.
    """

    __slots__ = (
        "reference_name",
        "reference_start",
        "reference_end",
        "cigartuples",
        "is_reverse",
    )

    def __init__(self, read_info: "ReadInfo"):
        self.reference_name = read_info.reference_name
        self.reference_start = read_info.reference_start
        self.reference_end = read_info.reference_end
        self.cigartuples = read_info.cigar_tuples
        self.is_reverse = read_info.is_reverse


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
        require_tags = REQUIRED_BAM_TAGS

    read_infos = []

    for aln in iter_bam_alignments(bam_path, min_mapq=min_mapq, require_tags=require_tags):
        try:
            read_info = ReadInfo(aln)
            # This one collects the whole BAM into a list with nothing else
            # holding the alignments alive, so the lazy reference sequence is
            # resolved here rather than pinning an AlignedSegment per read for
            # the life of the list. Use `iter_read_info_batches` to get the
            # laziness with bounded memory.
            read_info.materialize_reference_sequence()
            read_infos.append(read_info)
        except Exception as e:
            logger.warning(f"Skipping read {aln.query_name}: {e}")
            continue

    logger.info(f"Collected {len(read_infos)} read infos from {bam_path}")
    return read_infos


def iter_read_info_batches(
    bam_path: Path,
    batch_size: int = 5000,
    min_mapq: int = 0,
    require_tags: list[str] | None = None,
) -> Iterator[list[ReadInfo]]:
    """
    Yield batches of ReadInfo objects from a BAM file.

    Streaming alternative to collect_read_infos() that allows overlapping
    BAM reading with downstream processing.

    Unlike ``collect_read_infos`` this keeps :attr:`ReadInfo.reference_sequence`
    lazy: a batch is bounded, so pinning its alignments until each read is
    either used or pickled costs a few MB, and a run that never asks for the
    reference sequence (``anchor="basecall"``) never pays to rebuild it.

    Args:
        bam_path: Path to BAM file
        batch_size: Number of reads per batch
        min_mapq: Minimum mapping quality
        require_tags: List of required BAM tags (default: ["mv", "ns"])

    Yields:
        Lists of ReadInfo objects, each up to batch_size long
    """
    if require_tags is None:
        require_tags = REQUIRED_BAM_TAGS

    batch: list[ReadInfo] = []
    for aln in iter_bam_alignments(bam_path, min_mapq=min_mapq, require_tags=require_tags):
        try:
            batch.append(ReadInfo(aln))
        except Exception as e:
            logger.warning(f"Skipping read {aln.query_name}: {e}")
            continue
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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
        self.require_tags = require_tags if require_tags is not None else REQUIRED_BAM_TAGS
        self._bam = None

    def __enter__(self):
        """Open BAM file."""
        self._bam = pysam.AlignmentFile(str(self.bam_path), "rb")
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
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
