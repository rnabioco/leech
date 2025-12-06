"""
BED file reading and region-based position finding.

Provides functionality for loading BED files and finding training chunk
positions based on BED regions instead of motif search.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pysam

from leech.io.motif_search import map_reference_to_query_coords

logger = logging.getLogger("leech.io.bed_reader")


@dataclass
class BedRegion:
    """
    A single BED region.

    Attributes:
        chrom: Chromosome/reference name
        start: Start position (0-based)
        end: End position (0-based, exclusive)
        name: Optional name field (column 4), used as label
    """

    chrom: str
    start: int
    end: int
    name: str | None = None

    @property
    def center(self) -> int:
        """Return the center position of the region (0-based)."""
        return (self.start + self.end) // 2


class BedIndex:
    """
    Index BED regions by chromosome for efficient overlap queries.

    Stores regions indexed by chromosome name for quick lookup of
    regions overlapping a given genomic interval.
    """

    def __init__(self, regions: dict[str, list[BedRegion]] | None = None):
        """
        Initialize BedIndex.

        Args:
            regions: Optional dict mapping chromosome names to lists of BedRegions
        """
        self.regions: dict[str, list[BedRegion]] = regions if regions is not None else {}

    def add_region(self, region: BedRegion) -> None:
        """Add a region to the index."""
        if region.chrom not in self.regions:
            self.regions[region.chrom] = []
        self.regions[region.chrom].append(region)

    def get_overlapping_regions(
        self, chrom: str, start: int, end: int
    ) -> list[BedRegion]:
        """
        Find all BED regions overlapping a given interval.

        Args:
            chrom: Chromosome/reference name
            start: Start position (0-based)
            end: End position (0-based, exclusive)

        Returns:
            List of BedRegion objects that overlap [start, end)
        """
        if chrom not in self.regions:
            return []

        overlapping = []
        for region in self.regions[chrom]:
            # Check for overlap: regions overlap if neither is completely before the other
            if region.start < end and region.end > start:
                overlapping.append(region)

        return overlapping

    def get_all_chromosomes(self) -> list[str]:
        """Return list of all chromosomes with regions."""
        return list(self.regions.keys())

    def total_regions(self) -> int:
        """Return total number of regions across all chromosomes."""
        return sum(len(regions) for regions in self.regions.values())


def load_bed_regions(bed_path: Path) -> BedIndex:
    """
    Load BED file and create an indexed structure.

    Supports BED3 (chrom, start, end) through BED6 formats.
    The name field (column 4) is used as the label if present.

    Args:
        bed_path: Path to BED file

    Returns:
        BedIndex with all regions loaded

    Raises:
        ValueError: If BED file format is invalid
        FileNotFoundError: If BED file does not exist
    """
    if not bed_path.exists():
        raise FileNotFoundError(f"BED file not found: {bed_path}")

    index = BedIndex()

    with open(bed_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue

            fields = line.split("\t")

            # Validate minimum fields (BED3)
            if len(fields) < 3:
                raise ValueError(
                    f"Invalid BED format at line {line_num}: expected at least 3 fields, got {len(fields)}"
                )

            try:
                chrom = fields[0]
                start = int(fields[1])
                end = int(fields[2])
            except ValueError as e:
                raise ValueError(
                    f"Invalid BED format at line {line_num}: could not parse coordinates - {e}"
                ) from e

            # Validate coordinates
            if start < 0:
                raise ValueError(f"Invalid BED format at line {line_num}: start position cannot be negative")
            if end <= start:
                raise ValueError(f"Invalid BED format at line {line_num}: end must be greater than start")

            # Get name if present (column 4)
            name = None
            if len(fields) >= 4:
                name = fields[3] if fields[3] != "." else None

            region = BedRegion(chrom=chrom, start=start, end=end, name=name)
            index.add_region(region)

    logger.info(f"Loaded {index.total_regions()} regions from {bed_path}")

    if index.total_regions() == 0:
        logger.warning(f"BED file {bed_path} contains no regions")

    return index


class BedRegionSearcher:
    """
    Find focus positions based on BED regions.

    For each read, finds overlapping BED regions and returns the
    center of each region mapped to query coordinates.
    """

    def __init__(
        self,
        bed_index: BedIndex,
        skip_indels: bool = True,
        default_label: str | None = None,
    ):
        """
        Initialize BedRegionSearcher.

        Args:
            bed_index: Index of BED regions
            skip_indels: If True, skip positions where center falls in indel
            default_label: Label to use when BED name field is missing or '.'
        """
        self.bed_index = bed_index
        self.skip_indels = skip_indels
        self.default_label = default_label

    def find_positions_with_labels(
        self,
        read_id: str,
        sequence: str,
        alignment: pysam.AlignedSegment | None,
    ) -> list[tuple[int, str | None]]:
        """
        Find BED region centers overlapping this read, mapped to query coordinates.

        Args:
            read_id: Read identifier (for logging)
            sequence: Basecalled sequence (unused, for interface compatibility)
            alignment: BAM alignment (required)

        Returns:
            List of (query_position, label) tuples where:
            - query_position is the center of BED region in query coordinates
            - label is from BED name field or default_label
        """
        if alignment is None:
            logger.debug(f"No alignment for read {read_id}, skipping BED search")
            return []

        ref_name = alignment.reference_name
        if ref_name is None:
            return []

        ref_start = alignment.reference_start
        ref_end = alignment.reference_end
        if ref_end is None:
            return []

        # Find overlapping BED regions
        overlapping = self.bed_index.get_overlapping_regions(ref_name, ref_start, ref_end)

        results = []
        for region in overlapping:
            # Get center position in reference coordinates
            ref_center = region.center

            # Check if center is within aligned region
            if ref_center < ref_start or ref_center >= ref_end:
                continue

            # Map single base at center to query coordinates
            query_coords = map_reference_to_query_coords(
                alignment,
                ref_center,
                ref_center + 1,  # Single base
                skip_indels=self.skip_indels,
            )

            if query_coords is None:
                logger.debug(
                    f"Read {read_id}: Could not map BED region center "
                    f"({region.chrom}:{ref_center}) to query coordinates"
                )
                continue

            query_pos = query_coords[0]

            # Determine label: use BED name if present, else default
            label = region.name if region.name is not None else self.default_label

            results.append((query_pos, label))

        return results
