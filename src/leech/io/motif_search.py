"""
Motif search strategies for finding modification sites.

Provides different strategies for searching motifs in nanopore reads:
- BasecalledMotifSearcher: Search in basecalled sequence
- ReferenceMotifSearcher: Search in reference sequence (avoids basecalling errors)
"""

import logging
from abc import ABC, abstractmethod
from typing import NamedTuple

import pysam

logger = logging.getLogger("leech.io.motif_search")


class MotifMatch(NamedTuple):
    """One motif hit: a position plus the CIGAR-measured junction disruption.

    ``position`` is in whichever coordinate frame the searcher returns
    (reference-relative under ``anchor="reference"``, query-relative
    otherwise) — exactly what a bare ``int`` used to mean.

    ``junction_indel`` is ``mapped_len - len(motif)`` for the motif's mapped
    query span through the CIGAR: 0 when the alignment covers the motif at
    exactly its length, nonzero when an indel inside (or adjoining) the motif
    changed the mapped span's length. ``junction_mapped`` says whether that
    measurement was even possible — False when there is no alignment/CIGAR to
    measure against (:class:`BasecalledMotifSearcher`, or no-motif/all-bases
    mode) or the CIGAR mapping failed outright (the motif's edges do not both
    land in a match/mismatch block).

    Neither field gates whether a match is returned here — see
    :class:`ReferenceMotifSearcher` and issue #282, which introduced this: the
    charging classifier's honest failure mode is an uncharged read whose
    junction basecalls badly for ordinary reasons, and both prepare (as a
    per-chunk field for sampling/abstention) and predict (as an abstention
    rule) need this measurement even for motifs that ``require_query_mapping``
    keeps despite a bad mapping.
    """

    position: int
    junction_indel: int = 0
    junction_mapped: bool = False


def map_reference_to_query_coords(
    aln: pysam.AlignedSegment,
    ref_start: int,
    ref_end: int,
    skip_indels: bool = False,
    allow_edge_indels: bool = False,
) -> tuple[int, int] | None:
    """
    Map reference coordinates to query coordinates using CIGAR string.

    Args:
        aln: Aligned segment from BAM
        ref_start: Start position in reference (0-based)
        ref_end: End position in reference (0-based, exclusive)
        skip_indels: If True, return None if indels found in region
        allow_edge_indels: If True, only check for indels in core region (±1bp from edges)

    Returns:
        Tuple of (query_start, query_end) or None if mapping fails
        or indels detected (when skip_indels=True)

    Examples:
        >>> # Map reference positions 100-106 to query coordinates
        >>> query_coords = map_reference_to_query_coords(aln, 100, 106)
        >>> if query_coords:
        ...     query_start, query_end = query_coords
        ...     motif_seq = aln.query_sequence[query_start:query_end]
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
            # Check if ref_start falls within this block
            if query_start is None and ref_start >= ref_pos and ref_start < ref_pos + length:
                query_start = query_pos + (ref_start - ref_pos)
            if query_end is None and ref_end - 1 >= ref_pos and ref_end - 1 < ref_pos + length:
                query_end = query_pos + (ref_end - ref_pos)
            ref_pos += length
            query_pos += length

        # I: insertion (consumes query only)
        elif op == 1:  # BAM_CINS
            # Check if indel is in region (or core region if allow_edge_indels)
            if allow_edge_indels:
                # Only check core region (exclude ±3bp edges for 7bp motif)
                # For CCATGGC (7bp), this checks only middle position (amino acid site)
                motif_len = ref_end - ref_start
                edge_tolerance = min(3, motif_len // 2)  # ±3bp or half motif length
                core_start = ref_start + edge_tolerance
                core_end = ref_end - edge_tolerance
                if ref_pos >= core_start and ref_pos < core_end:
                    has_indel_in_region = True
            else:
                # Check entire region
                if ref_pos >= ref_start and ref_pos < ref_end:
                    has_indel_in_region = True
            query_pos += length

        # D: deletion (consumes reference only)
        elif op == 2:  # BAM_CDEL
            # Check if indel is in region (or core region if allow_edge_indels)
            if allow_edge_indels:
                # Only check core region (exclude ±3bp edges for 7bp motif)
                motif_len = ref_end - ref_start
                edge_tolerance = min(3, motif_len // 2)  # ±3bp or half motif length
                core_start = ref_start + edge_tolerance
                core_end = ref_end - edge_tolerance
                if ref_pos >= core_start and ref_pos + length > core_start:
                    has_indel_in_region = True
            else:
                # Check entire region
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


def find_motif_in_sequence(
    sequence: str, motif: str, start: int = 0, end: int | None = None
) -> list[int]:
    """
    Find all occurrences of motif in sequence.

    Args:
        sequence: DNA sequence to search
        motif: Motif to search for
        start: Start position in sequence (default: 0)
        end: End position in sequence (default: len(sequence))

    Returns:
        List of positions where motif starts (0-based)

    Examples:
        >>> positions = find_motif_in_sequence("ACGTCCAGGCTTCCAGGC", "CCAGGC")
        >>> print(positions)  # [4, 12]
    """
    if end is None:
        end = len(sequence)

    positions = []
    search_region = sequence[start:end].upper()
    motif_upper = motif.upper()
    pos = search_region.find(motif_upper)
    while pos != -1:
        positions.append(start + pos)
        pos = search_region.find(motif_upper, pos + 1)

    return positions


class MotifSearcher(ABC):
    """
    Abstract base class for motif search strategies.

    Subclasses implement different strategies for finding motifs in reads.
    """

    @abstractmethod
    def find_motif_positions(
        self, read_id: str, sequence: str, alignment: pysam.AlignedSegment | None, motif: str
    ) -> list[MotifMatch]:
        """
        Find positions of motif in read.

        Args:
            read_id: Read identifier
            sequence: Basecalled sequence
            alignment: BAM alignment (may be None for basecalled search)
            motif: Motif to search for

        Returns:
            List of :class:`MotifMatch` — position (0-based, in whichever
            coordinate frame this searcher returns) plus the CIGAR-measured
            junction disruption at that position, when one could be measured.
        """
        pass


class BasecalledMotifSearcher(MotifSearcher):
    """
    Search motif in basecalled sequence.

    This is the original behavior - search directly in the basecalled sequence.
    May be affected by basecalling errors at modification sites.

    Examples:
        >>> searcher = BasecalledMotifSearcher()
        >>> positions = searcher.find_motif_positions(
        ...     read_id="read_001",
        ...     sequence="ACGTCCAGGCTT",
        ...     alignment=None,
        ...     motif="CCAGGC"
        ... )
        >>> print(positions)  # [4]
    """

    def find_motif_positions(
        self, read_id: str, sequence: str, alignment: pysam.AlignedSegment | None, motif: str
    ) -> list[MotifMatch]:
        """
        Find motif in basecalled sequence.

        Args:
            read_id: Read identifier (unused)
            sequence: Basecalled sequence
            alignment: BAM alignment (unused)
            motif: Motif to search for

        Returns:
            List of positions where motif starts in basecalled sequence, each
            with ``junction_mapped=False`` -- there is no alignment/CIGAR here
            to measure a junction indel against.
        """
        return [MotifMatch(pos) for pos in find_motif_in_sequence(sequence, motif)]


class ReferenceMotifSearcher(MotifSearcher):
    """
    Search motif in reference sequence, then map to query.

    This avoids basecalling errors at modification sites by searching in the
    reference sequence and mapping positions to the query via CIGAR.

    Examples:
        >>> searcher = ReferenceMotifSearcher(
        ...     reference_sequences={"chr1": "ACGTCCAGGCTT..."},
        ...     skip_indels=True
        ... )
        >>> positions = searcher.find_motif_positions(
        ...     read_id="read_001",
        ...     sequence="...",
        ...     alignment=bam_alignment,
        ...     motif="CCAGGC"
        ... )
    """

    def __init__(
        self,
        reference_sequences: dict[str, str],
        skip_indels: bool = False,
        allow_edge_indels: bool = False,
        debug: bool = False,
        anchor: str = "reference",
        require_query_mapping: bool = True,
    ):
        """
        Initialize reference-based motif searcher.

        Args:
            reference_sequences: Dict mapping reference name to sequence
            skip_indels: If True, skip motif positions with indels in region
            allow_edge_indels: If True, only reject indels in core motif (not ±1bp edges)
            debug: If True, collect and log detailed statistics
            anchor: "reference" (return ref-relative coords, default) or "basecall" (query coords)
            require_query_mapping: If True (default), a motif is accepted only when
                it also maps cleanly to query coordinates through the CIGAR. Under
                ``anchor="reference"`` that mapping is used ONLY to accept or
                reject -- the returned coordinate is reference-relative and the
                query coordinates are discarded -- so the check is a quality gate
                rather than a requirement of window placement, and reads failing
                it can be kept without changing where any chunk is cut. Set False
                to keep them.

                This matters when the modification under study perturbs the
                basecall at the motif itself: on aminoacyl-tRNA the adduct
                mis-calls the CCA junction, and the resulting CIGAR indels drop
                28% of charged reads against 6% of uncharged. That is selection
                on the label, applied before any model sees the data.

                Only valid with ``anchor="reference"``; see ``find_motif_positions``.
        """
        if require_query_mapping is False and anchor != "reference":
            raise ValueError(
                "require_query_mapping=False is only valid with anchor='reference'. "
                f"Got anchor={anchor!r}: under 'basecall' the returned coordinate "
                "IS the query start, so the mapping cannot be skipped."
            )
        self.reference_sequences = reference_sequences
        self.skip_indels = skip_indels
        self.allow_edge_indels = allow_edge_indels
        self.debug = debug
        self.anchor = anchor
        self.require_query_mapping = require_query_mapping

        # Debug statistics
        self.stats = {
            "motifs_in_reference": 0,
            "failed_cigar_mapping": 0,
            "failed_indels": 0,
            "failed_length_check": 0,
            "accepted_without_query_mapping": 0,
            "successful": 0,
        }

    def get_stats(self):
        """Return accumulated statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0

    def find_motif_positions(
        self, read_id: str, sequence: str, alignment: pysam.AlignedSegment | None, motif: str
    ) -> list[MotifMatch]:
        """
        Find motif in reference, then map to query coordinates.

        Args:
            read_id: Read identifier
            sequence: Basecalled sequence (unused)
            alignment: BAM alignment (required for reference search)
            motif: Motif to search for

        Returns:
            List of :class:`MotifMatch` for positions where the motif starts
            in query sequence (or reference sequence, under
            ``anchor="reference"``) — see ``junction_indel``/``junction_mapped``
            below for what the CIGAR measurement means.
        """
        if alignment is None:
            logger.warning(f"Reference-based search requires alignment, but got None for {read_id}")
            return []

        # Get reference sequence
        ref_name = alignment.reference_name
        if ref_name not in self.reference_sequences:
            logger.warning(f"Reference {ref_name} not found for read {read_id}")
            return []

        ref_seq = self.reference_sequences[ref_name]

        # Find motif in reference (within aligned region)
        ref_start = alignment.reference_start
        ref_end = alignment.reference_end

        if ref_end is None:
            return []

        motif_positions = find_motif_in_sequence(ref_seq, motif, ref_start, ref_end)

        # Track statistics
        if self.debug:
            self.stats["motifs_in_reference"] += len(motif_positions)

        # Map each motif position to query coordinates
        query_positions: list[MotifMatch] = []
        motif_len = len(motif)

        for ref_motif_start in motif_positions:
            ref_motif_end = ref_motif_start + motif_len

            # Always measure the mapped query span through the CIGAR --
            # skip_indels=False regardless of self.skip_indels, so this walks
            # across an indel rather than refusing to (issue #282). This is
            # measurement, not gating: `allow_edge_indels` only affects
            # whether has_indel_in_region trips the `skip_indels`-driven early
            # return inside map_reference_to_query_coords, which never fires
            # here, so its value is irrelevant to what this call returns.
            #
            # In the common case (self.skip_indels is False), the acceptance
            # gate below reuses this exact call instead of re-walking the
            # CIGAR a second time with identical arguments.
            measured = map_reference_to_query_coords(
                alignment, ref_motif_start, ref_motif_end, skip_indels=False
            )
            if measured is not None:
                m_start, m_end = measured
                junction_indel = (m_end - m_start) - motif_len
                junction_mapped = True
            else:
                junction_indel = 0
                junction_mapped = False

            # Keep the motif without consulting the CIGAR for acceptance.
            # Legal only under anchor="reference" (enforced in __init__),
            # because there the returned coordinate is reference-relative and
            # LeechRead maps it to signal through `compute_ref_to_signal`,
            # which interpolates across indels. Nothing downstream needs the
            # query coordinates to place the chunk, so a read whose motif
            # basecalled badly is still positioned correctly -- but the
            # measurement above still ran, because this is exactly the
            # population issue #282's junction_indel/junction_mapped fields
            # are meant to flag.
            if not self.require_query_mapping:
                query_positions.append(
                    MotifMatch(
                        ref_motif_start - alignment.reference_start,
                        junction_indel,
                        junction_mapped,
                    )
                )
                if self.debug:
                    self.stats["accepted_without_query_mapping"] += 1
                    self.stats["successful"] += 1
                continue

            # Acceptance gate. skip_indels=False (the common, lenient case) is
            # identical to `measured` above -- reuse it rather than re-walking
            # the CIGAR. skip_indels=True is a stricter check (any indel in
            # the region rejects the position outright, even a compensating
            # ins+del pair whose net length matches), and needs its own call.
            query_coords = (
                map_reference_to_query_coords(
                    alignment,
                    ref_motif_start,
                    ref_motif_end,
                    skip_indels=True,
                    allow_edge_indels=self.allow_edge_indels,
                )
                if self.skip_indels
                else measured
            )

            if query_coords is None:
                if self.debug:
                    # `measured` is exactly the skip_indels=False re-walk this
                    # branch used to make on demand, kept for stats purposes:
                    # None means the CIGAR mapping itself failed, non-None
                    # means an indel was the reason skip_indels=True rejected
                    # this position.
                    if measured is None:
                        self.stats["failed_cigar_mapping"] += 1
                    else:
                        self.stats["failed_indels"] += 1
                continue  # Skip if indels or mapping failed

            query_start, query_end = query_coords

            # Sanity check: ensure we mapped a region of the expected length
            # When skip_indels=False, accept motifs within ±3bp (indels change length)
            # When skip_indels=True, require exact length (no indels should be present)
            mapped_len = query_end - query_start
            if self.skip_indels:
                # Strict check when filtering indels - must be exact length
                length_ok = mapped_len == motif_len
            else:
                # Lenient check when accepting indels - within ±3bp tolerance
                length_ok = abs(mapped_len - motif_len) <= 3

            if length_ok:
                if self.anchor == "reference":
                    # Return reference coordinate relative to aligned portion.
                    # When anchor="reference", LeechRead uses ref coords for
                    # sequence and seq_to_sig_map, so motif positions must also
                    # be in reference coords (relative to ref_start).
                    query_positions.append(
                        MotifMatch(
                            ref_motif_start - alignment.reference_start,
                            junction_indel,
                            junction_mapped,
                        )
                    )
                else:
                    query_positions.append(MotifMatch(query_start, junction_indel, junction_mapped))
                if self.debug:
                    self.stats["successful"] += 1
            else:
                if self.debug:
                    self.stats["failed_length_check"] += 1
                logger.debug(
                    f"Read {read_id}: Mapped motif has unexpected length "
                    f"({mapped_len} != {motif_len}), skipping"
                )

        return query_positions


def get_motif_searcher(
    mode: str,
    reference_sequences: dict[str, str] | None = None,
    skip_indels: bool = False,
    allow_edge_indels: bool = False,
    debug: bool = False,
    anchor: str = "reference",
    require_query_mapping: bool = True,
) -> MotifSearcher:
    """
    Factory function for creating motif searchers.

    Args:
        mode: Search mode ("bam" for basecalled, "fasta" for reference)
        reference_sequences: Dict of reference sequences (required for "fasta" mode)
        skip_indels: Whether to skip motif positions with indels (for "fasta" mode)
        allow_edge_indels: If True, only reject indels in core motif (for "fasta" mode)
        debug: If True, enable detailed statistics collection
        anchor: "reference" (default) or "basecall" — controls coordinate system of returned positions
        require_query_mapping: If False, keep reference motifs that do not map
            cleanly to query coordinates (for "fasta" mode with anchor="reference"
            only). See ReferenceMotifSearcher.

    Returns:
        MotifSearcher instance

    Raises:
        ValueError: If mode is invalid or required args missing

    Examples:
        >>> # Basecalled search
        >>> searcher = get_motif_searcher("bam")
        >>>
        >>> # Reference search
        >>> refs = {"chr1": "ACGTCCAGGC..."}
        >>> searcher = get_motif_searcher("fasta", reference_sequences=refs)
    """
    if mode == "bam":
        return BasecalledMotifSearcher()
    elif mode == "fasta":
        if reference_sequences is None:
            raise ValueError("reference_sequences required for reference-based motif search")
        return ReferenceMotifSearcher(
            reference_sequences,
            skip_indels,
            allow_edge_indels,
            debug,
            anchor=anchor,
            require_query_mapping=require_query_mapping,
        )
    else:
        raise ValueError(f"Invalid motif search mode: {mode}. Must be 'bam' or 'fasta'")
