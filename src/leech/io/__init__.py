"""
I/O utilities for reading BAM, POD5, and reference files.

This module provides efficient readers for nanopore data files with
support for batched access and filtering.
"""

from leech.io.bam_reader import (
    BAMReader,
    MockAlignment,
    ReadInfo,
    collect_read_infos,
    iter_bam_alignments,
)
from leech.io.motif_search import (
    BasecalledMotifSearcher,
    MotifSearcher,
    ReferenceMotifSearcher,
    find_motif_in_sequence,
    get_motif_searcher,
    map_reference_to_query_coords,
)
from leech.io.pod5_reader import (
    POD5Reader,
    _extract_pod5_metadata,
    read_pod5_signal,
    read_pod5_signals_batch,
)
from leech.io.reference import (
    ReferenceManager,
    extract_reference_from_bam,
    get_reference_sequences,
    load_reference_fasta,
)

__all__ = [
    # BAM reading
    "BAMReader",
    "MockAlignment",
    "ReadInfo",
    "iter_bam_alignments",
    "collect_read_infos",
    # POD5 reading
    "POD5Reader",
    "_extract_pod5_metadata",
    "read_pod5_signal",
    "read_pod5_signals_batch",
    # Reference sequences
    "ReferenceManager",
    "extract_reference_from_bam",
    "load_reference_fasta",
    "get_reference_sequences",
    # Motif search
    "MotifSearcher",
    "BasecalledMotifSearcher",
    "ReferenceMotifSearcher",
    "get_motif_searcher",
    "find_motif_in_sequence",
    "map_reference_to_query_coords",
]
