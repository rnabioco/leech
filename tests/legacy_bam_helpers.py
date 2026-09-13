"""``collect_read_infos``, moved out of ``leech.io.bam_reader`` in #277.

It had no production callers -- prepare's two-pass design uses
``collect_read_infos_from_bam`` (``preparation/parallel.py``) instead, which
streams via ``iter_read_info_batches`` rather than collecting the whole BAM
into a list up front. This one-shot version is kept here purely because
several test files build their read-info fixtures with it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from leech.constants import REQUIRED_BAM_TAGS
from leech.io.bam_reader import ReadInfo, iter_bam_alignments

logger = logging.getLogger("tests.legacy_bam_helpers")


def collect_read_infos(
    bam_path: Path,
    min_mapq: int = 0,
    require_tags: list[str] | None = None,
) -> list[ReadInfo]:
    """Collect lightweight read information from a BAM file.

    Args:
        bam_path: Path to BAM file
        min_mapq: Minimum mapping quality
        require_tags: List of required BAM tags (default: ["mv", "ns"])

    Returns:
        List of ReadInfo objects
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
            # the life of the list.
            read_info.materialize_reference_sequence()
            read_infos.append(read_info)
        except Exception as e:
            logger.warning(f"Skipping read {aln.query_name}: {e}")
            continue

    logger.info(f"Collected {len(read_infos)} read infos from {bam_path}")
    return read_infos
