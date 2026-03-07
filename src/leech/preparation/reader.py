"""
High-level read iteration combining BAM and POD5 data.

This module provides functions for iterating over aligned reads while
extracting features from both BAM alignments and POD5 signal data.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from leech.chunking import LeechRead
from leech.features import (
    MoveTable,
    compute_dwell_features,
    compute_ref_to_signal,
    compute_signal_features,
    extract_move_table,
    normalize_signal,
)
from leech.io import POD5Reader, iter_bam_alignments

logger = logging.getLogger("leech.preparation.reader")


def build_leech_read(
    read_id: str,
    sequence: str,
    raw_signal: np.ndarray,
    move_table: "MoveTable",
    reverse_signal: bool = True,
    metadata: dict | None = None,
    anchor: str = "basecall",
    reference_sequence: str | None = None,
    cigar_tuples: list[tuple[int, int]] | None = None,
    norm_method: str = "median_mad",
    pa_mean: float | None = None,
    pa_stdev: float | None = None,
    cal_offset: float | None = None,
    cal_scale: float | None = None,
    refine_signal_map: bool = False,
    signal_refiner: object | None = None,
) -> "LeechRead":
    """
    Build a LeechRead from raw components.

    Shared helper used by iter_bam_with_pod5(), parallel workers, and inference workers.

    Args:
        read_id: Read identifier
        sequence: Basecalled sequence
        raw_signal: Raw signal array (not yet reversed or normalized)
        move_table: Parsed MoveTable
        reverse_signal: Whether to reverse signal for RNA
        metadata: Additional metadata dict (will be merged)
        anchor: "basecall" (default) or "reference" for ref-anchored mode
        reference_sequence: Reference sequence (required when anchor="reference")
        cigar_tuples: CIGAR tuples (required when anchor="reference")
        norm_method: Signal normalization method
        pa_mean: Global shift for pa_scaling normalization
        pa_stdev: Global scale for pa_scaling normalization
        cal_offset: POD5 calibration offset (for pa_scaling)
        cal_scale: POD5 calibration scale (for pa_scaling)
        refine_signal_map: Whether to apply signal map refinement
        signal_refiner: SigMapRefiner instance (required if refine_signal_map=True)

    Returns:
        LeechRead with all features computed
    """
    if reverse_signal:
        raw_signal = raw_signal[::-1]

    norm_signal, norm_params = normalize_signal(
        raw_signal,
        method=norm_method,
        pa_mean=pa_mean,
        pa_stdev=pa_stdev,
        cal_offset=cal_offset,
        cal_scale=cal_scale,
    )
    query_to_sig_map = move_table.to_seq_to_sig_map()

    # Determine which sequence and mapping to use
    if anchor == "reference" and reference_sequence is not None and cigar_tuples is not None:
        # Reference-anchored mode: use ref sequence + ref->signal mapping
        ref_to_sig_map = compute_ref_to_signal(query_to_sig_map, cigar_tuples)

        # Trim signal to aligned region
        sig_start = int(ref_to_sig_map[0])
        sig_end = int(ref_to_sig_map[-1])
        norm_signal = norm_signal[sig_start:sig_end]

        # Shift mapping to be relative to trimmed signal
        seq_to_sig_map = ref_to_sig_map - sig_start
        use_sequence = reference_sequence
    else:
        seq_to_sig_map = query_to_sig_map
        use_sequence = sequence

    # Optional signal map refinement
    if refine_signal_map and signal_refiner is not None:
        from leech.signal_refine import SigMapRefiner

        if isinstance(signal_refiner, SigMapRefiner):
            seq_to_sig_map = signal_refiner.refine(norm_signal, use_sequence, seq_to_sig_map)

    dwells = np.diff(seq_to_sig_map)
    dwell_feats = compute_dwell_features(dwells)
    signal_feats = compute_signal_features(norm_signal, seq_to_sig_map)

    meta = {"normalization": norm_params}
    if metadata:
        meta.update(metadata)

    return LeechRead(
        read_id=read_id,
        sequence=use_sequence,
        signal=norm_signal,
        seq_to_sig_map=seq_to_sig_map,
        dwells=dwells,
        dwell_features=dwell_feats,
        signal_features=signal_feats,
        metadata=meta,
    )


def read_pod5_signal(pod5_path: Path, read_id: str) -> tuple[np.ndarray, dict]:
    """
    Read raw signal from POD5 file for a specific read.

    Args:
        pod5_path: Path to POD5 file
        read_id: Read identifier

    Returns:
        Tuple of (signal_array, metadata_dict)

    Note:
        This is a convenience wrapper. For batch reading or repeated access,
        use POD5Reader directly from leech.io.

    Examples:
        >>> signal, metadata = read_pod5_signal(Path("reads.pod5"), "read_001")
        >>> print(f"Signal length: {len(signal)}")
    """
    from leech.io import read_pod5_signal as _read_pod5_signal

    return _read_pod5_signal(pod5_path, read_id)


def iter_bam_with_pod5(
    bam_path: Path,
    pod5_path: Path,
    reference_fasta: Path | None = None,
    min_mapq: int = 0,
    require_tags: list[str] | None = None,
    reverse_signal: bool = True,
    anchor: str = "basecall",
    norm_method: str = "median_mad",
    pa_mean: float | None = None,
    pa_stdev: float | None = None,
    refine_signal_map: bool = False,
    signal_refiner: object | None = None,
) -> Iterator[LeechRead]:
    """
    Iterate over aligned reads, loading signal from POD5.

    This is the main high-level function for reading training data. It combines
    BAM alignments with POD5 signal and computes all features needed for
    model training.

    Args:
        bam_path: Path to BAM file with mv tags
        pod5_path: Path to POD5 file
        reference_fasta: Optional reference for MD tag parsing
        min_mapq: Minimum mapping quality
        require_tags: BAM tags that must be present (default: ["mv", "ns"])
        reverse_signal: Whether to reverse signal for RNA
        anchor: "basecall" or "reference" for ref-anchored mode
        norm_method: Signal normalization method
        pa_mean: Global shift for pa_scaling normalization
        pa_stdev: Global scale for pa_scaling normalization
        refine_signal_map: Whether to apply signal map refinement
        signal_refiner: SigMapRefiner instance (for refinement)

    Yields:
        LeechRead objects with full feature extraction
    """
    if require_tags is None:
        require_tags = ["mv", "ns"]

    with POD5Reader(pod5_path) as pod5_reader:
        for aln in iter_bam_alignments(bam_path, min_mapq=min_mapq, require_tags=require_tags):
            try:
                move_table = extract_move_table(aln)
                read_id = aln.query_name
                read_seq = aln.query_sequence
                if read_id is None or read_seq is None:
                    continue

                raw_signal, pod5_metadata = pod5_reader.get_signal(read_id)

                # Get calibration data for pa_scaling
                cal_offset = pod5_metadata.get("calibration_offset")
                cal_scale = pod5_metadata.get("calibration_scale")

                # Get reference sequence for anchor mode
                ref_seq = None
                cigar_tuples = None
                if anchor == "reference":
                    try:
                        ref_seq = aln.get_reference_sequence()
                    except Exception:
                        ref_seq = None
                    cigar_tuples = aln.cigartuples

                metadata = {
                    **pod5_metadata,
                    "mapping_quality": aln.mapping_quality,
                    "reference_name": aln.reference_name,
                    "reference_start": aln.reference_start,
                    "reference_end": aln.reference_end,
                    "is_reverse": aln.is_reverse,
                    "alignment": aln,
                }

                yield build_leech_read(
                    read_id=read_id,
                    sequence=read_seq,
                    raw_signal=raw_signal,
                    move_table=move_table,
                    reverse_signal=reverse_signal,
                    metadata=metadata,
                    anchor=anchor,
                    reference_sequence=ref_seq,
                    cigar_tuples=cigar_tuples,
                    norm_method=norm_method,
                    pa_mean=pa_mean,
                    pa_stdev=pa_stdev,
                    cal_offset=cal_offset,
                    cal_scale=cal_scale,
                    refine_signal_map=refine_signal_map,
                    signal_refiner=signal_refiner,
                )

            except Exception as e:
                logger.warning(f"Skipping read {aln.query_name}: {e}")
                continue
