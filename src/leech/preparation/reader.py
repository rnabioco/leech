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
from leech.configs import SignalConfig
from leech.constants import REQUIRED_BAM_TAGS
from leech.features import (
    MoveTable,
    compute_dwell_features,
    compute_kmer_residual_features,
    compute_ref_to_signal,
    compute_signal_features,
    compute_signal_residual,
    extract_move_table,
    normalize_signal,
)
from leech.io import POD5Reader, iter_bam_batches

logger = logging.getLogger("leech.preparation.reader")


def build_leech_read(
    read_id: str,
    sequence: str,
    raw_signal: np.ndarray,
    move_table: "MoveTable",
    signal_config: SignalConfig,
    metadata: dict | None = None,
    reference_sequence: str | None = None,
    cigar_tuples: list[tuple[int, int]] | None = None,
    cal_offset: float | None = None,
    cal_scale: float | None = None,
) -> "LeechRead":
    """
    Build a LeechRead from raw components.

    Shared helper used by iter_bam_with_pod5(), parallel workers, and inference workers.

    Args:
        read_id: Read identifier
        sequence: Basecalled sequence
        raw_signal: Raw signal array (not yet reversed or normalized)
        move_table: Parsed MoveTable
        signal_config: Signal processing configuration
        metadata: Additional metadata dict (will be merged)
        reference_sequence: Reference sequence (required when anchor="reference")
        cigar_tuples: CIGAR tuples (required when anchor="reference")
        cal_offset: POD5 calibration offset (for pa_scaling)
        cal_scale: POD5 calibration scale (for pa_scaling)

    Returns:
        LeechRead with all features computed
    """
    # Trim signal to basecalled region [ts:ns], then optionally reverse.
    ts = move_table.trim_offset
    ns = move_table.num_samples
    raw_signal = raw_signal[ts:ns]

    query_to_sig_map = move_table.to_seq_to_sig_map()
    # Shift mapping so it's relative to the trimmed signal [0, ns-ts]
    query_to_sig_map = query_to_sig_map - ts

    if signal_config.reverse_signal:
        sig_len = len(raw_signal)
        raw_signal = raw_signal[::-1]
        query_to_sig_map = sig_len - query_to_sig_map[::-1]

    norm_signal, norm_params = normalize_signal(
        raw_signal,
        method=signal_config.norm_method,
        pa_mean=signal_config.pa_mean,
        pa_stdev=signal_config.pa_stdev,
        cal_offset=cal_offset,
        cal_scale=cal_scale,
    )

    # Determine which sequence and mapping to use
    if (
        signal_config.anchor == "reference"
        and reference_sequence is not None
        and cigar_tuples is not None
    ):
        ref_to_sig_map = compute_ref_to_signal(query_to_sig_map, cigar_tuples)

        sig_start = int(ref_to_sig_map[0])
        sig_end = int(ref_to_sig_map[-1])
        norm_signal = norm_signal[sig_start:sig_end]

        seq_to_sig_map = ref_to_sig_map - sig_start
        use_sequence = reference_sequence
    else:
        seq_to_sig_map = query_to_sig_map
        use_sequence = sequence

    # Optional signal map refinement
    if signal_config.refine_signal_map and signal_config.signal_refiner is not None:
        from leech.signal_refine import SigMapRefiner

        if isinstance(signal_config.signal_refiner, SigMapRefiner):
            norm_signal, seq_to_sig_map = signal_config.signal_refiner.refine(
                norm_signal, use_sequence, seq_to_sig_map
            )

    dwells = np.diff(seq_to_sig_map)

    if signal_config.compute_features:
        dwell_feats = compute_dwell_features(dwells)
        signal_feats = compute_signal_features(norm_signal, seq_to_sig_map)
    else:
        dwell_feats = {}
        signal_feats = {}

    # Kmer residual features and signal-level residual
    sig_residual = None
    if (
        signal_config.compute_features
        and signal_config.signal_refiner is not None
        and hasattr(signal_config.signal_refiner, "kmer_to_level")
    ):
        kmer_residual_feats = compute_kmer_residual_features(
            norm_signal,
            seq_to_sig_map,
            use_sequence,
            signal_config.signal_refiner.kmer_to_level,
            signal_config.signal_refiner.kmer_len,
        )
        signal_feats.update(kmer_residual_feats)

        from leech.signal_refine import extract_levels

        expected_levels = extract_levels(
            use_sequence,
            signal_config.signal_refiner.kmer_to_level,
            signal_config.signal_refiner.kmer_len,
        )
        sig_residual = compute_signal_residual(norm_signal, seq_to_sig_map, expected_levels)

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
        signal_residual=sig_residual,
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
    """
    from leech.io import read_pod5_signal as _read_pod5_signal

    return _read_pod5_signal(pod5_path, read_id)


def iter_bam_with_pod5(
    bam_path: Path,
    pod5_path: Path,
    signal_config: SignalConfig,
    min_mapq: int = 0,
    require_tags: list[str] | None = None,
    reference_sequences: dict[str, str] | None = None,
    read_batch_size: int = 200_000,
) -> Iterator[LeechRead]:
    """
    Iterate over aligned reads, loading signal from POD5.

    Reads are processed in mega-batches (default 50K) to bound memory usage.
    Each batch: load BAM alignments, preload POD5 signals, yield LeechReads,
    then free the batch.

    Args:
        bam_path: Path to BAM file with mv tags
        pod5_path: Path to POD5 file
        signal_config: Signal processing configuration
        min_mapq: Minimum mapping quality
        require_tags: BAM tags that must be present (default: ["mv", "ns"])
        reference_sequences: Dict of reference sequences (for anchor mode)
        read_batch_size: Reads per mega-batch for memory-bounded streaming

    Yields:
        LeechRead objects with full feature extraction
    """
    if require_tags is None:
        require_tags = REQUIRED_BAM_TAGS

    with POD5Reader(pod5_path) as pod5_reader:
        for aln_batch in iter_bam_batches(
            bam_path, batch_size=read_batch_size, min_mapq=min_mapq, require_tags=require_tags
        ):
            # Preload POD5 signals for this batch only
            batch_read_ids = [
                aln.query_name
                for aln in aln_batch
                if aln.query_name is not None and aln.query_sequence is not None
            ]
            if not batch_read_ids:
                continue
            pod5_reader.preload(batch_read_ids)

            for aln in aln_batch:
                try:
                    move_table = extract_move_table(aln)
                    read_id = aln.query_name
                    read_seq = aln.query_sequence
                    if read_id is None or read_seq is None:
                        continue

                    raw_signal, pod5_metadata = pod5_reader.get_signal(read_id)

                    cal_offset = pod5_metadata.get("calibration_offset")
                    cal_scale = pod5_metadata.get("calibration_scale")

                    # Get reference sequence for anchor mode
                    ref_seq = None
                    cigar_tuples = None
                    if signal_config.anchor == "reference":
                        if reference_sequences and aln.reference_name in reference_sequences:
                            full_ref = reference_sequences[aln.reference_name]
                            ref_seq = full_ref[aln.reference_start : aln.reference_end]
                        else:
                            try:
                                ref_seq = aln.get_reference_sequence()
                            except Exception as e:
                                logger.warning(
                                    "Could not get reference sequence for %s (anchor=reference): %s",
                                    aln.query_name,
                                    e,
                                )
                                ref_seq = None
                        cigar_tuples = aln.cigartuples

                    # Extract optional CL tag
                    try:
                        cl_tag = aln.get_tag("CL")
                        cl_value = int(cl_tag[0]) if hasattr(cl_tag, "__getitem__") else int(cl_tag)
                    except (KeyError, TypeError):
                        cl_value = None

                    metadata = {
                        **pod5_metadata,
                        "mapping_quality": aln.mapping_quality,
                        "reference_name": aln.reference_name,
                        "reference_start": aln.reference_start,
                        "reference_end": aln.reference_end,
                        "is_reverse": aln.is_reverse,
                        "alignment": aln,
                        "cl_value": cl_value,
                    }

                    yield build_leech_read(
                        read_id=read_id,
                        sequence=read_seq,
                        raw_signal=raw_signal,
                        move_table=move_table,
                        signal_config=signal_config,
                        metadata=metadata,
                        reference_sequence=ref_seq,
                        cigar_tuples=cigar_tuples,
                        cal_offset=cal_offset,
                        cal_scale=cal_scale,
                    )

                except Exception as e:
                    logger.warning(f"Skipping read {aln.query_name}: {e}")
                    continue
