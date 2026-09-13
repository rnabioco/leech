"""
Builds a :class:`~leech.chunking.LeechRead` from raw BAM/POD5 components.

``build_leech_read`` is the shared feature-extraction step used by the
parallel dispatch workers (``preparation/parallel.py``) and by inference. The
sequential BAM+POD5 iterator that used to live here (``iter_bam_with_pod5``)
was retired in issue #275: ``prepare_training_data_parallel(num_workers=1)``
covers the same case through the same dispatcher every other worker count
uses.
"""

import logging
from pathlib import Path

import numpy as np

from leech.chunking import LeechRead
from leech.configs import SignalConfig
from leech.features import (
    MoveTable,
    compute_dwell_features,
    compute_kmer_residual_features,
    compute_ref_to_signal,
    compute_signal_features,
    compute_signal_residual,
    normalize_read_signal,
)

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

    Shared helper used by the parallel prepare workers and inference workers.

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

    norm_signal, norm_params = normalize_read_signal(
        raw_signal,
        method=signal_config.norm_method,
        pa_mean=signal_config.pa_mean,
        pa_stdev=signal_config.pa_stdev,
        cal_offset=cal_offset,
        cal_scale=cal_scale,
        # Nothing downstream reads this dict's contents (only
        # `leech_read.metadata["alignment"/"cl_value"/"reference_name"]` are
        # ever read back) -- skip the two full-signal np.median passes that
        # would otherwise exist only to populate it (issue #275).
        include_diagnostics=False,
    )

    # Determine which sequence and mapping to use. In ref-anchored mode we
    # crop ``norm_signal`` to the aligned region (so refinement and feature
    # extraction only see in-distribution data), but stash the full
    # pre-crop signal so ``LeechRead.get_chunk`` can recover real samples at
    # chunk-window edges instead of zero-padding (R4 in the coordinate audit).
    full_norm_signal: np.ndarray | None = None
    signal_offset: int = 0
    if (
        signal_config.anchor == "reference"
        and reference_sequence is not None
        and cigar_tuples is not None
    ):
        ref_to_sig_map = compute_ref_to_signal(query_to_sig_map, cigar_tuples)

        sig_start = int(ref_to_sig_map[0])
        sig_end = int(ref_to_sig_map[-1])
        full_norm_signal = norm_signal
        signal_offset = sig_start
        norm_signal = norm_signal[sig_start:sig_end]

        seq_to_sig_map = ref_to_sig_map - sig_start
        use_sequence = reference_sequence
    else:
        seq_to_sig_map = query_to_sig_map
        use_sequence = sequence

    # `extract_levels(use_sequence, ...)` used to be called up to three times
    # per read with byte-identical arguments -- once inside `refine()`, once
    # here for the kmer-residual features, once more for the signal residual
    # (issue #275). Computed once and threaded through instead. It depends
    # only on `use_sequence` and the refiner's table/kmer_len/center_idx, none
    # of which refinement changes (`refine()` only moves `seq_to_sig_map`
    # boundaries; the returned `signal` and the sequence are unchanged), so
    # one value is valid for both the refinement DP band and the post-
    # refinement residual features below.
    expected_levels: np.ndarray | None = None
    if signal_config.signal_refiner is not None and hasattr(
        signal_config.signal_refiner, "kmer_to_level"
    ):
        from leech.signal_refine import extract_levels

        expected_levels = extract_levels(
            use_sequence,
            signal_config.signal_refiner.kmer_to_level,
            signal_config.signal_refiner.kmer_len,
            center_idx=signal_config.signal_refiner.center_idx,
        )

    # Optional signal map refinement
    if signal_config.refine_signal_map and signal_config.signal_refiner is not None:
        from leech.signal_refine import SigMapRefiner

        if isinstance(signal_config.signal_refiner, SigMapRefiner):
            norm_signal, seq_to_sig_map = signal_config.signal_refiner.refine(
                norm_signal, use_sequence, seq_to_sig_map, expected_levels=expected_levels
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
    if signal_config.compute_features and expected_levels is not None:
        # `center_idx` matters here, not just in refinement: it decides which
        # base of each k-mer window the expected level is attributed to, so
        # leaving it at extract_levels' default while the refiner used its own
        # would offset every residual feature against the boundaries that
        # produced it. The Rust pipeline passes one `kmer_center_idx` to both.
        kmer_residual_feats = compute_kmer_residual_features(
            norm_signal,
            seq_to_sig_map,
            use_sequence,
            signal_config.signal_refiner.kmer_to_level,
            signal_config.signal_refiner.kmer_len,
            center_idx=signal_config.signal_refiner.center_idx,
            expected_levels=expected_levels,
        )
        signal_feats.update(kmer_residual_feats)
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
        full_signal=full_norm_signal,
        signal_offset=signal_offset,
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
