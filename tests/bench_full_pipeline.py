"""
Full pipeline benchmark: Python vs Rust extraction on real data.

Measures the complete per-read cost including POD5 reading, normalization,
move table parsing, feature computation, and chunk extraction.
"""

import time
from pathlib import Path

import numpy as np
import pysam
from escapepod import Reader

from leech.features import (
    MoveTable,
    compute_dwell_features,
    compute_signal_features,
    extract_move_table,
    normalize_signal,
)

BAM = Path(
    "/beevol/home/jhessel/devel/rnabioco/2026-aa-trna-models/"
    "results-production/data/filtered/LeuRS_leu_b3/cognate.bam"
)
POD5 = Path(
    "/beevol/home/jhessel/devel/rnabioco/2026-aa-trna-models/"
    "results-production/data/filtered/LeuRS_leu_b3/cognate.pod5"
)


def collect_bam_data(bam_path, max_reads=2000):
    reads = []
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for aln in bam.fetch(until_eof=True):
            if aln.query_name is None or aln.query_sequence is None:
                continue
            try:
                mt = extract_move_table(aln)
            except Exception:
                continue
            reads.append(
                {
                    "read_id": aln.query_name,
                    "sequence": aln.query_sequence,
                    "stride": mt.stride,
                    "moves": mt.moves,
                    "num_samples": mt.num_samples,
                    "trim_offset": mt.trim_offset,
                }
            )
            if len(reads) >= max_reads:
                break
    return reads


# ---------------------------------------------------------------------------
# POD5 reading benchmarks (pure Python pod5 vs Rust escapepod)
# ---------------------------------------------------------------------------


def python_pod5_read(read_ids, pod5_path):
    """Escapepod Python batch read (NO leech_core fallback)."""
    reader = Reader(str(pod5_path))
    reads = reader.get_reads(read_ids)
    signals = reader.get_signals(reads)
    return dict(signals)


def rust_pod5_read(read_ids, pod5_path):
    """Rust escapepod batch read."""
    from leech_core import read_pod5_batch

    return read_pod5_batch(str(pod5_path), read_ids)


# ---------------------------------------------------------------------------
# Full pipeline benchmarks
# ---------------------------------------------------------------------------


def python_full_pipeline(reads, pod5_path, reverse=True):
    """Full Python extraction: escapepod + numpy pipeline."""
    read_ids = [r["read_id"] for r in reads]

    # POD5 read via escapepod
    pod5_cache = python_pod5_read(read_ids, pod5_path)

    results = []
    for r in reads:
        raw_signal = pod5_cache.get(r["read_id"])
        if raw_signal is None:
            continue

        ts = r["trim_offset"]
        ns = r["num_samples"]
        trimmed = raw_signal[ts:ns].astype(np.float32)
        if reverse:
            trimmed = trimmed[::-1].copy()

        mt = MoveTable(
            stride=r["stride"],
            moves=r["moves"],
            read_id=r["read_id"],
            num_samples=ns,
            trim_offset=ts,
        )
        sig_map = mt.to_seq_to_sig_map() - ts
        if reverse:
            sig_map = (ns - ts) - sig_map[::-1]

        norm, _ = normalize_signal(trimmed, method="median_mad")
        dwells = np.diff(sig_map).astype(np.float32)
        compute_dwell_features(dwells)
        compute_signal_features(norm, sig_map)
        results.append((norm, sig_map, dwells))

    return results


def rust_full_pipeline(reads, pod5_path, reverse=True):
    """Full Rust extraction: escapepod POD5 + Rust pipeline."""
    from leech_core import extract_inference_chunks

    read_ids = [r["read_id"] for r in reads]
    sequences = [r["sequence"] for r in reads]
    mv_strides = [r["stride"] for r in reads]
    mv_arrays = [r["moves"].tolist() for r in reads]
    num_samples = [r["num_samples"] for r in reads]
    trim_offsets = [r["trim_offset"] for r in reads]
    motif_positions = [[30] for _ in reads]

    chunks = extract_inference_chunks(
        str(pod5_path),
        read_ids=read_ids,
        sequences=sequences,
        mv_strides=mv_strides,
        mv_arrays=mv_arrays,
        num_samples_list=num_samples,
        trim_offsets=trim_offsets,
        signal_context_left=200,
        signal_context_right=200,
        kmer_context=5,
        motif_positions=motif_positions,
        signal_len=400,
        compute_features=True,
        reverse_signal=reverse,
    )
    return chunks


def bench(func, args, n_runs, label):
    # Warmup
    func(*args)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)

    mean_t = np.mean(times)
    std_t = np.std(times)
    n_items = len(args[0])
    per_item = mean_t / n_items * 1e6

    print(f"  {label:>30s}: {mean_t:.3f}s ± {std_t:.3f}s  ({per_item:.1f} µs/read)")
    return mean_t


if __name__ == "__main__":
    print("Collecting BAM data...")
    reads_500 = collect_bam_data(BAM, max_reads=500)
    reads_2k = collect_bam_data(BAM, max_reads=2000)
    reads_all = collect_bam_data(BAM, max_reads=20000)
    print(f"  500-batch: {len(reads_500)}, 2k-batch: {len(reads_2k)}, all: {len(reads_all)}")

    print()
    print("=" * 72)
    print("1. POD5 batch reading: Python pod5 vs Rust escapepod")
    print("=" * 72)

    for batch, label in [
        (reads_500, "500 reads"),
        (reads_2k, "2000 reads"),
        (reads_all, f"{len(reads_all)} reads"),
    ]:
        ids = [r["read_id"] for r in batch]
        print(f"\n{label}:")
        py_t = bench(python_pod5_read, (ids, POD5), 3, "Python (pod5 library)")
        rs_t = bench(rust_pod5_read, (ids, POD5), 3, "Rust (escapepod-rs)")
        print(f"  {'POD5 speedup':>30s}: {py_t / rs_t:.1f}x")

    print()
    print("=" * 72)
    print("2. Full pipeline: POD5 + normalize + sig_map + features + chunk")
    print("=" * 72)

    for batch, label in [
        (reads_500, "500 reads"),
        (reads_2k, "2000 reads"),
        (reads_all, f"{len(reads_all)} reads"),
    ]:
        print(f"\n{label}:")
        py_t = bench(python_full_pipeline, (batch, POD5), 3, "Python (full)")
        rs_t = bench(rust_full_pipeline, (batch, POD5), 3, "Rust (full monolithic)")
        print(f"  {'Full pipeline speedup':>30s}: {py_t / rs_t:.1f}x")

    print()
    print("=" * 72)
    print("3. Breakdown: Python pipeline component timing (2000 reads)")
    print("=" * 72)

    ids = [r["read_id"] for r in reads_2k]

    # POD5 reading
    t0 = time.perf_counter()
    pod5_cache = python_pod5_read(ids, POD5)
    t_pod5 = time.perf_counter() - t0

    # Per-read processing
    t_norm = 0
    t_sigmap = 0
    t_features = 0
    for r in reads_2k:
        raw = pod5_cache.get(r["read_id"])
        if raw is None:
            continue
        ts, ns = r["trim_offset"], r["num_samples"]
        trimmed = raw[ts:ns].astype(np.float32)[::-1].copy()

        t0 = time.perf_counter()
        norm, _ = normalize_signal(trimmed, method="median_mad")
        t_norm += time.perf_counter() - t0

        t0 = time.perf_counter()
        mt = MoveTable(
            stride=r["stride"],
            moves=r["moves"],
            read_id=r["read_id"],
            num_samples=ns,
            trim_offset=ts,
        )
        sig_map = mt.to_seq_to_sig_map() - ts
        sig_map = (ns - ts) - sig_map[::-1]
        dwells = np.diff(sig_map).astype(np.float32)
        t_sigmap += time.perf_counter() - t0

        t0 = time.perf_counter()
        compute_dwell_features(dwells)
        compute_signal_features(norm, sig_map)
        t_features += time.perf_counter() - t0

    total = t_pod5 + t_norm + t_sigmap + t_features
    n = len(reads_2k)
    print(
        f"  {'POD5 reading':>20s}: {t_pod5:.3f}s  ({t_pod5 / n * 1e6:.1f} µs/read, {t_pod5 / total * 100:.0f}%)"
    )
    print(
        f"  {'Normalization':>20s}: {t_norm:.3f}s  ({t_norm / n * 1e6:.1f} µs/read, {t_norm / total * 100:.0f}%)"
    )
    print(
        f"  {'Sig map + dwells':>20s}: {t_sigmap:.3f}s  ({t_sigmap / n * 1e6:.1f} µs/read, {t_sigmap / total * 100:.0f}%)"
    )
    print(
        f"  {'Features (dwell+sig)':>20s}: {t_features:.3f}s  ({t_features / n * 1e6:.1f} µs/read, {t_features / total * 100:.0f}%)"
    )
    print(f"  {'Total':>20s}: {total:.3f}s  ({total / n * 1e6:.1f} µs/read)")
    print()
