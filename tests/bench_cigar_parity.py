#!/usr/bin/env python3
"""
Direct comparison of Python vs Rust ref-to-signal mapping on real CIGAR data.

Isolates the CIGAR parsing path to identify precision differences between:
- Python: make_ref_to_query_mapping → np.interp → compute_ref_to_signal
- Rust: compute_ref_to_signal (direct CIGAR walk with integer knot arithmetic)

Reports per-read statistics on signal index differences and characterizes
which CIGAR patterns (insertions, deletions, mismatches) cause divergence.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pysam

from leech.features import (
    MoveTable,
    compute_ref_to_signal,
    extract_move_table,
    normalize_signal,
)

# Skip if leech_core not built
try:
    from leech._leech_core import _test_ref_to_signal

    from leech._rust_accel import _rs_test_process_read

    HAS_REF_TO_SIG = True
except ImportError:
    HAS_REF_TO_SIG = False
    try:
        from leech._rust_accel import _rs_test_process_read
    except ImportError:
        print("ERROR: leech_core not built")
        sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent.parent
BAM = ROOT / "results/production/data/filtered/LeuRS_leu_b3/cognate.bam"
POD5 = ROOT / "results/production/data/filtered/LeuRS_leu_b3/cognate.pod5"


def classify_cigar(cigar_tuples):
    """Classify CIGAR string complexity."""
    ops = Counter()
    for op, length in cigar_tuples:
        ops[op] += length
    has_ins = ops.get(1, 0) > 0
    has_del = ops.get(2, 0) > 0
    has_mismatch = ops.get(8, 0) > 0  # X op
    n_indel_events = sum(1 for op, _ in cigar_tuples if op in (1, 2))
    total_indel_bases = ops.get(1, 0) + ops.get(2, 0)
    return {
        "has_ins": has_ins,
        "has_del": has_del,
        "has_mismatch": has_mismatch,
        "n_indel_events": n_indel_events,
        "total_indel_bases": total_indel_bases,
        "ins_bases": ops.get(1, 0),
        "del_bases": ops.get(2, 0),
    }


def compare_ref_to_signal_direct(max_reads=2000):
    """Compare Python vs Rust ref_to_signal on real reads, bypassing full pipeline."""
    if not BAM.exists():
        print(f"ERROR: BAM not found: {BAM}")
        sys.exit(1)

    stats = {
        "total": 0,
        "exact_match": 0,
        "off_by_one": 0,  # max diff == 1
        "larger_diff": 0,
        "max_diff_seen": 0,
        "by_indel_count": {},  # n_indel_events -> {exact, off_by_one, larger}
    }

    worst_reads = []  # (max_diff, read_id, cigar_info, n_diff_positions)

    with pysam.AlignmentFile(str(BAM), "rb") as bam:
        for i, aln in enumerate(bam.fetch(until_eof=True)):
            if i >= max_reads:
                break
            if aln.query_name is None or aln.query_sequence is None:
                continue
            if not aln.has_tag("mv") or not aln.has_tag("ns"):
                continue

            cigar_tuples = aln.cigartuples
            if cigar_tuples is None:
                continue

            try:
                mt = extract_move_table(aln)
            except Exception:
                continue

            # Build query_to_signal (Python)
            sig_map = mt.to_seq_to_sig_map()

            # Python ref_to_signal
            py_ref_to_sig = compute_ref_to_signal(sig_map, cigar_tuples)

            # Rust: we need to call the Rust compute_ref_to_signal
            # Convert cigar_tuples to the format Rust expects: list of (op, len)
            cigar_ops = [(op, length) for op, length in cigar_tuples]

            if HAS_REF_TO_SIG:
                rs_ref_to_sig = np.array(
                    _test_ref_to_signal(sig_map.tolist(), cigar_ops),
                    dtype=np.int64,
                )
            else:
                # Fall back: compare via full pipeline (less isolated)
                continue

            # Compare
            cigar_info = classify_cigar(cigar_tuples)
            n_indels = cigar_info["n_indel_events"]

            if len(py_ref_to_sig) != len(rs_ref_to_sig):
                print(
                    f"  LENGTH MISMATCH: {aln.query_name[:12]}... "
                    f"py={len(py_ref_to_sig)} rs={len(rs_ref_to_sig)} "
                    f"indels={n_indels}"
                )
                stats["larger_diff"] += 1
                stats["total"] += 1
                continue

            diff = np.abs(py_ref_to_sig - rs_ref_to_sig)
            max_diff = int(diff.max()) if len(diff) > 0 else 0
            n_diff_positions = int(np.sum(diff > 0))

            stats["total"] += 1
            if max_diff == 0:
                stats["exact_match"] += 1
            elif max_diff == 1:
                stats["off_by_one"] += 1
            else:
                stats["larger_diff"] += 1

            stats["max_diff_seen"] = max(stats["max_diff_seen"], max_diff)

            # Track by indel count
            key = min(n_indels, 10)  # bucket 10+
            if key not in stats["by_indel_count"]:
                stats["by_indel_count"][key] = {
                    "exact": 0,
                    "off_by_one": 0,
                    "larger": 0,
                    "total": 0,
                }
            bucket = stats["by_indel_count"][key]
            bucket["total"] += 1
            if max_diff == 0:
                bucket["exact"] += 1
            elif max_diff == 1:
                bucket["off_by_one"] += 1
            else:
                bucket["larger"] += 1

            if max_diff > 0:
                worst_reads.append(
                    (max_diff, aln.query_name, cigar_info, n_diff_positions, len(diff))
                )

    return stats, worst_reads


def compare_full_pipeline(max_reads=500):
    """Compare full Python vs Rust feature extraction on real reads.

    Tests the end-to-end path including normalization, sig_map, dwell, features.
    Uses the test harness function _rs_test_process_read which mirrors the
    Python pipeline exactly.
    """
    from escapepod import Reader

    stats = {
        "total": 0,
        "signal_exact": 0,
        "signal_close": 0,  # within 1e-5
        "signal_differ": 0,
        "feature_exact": 0,
        "feature_close": 0,
        "feature_differ": 0,
        "max_signal_diff": 0.0,
        "max_feature_diff": 0.0,
    }

    pod5 = Reader(str(POD5))
    read_infos = []

    with pysam.AlignmentFile(str(BAM), "rb") as bam:
        for i, aln in enumerate(bam.fetch(until_eof=True)):
            if i >= max_reads:
                break
            if aln.query_name is None or not aln.has_tag("mv") or not aln.has_tag("ns"):
                continue
            try:
                mt = extract_move_table(aln)
            except Exception:
                continue
            read_infos.append(
                {
                    "read_id": aln.query_name,
                    "stride": mt.stride,
                    "moves": mt.moves,
                    "num_samples": mt.num_samples,
                    "trim_offset": mt.trim_offset,
                }
            )

    # Fetch signals in bulk
    rids = [r["read_id"] for r in read_infos]
    read_datas = pod5.get_reads(rids)
    sig_by_rid = {}
    for rd in read_datas:
        sig_by_rid[rd.read_id] = pod5.get_signal(rd)

    for r in read_infos:
        raw = sig_by_rid.get(r["read_id"])
        if raw is None:
            continue

        # Python path
        ts = r["trim_offset"]
        ns = r["num_samples"]
        trimmed = raw[ts:ns].astype(np.float32)[::-1].copy()
        py_norm, _ = normalize_signal(trimmed, method="median_mad")

        mt = MoveTable(
            stride=r["stride"],
            moves=r["moves"],
            read_id=r["read_id"],
            num_samples=ns,
            trim_offset=ts,
        )
        py_map = mt.to_seq_to_sig_map() - ts
        py_map = (ns - ts) - py_map[::-1]

        # Rust path
        rs_norm, rs_map, rs_dwells, rs_feats = _rs_test_process_read(
            raw.tolist(),
            r["moves"].tolist(),
            r["stride"],
            ts,
            ns,
            reverse_signal=True,
        )

        stats["total"] += 1

        # Signal comparison
        if len(py_norm) == len(rs_norm):
            sig_diff = np.max(np.abs(py_norm - np.array(rs_norm)))
            stats["max_signal_diff"] = max(stats["max_signal_diff"], sig_diff)
            if sig_diff == 0.0:
                stats["signal_exact"] += 1
            elif sig_diff < 1e-5:
                stats["signal_close"] += 1
            else:
                stats["signal_differ"] += 1

        # Feature comparison
        py_dwells = np.diff(py_map).astype(np.float32)
        from leech.features import compute_dwell_features, compute_signal_features

        py_dwell_feats = compute_dwell_features(py_dwells)
        py_sig_feats = compute_signal_features(py_norm, py_map)
        feat_rows = []
        for key in ["dwell", "dwell_log", "dwell_mean", "dwell_std", "dwell_ratio"]:
            feat_rows.append(py_dwell_feats[key])
        for key in ["level_mean", "level_median", "level_std", "level_range"]:
            feat_rows.append(py_sig_feats[key])
        py_features = np.stack(feat_rows, axis=0)

        rs_features = np.array(rs_feats)
        if py_features.shape == rs_features.shape:
            feat_diff = np.max(np.abs(py_features - rs_features))
            stats["max_feature_diff"] = max(stats["max_feature_diff"], feat_diff)
            if feat_diff == 0.0:
                stats["feature_exact"] += 1
            elif feat_diff < 1e-5:
                stats["feature_close"] += 1
            else:
                stats["feature_differ"] += 1

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CIGAR parity benchmark: Python vs Rust")
    parser.add_argument("--reads", type=int, default=2000, help="Max reads")
    parser.add_argument(
        "--full-pipeline", action="store_true", help="Also run full pipeline comparison"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CIGAR ref-to-signal parity: Python vs Rust")
    print("=" * 70)

    if HAS_REF_TO_SIG:
        stats, worst = compare_ref_to_signal_direct(args.reads)

        print(f"\nTotal reads compared: {stats['total']}")
        print(
            f"  Exact match:   {stats['exact_match']} ({100 * stats['exact_match'] / max(1, stats['total']):.1f}%)"
        )
        print(
            f"  Off-by-one:    {stats['off_by_one']} ({100 * stats['off_by_one'] / max(1, stats['total']):.1f}%)"
        )
        print(
            f"  Larger diff:   {stats['larger_diff']} ({100 * stats['larger_diff'] / max(1, stats['total']):.1f}%)"
        )
        print(f"  Max diff seen: {stats['max_diff_seen']}")

        print("\n--- By indel complexity ---")
        print(
            f"  {'Indels':>8s}  {'Total':>6s}  {'Exact':>6s}  {'Off1':>6s}  {'Larger':>6s}  {'ExactPct':>8s}"
        )
        for k in sorted(stats["by_indel_count"].keys()):
            b = stats["by_indel_count"][k]
            label = f"{k}+" if k == 10 else str(k)
            pct = 100 * b["exact"] / max(1, b["total"])
            print(
                f"  {label:>8s}  {b['total']:>6d}  {b['exact']:>6d}  {b['off_by_one']:>6d}  {b['larger']:>6d}  {pct:>7.1f}%"
            )

        if worst:
            worst.sort(reverse=True)
            print(f"\n--- Worst {min(10, len(worst))} reads ---")
            for max_diff, rid, info, n_diff, total in worst[:10]:
                print(
                    f"  {rid[:16]}... max_diff={max_diff:>3d}  "
                    f"diff_positions={n_diff}/{total}  "
                    f"ins={info['ins_bases']}  del={info['del_bases']}  "
                    f"indel_events={info['n_indel_events']}"
                )
    else:
        print("SKIP: _test_ref_to_signal not exported from leech_core")
        print("  (Need to expose compute_ref_to_signal as a test function)")

    if args.full_pipeline:
        print(f"\n{'=' * 70}")
        print("Full pipeline comparison: Python vs Rust (normalization + features)")
        print(f"{'=' * 70}")

        fstats = compare_full_pipeline(min(args.reads, 500))
        print(f"\nTotal reads: {fstats['total']}")
        print("\nSignal (normalized):")
        print(f"  Exact:    {fstats['signal_exact']}")
        print(f"  Close:    {fstats['signal_close']} (< 1e-5)")
        print(f"  Differ:   {fstats['signal_differ']}")
        print(f"  Max diff: {fstats['max_signal_diff']:.2e}")
        print("\nFeatures (9-channel):")
        print(f"  Exact:    {fstats['feature_exact']}")
        print(f"  Close:    {fstats['feature_close']} (< 1e-5)")
        print(f"  Differ:   {fstats['feature_differ']}")
        print(f"  Max diff: {fstats['max_feature_diff']:.2e}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
