"""
Benchmark Rust vs Python extraction pipeline.

Measures per-read time for: normalize → sig_map → dwell → features.
"""

import time

import numpy as np
from leech_core import _test_process_read

from leech.features import (
    MoveTable,
    compute_dwell_features,
    compute_signal_features,
    normalize_read_signal,
)


def make_read(num_bases=80, mean_dwell=12, stride=5, trim_offset=10, seed=42):
    rng = np.random.RandomState(seed)
    total_positions = num_bases * mean_dwell
    moves = np.zeros(total_positions, dtype=np.uint8)
    move_positions = np.linspace(0, total_positions - 1, num_bases, dtype=int)
    moves[move_positions] = 1
    num_samples = trim_offset + total_positions * stride + 20
    raw_signal = (rng.normal(600, 50, num_samples)).astype(np.int16)
    return {
        "raw_signal": raw_signal,
        "moves": moves,
        "stride": stride,
        "trim_offset": trim_offset,
        "num_samples": num_samples,
        "num_bases": num_bases,
    }


def python_pipeline(read, reverse=True):
    raw = read["raw_signal"]
    ts = read["trim_offset"]
    ns = read["num_samples"]
    trimmed = raw[ts:ns].astype(np.float32)
    if reverse:
        trimmed = trimmed[::-1].copy()

    mt = MoveTable(
        stride=read["stride"],
        moves=read["moves"],
        read_id="test",
        num_samples=ns,
        trim_offset=ts,
    )
    py_map = mt.to_seq_to_sig_map() - ts
    if reverse:
        py_map = (ns - ts) - py_map[::-1]

    norm, _ = normalize_read_signal(trimmed, method="median_mad")
    dwells = np.diff(py_map).astype(np.float32)
    dwell_feats = compute_dwell_features(dwells)
    sig_feats = compute_signal_features(norm, py_map)
    return norm, py_map, dwells, dwell_feats, sig_feats


def rust_pipeline(sig_list, mv_list, stride, trim, ns, reverse=True):
    """Call Rust with pre-converted lists (amortizes .tolist() cost)."""
    return _test_process_read(sig_list, mv_list, stride, trim, ns, reverse_signal=reverse)


def bench(func, args, n_calls, label):
    # Warmup
    for _ in range(min(20, n_calls)):
        func(*args)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        func(*args)
    elapsed = time.perf_counter() - t0
    per_call = elapsed / n_calls * 1e6  # microseconds
    print(f"  {label:>20s}: {per_call:8.1f} µs/read  ({n_calls} calls in {elapsed:.2f}s)")
    return per_call


if __name__ == "__main__":
    configs = [
        (30, 8, "small (30 bases)"),
        (80, 12, "medium (80 bases)"),
        (200, 15, "large (200 bases)"),
        (500, 10, "xlarge (500 bases)"),
    ]

    print("=" * 72)
    print("Rust vs Python per-read extraction benchmark")
    print("  Note: real pipeline reads POD5 in Rust (no list conversion)")
    print("=" * 72)

    for num_bases, mean_dwell, label in configs:
        read = make_read(num_bases=num_bases, mean_dwell=mean_dwell)
        sig_len = read["num_samples"] - read["trim_offset"]
        print(f"\n{label} — {sig_len} signal samples, stride={read['stride']}")

        # Pre-convert to lists (amortize conversion cost)
        sig_list = read["raw_signal"].tolist()
        mv_list = read["moves"].tolist()

        # Measure conversion overhead
        n = 2000 if num_bases <= 200 else 500
        t0 = time.perf_counter()
        for _ in range(n):
            _ = read["raw_signal"].tolist()
            _ = read["moves"].tolist()
        conv_us = (time.perf_counter() - t0) / n * 1e6
        print(f"  {'list conversion':>20s}: {conv_us:8.1f} µs/read")

        py_us = bench(
            lambda r=read: python_pipeline(r),
            (),
            n,
            "Python (full)",
        )
        rs_with_conv = bench(
            lambda r=read: _test_process_read(
                r["raw_signal"].tolist(),
                r["moves"].tolist(),
                r["stride"],
                r["trim_offset"],
                r["num_samples"],
                True,
            ),
            (),
            n,
            "Rust (with .tolist())",
        )
        rs_no_conv = bench(
            rust_pipeline,
            (sig_list, mv_list, read["stride"], read["trim_offset"], read["num_samples"]),
            n,
            "Rust (pre-converted)",
        )
        print(f"  {'Speedup (fair)':>20s}: {py_us / rs_no_conv:.1f}x")
        print(f"  {'Speedup (w/ conv)':>20s}: {py_us / rs_with_conv:.1f}x")

    print("\n" + "=" * 72)
    print("In production: POD5 signal stays in Rust (no list conversion),")
    print("so 'Rust (pre-converted)' is the relevant comparison.")
    print("Additional wins not measured here:")
    print("  - POD5 reading: 25-43x faster (escapepod vs python-pod5)")
    print("  - No LeechRead/dict object creation per read")
    print("  - No Python GIL contention across threads")
    print("=" * 72)
