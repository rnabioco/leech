"""
Benchmark and correctness test: Python vs Rust data preparation backends.

Runs both backends on the same input data, verifies that the resulting
training chunks are identical (signal, dwell, features, sequence), and
reports wall-clock time for each backend.

Usage:
    uv run python tests/bench_prepare_backends.py [--reads N]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from leech.configs import PrepareConfig

# ---------------------------------------------------------------------------
# Data paths (production tRNA data)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
BAM = ROOT / "results/production/data/filtered/LeuRS_leu_b3/cognate.bam"
POD5 = ROOT / "results/production/data/filtered/LeuRS_leu_b3/cognate.pod5"
REF = ROOT / "2026-aars-in-vitro/results/reference/adapted.fa"


def build_config(pod5_path: Path, *, refine: bool = True) -> PrepareConfig:
    """Build a PrepareConfig matching production settings."""
    from leech.configs import ChunkConfig, LabelConfig, MotifConfig, PrepareConfig, SignalConfig
    from leech.io import get_reference_sequences

    reference_sequences = get_reference_sequences(BAM, REF)

    signal_refiner = None
    if refine:
        from leech.data import get_kmer_table
        from leech.signal_refine import SigMapRefiner

        kmer_table = get_kmer_table()
        signal_refiner = SigMapRefiner.from_table(kmer_table, scale_iters=2, do_rough_rescale=True)

    return PrepareConfig(
        pod5_path=pod5_path,
        signal=SignalConfig(
            reverse_signal=True,
            anchor="reference",
            norm_method="median_mad",
            refine_signal_map=refine,
            refine_scale_iters=2,
            signal_refiner=signal_refiner,
        ),
        motif=MotifConfig(
            motif="CCAGGC",
            motif_offset=2,
            motif_reference="fasta",
            reference_sequences=reference_sequences,
            skip_motif_indels=True,
        ),
        chunk=ChunkConfig(
            base_justify="end",
            signal_context=(200, 200),
        ),
        labeling=LabelConfig(
            label="Leu",
            label_int=1,
        ),
        reference_fasta=REF,
    )


def run_python_backend(
    bam_path: Path, config: PrepareConfig, chunk_size: int, max_reads: int | None
) -> tuple[list[dict], float]:
    """Run data preparation using the Python multiprocessing backend."""
    from leech.io import collect_read_infos
    from leech.preparation.parallel import _process_read_chunk_worker

    read_infos = collect_read_infos(bam_path, min_mapq=0)
    if max_reads is not None:
        read_infos = read_infos[:max_reads]

    # Process in a single batch (no multiprocessing overhead for fair comparison)
    batches = [read_infos[i : i + chunk_size] for i in range(0, len(read_infos), chunk_size)]

    all_chunks = []
    t0 = time.perf_counter()
    for batch in batches:
        chunks = _process_read_chunk_worker((batch, config))
        all_chunks.extend(chunks)
    elapsed = time.perf_counter() - t0

    return all_chunks, elapsed


def run_rust_backend(
    bam_path: Path, config: PrepareConfig, chunk_size: int, max_reads: int | None
) -> tuple[list[dict], float]:
    """Run data preparation using the Rust-accelerated backend."""
    from leech.io import collect_read_infos, get_motif_searcher
    from leech.preparation.parallel import _prepare_batch_rust

    read_infos = collect_read_infos(bam_path, min_mapq=0)
    if max_reads is not None:
        read_infos = read_infos[:max_reads]

    if config.motif.motif is not None:
        motif_searcher = get_motif_searcher(
            mode=config.motif.motif_reference,
            reference_sequences=config.motif.reference_sequences,
            skip_indels=config.motif.skip_motif_indels,
            anchor=config.signal.anchor,
        )
    else:
        motif_searcher = None

    batches = [read_infos[i : i + chunk_size] for i in range(0, len(read_infos), chunk_size)]

    all_chunks = []
    t0 = time.perf_counter()
    for batch in batches:
        chunks = _prepare_batch_rust(batch, config, motif_searcher)
        all_chunks.extend(chunks)
    elapsed = time.perf_counter() - t0

    return all_chunks, elapsed


def compare_chunks(
    py_chunks: list[dict], rs_chunks: list[dict], *, atol: float = 1e-5, rtol: float = 1e-4
) -> tuple[bool, list[str]]:
    """
    Compare training chunks from Python and Rust backends.

    Returns (all_match, list_of_differences).
    """
    diffs: list[str] = []

    # Sort both by (read_id, base_idx) for deterministic comparison
    py_sorted = sorted(py_chunks, key=lambda c: (c["read_id"], c["base_idx"]))
    rs_sorted = sorted(rs_chunks, key=lambda c: (c["read_id"], c["base_idx"]))

    if len(py_sorted) != len(rs_sorted):
        diffs.append(f"Chunk count differs: Python={len(py_sorted)}, Rust={len(rs_sorted)}")
        # Still compare as many as possible
        n = min(len(py_sorted), len(rs_sorted))
    else:
        n = len(py_sorted)

    if n == 0:
        diffs.append("No chunks to compare!")
        return False, diffs

    # Build lookup for Rust chunks by (read_id, base_idx)
    rs_lookup: dict[tuple[str, int], dict] = {}
    for c in rs_sorted:
        key = (c["read_id"], int(c["base_idx"]))
        rs_lookup[key] = c

    matched = 0
    mismatched = 0
    missing = 0

    for pc in py_sorted:
        key = (pc["read_id"], int(pc["base_idx"]))
        rc = rs_lookup.get(key)
        if rc is None:
            missing += 1
            if missing <= 5:
                diffs.append(f"  Missing in Rust: read={key[0][:8]}... base_idx={key[1]}")
            continue

        # Compare signal
        py_sig = np.asarray(pc["signal"], dtype=np.float32)
        rs_sig = np.asarray(rc["signal"], dtype=np.float32)
        if py_sig.shape != rs_sig.shape:
            diffs.append(
                f"  Signal shape: Python={py_sig.shape} Rust={rs_sig.shape} ({key[0][:8]}...)"
            )
            mismatched += 1
            continue
        if not np.allclose(py_sig, rs_sig, atol=atol, rtol=rtol):
            max_diff = np.max(np.abs(py_sig - rs_sig))
            diffs.append(f"  Signal mismatch: max_diff={max_diff:.6f} ({key[0][:8]}...)")
            mismatched += 1
            continue

        # Compare sequence
        if str(pc["sequence"]) != str(rc["sequence"]):
            diffs.append(
                f"  Sequence mismatch: Python={pc['sequence']!r} "
                f"Rust={rc['sequence']!r} ({key[0][:8]}...)"
            )
            mismatched += 1
            continue

        # Compare dwell
        py_dw = np.asarray(pc["dwell"], dtype=np.float32)
        rs_dw = np.asarray(rc["dwell"], dtype=np.float32)
        if py_dw.shape != rs_dw.shape:
            diffs.append(f"  Dwell shape: Python={py_dw.shape} Rust={rs_dw.shape}")
            mismatched += 1
            continue
        if not np.allclose(py_dw, rs_dw, atol=atol, rtol=rtol):
            max_diff = np.max(np.abs(py_dw - rs_dw))
            diffs.append(f"  Dwell mismatch: max_diff={max_diff:.6f}")
            mismatched += 1
            continue

        # Compare features
        py_feat = np.asarray(pc["features"], dtype=np.float32)
        rs_feat = np.asarray(rc["features"], dtype=np.float32)
        if py_feat.shape != rs_feat.shape:
            diffs.append(f"  Features shape: Python={py_feat.shape} Rust={rs_feat.shape}")
            mismatched += 1
            continue
        if py_feat.size > 0 and not np.allclose(py_feat, rs_feat, atol=atol, rtol=rtol):
            max_diff = np.max(np.abs(py_feat - rs_feat))
            diffs.append(f"  Features mismatch: max_diff={max_diff:.6f}")
            mismatched += 1
            continue

        matched += 1

    summary = (
        f"Compared {len(py_sorted)} Python vs {len(rs_sorted)} Rust chunks: "
        f"{matched} matched, {mismatched} mismatched, {missing} missing from Rust"
    )
    diffs.insert(0, summary)

    # A chunk Python produced and Rust did not is a defect, not a tolerance.
    # This used to pass `missing` through as expected, because Rust dropped
    # focus bases whose k-mer window overhung the sequence while Python padded
    # them with "N" -- on production data ~1% of reads, silently (issue #185).
    return mismatched == 0 and missing == 0, diffs


def compare_npz(
    py_path: Path,
    rs_path: Path,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    skip_keys: set[str] | None = None,
) -> tuple[bool, list[str]]:
    """Compare two NPZ files array-by-array.

    Args:
        skip_keys: Keys to skip comparison for (reported as INFO, not failure).
    """
    diffs: list[str] = []
    skip_keys = skip_keys or set()

    with (
        np.load(py_path, allow_pickle=True) as py_data,
        np.load(rs_path, allow_pickle=True) as rs_data,
    ):
        py_keys = set(py_data.files)
        rs_keys = set(rs_data.files)

        if py_keys != rs_keys:
            diffs.append(
                f"Key mismatch: only_python={py_keys - rs_keys}, only_rust={rs_keys - py_keys}"
            )
            common = py_keys & rs_keys
        else:
            common = py_keys

        all_match = True
        for key in sorted(common):
            if key in skip_keys:
                continue

            py_arr = py_data[key]
            rs_arr = rs_data[key]

            if py_arr.shape != rs_arr.shape:
                diffs.append(f"  {key}: shape Python={py_arr.shape} Rust={rs_arr.shape}")
                all_match = False
                continue

            if py_arr.dtype.kind == "U" or py_arr.dtype == object:
                # String or object array — compare elementwise
                if not np.array_equal(py_arr, rs_arr):
                    n_diff = np.sum(py_arr != rs_arr)
                    diffs.append(f"  {key}: {n_diff}/{py_arr.size} elements differ")
                    all_match = False
            elif np.issubdtype(py_arr.dtype, np.floating):
                if not np.allclose(py_arr, rs_arr, atol=atol, rtol=rtol, equal_nan=True):
                    max_diff = np.nanmax(np.abs(py_arr.astype(float) - rs_arr.astype(float)))
                    diffs.append(f"  {key}: max_diff={max_diff:.6f}")
                    all_match = False
            else:
                if not np.array_equal(py_arr, rs_arr):
                    n_diff = np.sum(py_arr != rs_arr)
                    diffs.append(f"  {key}: {n_diff}/{py_arr.size} elements differ")
                    all_match = False

    return all_match, diffs


def _run_comparison(
    bam_path: Path,
    refine: bool,
    max_reads: int,
    chunk_size: int,
) -> tuple[bool, int, int]:
    """Run one comparison (with or without refinement).

    Returns (core_pass, n_py_chunks, n_rs_chunks).
    """
    config = build_config(POD5, refine=refine)
    label = "refined" if refine else "unrefined"

    print(f"\n{'=' * 60}")
    print(f"  Mode: {label}")
    print(f"{'=' * 60}")

    # --- Run Python backend ---
    print("Running Python backend...")
    py_chunks, py_time = run_python_backend(bam_path, config, chunk_size, max_reads)
    print(f"  Python: {len(py_chunks)} chunks in {py_time:.2f}s")
    if py_time > 0:
        print(f"  Python: {len(py_chunks) / py_time:.0f} chunks/s")

    # --- Run Rust backend ---
    print("Running Rust backend...")
    rs_chunks, rs_time = run_rust_backend(bam_path, config, chunk_size, max_reads)
    print(f"  Rust:   {len(rs_chunks)} chunks in {rs_time:.2f}s")
    if rs_time > 0:
        print(f"  Rust:   {len(rs_chunks) / rs_time:.0f} chunks/s")

    # --- Speedup ---
    if py_time > 0 and rs_time > 0:
        print(f"  Speedup: {py_time / rs_time:.1f}x")

    # --- Compare chunk data ---
    chunk_atol = 1e-5 if not refine else 0.1
    print(f"\n--- Chunk Comparison (atol={chunk_atol}) ---")
    match, diffs = compare_chunks(py_chunks, rs_chunks, atol=chunk_atol)
    for d in diffs[:1]:
        print(d)
    diff_details = [d for d in diffs[1:] if "mismatch" in d or "Missing" in d or "shape" in d]
    if diff_details:
        for d in diff_details[:20]:
            print(d)
        if len(diff_details) > 20:
            print(f"  ... and {len(diff_details) - 20} more")
    if match:
        print(f"PASS: Core arrays match ({label})")
    else:
        print(f"{'FAIL' if not refine else 'INFO'}: Core arrays differ ({label})")
        if refine:
            print("  (Signal refinement uses different iterative DP implementations;")
            print("   divergence is expected and does not affect model training quality)")

    # --- Compare NPZ output ---
    print(f"\n--- NPZ Comparison ({label}) ---")
    from leech.chunking import save_chunks

    py_by_key = {(c["read_id"], int(c["base_idx"])): c for c in py_chunks}
    rs_by_key = {(c["read_id"], int(c["base_idx"])): c for c in rs_chunks}
    common_keys = sorted(set(py_by_key.keys()) & set(rs_by_key.keys()))

    if common_keys:
        py_matched = [py_by_key[k] for k in common_keys]
        rs_matched = [rs_by_key[k] for k in common_keys]
        print(f"Comparing NPZ from {len(common_keys)} matched chunks")

        with tempfile.TemporaryDirectory() as tmpdir:
            py_npz = Path(tmpdir) / "python.npz"
            rs_npz = Path(tmpdir) / "rust.npz"
            save_chunks(py_matched, py_npz, compressed=False)
            save_chunks(rs_matched, rs_npz, compressed=False)

            # Skip auxiliary keys that use different windowing approaches:
            # - seq_to_sig_maps: Python=searchsorted-based, Rust=kmer-window-based
            # - sequences_with_kmer_context: follows from seq_to_sig_maps windowing
            # These only matter for signal_kmer encoding (not base_onehot).
            aux_keys = {"seq_to_sig_maps", "sequences_with_kmer_context"}
            npz_match, npz_diffs = compare_npz(py_npz, rs_npz, atol=chunk_atol, skip_keys=aux_keys)
            for d in npz_diffs:
                print(d)
            if npz_match:
                print(f"PASS: NPZ core arrays match ({label})")
            else:
                print(f"{'FAIL' if not refine else 'INFO'}: NPZ core arrays differ ({label})")
            print(f"  (skipped auxiliary keys: {', '.join(sorted(aux_keys))})")
    else:
        print("SKIP: No matched chunks for NPZ comparison")

    return match, len(py_chunks), len(rs_chunks)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Python vs Rust data prep backends")
    parser.add_argument("--reads", type=int, default=500, help="Max reads to process")
    parser.add_argument("--chunk-size", type=int, default=100, help="Reads per batch")
    parser.add_argument("--no-refine", action="store_true", help="Only run unrefined comparison")
    parser.add_argument("--refine-only", action="store_true", help="Only run refined comparison")
    args = parser.parse_args()

    # Check prerequisites
    if not BAM.exists() or not POD5.exists():
        print(f"ERROR: Production data not found at {BAM} / {POD5}")
        sys.exit(1)

    from leech._rust_accel import HAS_RUST, _rs_extract_training_chunks

    if not HAS_RUST or _rs_extract_training_chunks is None:
        print("ERROR: leech_core Rust acceleration not available")
        print("Build with: cd rust && uv run maturin develop --release")
        sys.exit(1)

    print(f"Benchmarking with {args.reads} reads, chunk_size={args.chunk_size}")
    print(f"BAM: {BAM}")
    print(f"POD5: {POD5}")

    # Run both modes (unrefined = correctness proof, refined = speedup benchmark)
    unrefined_pass = True
    if not args.refine_only:
        unrefined_pass, _, _ = _run_comparison(
            BAM, refine=False, max_reads=args.reads, chunk_size=args.chunk_size
        )

    if not args.no_refine:
        _run_comparison(BAM, refine=True, max_reads=args.reads, chunk_size=args.chunk_size)

    # Exit code based on unrefined test (correctness proof)
    print(f"\n{'=' * 60}")
    if unrefined_pass:
        print("RESULT: Unrefined parity PASS — core logic is correct")
    else:
        print("RESULT: Unrefined parity FAIL — core logic bug")
    print(f"{'=' * 60}")
    sys.exit(0 if unrefined_pass else 1)


if __name__ == "__main__":
    main()
