#!/usr/bin/env python3
"""
Benchmark CCA classification: Remora vs Leech.

Runs both tools on the same reads and compares:
1. Wall-clock time
2. Classification agreement (ML scores)

Usage:
    # Quick test (1000 reads)
    uv run python benchmarks/benchmark_cca.py \
        --pod5 /path/to/sample.pod5 \
        --bam /path/to/sample.tagged.bam \
        --model /path/to/cca_classifier.pt \
        --num-reads 1000

    # Full benchmark
    uv run python benchmarks/benchmark_cca.py \
        --pod5 /path/to/sample.pod5 \
        --bam /path/to/sample.tagged.bam \
        --model /path/to/cca_classifier.pt \
        --num-reads 50000 --workers 4

    # Use defaults from 2026-aars-in-vitro pipeline
    uv run python benchmarks/benchmark_cca.py --num-reads 10000
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pysam

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("benchmark_cca")

# Default paths from 2026-aars-in-vitro pipeline
PIPELINE_DIR = Path("/beevol/home/jhessel/devel/rnabioco/2026-aars-in-vitro")
DEFAULT_SAMPLE = "AsnRS_asn_b1"
DEFAULT_MODEL = PIPELINE_DIR / "aa-tRNA-seq-pipeline/resources/models/cca_classifier.pt"
DEFAULT_BAM = (
    PIPELINE_DIR / f"results/bam/tagged/{DEFAULT_SAMPLE}/{DEFAULT_SAMPLE}.tagged.bam"
)
DEFAULT_POD5 = (
    PIPELINE_DIR
    / f"results/demux/pod5/{DEFAULT_SAMPLE}/{DEFAULT_SAMPLE}.pod5"
)
REMORA_BIN = (
    PIPELINE_DIR / "aa-tRNA-seq-pipeline/.pixi/envs/default/bin/remora"
)


def subset_bam(bam_path: Path, output_path: Path, num_reads: int) -> int:
    """Subset BAM to first N reads with mv tags. Returns actual count."""
    count = 0
    with pysam.AlignmentFile(str(bam_path), "rb") as bam_in:
        with pysam.AlignmentFile(str(output_path), "wb", template=bam_in) as bam_out:
            for aln in bam_in.fetch(until_eof=True):
                if count >= num_reads:
                    break
                if aln.has_tag("mv") and aln.has_tag("ns"):
                    bam_out.write(aln)
                    count += 1
    # Index
    pysam.index(str(output_path))
    return count


def run_remora(
    pod5_path: Path,
    bam_path: Path,
    model_path: Path,
    output_path: Path,
    num_reads: int | None = None,
    num_workers: int = 2,
) -> float:
    """Run remora inference, return wall-clock seconds."""
    w = max(1, num_workers)
    cmd = [
        str(REMORA_BIN),
        "infer",
        "from_pod5_and_bam",
        str(pod5_path),
        str(bam_path),
        "--model",
        str(model_path),
        "--out-bam",
        str(output_path),
        "--reference-anchored",
        "--num-extract-alignment-workers",
        str(w),
        "--num-prepare-read-workers",
        str(w),
        "--num-prepare-nn-input-workers",
        str(w),
        "--num-post-process-workers",
        str(w),
    ]
    if num_reads is not None:
        cmd.extend(["--num-reads", str(num_reads)])

    logger.info(f"Running remora: {' '.join(cmd)}")
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        logger.error(f"Remora failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
        sys.exit(1)

    logger.info(f"Remora finished in {elapsed:.1f}s")
    return elapsed


def run_leech(
    pod5_path: Path,
    bam_path: Path,
    model_path: Path,
    output_path: Path,
    num_workers: int = 0,
) -> float:
    """Run leech inference, return wall-clock seconds."""
    cmd = [
        "uv",
        "run",
        "leech",
        "predict",
        "--model",
        str(model_path),
        "--pod5",
        str(pod5_path),
        "--bam",
        str(bam_path),
        "--output",
        str(output_path),
        "--device",
        "cpu",
        "--motif",
        "CCAGGC",
        "--motif-offset",
        "3",
        "--reference-anchored",
        "--workers",
        str(num_workers),
    ]

    logger.info(f"Running leech: {' '.join(cmd)}")
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        logger.error(f"Leech failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
        sys.exit(1)

    logger.info(f"Leech finished in {elapsed:.1f}s")
    return elapsed


def extract_remora_scores(bam_path: Path) -> dict[str, int]:
    """Extract read_id -> ML score from remora output BAM.

    Remora writes MM:Z:G+X?,N; and ML:B:C,score for CCA classification.
    Only counts reads where MM tag contains 'G+X?' (the CCA classifier output).
    """
    scores = {}
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for aln in bam.fetch(until_eof=True):
            if not aln.query_name:
                continue
            if aln.has_tag("MM") and aln.has_tag("ML"):
                mm = aln.get_tag("MM")
                if isinstance(mm, str) and "G+X?" in mm:
                    ml = aln.get_tag("ML")
                    if hasattr(ml, "__len__") and len(ml) > 0:
                        scores[aln.query_name] = int(ml[0])
    return scores


def extract_leech_scores(bam_path: Path) -> dict[str, int]:
    """Extract read_id -> ML score from leech output BAM.

    Leech writes MP:B:i (positions) and ML:B:C (scores) tags.
    Only counts reads that have an MP tag (leech's output marker).
    """
    scores = {}
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for aln in bam.fetch(until_eof=True):
            if not aln.query_name:
                continue
            if aln.has_tag("MP") and aln.has_tag("ML"):
                ml = aln.get_tag("ML")
                if hasattr(ml, "__len__") and len(ml) > 0:
                    scores[aln.query_name] = int(ml[0])
    return scores


def compare_scores(
    remora_scores: dict[str, int],
    leech_scores: dict[str, int],
) -> dict:
    """Compare ML scores between remora and leech outputs."""
    common_reads = set(remora_scores.keys()) & set(leech_scores.keys())
    remora_only = set(remora_scores.keys()) - set(leech_scores.keys())
    leech_only = set(leech_scores.keys()) - set(remora_scores.keys())

    if not common_reads:
        return {
            "common_reads": 0,
            "remora_only": len(remora_only),
            "leech_only": len(leech_only),
            "exact_match": 0,
            "exact_match_pct": 0.0,
        }

    exact_match = 0
    diffs = []
    class_agree = 0  # Both agree on charged (>=128) vs uncharged (<128)
    for rid in common_reads:
        r_score = remora_scores[rid]
        l_score = leech_scores[rid]
        if r_score == l_score:
            exact_match += 1
        diffs.append(abs(r_score - l_score))
        # Classification agreement (threshold at 128 = 0.5 probability)
        if (r_score >= 128) == (l_score >= 128):
            class_agree += 1

    diffs = np.array(diffs)

    return {
        "common_reads": len(common_reads),
        "remora_only": len(remora_only),
        "leech_only": len(leech_only),
        "remora_total": len(remora_scores),
        "leech_total": len(leech_scores),
        "exact_match": exact_match,
        "exact_match_pct": 100.0 * exact_match / len(common_reads),
        "classification_agree": class_agree,
        "classification_agree_pct": 100.0 * class_agree / len(common_reads),
        "mean_abs_diff": float(diffs.mean()),
        "median_abs_diff": float(np.median(diffs)),
        "max_abs_diff": int(diffs.max()),
        "p95_abs_diff": float(np.percentile(diffs, 95)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CCA classification: Remora vs Leech"
    )
    parser.add_argument(
        "--pod5",
        type=Path,
        default=DEFAULT_POD5,
        help="POD5 file",
    )
    parser.add_argument(
        "--bam",
        type=Path,
        default=DEFAULT_BAM,
        help="Input BAM file (with mv tags)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Remora CCA classifier model (.pt)",
    )
    parser.add_argument(
        "--num-reads",
        type=int,
        default=10000,
        help="Number of reads to benchmark (default: 10000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of workers for both tools (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files (default: temp dir)",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep output BAM files after comparison",
    )
    parser.add_argument(
        "--skip-remora",
        action="store_true",
        help="Skip remora run (use existing output)",
    )
    parser.add_argument(
        "--skip-leech",
        action="store_true",
        help="Skip leech run (use existing output)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.pod5.exists():
        logger.error(f"POD5 not found: {args.pod5}")
        sys.exit(1)
    if not args.bam.exists():
        logger.error(f"BAM not found: {args.bam}")
        sys.exit(1)
    if not args.model.exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)

    # Setup output directory
    if args.output_dir:
        outdir = args.output_dir
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = Path(tempfile.mkdtemp(prefix="benchmark_cca_"))

    logger.info(f"Output directory: {outdir}")
    logger.info(f"Benchmarking with {args.num_reads} reads")

    # Subset BAM for leech (remora uses --num-reads on the full BAM)
    subset_bam_path = outdir / "subset.bam"
    logger.info("Subsetting BAM...")
    actual_reads = subset_bam(args.bam, subset_bam_path, args.num_reads)
    logger.info(f"Subset to {actual_reads} reads")

    remora_out = outdir / "remora_output.bam"
    leech_out = outdir / "leech_output.bam"

    results = {
        "num_reads": actual_reads,
        "workers": args.workers,
        "sample_bam": str(args.bam),
        "model": str(args.model),
    }

    # Run remora (uses subset BAM + --num-reads for consistent read set)
    if not args.skip_remora:
        remora_time = run_remora(
            args.pod5, subset_bam_path, args.model, remora_out,
            num_reads=actual_reads, num_workers=args.workers,
        )
        results["remora_time_s"] = remora_time
        results["remora_reads_per_sec"] = actual_reads / remora_time
    else:
        logger.info("Skipping remora run")

    # Run leech (uses same subset BAM)
    if not args.skip_leech:
        leech_time = run_leech(
            args.pod5, subset_bam_path, args.model, leech_out, args.workers
        )
        results["leech_time_s"] = leech_time
        results["leech_reads_per_sec"] = actual_reads / leech_time

    # Compare outputs
    if remora_out.exists() and leech_out.exists():
        logger.info("Extracting scores...")
        remora_scores = extract_remora_scores(remora_out)
        leech_scores = extract_leech_scores(leech_out)

        logger.info(
            f"Remora: {len(remora_scores)} scored reads, "
            f"Leech: {len(leech_scores)} scored reads"
        )

        comparison = compare_scores(remora_scores, leech_scores)
        results["comparison"] = comparison

        if "remora_time_s" in results and "leech_time_s" in results:
            speedup = results["remora_time_s"] / results["leech_time_s"]
            results["speedup"] = speedup

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS: CCA Classification")
    print("=" * 60)
    print(f"Reads:   {results['num_reads']}")
    print(f"Workers: {results['workers']}")

    if "remora_time_s" in results:
        print(f"\nRemora:  {results['remora_time_s']:.1f}s "
              f"({results['remora_reads_per_sec']:.0f} reads/s)")
    if "leech_time_s" in results:
        print(f"Leech:   {results['leech_time_s']:.1f}s "
              f"({results['leech_reads_per_sec']:.0f} reads/s)")
    if "speedup" in results:
        print(f"Speedup: {results['speedup']:.2f}x")

    if "comparison" in results:
        c = results["comparison"]
        print(f"\n--- Score Comparison ---")
        print(f"Common reads:           {c['common_reads']}")
        print(f"Remora-only reads:      {c['remora_only']}")
        print(f"Leech-only reads:       {c['leech_only']}")
        if c["common_reads"] > 0:
            print(f"Exact ML match:         {c['exact_match']}/{c['common_reads']} "
                  f"({c['exact_match_pct']:.1f}%)")
            print(f"Classification agree:   {c['classification_agree']}/{c['common_reads']} "
                  f"({c['classification_agree_pct']:.1f}%)")
            print(f"Mean |diff|:            {c['mean_abs_diff']:.2f}")
            print(f"Median |diff|:          {c['median_abs_diff']:.1f}")
            print(f"95th %ile |diff|:       {c['p95_abs_diff']:.1f}")
            print(f"Max |diff|:             {c['max_abs_diff']}")

    print("=" * 60)

    # Save JSON results
    results_path = outdir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    if not args.keep_outputs and not args.output_dir:
        logger.info(f"Temp dir: {outdir} (use --keep-outputs to preserve)")


if __name__ == "__main__":
    main()
