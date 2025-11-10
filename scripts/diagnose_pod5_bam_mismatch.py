#!/usr/bin/env python3
"""
Diagnostic script to identify read ID mismatches between POD5 and BAM files.

This helps debug the "Read not found in POD5" warnings during data preparation.
"""

import sys
from pathlib import Path

import pysam
from pod5 import DatasetReader


def analyze_mismatch(bam_path: Path, pod5_path: Path, sample_size: int = 1000):
    """
    Analyze read ID mismatches between BAM and POD5 files.

    Args:
        bam_path: Path to BAM file
        pod5_path: Path to POD5 file
        sample_size: Number of BAM reads to sample for analysis
    """
    print(f"\n{'='*80}")
    print(f"POD5/BAM Read ID Mismatch Diagnostic")
    print(f"{'='*80}\n")

    print(f"BAM file:  {bam_path}")
    print(f"POD5 file: {pod5_path}")
    print()

    # 1. Count total reads in POD5
    print("Step 1: Counting reads in POD5...")
    pod5_read_ids = set()
    with DatasetReader(pod5_path) as reader:
        for read_record in reader.reads():
            pod5_read_ids.add(str(read_record.read_id))

    total_pod5_reads = len(pod5_read_ids)
    print(f"  Total reads in POD5: {total_pod5_reads:,}")

    # 2. Sample reads from BAM
    print(f"\nStep 2: Sampling {sample_size} reads from BAM...")
    bam_read_ids = []
    bam_read_ids_set = set()

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        total_bam_reads = 0
        for aln in bam:
            if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                continue

            read_id = aln.query_name
            if read_id:
                if len(bam_read_ids) < sample_size:
                    bam_read_ids.append(read_id)
                bam_read_ids_set.add(read_id)
                total_bam_reads += 1

    print(f"  Total aligned reads in BAM: {total_bam_reads:,}")
    print(f"  Unique read IDs in BAM: {len(bam_read_ids_set):,}")
    print(f"  Sampled reads for analysis: {len(bam_read_ids):,}")

    # 3. Check for matches
    print("\nStep 3: Checking for read ID matches...")
    matches = []
    mismatches = []

    for read_id in bam_read_ids:
        if read_id in pod5_read_ids:
            matches.append(read_id)
        else:
            mismatches.append(read_id)

    match_rate = len(matches) / len(bam_read_ids) * 100 if bam_read_ids else 0

    print(f"  Matches: {len(matches):,} ({match_rate:.1f}%)")
    print(f"  Mismatches: {len(mismatches):,} ({100-match_rate:.1f}%)")

    # 4. Analyze mismatch patterns
    if mismatches:
        print("\nStep 4: Analyzing mismatch patterns...")
        print("\n  Sample of BAM read IDs NOT in POD5:")
        for read_id in mismatches[:10]:
            print(f"    {read_id}")

        print("\n  Sample of POD5 read IDs (for comparison):")
        for read_id in list(pod5_read_ids)[:10]:
            print(f"    {read_id}")

        # Check for format differences
        print("\n  Format analysis:")
        bam_sample = mismatches[0] if mismatches else bam_read_ids[0]
        pod5_sample = list(pod5_read_ids)[0]

        print(f"    BAM read ID length:  {len(bam_sample)} chars")
        print(f"    POD5 read ID length: {len(pod5_sample)} chars")
        print(f"    BAM contains '-':    {'-' in bam_sample}")
        print(f"    POD5 contains '-':   {'-' in pod5_sample}")

        # Check if UUIDs match with different formatting
        bam_normalized = [r.replace('-', '').lower() for r in mismatches[:100]]
        pod5_normalized = {r.replace('-', '').lower(): r for r in pod5_read_ids}

        format_matches = sum(1 for bn in bam_normalized if bn in pod5_normalized)

        if format_matches > 0:
            print(f"\n  ⚠️  FOUND FORMAT ISSUE: {format_matches} reads match after normalizing format!")
            print("      The read IDs are the same but formatted differently.")

    # 5. Summary and recommendations
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    if match_rate > 95:
        print("✓ Good: >95% of BAM reads found in POD5")
        print(f"  {len(mismatches)} missing reads may be expected (QC failures, etc.)")
    elif match_rate > 50:
        print(f"⚠️  Moderate mismatch: {100-match_rate:.1f}% of BAM reads not in POD5")
        print("  Possible causes:")
        print("    - Incomplete POD5 merge (some source files missing)")
        print("    - BAM basecalled from different POD5 set")
        print("    - POD5 was filtered after basecalling")
    else:
        print(f"❌ Major mismatch: {100-match_rate:.1f}% of BAM reads not in POD5")
        print("  Likely causes:")
        print("    - Wrong POD5 file (different sequencing run)")
        print("    - Read ID format incompatibility")
        print("    - Corrupted merge")

    print(f"\nEstimated success rate for full dataset: ~{match_rate:.1f}%")
    print(f"Expected warnings: ~{int(total_bam_reads * (100-match_rate) / 100):,} reads")
    print()

    return {
        'total_pod5_reads': total_pod5_reads,
        'total_bam_reads': total_bam_reads,
        'match_rate': match_rate,
        'matches': len(matches),
        'mismatches': len(mismatches)
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python diagnose_pod5_bam_mismatch.py <bam_file> <pod5_file>")
        print("\nExample:")
        print("  python diagnose_pod5_bam_mismatch.py alignments.bam reads_merged.pod5")
        sys.exit(1)

    bam_path = Path(sys.argv[1])
    pod5_path = Path(sys.argv[2])

    if not bam_path.exists():
        print(f"Error: BAM file not found: {bam_path}")
        sys.exit(1)

    if not pod5_path.exists():
        print(f"Error: POD5 file not found: {pod5_path}")
        sys.exit(1)

    analyze_mismatch(bam_path, pod5_path)


if __name__ == "__main__":
    main()
