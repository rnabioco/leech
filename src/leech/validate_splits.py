"""
Utility to validate data splits and detect potential data leakage.

This script checks that:
1. No read appears in multiple splits (train/val/test)
2. All chunks are assigned to exactly one split
3. Provides detailed leakage reports if issues are found
"""

import logging
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from leech.data_prep import load_chunks

logger = logging.getLogger("leech.validate_splits")


def collect_read_ids(chunks: List[Dict]) -> Set[str]:
    """Collect unique read IDs from a list of chunks."""
    return set(chunk["read_id"] for chunk in chunks)


def detect_leakage(
    train_path: Path | None = None,
    val_path: Path | None = None,
    test_path: Path | None = None,
    train_chunks: List[Dict] | None = None,
    val_chunks: List[Dict] | None = None,
    test_chunks: List[Dict] | None = None,
) -> Dict:
    """
    Detect data leakage between train/val/test splits.

    Args:
        train_path: Path to train.npz file
        val_path: Path to val.npz file
        test_path: Path to test.npz file
        train_chunks: Pre-loaded train chunks (alternative to train_path)
        val_chunks: Pre-loaded val chunks (alternative to val_path)
        test_chunks: Pre-loaded test chunks (alternative to test_path)

    Returns:
        Dictionary with leakage report:
        {
            "has_leakage": bool,
            "train_val_leak": set of leaked read IDs,
            "train_test_leak": set of leaked read IDs,
            "val_test_leak": set of leaked read IDs,
            "statistics": dict with counts and percentages,
            "severity": "none" | "minor" | "moderate" | "severe"
        }
    """
    # Load chunks if paths provided
    if train_path and train_chunks is None:
        logger.info(f"Loading train chunks from {train_path}")
        train_chunks = load_chunks(train_path)

    if val_path and val_chunks is None:
        logger.info(f"Loading val chunks from {val_path}")
        val_chunks = load_chunks(val_path)

    if test_path and test_chunks is None:
        logger.info(f"Loading test chunks from {test_path}")
        test_chunks = load_chunks(test_path)

    # Collect read IDs from each split
    splits = {}
    if train_chunks:
        splits["train"] = collect_read_ids(train_chunks)
    if val_chunks:
        splits["val"] = collect_read_ids(val_chunks)
    if test_chunks:
        splits["test"] = collect_read_ids(test_chunks)

    # Detect leakage
    train_val_leak = set()
    train_test_leak = set()
    val_test_leak = set()

    if "train" in splits and "val" in splits:
        train_val_leak = splits["train"] & splits["val"]

    if "train" in splits and "test" in splits:
        train_test_leak = splits["train"] & splits["test"]

    if "val" in splits and "test" in splits:
        val_test_leak = splits["val"] & splits["test"]

    # Calculate statistics
    total_reads = len(set().union(*splits.values()))
    total_leaked = len(train_val_leak | train_test_leak | val_test_leak)

    has_leakage = total_leaked > 0
    leak_percentage = (total_leaked / total_reads * 100) if total_reads > 0 else 0

    # Determine severity
    if leak_percentage == 0:
        severity = "none"
    elif leak_percentage < 1:
        severity = "minor"
    elif leak_percentage < 5:
        severity = "moderate"
    else:
        severity = "severe"

    statistics = {
        "total_unique_reads": total_reads,
        "total_leaked_reads": total_leaked,
        "leak_percentage": leak_percentage,
        "train_reads": len(splits.get("train", set())),
        "val_reads": len(splits.get("val", set())),
        "test_reads": len(splits.get("test", set())),
        "train_val_leaked": len(train_val_leak),
        "train_test_leaked": len(train_test_leak),
        "val_test_leaked": len(val_test_leak),
    }

    # Chunk-level statistics
    if train_chunks:
        statistics["train_chunks"] = len(train_chunks)
    if val_chunks:
        statistics["val_chunks"] = len(val_chunks)
    if test_chunks:
        statistics["test_chunks"] = len(test_chunks)

    return {
        "has_leakage": has_leakage,
        "train_val_leak": train_val_leak,
        "train_test_leak": train_test_leak,
        "val_test_leak": val_test_leak,
        "statistics": statistics,
        "severity": severity,
    }


def print_leakage_report(report: Dict, verbose: bool = False):
    """Print a formatted leakage report."""
    print("\n" + "=" * 70)
    print("DATA LEAKAGE VALIDATION REPORT")
    print("=" * 70)

    stats = report["statistics"]
    severity = report["severity"]

    # Overall status
    if report["has_leakage"]:
        status_symbol = "❌" if severity in ["moderate", "severe"] else "⚠️"
        print(f"\n{status_symbol} LEAKAGE DETECTED - Severity: {severity.upper()}")
    else:
        print("\n✅ NO LEAKAGE DETECTED - All splits are clean!")

    # Statistics
    print("\n" + "-" * 70)
    print("STATISTICS")
    print("-" * 70)
    print(f"Total unique reads: {stats['total_unique_reads']}")
    print(f"  Train reads: {stats.get('train_reads', 0)} ({stats.get('train_chunks', 0)} chunks)")
    print(f"  Val reads: {stats.get('val_reads', 0)} ({stats.get('val_chunks', 0)} chunks)")
    print(f"  Test reads: {stats.get('test_reads', 0)} ({stats.get('test_chunks', 0)} chunks)")

    if report["has_leakage"]:
        print(f"\nLeaked reads: {stats['total_leaked_reads']} ({stats['leak_percentage']:.2f}%)")
        print(f"  Train ↔ Val: {stats['train_val_leaked']} reads")
        print(f"  Train ↔ Test: {stats['train_test_leaked']} reads")
        print(f"  Val ↔ Test: {stats['val_test_leaked']} reads")

        # Show leaked read IDs if verbose
        if verbose:
            print("\n" + "-" * 70)
            print("LEAKED READ IDs")
            print("-" * 70)

            if report["train_val_leak"]:
                print(f"\nTrain ↔ Val ({len(report['train_val_leak'])} reads):")
                for read_id in sorted(report["train_val_leak"])[:10]:
                    print(f"  - {read_id}")
                if len(report["train_val_leak"]) > 10:
                    print(f"  ... and {len(report['train_val_leak']) - 10} more")

            if report["train_test_leak"]:
                print(f"\nTrain ↔ Test ({len(report['train_test_leak'])} reads):")
                for read_id in sorted(report["train_test_leak"])[:10]:
                    print(f"  - {read_id}")
                if len(report["train_test_leak"]) > 10:
                    print(f"  ... and {len(report['train_test_leak']) - 10} more")

            if report["val_test_leak"]:
                print(f"\nVal ↔ Test ({len(report['val_test_leak'])} reads):")
                for read_id in sorted(report["val_test_leak"])[:10]:
                    print(f"  - {read_id}")
                if len(report["val_test_leak"]) > 10:
                    print(f"  ... and {len(report['val_test_leak']) - 10} more")

    # Recommendations
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS")
    print("-" * 70)

    if report["has_leakage"]:
        print("\n⚠️  Data leakage detected! This can lead to:")
        print("   • Inflated validation metrics")
        print("   • Overfitting to specific reads")
        print("   • Poor generalization to new data")
        print("\n💡 To fix this issue:")
        print("   1. Use `leech merge-and-split` to merge samples and split at read level")
        print("   2. Or use `--no-split` during prepare, then merge-and-split afterward")
        print("   3. Ensure you're using split_chunks_by_read() in your code")
    else:
        print("\n✅ Your data splits are clean!")
        print("   • No reads appear in multiple splits")
        print("   • Safe to proceed with training")

    print("\n" + "=" * 70 + "\n")


def validate_directory(data_dir: Path, verbose: bool = False) -> Dict:
    """
    Validate splits in a directory containing train/val/test files.

    Args:
        data_dir: Directory containing train.npz, val.npz, test.npz
        verbose: Print detailed leakage information

    Returns:
        Leakage report dictionary
    """
    train_path = data_dir / "train.npz"
    val_path = data_dir / "val.npz"
    test_path = data_dir / "test.npz"

    # Check which files exist
    existing_files = []
    if train_path.exists():
        existing_files.append(f"train.npz")
    if val_path.exists():
        existing_files.append(f"val.npz")
    if test_path.exists():
        existing_files.append(f"test.npz")

    if len(existing_files) < 2:
        raise ValueError(
            f"Need at least 2 split files to check for leakage. "
            f"Found: {', '.join(existing_files) if existing_files else 'none'}"
        )

    logger.info(f"Validating splits in {data_dir}")
    logger.info(f"Found files: {', '.join(existing_files)}")

    # Detect leakage
    report = detect_leakage(
        train_path=train_path if train_path.exists() else None,
        val_path=val_path if val_path.exists() else None,
        test_path=test_path if test_path.exists() else None,
    )

    # Print report
    print_leakage_report(report, verbose=verbose)

    return report


if __name__ == "__main__":
    import click

    @click.command()
    @click.argument("data_dir", type=click.Path(exists=True, path_type=Path))
    @click.option(
        "-v",
        "--verbose",
        is_flag=True,
        help="Print detailed leakage information including read IDs",
    )
    @click.option(
        "--log-level",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
        default="INFO",
        help="Logging level",
    )
    def main(data_dir, verbose, log_level):
        """Validate data splits and detect data leakage.

        DATA_DIR: Directory containing train.npz, val.npz, test.npz files
        """
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(levelname)s: %(message)s",
        )

        # Validate
        report = validate_directory(data_dir, verbose=verbose)

        # Exit with error code if leakage detected
        exit_code = 1 if report["has_leakage"] else 0
        exit(exit_code)

    main()
