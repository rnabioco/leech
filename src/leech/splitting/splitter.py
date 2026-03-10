"""
Read-level data splitting to prevent data leakage.

Provides functions for splitting training data at the read level (not chunk level)
to ensure no molecule appears in both training and validation sets.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from leech.chunking import load_chunks, save_chunks

logger = logging.getLogger("leech.splitting.splitter")


def split_chunks_by_read(
    chunks: list[dict],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split chunks into train/val/test sets at the READ level to prevent data leakage.

    Groups chunks by read_id, then splits the read IDs into train/val/test sets.
    This ensures that no read appears in multiple splits, preventing the model
    from seeing similar signals from the same molecule during training and validation.

    Args:
        chunks: List of chunk dictionaries (must have 'read_id' key)
        train_frac: Fraction of reads for training
        val_frac: Fraction of reads for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_chunks, val_chunks, test_chunks)

    Raises:
        ValueError: If fractions don't sum to <= 1.0

    Example:
        >>> chunks = load_chunks(Path("all_chunks.npz"))
        >>> train, val, test = split_chunks_by_read(chunks, seed=42)
        >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    """
    if train_frac + val_frac > 1.0:
        raise ValueError(f"train_frac ({train_frac}) + val_frac ({val_frac}) must be <= 1.0")

    # Set seed if provided
    if seed is not None:
        random.seed(seed)

    # Group chunks by read_id
    read_to_chunks: dict[str, list[dict]] = {}
    for chunk in chunks:
        read_id = chunk["read_id"]
        if read_id not in read_to_chunks:
            read_to_chunks[read_id] = []
        read_to_chunks[read_id].append(chunk)

    # Get list of unique read IDs and shuffle
    read_ids = list(read_to_chunks.keys())
    random.shuffle(read_ids)

    # Split read IDs
    n_reads = len(read_ids)
    n_train = int(n_reads * train_frac)
    n_val = int(n_reads * val_frac)

    train_read_ids = set(read_ids[:n_train])
    val_read_ids = set(read_ids[n_train : n_train + n_val])
    test_read_ids = set(read_ids[n_train + n_val :])

    # Assign chunks to splits based on read ID
    train_chunks = []
    val_chunks = []
    test_chunks = []

    for read_id, read_chunks in read_to_chunks.items():
        if read_id in train_read_ids:
            train_chunks.extend(read_chunks)
        elif read_id in val_read_ids:
            val_chunks.extend(read_chunks)
        elif read_id in test_read_ids:
            test_chunks.extend(read_chunks)

    logger.info(f"Split {n_reads} reads into train/val/test")
    logger.info(
        f"  Train: {len(train_read_ids)} reads ({len(train_chunks)} chunks, "
        f"{len(train_chunks) / len(chunks) * 100:.1f}%)"
    )
    logger.info(
        f"  Val: {len(val_read_ids)} reads ({len(val_chunks)} chunks, "
        f"{len(val_chunks) / len(chunks) * 100:.1f}%)"
    )
    logger.info(
        f"  Test: {len(test_read_ids)} reads ({len(test_chunks)} chunks, "
        f"{len(test_chunks) / len(chunks) * 100:.1f}%)"
    )

    return train_chunks, val_chunks, test_chunks


def merge_and_split_chunks(
    input_paths: list[Path],
    output_dir: Path | None = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int | None = None,
    relabel_pairwise: tuple[str | list[str], str | list[str]] | None = None,
) -> tuple[list[dict], list[dict], list[dict]] | dict[str, Any]:
    """
    Merge multiple chunk files and split at read level to prevent data leakage.

    This implements the correct workflow for multi-sample datasets:
    1. Load and merge all chunks from different samples
    2. Split merged data at the READ level into train/val/test
    3. Optionally relabel chunks for pairwise classification
    4. Optionally save splits to disk

    This prevents data leakage that occurs when splitting each sample independently
    and then merging the splits, which can allow reads from the same molecule to
    appear in both training and validation sets.

    Memory-efficient implementation: First pass collects only read IDs and metadata
    to determine splits, second pass loads and assigns chunks to appropriate splits.

    Args:
        input_paths: List of paths to .npz chunk files to merge
        output_dir: Directory to save splits (if None, only return chunks without saving)
        train_frac: Fraction of reads for training
        val_frac: Fraction of reads for validation
        seed: Random seed for reproducibility
        relabel_pairwise: Optional tuple of (group1, group2) for pairwise comparison.
            Each group can be a single label (str) or multiple labels (list[str]).
            Chunks matching group1 get label_int=0, chunks matching group2 get label_int=1.
            Examples:
                ("Ala", "Gly") - Single label per group
                (["Lys", "Arg"], ["Glu", "Asp"]) - Multiple labels per group (basic vs acidic)

    Returns:
        If output_dir is None: Tuple of (train_chunks, val_chunks, test_chunks)
        If output_dir provided: Dictionary with statistics:
        {
            'n_total': int,
            'n_train': int,
            'n_val': int,
            'n_test': int,
            'output_files': {'train': Path, 'val': Path, 'test': Path}
        }

    Example:
        >>> # Merge charged and uncharged samples, then split
        >>> result = merge_and_split_chunks(
        ...     [Path("charged_all.npz"), Path("uncharged_all.npz")],
        ...     output_dir=Path("merged"),
        ...     train_frac=0.7,
        ...     val_frac=0.15,
        ...     seed=42
        ... )

        >>> # Pairwise amino acid comparison with relabeling
        >>> result = merge_and_split_chunks(
        ...     [Path("ala_all.npz"), Path("gly_all.npz")],
        ...     output_dir=Path("merged/Ala_vs_Gly"),
        ...     relabel_pairwise=("Ala", "Gly"),
        ...     seed=42
        ... )
    """
    from leech.util import setup_random_seed

    if train_frac + val_frac > 1.0:
        raise ValueError(f"train_frac ({train_frac}) + val_frac ({val_frac}) must be <= 1.0")

    # Setup seed and output directory if provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        setup_random_seed(seed, output_dir)
    elif seed is not None:
        # If no output_dir but seed provided, just set the seed
        random.seed(seed)
        logger.info(f"Using provided seed: {seed}")

    logger.info(f"Merging {len(input_paths)} chunk files (memory-efficient mode)")

    # First pass: collect only read IDs from all files (minimal memory)
    logger.info("Pass 1: Collecting read IDs for split assignment")
    all_read_ids: set[str] = set()
    file_chunk_counts = []

    for chunk_path in input_paths:
        logger.info(f"  Scanning {chunk_path}")
        # Load only read_ids array (much smaller than full data)
        with np.load(chunk_path, allow_pickle=True) as data:
            read_ids = data["read_ids"]
            all_read_ids.update(str(rid) for rid in read_ids)
            file_chunk_counts.append(len(read_ids))
            logger.info(f"    Found {len(read_ids)} chunks")

    logger.info(f"Total unique reads across all files: {len(all_read_ids)}")

    # Determine read-level splits
    if seed is not None:
        random.seed(seed)

    read_ids_list = list(all_read_ids)
    random.shuffle(read_ids_list)

    n_reads = len(read_ids_list)
    n_train = int(n_reads * train_frac)
    n_val = int(n_reads * val_frac)

    train_read_ids = set(read_ids_list[:n_train])
    val_read_ids = set(read_ids_list[n_train : n_train + n_val])
    test_read_ids = set(read_ids_list[n_train + n_val :])

    logger.info(f"Split {n_reads} unique reads:")
    logger.info(
        f"  Train: {len(train_read_ids)} reads ({len(train_read_ids) / n_reads * 100:.1f}%)"
    )
    logger.info(f"  Val: {len(val_read_ids)} reads ({len(val_read_ids) / n_reads * 100:.1f}%)")
    logger.info(f"  Test: {len(test_read_ids)} reads ({len(test_read_ids) / n_reads * 100:.1f}%)")

    # Second pass: load chunks and assign to splits (one file at a time)
    logger.info("Pass 2: Loading chunks and assigning to splits")
    train_chunks = []
    val_chunks = []
    test_chunks = []

    for chunk_path in input_paths:
        logger.info(f"  Processing {chunk_path}")
        # Load chunks from this file using the standard loader
        chunks = load_chunks(chunk_path)

        # Relabel chunks if pairwise comparison is requested
        if relabel_pairwise is not None:
            group1, group2 = relabel_pairwise
            # Normalize to lists for uniform handling
            group1_labels = [group1] if isinstance(group1, str) else group1
            group2_labels = [group2] if isinstance(group2, str) else group2

            relabeled_count = 0
            skipped_count = 0
            for chunk in chunks:
                chunk_label = chunk.get("label")
                if chunk_label in group1_labels:
                    chunk["label_int"] = 0
                    relabeled_count += 1
                elif chunk_label in group2_labels:
                    chunk["label_int"] = 1
                    relabeled_count += 1
                else:
                    # Skip chunks that don't match either group
                    logger.warning(
                        f"Chunk with label='{chunk_label}' does not match "
                        f"pairwise comparison (group1={group1_labels}, group2={group2_labels}), keeping original label"
                    )
                    skipped_count += 1
            if relabeled_count > 0:
                logger.info(
                    f"    Relabeled {relabeled_count} chunks for pairwise comparison "
                    f"(group1={group1_labels}→0, group2={group2_labels}→1)"
                )
            if skipped_count > 0:
                logger.warning(f"    Skipped {skipped_count} chunks with mismatched labels")

        # Assign to appropriate split
        for chunk in chunks:
            read_id = chunk["read_id"]
            if read_id in train_read_ids:
                train_chunks.append(chunk)
            elif read_id in val_read_ids:
                val_chunks.append(chunk)
            elif read_id in test_read_ids:
                test_chunks.append(chunk)

        logger.info(f"    Assigned {len(chunks)} chunks")
        # Clear chunks to free memory before loading next file
        del chunks

    n_total = len(train_chunks) + len(val_chunks) + len(test_chunks)
    logger.info("Final chunk distribution:")
    logger.info(f"  Train: {len(train_chunks)} chunks")
    logger.info(f"  Val: {len(val_chunks)} chunks")
    logger.info(f"  Test: {len(test_chunks)} chunks")

    # Check label distribution and warn if all labels are the same
    all_chunks_combined = train_chunks + val_chunks + test_chunks
    unique_labels = {chunk["label"] for chunk in all_chunks_combined if chunk["label"] is not None}
    if len(unique_labels) == 1:
        logger.warning(
            f"⚠️  WARNING: All chunks have the same label ({list(unique_labels)[0]})! "
            "This suggests pairwise relabeling may not have worked correctly. "
            "Check that label values match the relabel_pairwise argument."
        )
    elif len(unique_labels) > 0:
        label_counts: dict[str, int] = {}
        for chunk in all_chunks_combined:
            label = chunk["label"]
            if label is not None:
                label_counts[label] = label_counts.get(label, 0) + 1
        logger.info(f"Label distribution: {label_counts}")

    # If output_dir provided, save splits and return statistics
    if output_dir is not None:
        train_file = output_dir / "train.npz"
        val_file = output_dir / "val.npz"
        test_file = output_dir / "test.npz"

        save_chunks(train_chunks, train_file)
        logger.info(f"Saved {len(train_chunks)} train chunks to {train_file}")

        save_chunks(val_chunks, val_file)
        logger.info(f"Saved {len(val_chunks)} val chunks to {val_file}")

        save_chunks(test_chunks, test_file)
        logger.info(f"Saved {len(test_chunks)} test chunks to {test_file}")

        return {
            "n_total": n_total,
            "n_train": len(train_chunks),
            "n_val": len(val_chunks),
            "n_test": len(test_chunks),
            "output_files": {
                "train": train_file,
                "val": val_file,
                "test": test_file,
            },
        }
    else:
        # Legacy behavior: return tuple of chunks without saving
        return train_chunks, val_chunks, test_chunks


def merge_and_kfold_split_chunks(
    input_paths: list[Path],
    output_dir: Path,
    k_fold: int,
    seed: int | None = None,
    relabel_pairwise: tuple[str | list[str], str | list[str]] | None = None,
) -> dict[str, Any]:
    """
    Merge multiple chunk files and split into k folds at read level for cross-validation.

    This implements k-fold cross-validation with read-level splitting to prevent
    data leakage. For each fold i:
    - test = partition[i]
    - val = partition[(i+1) % k]
    - train = all remaining partitions

    Memory-efficient implementation: First pass collects only read IDs to determine
    fold assignments, second pass loads and assigns chunks to appropriate folds.

    Args:
        input_paths: List of paths to .npz chunk files to merge
        output_dir: Directory to save fold splits (fold_0/, fold_1/, ...)
        k_fold: Number of folds (must be >= 3)
        seed: Random seed for reproducibility
        relabel_pairwise: Optional tuple of (group1, group2) for pairwise comparison.
            Each group can be a single label (str) or multiple labels (list[str]).
            Chunks matching group1 get label_int=0, chunks matching group2 get label_int=1.

    Returns:
        Dictionary with statistics:
        {
            'k_fold': int,
            'n_total': int,
            'folds': [
                {'n_train': int, 'n_val': int, 'n_test': int, 'output_files': {...}},
                ...
            ]
        }

    Raises:
        ValueError: If k_fold < 3

    Example:
        >>> result = merge_and_kfold_split_chunks(
        ...     [Path("ala.npz"), Path("gly.npz")],
        ...     output_dir=Path("kfold/Ala_vs_Gly"),
        ...     k_fold=5,
        ...     relabel_pairwise=("Ala", "Gly"),
        ...     seed=42
        ... )
    """
    from leech.util import setup_random_seed

    if k_fold < 3:
        raise ValueError(f"k_fold must be >= 3, got {k_fold}")

    # Setup seed and output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_random_seed(seed, output_dir)

    logger.info(f"Merging {len(input_paths)} chunk files for {k_fold}-fold cross-validation")

    # ---- Pass 1: collect read IDs ----
    logger.info("Pass 1: Collecting read IDs for fold assignment")
    all_read_ids: set[str] = set()
    file_chunk_counts = []

    for chunk_path in input_paths:
        logger.info(f"  Scanning {chunk_path}")
        with np.load(chunk_path, allow_pickle=True) as data:
            read_ids = data["read_ids"]
            all_read_ids.update(str(rid) for rid in read_ids)
            file_chunk_counts.append(len(read_ids))
            logger.info(f"    Found {len(read_ids)} chunks")

    logger.info(f"Total unique reads across all files: {len(all_read_ids)}")

    # Shuffle read IDs deterministically
    if seed is not None:
        random.seed(seed)

    read_ids_list = list(all_read_ids)
    random.shuffle(read_ids_list)

    # Partition into k groups
    partitions = np.array_split(read_ids_list, k_fold)
    partition_sets = [set(p.tolist()) for p in partitions]

    for i, part in enumerate(partition_sets):
        logger.info(f"  Partition {i}: {len(part)} reads")

    # Build fold assignments: for each fold i, determine train/val/test read sets
    fold_read_assignments: list[dict[str, set[str]]] = []
    for i in range(k_fold):
        test_ids = partition_sets[i]
        val_ids = partition_sets[(i + 1) % k_fold]
        train_ids: set[str] = set()
        for j in range(k_fold):
            if j != i and j != (i + 1) % k_fold:
                train_ids.update(partition_sets[j])
        fold_read_assignments.append({
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        })
        logger.info(
            f"  Fold {i}: train={len(train_ids)} reads, "
            f"val={len(val_ids)} reads, test={len(test_ids)} reads"
        )

    # ---- Pass 2: load chunks and assign to folds ----
    logger.info("Pass 2: Loading chunks and assigning to folds")

    # Initialize accumulators for each fold
    fold_chunks: list[dict[str, list[dict]]] = [
        {"train": [], "val": [], "test": []} for _ in range(k_fold)
    ]

    for chunk_path in input_paths:
        logger.info(f"  Processing {chunk_path}")
        chunks = load_chunks(chunk_path)

        # Relabel chunks if pairwise comparison is requested
        if relabel_pairwise is not None:
            group1, group2 = relabel_pairwise
            group1_labels = [group1] if isinstance(group1, str) else group1
            group2_labels = [group2] if isinstance(group2, str) else group2

            relabeled_count = 0
            skipped_count = 0
            for chunk in chunks:
                chunk_label = chunk.get("label")
                if chunk_label in group1_labels:
                    chunk["label_int"] = 0
                    relabeled_count += 1
                elif chunk_label in group2_labels:
                    chunk["label_int"] = 1
                    relabeled_count += 1
                else:
                    logger.warning(
                        f"Chunk with label='{chunk_label}' does not match "
                        f"pairwise comparison (group1={group1_labels}, group2={group2_labels}), keeping original label"
                    )
                    skipped_count += 1
            if relabeled_count > 0:
                logger.info(
                    f"    Relabeled {relabeled_count} chunks for pairwise comparison "
                    f"(group1={group1_labels}->0, group2={group2_labels}->1)"
                )
            if skipped_count > 0:
                logger.warning(f"    Skipped {skipped_count} chunks with mismatched labels")

        # Distribute chunks to folds based on read ID membership
        for chunk in chunks:
            read_id = chunk["read_id"]
            for i in range(k_fold):
                assignments = fold_read_assignments[i]
                if read_id in assignments["test"]:
                    fold_chunks[i]["test"].append(chunk)
                elif read_id in assignments["val"]:
                    fold_chunks[i]["val"].append(chunk)
                elif read_id in assignments["train"]:
                    fold_chunks[i]["train"].append(chunk)

        logger.info(f"    Assigned {len(chunks)} chunks")
        del chunks

    # Save each fold
    folds_stats: list[dict[str, Any]] = []
    n_total = 0

    for i in range(k_fold):
        fold_dir = output_dir / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_file = fold_dir / "train.npz"
        val_file = fold_dir / "val.npz"
        test_file = fold_dir / "test.npz"

        train_chunks = fold_chunks[i]["train"]
        val_chunks = fold_chunks[i]["val"]
        test_chunks = fold_chunks[i]["test"]

        save_chunks(train_chunks, train_file)
        save_chunks(val_chunks, val_file)
        save_chunks(test_chunks, test_file)

        fold_total = len(train_chunks) + len(val_chunks) + len(test_chunks)
        if i == 0:
            n_total = fold_total

        logger.info(
            f"Fold {i}: train={len(train_chunks)}, val={len(val_chunks)}, "
            f"test={len(test_chunks)} chunks -> {fold_dir}"
        )

        folds_stats.append({
            "n_train": len(train_chunks),
            "n_val": len(val_chunks),
            "n_test": len(test_chunks),
            "output_files": {
                "train": train_file,
                "val": val_file,
                "test": test_file,
            },
        })

    logger.info(f"Saved {k_fold} folds to {output_dir}")

    return {
        "k_fold": k_fold,
        "n_total": n_total,
        "folds": folds_stats,
    }


def parse_comparison_spec(tsv_path: Path) -> list[tuple[str, list[str], str, list[str]]]:
    """
    Parse TSV comparison spec file into list of comparison specifications.

    The TSV file should have 4 columns with NO header:
    - Column 1: meta_label_1 (e.g., "basic")
    - Column 2: label_set_1 (comma-separated, e.g., "Lys,Arg")
    - Column 3: meta_label_2 (e.g., "acidic")
    - Column 4: label_set_2 (comma-separated, e.g., "Glu,Asp")

    Args:
        tsv_path: Path to TSV file

    Returns:
        List of tuples: (meta_label_1, labels_1, meta_label_2, labels_2)
        where labels_1 and labels_2 are lists of label strings

    Example:
        >>> specs = parse_comparison_spec(Path("comparisons.tsv"))
        >>> specs[0]
        ('basic', ['Lys', 'Arg'], 'acidic', ['Glu', 'Asp'])
    """
    comparisons = []
    with open(tsv_path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid format at line {line_num}: expected 4 tab-separated columns, "
                    f"got {len(parts)}. Line: {line}"
                )

            meta_label_1, label_set_1, meta_label_2, label_set_2 = parts

            # Parse comma-separated label sets
            labels_1 = [label.strip() for label in label_set_1.split(",")]
            labels_2 = [label.strip() for label in label_set_2.split(",")]

            comparisons.append((meta_label_1, labels_1, meta_label_2, labels_2))

    logger.info(f"Parsed {len(comparisons)} comparisons from {tsv_path}")
    return comparisons


def process_comparison_spec(
    chunk_dirs: list[Path],
    comparison_spec: Path,
    output_dir: Path,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Process all comparisons from a TSV spec file.

    For each comparison in the spec:
    1. Create output subdirectory named "{meta_label_1}_vs_{meta_label_2}"
    2. Run merge_and_split_chunks with relabel_pairwise=(labels_1, labels_2)
    3. Save metadata.json documenting the comparison

    Args:
        chunk_dirs: List of directories containing .npz chunk files
        comparison_spec: Path to TSV comparison specification file
        output_dir: Base output directory (subdirs created for each comparison)
        train_frac: Fraction of reads for training
        val_frac: Fraction of reads for validation
        seed: Random seed for reproducibility

    Returns:
        Dictionary with processing results:
        {
            'n_comparisons': int,
            'comparisons': {
                'basic_vs_acidic': {'n_train': int, 'n_val': int, 'n_test': int},
                ...
            }
        }

    Example:
        >>> result = process_comparison_spec(
        ...     chunk_dirs=[Path("chunks/Lys"), Path("chunks/Arg"), Path("chunks/Glu")],
        ...     comparison_spec=Path("comparisons.tsv"),
        ...     output_dir=Path("splits/"),
        ...     seed=42
        ... )
    """
    # Parse comparison specifications
    comparisons = parse_comparison_spec(comparison_spec)

    # Collect all chunk files from input directories
    chunk_files = []
    for chunk_dir in chunk_dirs:
        if not chunk_dir.exists():
            logger.warning(f"Chunk directory does not exist: {chunk_dir}")
            continue
        chunk_files.extend(sorted(chunk_dir.glob("*.npz")))

    if not chunk_files:
        raise ValueError(f"No .npz chunk files found in {chunk_dirs}")

    logger.info(f"Found {len(chunk_files)} chunk files across {len(chunk_dirs)} directories")

    # Process each comparison
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for meta_label_1, labels_1, meta_label_2, labels_2 in comparisons:
        comparison_name = f"{meta_label_1}_vs_{meta_label_2}"
        comparison_dir = output_dir / comparison_name

        logger.info(f"\nProcessing comparison: {comparison_name}")
        logger.info(f"  Group 0 ({meta_label_1}): {labels_1}")
        logger.info(f"  Group 1 ({meta_label_2}): {labels_2}")

        # Run merge and split with pairwise relabeling
        result = merge_and_split_chunks(
            input_paths=chunk_files,
            output_dir=comparison_dir,
            train_frac=train_frac,
            val_frac=val_frac,
            seed=seed,
            relabel_pairwise=(labels_1, labels_2),
        )

        # Type narrowing: result is always a dict when output_dir is provided
        assert isinstance(result, dict)

        # Save metadata documenting the comparison
        metadata = {
            "comparison": comparison_name,
            "group_0": {"meta_label": meta_label_1, "labels": labels_1},
            "group_1": {"meta_label": meta_label_2, "labels": labels_2},
            "train_frac": train_frac,
            "val_frac": val_frac,
            "seed": seed,
            "n_train": result["n_train"],
            "n_val": result["n_val"],
            "n_test": result["n_test"],
            "n_total": result["n_total"],
        }

        metadata_file = comparison_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_file}")

        results[comparison_name] = {
            "n_train": result["n_train"],
            "n_val": result["n_val"],
            "n_test": result["n_test"],
        }

    return {"n_comparisons": len(comparisons), "comparisons": results}
