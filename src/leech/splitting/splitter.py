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

from leech.chunking import load_chunks

logger = logging.getLogger("leech.splitting.splitter")


def _source_group_from_path(chunk_path: Path) -> str:
    """Extract source group name from a chunk file path.

    For charging paths like ``.../charging/ThrRS_thr_b1/charged/all.npz``,
    the parent dir is ``charged`` or ``uncharged``, so we go up one more level
    to get the sample name (``ThrRS_thr_b1``).

    For standard paths like ``.../Ala/all.npz``, the parent dir is the label.
    """
    parent_name = chunk_path.parent.name
    if parent_name in ("charged", "uncharged"):
        return chunk_path.parent.parent.name
    return parent_name


def _split_by_group(
    read_to_group: dict[str, str],
    read_to_label: dict[str, str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[set[str], set[str], set[str]]:
    """Assign reads to splits at the group level, stratified per label.

    For each label, its unique groups are shuffled and divided into train/val/test.
    Labels with only 1 group go entirely to train.  Labels with 2 groups get one
    in train and one in test (no val).  Labels with 3+ groups use the requested
    fractions.

    Returns:
        (train_read_ids, val_read_ids, test_read_ids)
    """
    from collections import defaultdict

    # Build label -> set of groups
    label_groups: dict[str, set[str]] = defaultdict(set)
    for rid, grp in read_to_group.items():
        label = read_to_label[rid]
        label_groups[label].add(grp)

    # Build group -> set of read IDs
    group_reads: dict[str, set[str]] = defaultdict(set)
    for rid, grp in read_to_group.items():
        group_reads[grp].add(rid)

    # Assign groups to splits per label
    group_to_split: dict[str, str] = {}
    for label in sorted(label_groups):
        groups = sorted(label_groups[label])
        random.shuffle(groups)
        n = len(groups)

        if n == 1:
            # Single group: train only
            for g in groups:
                group_to_split[g] = "train"
            logger.info(f"  {label}: 1 group -> train only")
        elif n == 2:
            # Two groups: one train, one test
            group_to_split[groups[0]] = "train"
            group_to_split[groups[1]] = "test"
            logger.info(f"  {label}: 2 groups -> train={groups[0]}, test={groups[1]}")
        else:
            # 3+ groups: split by fraction
            n_train = max(1, int(n * train_frac))
            n_val = max(1, int(n * val_frac))
            # Ensure at least 1 in test
            if n_train + n_val >= n:
                n_val = max(1, n - n_train - 1)
            for g in groups[:n_train]:
                group_to_split[g] = "train"
            for g in groups[n_train : n_train + n_val]:
                group_to_split[g] = "val"
            for g in groups[n_train + n_val :]:
                group_to_split[g] = "test"
            train_g = groups[:n_train]
            val_g = groups[n_train : n_train + n_val]
            test_g = groups[n_train + n_val :]
            logger.info(f"  {label}: {n} groups -> train={train_g}, val={val_g}, test={test_g}")

    # Map reads to splits
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    test_ids: set[str] = set()
    for grp, split in group_to_split.items():
        rids = group_reads[grp]
        if split == "train":
            train_ids.update(rids)
        elif split == "val":
            val_ids.update(rids)
        else:
            test_ids.update(rids)

    logger.info(
        f"Group-level split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test reads"
    )
    return train_ids, val_ids, test_ids


def _merge_arrays_by_split(
    input_paths: list[Path],
    split_read_ids: dict[str, set[str]],
    output_paths: dict[str, Path],
    label_overrides: dict[Path, tuple[int, str]] | None = None,
    source_group_overrides: dict[Path, str] | None = None,
) -> dict[str, int]:
    """Merge input .npz files into per-split output files at the array level.

    Instead of creating Python dicts for each chunk, this operates directly on
    numpy arrays: load -> mask -> slice -> accumulate -> concatenate -> save.

    Args:
        input_paths: Paths to input .npz chunk files.
        split_read_ids: Mapping from split name (e.g. "train") to set of read IDs.
        output_paths: Mapping from split name to output .npz path.
        label_overrides: Optional per-file (label_int, label_str) override for multiclass.
        source_group_overrides: Optional per-file source_group string override.

    Returns:
        Mapping from split name to number of chunks saved.
    """
    split_names = list(split_read_ids.keys())

    # Accumulators: split_name -> array_key -> list of arrays
    accumulators: dict[str, dict[str, list[np.ndarray]]] = {s: {} for s in split_names}

    for chunk_path in input_paths:
        with np.load(chunk_path, allow_pickle=True) as data:
            # Get read_ids to build masks
            read_ids_arr = data["read_ids"]
            read_ids_str = np.array([str(r) for r in read_ids_arr])

            # Build boolean mask per split
            masks: dict[str, np.ndarray] = {}
            for sname, rid_set in split_read_ids.items():
                masks[sname] = np.array([r in rid_set for r in read_ids_str], dtype=bool)

            # Collect all array keys from the file
            array_keys = list(data.keys())

            for sname in split_names:
                mask = masks[sname]
                if not mask.any():
                    continue

                count = int(mask.sum())
                for key in array_keys:
                    arr = data[key]
                    sliced = arr[mask]

                    # Apply label overrides
                    if label_overrides is not None and chunk_path in label_overrides:
                        lint, lstr = label_overrides[chunk_path]
                        if key == "labels_int":
                            sliced = np.full(sliced.shape, lint, dtype=sliced.dtype)
                        elif key == "labels":
                            # Let numpy infer dtype to avoid truncation
                            sliced = np.array([lstr] * count, dtype=str)

                    # Apply source_group overrides
                    if source_group_overrides is not None and chunk_path in source_group_overrides:
                        if key == "source_groups":
                            sg = source_group_overrides[chunk_path]
                            # Let numpy infer dtype to avoid truncation
                            sliced = np.array([sg] * count, dtype=str)

                    if key not in accumulators[sname]:
                        accumulators[sname][key] = []
                    accumulators[sname][key].append(sliced)

    # Concatenate and save
    counts: dict[str, int] = {}
    for sname in split_names:
        acc = accumulators[sname]
        if not acc:
            counts[sname] = 0
            continue

        save_kwargs: dict[str, np.ndarray] = {}
        for key, arr_list in acc.items():
            save_kwargs[key] = np.concatenate(arr_list)

        n_chunks = len(save_kwargs.get("read_ids", np.array([])))
        counts[sname] = n_chunks

        out_path = output_paths[sname]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **save_kwargs)
        logger.info(f"Saved {n_chunks} {sname} chunks to {out_path}")

    return counts


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

    # If output_dir provided, use fast array-level merge
    if output_dir is not None:
        train_file = output_dir / "train.npz"
        val_file = output_dir / "val.npz"
        test_file = output_dir / "test.npz"

        # Build overrides for pairwise relabeling
        label_overrides = None
        if relabel_pairwise is not None:
            group1, group2 = relabel_pairwise
            group1_labels = [group1] if isinstance(group1, str) else group1
            group2_labels = [group2] if isinstance(group2, str) else group2
            label_overrides = {}
            for chunk_path in input_paths:
                # Determine which group this file belongs to based on its label
                with np.load(chunk_path, allow_pickle=True) as data:
                    file_labels = data["labels"]
                    if len(file_labels) > 0:
                        first_label = str(file_labels[0])
                        if first_label in group1_labels:
                            label_overrides[chunk_path] = (0, first_label)
                        elif first_label in group2_labels:
                            label_overrides[chunk_path] = (1, first_label)

        # Build source group overrides
        source_group_overrides = {p: _source_group_from_path(p) for p in input_paths}

        counts = _merge_arrays_by_split(
            input_paths=input_paths,
            split_read_ids={
                "train": train_read_ids,
                "val": val_read_ids,
                "test": test_read_ids,
            },
            output_paths={"train": train_file, "val": val_file, "test": test_file},
            label_overrides=label_overrides,
            source_group_overrides=source_group_overrides,
        )

        n_total = sum(counts.values())
        logger.info("Final chunk distribution:")
        logger.info(f"  Train: {counts['train']} chunks")
        logger.info(f"  Val: {counts['val']} chunks")
        logger.info(f"  Test: {counts['test']} chunks")

        return {
            "n_total": n_total,
            "n_train": counts["train"],
            "n_val": counts["val"],
            "n_test": counts["test"],
            "output_files": {
                "train": train_file,
                "val": val_file,
                "test": test_file,
            },
        }
    else:
        # Legacy behavior: return tuple of chunks (uses dict path)
        logger.info("Pass 2: Loading chunks and assigning to splits")
        train_chunks: list[dict] = []
        val_chunks: list[dict] = []
        test_chunks: list[dict] = []

        for chunk_path in input_paths:
            logger.info(f"  Processing {chunk_path}")
            chunks = load_chunks(chunk_path)

            source_group = _source_group_from_path(chunk_path)
            for chunk in chunks:
                chunk["source_group"] = source_group

            if relabel_pairwise is not None:
                group1, group2 = relabel_pairwise
                group1_labels = [group1] if isinstance(group1, str) else group1
                group2_labels = [group2] if isinstance(group2, str) else group2
                for chunk in chunks:
                    chunk_label = chunk.get("label")
                    if chunk_label in group1_labels:
                        chunk["label_int"] = 0
                    elif chunk_label in group2_labels:
                        chunk["label_int"] = 1

            for chunk in chunks:
                read_id = chunk["read_id"]
                if read_id in train_read_ids:
                    train_chunks.append(chunk)
                elif read_id in val_read_ids:
                    val_chunks.append(chunk)
                elif read_id in test_read_ids:
                    test_chunks.append(chunk)
            del chunks

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
        fold_read_assignments.append(
            {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            }
        )
        logger.info(
            f"  Fold {i}: train={len(train_ids)} reads, "
            f"val={len(val_ids)} reads, test={len(test_ids)} reads"
        )

    # ---- Pass 2: merge arrays per fold ----
    logger.info("Pass 2: Merging arrays per fold")

    # Build source group overrides (same for all folds)
    source_group_overrides = {p: _source_group_from_path(p) for p in input_paths}

    # Build label overrides for pairwise relabeling
    label_overrides = None
    if relabel_pairwise is not None:
        group1, group2 = relabel_pairwise
        group1_labels = [group1] if isinstance(group1, str) else group1
        group2_labels = [group2] if isinstance(group2, str) else group2
        label_overrides = {}
        for chunk_path in input_paths:
            with np.load(chunk_path, allow_pickle=True) as data:
                file_labels = data["labels"]
                if len(file_labels) > 0:
                    first_label = str(file_labels[0])
                    if first_label in group1_labels:
                        label_overrides[chunk_path] = (0, first_label)
                    elif first_label in group2_labels:
                        label_overrides[chunk_path] = (1, first_label)

    folds_stats: list[dict[str, Any]] = []
    n_total = 0

    for i in range(k_fold):
        fold_dir = output_dir / f"fold_{i}"

        counts = _merge_arrays_by_split(
            input_paths=input_paths,
            split_read_ids=fold_read_assignments[i],
            output_paths={
                "train": fold_dir / "train.npz",
                "val": fold_dir / "val.npz",
                "test": fold_dir / "test.npz",
            },
            label_overrides=label_overrides,
            source_group_overrides=source_group_overrides,
        )

        fold_total = sum(counts.values())
        if i == 0:
            n_total = fold_total

        logger.info(
            f"Fold {i}: train={counts['train']}, val={counts['val']}, "
            f"test={counts['test']} chunks -> {fold_dir}"
        )

        folds_stats.append(
            {
                "n_train": counts["train"],
                "n_val": counts["val"],
                "n_test": counts["test"],
                "output_files": {
                    "train": fold_dir / "train.npz",
                    "val": fold_dir / "val.npz",
                    "test": fold_dir / "test.npz",
                },
            }
        )

    logger.info(f"Saved {k_fold} folds to {output_dir}")

    return {
        "k_fold": k_fold,
        "n_total": n_total,
        "folds": folds_stats,
    }


def merge_and_split_multiclass(
    input_paths: list[Path],
    labels: list[str],
    output_dir: Path,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int | None = None,
    split_by: str | None = None,
) -> dict[str, Any]:
    """Merge N chunk files into a multi-class dataset with label_int 0..N-1.

    Each input path is assigned the corresponding label from ``labels``.
    Read-level splitting prevents data leakage.

    Args:
        input_paths: Paths to .npz chunk files (one per class).
        labels: Class label for each path (same order/length as input_paths).
        output_dir: Directory to write train.npz / val.npz / test.npz.
        train_frac: Training fraction.
        val_frac: Validation fraction.
        seed: Random seed.
        split_by: Optional NPZ field name (e.g., ``"reference_names"``) to split
            by group instead of by read.  All reads sharing a group value are
            assigned to the same split.  Groups are allocated per-label so that
            each label with ≥2 groups has at least one group in test.

    Returns:
        Statistics dict with n_total, n_train, n_val, n_test, label_map.
    """
    from leech.util import setup_random_seed

    if len(input_paths) != len(labels):
        raise ValueError(
            f"input_paths ({len(input_paths)}) and labels ({len(labels)}) must have same length"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_random_seed(seed, output_dir)

    # Build label -> int mapping (stable alphabetical order)
    unique_labels = sorted(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    logger.info(f"Multi-class label map ({len(unique_labels)} classes): {label_to_int}")

    # Save label map for downstream use
    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_to_int, f, indent=2)

    # First pass: collect read IDs (and group values if split_by is set)
    logger.info("Pass 1: Collecting read IDs")
    all_read_ids: set[str] = set()
    # read_id -> group value mapping (only used when split_by is set)
    read_to_group: dict[str, str] = {}
    # read_id -> label mapping (for per-label group assignment)
    read_to_label: dict[str, str] = {}
    for chunk_path, label in zip(input_paths, labels, strict=True):
        with np.load(chunk_path, allow_pickle=True) as data:
            read_ids = data["read_ids"]
            all_read_ids.update(str(rid) for rid in read_ids)
            if split_by is not None:
                if split_by not in data:
                    raise ValueError(
                        f"--split-by field '{split_by}' not found in {chunk_path}. "
                        f"Available fields: {list(data.keys())}"
                    )
                group_vals = data[split_by]
                for rid, gv in zip(read_ids, group_vals, strict=True):
                    rid_str = str(rid)
                    read_to_group[rid_str] = str(gv)
                    read_to_label[rid_str] = label

    logger.info(f"Total unique reads: {len(all_read_ids)}")

    # Split reads
    if seed is not None:
        random.seed(seed)

    if split_by is not None:
        # Group-level split: assign entire groups to splits, stratified per label
        train_read_ids, val_read_ids, test_read_ids = _split_by_group(
            read_to_group=read_to_group,
            read_to_label=read_to_label,
            train_frac=train_frac,
            val_frac=val_frac,
        )
    else:
        read_ids_list = list(all_read_ids)
        random.shuffle(read_ids_list)

        n_reads = len(read_ids_list)
        n_train = int(n_reads * train_frac)
        n_val = int(n_reads * val_frac)

        train_read_ids = set(read_ids_list[:n_train])
        val_read_ids = set(read_ids_list[n_train : n_train + n_val])
        test_read_ids = set(read_ids_list[n_train + n_val :])

    # Second pass: array-level merge with label overrides
    logger.info("Pass 2: Merging arrays with label overrides")

    label_overrides = {}
    source_group_overrides = {}
    for chunk_path, label in zip(input_paths, labels, strict=True):
        label_overrides[chunk_path] = (label_to_int[label], label)
        source_group_overrides[chunk_path] = _source_group_from_path(chunk_path)

    counts = _merge_arrays_by_split(
        input_paths=input_paths,
        split_read_ids={
            "train": train_read_ids,
            "val": val_read_ids,
            "test": test_read_ids,
        },
        output_paths={
            "train": output_dir / "train.npz",
            "val": output_dir / "val.npz",
            "test": output_dir / "test.npz",
        },
        label_overrides=label_overrides,
        source_group_overrides=source_group_overrides,
    )

    n_total = sum(counts.values())
    logger.info(
        f"Multi-class split: train={counts['train']}, val={counts['val']}, test={counts['test']}"
    )

    return {
        "n_total": n_total,
        "n_train": counts["train"],
        "n_val": counts["val"],
        "n_test": counts["test"],
        "label_map": label_to_int,
        "output_files": {
            "train": output_dir / "train.npz",
            "val": output_dir / "val.npz",
            "test": output_dir / "test.npz",
        },
    }


def merge_and_kfold_split_multiclass(
    input_paths: list[Path],
    labels: list[str],
    output_dir: Path,
    k_fold: int,
    seed: int | None = None,
) -> dict[str, Any]:
    """Merge N chunk files and split into k folds for multi-class cross-validation.

    Args:
        input_paths: Paths to .npz chunk files (one per class).
        labels: Class label for each path.
        output_dir: Base output directory (fold_0/, fold_1/, ...).
        k_fold: Number of folds (>= 3).
        seed: Random seed.

    Returns:
        Statistics dict with k_fold, n_total, label_map, folds.
    """
    from leech.util import setup_random_seed

    if k_fold < 3:
        raise ValueError(f"k_fold must be >= 3, got {k_fold}")
    if len(input_paths) != len(labels):
        raise ValueError(
            f"input_paths ({len(input_paths)}) and labels ({len(labels)}) must have same length"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_random_seed(seed, output_dir)

    # Build label map
    unique_labels = sorted(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    logger.info(f"Multi-class label map ({len(unique_labels)} classes): {label_to_int}")

    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_to_int, f, indent=2)

    # Collect read IDs
    all_read_ids: set[str] = set()
    for chunk_path in input_paths:
        with np.load(chunk_path, allow_pickle=True) as data:
            all_read_ids.update(str(rid) for rid in data["read_ids"])

    # Partition reads into folds
    if seed is not None:
        random.seed(seed)
    read_ids_list = list(all_read_ids)
    random.shuffle(read_ids_list)

    partitions = np.array_split(read_ids_list, k_fold)
    partition_sets = [set(p.tolist()) for p in partitions]

    # Build fold assignments
    fold_read_assignments = []
    for i in range(k_fold):
        test_ids = partition_sets[i]
        val_ids = partition_sets[(i + 1) % k_fold]
        train_ids: set[str] = set()
        for j in range(k_fold):
            if j != i and j != (i + 1) % k_fold:
                train_ids.update(partition_sets[j])
        fold_read_assignments.append({"train": train_ids, "val": val_ids, "test": test_ids})

    # Build overrides
    label_overrides = {}
    source_group_overrides = {}
    for chunk_path, label in zip(input_paths, labels, strict=True):
        label_overrides[chunk_path] = (label_to_int[label], label)
        source_group_overrides[chunk_path] = _source_group_from_path(chunk_path)

    # Load all input files once into RAM to avoid re-reading per fold
    cached_inputs: list[tuple[Path, dict[str, np.ndarray], np.ndarray]] = []
    for chunk_path in input_paths:
        with np.load(chunk_path, allow_pickle=True) as data:
            cached = {key: data[key] for key in data.keys()}
        read_ids_str = np.array([str(r) for r in cached["read_ids"]])
        cached_inputs.append((chunk_path, cached, read_ids_str))

    # Merge arrays per fold using cached data
    folds_stats: list[dict[str, Any]] = []
    n_total = 0
    for i in range(k_fold):
        fold_dir = output_dir / f"fold_{i}"
        split_names = ["train", "val", "test"]
        accumulators: dict[str, dict[str, list[np.ndarray]]] = {s: {} for s in split_names}

        for chunk_path, cached, read_ids_str in cached_inputs:
            masks = {
                sname: np.array([r in rid_set for r in read_ids_str], dtype=bool)
                for sname, rid_set in fold_read_assignments[i].items()
            }
            array_keys = [k for k in cached if not k.startswith("_")]

            for sname in split_names:
                mask = masks[sname]
                if not mask.any():
                    continue
                count = int(mask.sum())
                for key in array_keys:
                    sliced = cached[key][mask]

                    # Apply label overrides
                    if label_overrides is not None and chunk_path in label_overrides:
                        lint, lstr = label_overrides[chunk_path]
                        if key == "labels_int":
                            sliced = np.full(sliced.shape, lint, dtype=sliced.dtype)
                        elif key == "labels":
                            sliced = np.array([lstr] * count, dtype=str)

                    # Apply source_group overrides
                    if source_group_overrides is not None and chunk_path in source_group_overrides:
                        if key == "source_groups":
                            sg = source_group_overrides[chunk_path]
                            sliced = np.array([sg] * count, dtype=str)

                    accumulators[sname].setdefault(key, []).append(sliced)

        # Concatenate and save
        counts: dict[str, int] = {}
        output_paths = {
            "train": fold_dir / "train.npz",
            "val": fold_dir / "val.npz",
            "test": fold_dir / "test.npz",
        }
        for sname in split_names:
            acc = accumulators[sname]
            if not acc:
                counts[sname] = 0
                continue

            save_kwargs: dict[str, np.ndarray] = {}
            for key, arr_list in acc.items():
                save_kwargs[key] = np.concatenate(arr_list)

            n_chunks = len(save_kwargs.get("read_ids", np.array([])))
            counts[sname] = n_chunks

            out_path = output_paths[sname]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_path, **save_kwargs)
            logger.info(f"Saved {n_chunks} {sname} chunks to {out_path}")

        fold_total = sum(counts.values())
        if i == 0:
            n_total = fold_total

        folds_stats.append(
            {
                "n_train": counts["train"],
                "n_val": counts["val"],
                "n_test": counts["test"],
                "output_files": output_paths,
            }
        )

    del cached_inputs

    return {
        "k_fold": k_fold,
        "n_total": n_total,
        "label_map": label_to_int,
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
