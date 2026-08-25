"""
Read-level data splitting to prevent data leakage.

Provides functions for splitting training data at the read level (not chunk level)
to ensure no molecule appears in both training and validation sets.
"""

import json
import logging
import random
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from leech.chunking import (
    csr_from_object_rows,
    csr_gather_index,
    csr_offsets_from_lens,
    iter_npz_row_blocks,
    load_chunks,
    npz_array_members,
)

logger = logging.getLogger("leech.splitting.splitter")

# npz members holding the CSR base-to-signal maps; merged by row gather, not
# by boolean mask, so they are excluded from the generic member loop.
_S2S_MEMBERS = frozenset({"seq_to_sig_values", "seq_to_sig_offsets", "seq_to_sig_maps"})

# The three CSR spellings are one logical field, so a merge of mixed-vintage
# inputs must not read as a member-set mismatch.
_S2S_FIELD = "<seq_to_sig_map>"

# The split names every merge writes, in the order they are written.
_SPLITS = ("train", "val", "test")


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


def _split_codes(read_ids: np.ndarray, split_read_ids: dict[str, set[str]]) -> np.ndarray:
    """Per-row split index for a file's ``read_ids`` column (-1 = no split).

    One dict lookup per row, in one pass, instead of one Python membership
    comprehension per split over a freshly stringified column: the old
    ``np.array([str(r) for r in read_ids])`` plus a ``[r in rid_set ...]``
    comprehension per split cost 802 ms per 500k rows x 3 splits, and minted a
    str object per row per split.

    Rows are converted a block at a time so the transient list of Python
    strings stays bounded rather than being one object per chunk in the corpus.

    The splits must be disjoint — one code per row is the whole point — which
    every producer here guarantees: fractional, k-fold and group-level
    assignment all partition the read ids.
    """
    lookup: dict[str, int] = {}
    for code, rid_set in enumerate(split_read_ids.values()):
        for rid in rid_set:
            lookup[rid] = code

    # Pass 1 keys the split sets by str(rid); U-dtype elements come back from
    # tolist() as exactly that string, anything else has to go through str().
    is_text = read_ids.dtype.kind == "U"
    codes = np.empty(len(read_ids), dtype=np.int8)
    block = 1 << 16
    for start in range(0, len(read_ids), block):
        values = read_ids[start : start + block].tolist()
        if not is_text:
            values = [str(v) for v in values]
        codes[start : start + len(values)] = [lookup.get(v, -1) for v in values]
    return codes


@dataclass
class _InputPlan:
    """What one input file contributes to each split, decided without payload."""

    path: Path
    members: list[str]  # every member of the file, in file order
    streamable: dict[str, tuple[tuple[int, ...], np.dtype]]  # name -> (shape, dtype)
    rows: dict[str, np.ndarray]  # split -> ascending row indices
    s2s_offsets: np.ndarray | None  # CSR row offsets, None if the file has no maps
    label_int: int | None = None  # per-file labels_int override
    text_overrides: dict[str, str] = field(default_factory=dict)  # member -> constant


def _logical_members(members: Collection[str]) -> set[str]:
    """Member names with the three CSR spellings folded into one field name."""
    logical = {name for name in members if name not in _S2S_MEMBERS}
    if not _S2S_MEMBERS.isdisjoint(members):
        logical.add(_S2S_FIELD)
    return logical


def _validate_member_sets(plans: list["_InputPlan"]) -> None:
    """Refuse to merge inputs whose member sets differ.

    A file written before ``focus_signal_pos`` (or before the residual channel,
    or storing a member ragged that the others store flat) contributes no rows
    for that member but full rows for every other one, so the merged file used
    to carry a short column beside full ones: every chunk past the short
    column's end raises on read, and — with the inputs in the other order —
    every affected chunk silently gets another read's value, i.e. the wrong
    asymmetric signal crop, with no warning.
    """
    reference = _logical_members(plans[0].members)
    for plan in plans[1:]:
        logical = _logical_members(plan.members)
        if logical == reference:
            continue
        missing = sorted(reference - logical)
        extra = sorted(logical - reference)
        detail = []
        if missing:
            detail.append(f"{plan.path} is missing {missing}")
        if extra:
            detail.append(f"{plan.path} has extra {extra}")
        raise ValueError(
            "Cannot merge chunk files with different member sets: "
            + "; ".join(detail)
            + f" (compared against {plans[0].path}). Re-prepare the odd file(s) "
            "with the same leech version, or merge only files that agree."
        )


def _load_s2s_offsets(path: Path, members: Collection[str]) -> np.ndarray | None:
    """A file's CSR row offsets for the base-to-signal maps, or None if absent."""
    if "seq_to_sig_values" in members:
        with np.load(path, allow_pickle=False) as data:
            return data["seq_to_sig_offsets"]
    if "seq_to_sig_maps" in members:
        with np.load(path, allow_pickle=True) as data:
            rows = data["seq_to_sig_maps"]
            lens = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
        return csr_offsets_from_lens(lens)
    return None


def _load_s2s_csr(path: Path, members: Collection[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """A file's base-to-signal maps as ``(values, offsets)``, or None if absent.

    Legacy object arrays are normalized to CSR so a merge of mixed-vintage
    inputs writes one format.
    """
    if "seq_to_sig_values" in members:
        with np.load(path, allow_pickle=False) as data:
            return data["seq_to_sig_values"], data["seq_to_sig_offsets"]
    if "seq_to_sig_maps" in members:
        with np.load(path, allow_pickle=True) as data:
            return csr_from_object_rows(data["seq_to_sig_maps"])
    return None


def _plan_inputs(
    input_paths: list[Path],
    split_read_ids: dict[str, set[str]],
    label_overrides: dict[Path, tuple[int, str]] | None,
    source_group_overrides: dict[Path, str] | None,
) -> list[_InputPlan]:
    """Pass 1: which rows of each input land in which split, and in what shape.

    Reads only ``read_ids``, the CSR row offsets and the members' .npy headers
    — never a payload member — so no input file is ever fully resident here.
    """
    plans: list[_InputPlan] = []
    for chunk_path in input_paths:
        with np.load(chunk_path, allow_pickle=True) as data:
            members = list(data.keys())
            codes = _split_codes(data["read_ids"], split_read_ids)

        text_overrides: dict[str, str] = {}
        label_int: int | None = None
        if label_overrides is not None and chunk_path in label_overrides:
            label_int, label_str = label_overrides[chunk_path]
            text_overrides["labels"] = label_str
        if source_group_overrides is not None and chunk_path in source_group_overrides:
            text_overrides["source_groups"] = source_group_overrides[chunk_path]

        plans.append(
            _InputPlan(
                path=chunk_path,
                members=members,
                streamable=npz_array_members(chunk_path),
                rows={
                    sname: np.nonzero(codes == code)[0] for code, sname in enumerate(split_read_ids)
                },
                s2s_offsets=_load_s2s_offsets(chunk_path, members),
                label_int=label_int,
                text_overrides={k: v for k, v in text_overrides.items() if k in members},
            )
        )
    _validate_member_sets(plans)
    return plans


def _output_specs(plans: list[_InputPlan]) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Row shape and promoted dtype of every output member that can be streamed.

    Only members every input stores in a row-streamable form qualify; the rest
    fall back to load-mask-concatenate. A member whose rows a file overrides
    with a constant contributes the constant's dtype, not the stored column's,
    so the merged column stays exactly as wide as the values actually written.
    """
    common = set(plans[0].streamable)
    for plan in plans[1:]:
        common &= set(plan.streamable)

    specs: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    for name in plans[0].members:
        if name in _S2S_MEMBERS or name not in common:
            continue
        row_shape = plans[0].streamable[name][0][1:]
        dtype: np.dtype | None = None
        for plan in plans:
            shape, member_dtype = plan.streamable[name]
            if shape[1:] != row_shape:
                raise ValueError(
                    f"Cannot merge chunk files: member '{name}' has row shape "
                    f"{shape[1:]} in {plan.path} but {row_shape} in {plans[0].path}"
                )
            if name in plan.text_overrides:
                member_dtype = np.array([plan.text_overrides[name]], dtype=str).dtype
            dtype = member_dtype if dtype is None else np.result_type(dtype, member_dtype)
        specs[name] = (row_shape, np.dtype(dtype))
    return specs


def _merge_arrays_by_split(
    input_paths: list[Path],
    split_read_ids: dict[str, set[str]],
    output_paths: dict[str, Path],
    label_overrides: dict[Path, tuple[int, str]] | None = None,
    source_group_overrides: dict[Path, str] | None = None,
) -> dict[str, int]:
    """Merge input .npz files into per-split output files at the array level.

    Two passes, neither holding an input file's payload: pass 1 decides which
    rows of each input go to which split and how wide every output member has
    to be, pass 2 preallocates one array per output member and fills it from
    :func:`~leech.chunking.iter_npz_row_blocks`. The old accumulate-then-
    concatenate shape peaked at the whole merged corpus plus the largest split,
    which is the stage that produced the 41 GB peak in #211.

    Args:
        input_paths: Paths to input .npz chunk files.
        split_read_ids: Mapping from split name (e.g. "train") to set of read IDs.
        output_paths: Mapping from split name to output .npz path.
        label_overrides: Optional per-file (label_int, label_str) override for multiclass.
        source_group_overrides: Optional per-file source_group string override.

    Returns:
        Mapping from split name to number of chunks saved.

    Raises:
        ValueError: If the inputs do not all carry the same members, or a
            member's row shape differs between them.
    """
    split_names = list(split_read_ids.keys())
    if not input_paths:
        return dict.fromkeys(split_names, 0)

    plans = _plan_inputs(input_paths, split_read_ids, label_overrides, source_group_overrides)
    # An input none of whose reads made it into a split contributes nothing —
    # not even its dtypes, which would otherwise widen a column past what the
    # merged file actually holds.
    plans = [plan for plan in plans if any(plan.rows[sname].size for sname in split_names)]
    if not plans:
        return dict.fromkeys(split_names, 0)
    specs = _output_specs(plans)

    # Members no .npy header could be read for: object dtype (the legacy
    # pickled ragged signals/dwells/features) and anything else
    # iter_npz_row_blocks refuses. Those still go through
    # load-mask-concatenate; pickled rows are pointers, so the copies are cheap.
    boxed_members = [
        name for name in plans[0].members if name not in _S2S_MEMBERS and name not in specs
    ]

    counts = {sname: int(sum(plan.rows[sname].size for plan in plans)) for sname in split_names}
    any_maps = plans[0].s2s_offsets is not None

    # Preallocate one array per output member per non-empty split.
    outputs: dict[str, dict[str, np.ndarray]] = {}
    boxed: dict[str, dict[str, list[np.ndarray]]] = {}
    s2s_values_out: dict[str, np.ndarray] = {}
    s2s_lens: dict[str, list[np.ndarray]] = {sname: [] for sname in split_names}
    s2s_dtype = np.dtype(np.int32)
    for plan in plans:
        stored = plan.streamable.get("seq_to_sig_values")
        if stored is not None:
            s2s_dtype = np.result_type(s2s_dtype, stored[1])
    for sname in split_names:
        if counts[sname] == 0:
            continue
        outputs[sname] = {
            name: np.empty((counts[sname], *row_shape), dtype=dtype)
            for name, (row_shape, dtype) in specs.items()
        }
        boxed[sname] = {name: [] for name in boxed_members}
        if any_maps:
            total = 0
            for plan in plans:
                rows = plan.rows[sname]
                offsets = plan.s2s_offsets
                assert offsets is not None
                total += int(offsets[rows + 1].sum() - offsets[rows].sum())
            s2s_values_out[sname] = np.empty(total, dtype=s2s_dtype)

    # Pass 2: fill. `written` tracks the next free row of each output.
    written = dict.fromkeys(split_names, 0)
    s2s_written = dict.fromkeys(split_names, 0)
    for plan in plans:
        base = dict(written)
        skip = set(plan.text_overrides)
        if plan.label_int is not None:
            skip.add("labels_int")
        streamed = [name for name in specs if name not in skip]

        if streamed:
            for row_start, blocks in iter_npz_row_blocks(plan.path, streamed):
                n_block = len(blocks[streamed[0]])
                for sname in split_names:
                    rows = plan.rows[sname]
                    lo = int(np.searchsorted(rows, row_start))
                    hi = int(np.searchsorted(rows, row_start + n_block))
                    if lo == hi:
                        continue
                    local = rows[lo:hi] - row_start
                    dst = base[sname] + lo
                    for name in streamed:
                        outputs[sname][name][dst : dst + (hi - lo)] = blocks[name][local]

        if boxed_members:
            with np.load(plan.path, allow_pickle=True) as data:
                for name in boxed_members:
                    column = data[name]
                    for sname in split_names:
                        rows = plan.rows[sname]
                        if rows.size:
                            boxed[sname][name].append(_override_rows(plan, name, column[rows]))

        for sname in split_names:
            rows = plan.rows[sname]
            if not rows.size:
                continue
            start = base[sname]
            for name, value in plan.text_overrides.items():
                if name in outputs[sname]:
                    outputs[sname][name][start : start + rows.size] = value
            if plan.label_int is not None and "labels_int" in outputs[sname]:
                outputs[sname]["labels_int"][start : start + rows.size] = plan.label_int
            written[sname] = start + rows.size

        # Base-to-signal maps are CSR (values + offsets), so they are selected
        # by gathering rows rather than by masking, and their offsets are
        # rebuilt from the accumulated row lengths at save time.
        s2s = _load_s2s_csr(plan.path, plan.members)
        for sname in split_names:
            rows = plan.rows[sname]
            if not rows.size:
                continue
            if s2s is None:
                # No maps in this file; keep the row count aligned so a merge
                # with files that do have them stays row-indexable.
                s2s_lens[sname].append(np.zeros(rows.size, dtype=np.int64))
                continue
            values, offsets = s2s
            lens, _col, src = csr_gather_index(offsets, rows)
            s2s_lens[sname].append(lens)
            gathered = int(lens.sum())
            at = s2s_written[sname]
            s2s_values_out[sname][at : at + gathered] = values[src]
            s2s_written[sname] = at + gathered
        del s2s

    # Assemble and save one split at a time, dropping each before the next.
    saved_counts: dict[str, int] = {}
    for sname in split_names:
        n_chunks = counts[sname]
        if n_chunks == 0:
            saved_counts[sname] = 0
            continue

        # Member order follows the first input that contributes rows, which is
        # the order the accumulators used to be keyed in.
        order = next(plan for plan in plans if plan.rows[sname].size)
        save_kwargs: dict[str, np.ndarray] = {}
        if any_maps:
            save_kwargs["seq_to_sig_values"] = s2s_values_out[sname]
        for name in order.members:
            if name in _S2S_MEMBERS:
                continue
            if name in outputs[sname]:
                save_kwargs[name] = outputs[sname][name]
            else:
                save_kwargs[name] = np.concatenate(boxed[sname][name])
        if s2s_lens[sname]:
            lens = np.concatenate(s2s_lens[sname])
            save_kwargs["seq_to_sig_offsets"] = csr_offsets_from_lens(lens)
            save_kwargs.setdefault("seq_to_sig_values", np.empty(0, dtype=np.int32))

        _assert_row_counts(save_kwargs, n_chunks, output_paths[sname])
        saved_counts[sname] = n_chunks

        out_path = output_paths[sname]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **save_kwargs)
        logger.info(f"Saved {n_chunks} {sname} chunks to {out_path}")

        save_kwargs.clear()
        outputs.pop(sname, None)
        boxed.pop(sname, None)
        s2s_values_out.pop(sname, None)

    return saved_counts


def _override_rows(plan: _InputPlan, name: str, sliced: np.ndarray) -> np.ndarray:
    """The rows a boxed member contributes, after this file's overrides."""
    if name in plan.text_overrides:
        # Let numpy infer dtype to avoid truncation
        return np.array([plan.text_overrides[name]] * len(sliced), dtype=str)
    if name == "labels_int" and plan.label_int is not None:
        return np.full(sliced.shape, plan.label_int, dtype=sliced.dtype)
    return sliced


def _assert_row_counts(save_kwargs: dict[str, np.ndarray], n_chunks: int, out_path: Path) -> None:
    """Every output member must have one row per chunk before it is written.

    The check that would have caught a member-set mismatch: a merged file whose
    columns disagree on how many chunks it holds is not a file anything
    downstream can read correctly, and without this the failure mode is a wrong
    value rather than an error.
    """
    for name, arr in save_kwargs.items():
        if name == "seq_to_sig_values":
            continue  # flat CSR payload, addressed through the offsets
        expected = n_chunks + 1 if name == "seq_to_sig_offsets" else n_chunks
        if len(arr) != expected:
            raise ValueError(
                f"Refusing to write {out_path}: member '{name}' has {len(arr)} rows, "
                f"expected {expected} for {n_chunks} chunks"
            )


def _collect_read_index(
    input_paths: list[Path],
    *,
    split_by: str | None = None,
    labels: list[str] | None = None,
) -> tuple[set[str], dict[str, str], dict[str, str]]:
    """Pass 1 of every merge: the unique read ids across a set of chunk files.

    Args:
        input_paths: Paths to .npz chunk files.
        split_by: Optional npz member whose value groups reads for a
            group-level split (see :func:`_split_by_group`).
        labels: Per-file class label, required when ``split_by`` is set so
            groups can be allocated per label.

    Returns:
        ``(all_read_ids, read_to_group, read_to_label)``. The last two are
        empty unless ``split_by`` is set.

    Raises:
        ValueError: If ``split_by`` names a member an input does not have.
    """
    all_read_ids: set[str] = set()
    read_to_group: dict[str, str] = {}
    read_to_label: dict[str, str] = {}

    for index, chunk_path in enumerate(input_paths):
        logger.info(f"  Scanning {chunk_path}")
        # Only read_ids is read: much smaller than the payload members.
        with np.load(chunk_path, allow_pickle=True) as data:
            read_ids = data["read_ids"]
            all_read_ids.update(str(rid) for rid in read_ids)
            logger.info(f"    Found {len(read_ids)} chunks")
            if split_by is None:
                continue
            if split_by not in data:
                raise ValueError(
                    f"--split-by field '{split_by}' not found in {chunk_path}. "
                    f"Available fields: {list(data.keys())}"
                )
            label = labels[index] if labels is not None else ""
            for rid, group_value in zip(read_ids, data[split_by], strict=True):
                rid_str = str(rid)
                read_to_group[rid_str] = str(group_value)
                read_to_label[rid_str] = label

    logger.info(f"Total unique reads across all files: {len(all_read_ids)}")
    return all_read_ids, read_to_group, read_to_label


def _assign_splits(
    all_read_ids: set[str], train_frac: float, val_frac: float
) -> dict[str, set[str]]:
    """Partition read ids into train/val/test by fraction.

    Consumes the module-level ``random`` stream, which the caller has already
    seeded — the seed is the only input to the assignment, so the set is sorted
    before it is shuffled (set iteration order is PYTHONHASHSEED-dependent).
    """
    read_ids_list = sorted(all_read_ids)
    random.shuffle(read_ids_list)

    n_reads = len(read_ids_list)
    n_train = int(n_reads * train_frac)
    n_val = int(n_reads * val_frac)

    splits = {
        "train": set(read_ids_list[:n_train]),
        "val": set(read_ids_list[n_train : n_train + n_val]),
        "test": set(read_ids_list[n_train + n_val :]),
    }
    if n_reads:
        logger.info(f"Split {n_reads} unique reads:")
        for sname in _SPLITS:
            n_split = len(splits[sname])
            logger.info(f"  {sname.title()}: {n_split} reads ({n_split / n_reads * 100:.1f}%)")
    return splits


def _assign_kfold_splits(all_read_ids: set[str], k_fold: int) -> list[dict[str, set[str]]]:
    """Read-id assignments for each of ``k_fold`` folds.

    For fold ``i``: test is partition ``i``, val is partition ``i + 1``, train
    is everything else. Same seeding contract as :func:`_assign_splits`.
    """
    read_ids_list = sorted(all_read_ids)
    random.shuffle(read_ids_list)

    partitions = [set(part.tolist()) for part in np.array_split(read_ids_list, k_fold)]
    for i, part in enumerate(partitions):
        logger.info(f"  Partition {i}: {len(part)} reads")

    assignments: list[dict[str, set[str]]] = []
    for i in range(k_fold):
        val_index = (i + 1) % k_fold
        train_ids: set[str] = set()
        for j in range(k_fold):
            if j != i and j != val_index:
                train_ids.update(partitions[j])
        assignments.append(
            {"train": train_ids, "val": partitions[val_index], "test": partitions[i]}
        )
        logger.info(
            f"  Fold {i}: train={len(train_ids)} reads, "
            f"val={len(partitions[val_index])} reads, test={len(partitions[i])} reads"
        )
    return assignments


def _build_label_overrides(
    input_paths: list[Path],
    relabel_pairwise: tuple[str | list[str], str | list[str]] | None,
) -> dict[Path, tuple[int, str]] | None:
    """Per-file ``(label_int, label_str)`` overrides for a pairwise comparison.

    A file's group is decided by its first stored label, so every chunk in a
    file must share one label — which is how ``data prepare`` writes them.
    """
    if relabel_pairwise is None:
        return None

    group1, group2 = relabel_pairwise
    group1_labels = [group1] if isinstance(group1, str) else group1
    group2_labels = [group2] if isinstance(group2, str) else group2

    overrides: dict[Path, tuple[int, str]] = {}
    for chunk_path in input_paths:
        with np.load(chunk_path, allow_pickle=True) as data:
            file_labels = data["labels"]
            if len(file_labels) == 0:
                continue
            first_label = str(file_labels[0])
            if first_label in group1_labels:
                overrides[chunk_path] = (0, first_label)
            elif first_label in group2_labels:
                overrides[chunk_path] = (1, first_label)
    return overrides


def _build_label_map(labels: list[str], output_dir: Path) -> dict[str, int]:
    """Class-to-int map in stable alphabetical order, written to ``label_map.json``."""
    unique_labels = sorted(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    logger.info(f"Multi-class label map ({len(unique_labels)} classes): {label_to_int}")

    with open(output_dir / "label_map.json", "w") as f:
        json.dump(label_to_int, f, indent=2)
    return label_to_int


def _multiclass_label_overrides(
    input_paths: list[Path], labels: list[str], label_to_int: dict[str, int]
) -> dict[Path, tuple[int, str]]:
    """Per-file ``(label_int, label_str)`` overrides for a multi-class merge.

    Unlike the pairwise case the label is given per input path, so nothing has
    to be read from the file to decide it.
    """
    return {
        chunk_path: (label_to_int[label], label)
        for chunk_path, label in zip(input_paths, labels, strict=True)
    }


def _split_output_paths(output_dir: Path) -> dict[str, Path]:
    """``{"train": .../train.npz, ...}`` for a directory of splits."""
    return {sname: output_dir / f"{sname}.npz" for sname in _SPLITS}


def _split_stats(counts: dict[str, int], output_paths: dict[str, Path]) -> dict[str, Any]:
    """The statistics dict every single-split merge returns."""
    return {
        "n_total": sum(counts.values()),
        "n_train": counts["train"],
        "n_val": counts["val"],
        "n_test": counts["test"],
        "output_files": dict(output_paths),
    }


def _merge_folds(
    input_paths: list[Path],
    fold_read_assignments: list[dict[str, set[str]]],
    output_dir: Path,
    label_overrides: dict[Path, tuple[int, str]] | None,
    source_group_overrides: dict[Path, str] | None,
) -> tuple[int, list[dict[str, Any]]]:
    """Merge one set of inputs into ``fold_i/{train,val,test}.npz`` per fold.

    Returns:
        ``(n_total, folds_stats)`` where ``n_total`` is fold 0's chunk count —
        every fold holds the same chunks, only partitioned differently.
    """
    folds_stats: list[dict[str, Any]] = []
    n_total = 0

    for i, assignment in enumerate(fold_read_assignments):
        fold_dir = output_dir / f"fold_{i}"
        output_paths = _split_output_paths(fold_dir)

        counts = _merge_arrays_by_split(
            input_paths=input_paths,
            split_read_ids=assignment,
            output_paths=output_paths,
            label_overrides=label_overrides,
            source_group_overrides=source_group_overrides,
        )

        if i == 0:
            n_total = sum(counts.values())

        logger.info(
            f"Fold {i}: train={counts['train']}, val={counts['val']}, "
            f"test={counts['test']} chunks -> {fold_dir}"
        )
        folds_stats.append(
            {
                "n_train": counts["train"],
                "n_val": counts["val"],
                "n_test": counts["test"],
                "output_files": output_paths,
            }
        )

    logger.info(f"Saved {len(fold_read_assignments)} folds to {output_dir}")
    return n_total, folds_stats


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

    # Sort before shuffling. `read_to_chunks` is keyed in chunk arrival order,
    # which is a property of how the corpus was prepared, not of the data: the
    # Python prepare backend returns batches through `imap_unordered`, so the
    # same POD5/BAM at the same seed lands reads in different splits run to
    # run. Sorting makes the seed the only input to the assignment. Same
    # reason `_split_by_group` sorts its groups first.
    read_ids = sorted(read_to_chunks)
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
    from leech.model_loading import setup_random_seed

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
    all_read_ids, _groups, _labels = _collect_read_index(input_paths)

    # Determine read-level splits
    if seed is not None:
        random.seed(seed)
    split_read_ids = _assign_splits(all_read_ids, train_frac, val_frac)

    # If output_dir provided, use fast array-level merge
    if output_dir is not None:
        output_paths = _split_output_paths(output_dir)

        counts = _merge_arrays_by_split(
            input_paths=input_paths,
            split_read_ids=split_read_ids,
            output_paths=output_paths,
            label_overrides=_build_label_overrides(input_paths, relabel_pairwise),
            source_group_overrides={p: _source_group_from_path(p) for p in input_paths},
        )

        logger.info("Final chunk distribution:")
        for sname in _SPLITS:
            logger.info(f"  {sname.title()}: {counts[sname]} chunks")

        return _split_stats(counts, output_paths)
    else:
        # Legacy behavior: return tuple of chunks (uses dict path)
        logger.info("Pass 2: Loading chunks and assigning to splits")
        split_chunks: dict[str, list[dict]] = {sname: [] for sname in _SPLITS}

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
                for sname in _SPLITS:
                    if chunk["read_id"] in split_read_ids[sname]:
                        split_chunks[sname].append(chunk)
                        break
            del chunks

        return split_chunks["train"], split_chunks["val"], split_chunks["test"]


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
    from leech.model_loading import setup_random_seed

    if k_fold < 3:
        raise ValueError(f"k_fold must be >= 3, got {k_fold}")

    # Setup seed and output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_random_seed(seed, output_dir)

    logger.info(f"Merging {len(input_paths)} chunk files for {k_fold}-fold cross-validation")

    # ---- Pass 1: collect read IDs ----
    logger.info("Pass 1: Collecting read IDs for fold assignment")
    all_read_ids, _groups, _labels = _collect_read_index(input_paths)

    # Shuffle read IDs deterministically
    if seed is not None:
        random.seed(seed)
    fold_read_assignments = _assign_kfold_splits(all_read_ids, k_fold)

    # ---- Pass 2: merge arrays per fold ----
    logger.info("Pass 2: Merging arrays per fold")
    n_total, folds_stats = _merge_folds(
        input_paths=input_paths,
        fold_read_assignments=fold_read_assignments,
        output_dir=output_dir,
        label_overrides=_build_label_overrides(input_paths, relabel_pairwise),
        source_group_overrides={p: _source_group_from_path(p) for p in input_paths},
    )

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
    from leech.model_loading import setup_random_seed

    if len(input_paths) != len(labels):
        raise ValueError(
            f"input_paths ({len(input_paths)}) and labels ({len(labels)}) must have same length"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_random_seed(seed, output_dir)

    label_to_int = _build_label_map(labels, output_dir)

    # First pass: collect read IDs (and group values if split_by is set)
    logger.info("Pass 1: Collecting read IDs")
    all_read_ids, read_to_group, read_to_label = _collect_read_index(
        input_paths, split_by=split_by, labels=labels
    )

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
        split_read_ids = {
            "train": train_read_ids,
            "val": val_read_ids,
            "test": test_read_ids,
        }
    else:
        split_read_ids = _assign_splits(all_read_ids, train_frac, val_frac)

    # Second pass: array-level merge with label overrides
    logger.info("Pass 2: Merging arrays with label overrides")
    output_paths = _split_output_paths(output_dir)

    counts = _merge_arrays_by_split(
        input_paths=input_paths,
        split_read_ids=split_read_ids,
        output_paths=output_paths,
        label_overrides=_multiclass_label_overrides(input_paths, labels, label_to_int),
        source_group_overrides={p: _source_group_from_path(p) for p in input_paths},
    )

    logger.info(
        f"Multi-class split: train={counts['train']}, val={counts['val']}, test={counts['test']}"
    )

    return _split_stats(counts, output_paths) | {"label_map": label_to_int}


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
    from leech.model_loading import setup_random_seed

    if k_fold < 3:
        raise ValueError(f"k_fold must be >= 3, got {k_fold}")
    if len(input_paths) != len(labels):
        raise ValueError(
            f"input_paths ({len(input_paths)}) and labels ({len(labels)}) must have same length"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_random_seed(seed, output_dir)

    label_to_int = _build_label_map(labels, output_dir)

    # ---- Pass 1: collect read IDs ----
    logger.info("Pass 1: Collecting read IDs for fold assignment")
    all_read_ids, _groups, _labels = _collect_read_index(input_paths)

    # Partition reads into folds
    if seed is not None:
        random.seed(seed)
    fold_read_assignments = _assign_kfold_splits(all_read_ids, k_fold)

    # ---- Pass 2: merge arrays per fold ----
    #
    # This used to be an inline copy of _merge_arrays_by_split that masked
    # *every* member with the per-chunk boolean mask. It never learned about
    # the CSR seq_to_sig_values/seq_to_sig_offsets pair, so it raised
    # IndexError on any corpus the current save_chunks writes; it also cached
    # every input file decompressed for the whole run and then copied the
    # corpus twice per fold. There is one merge, and this is it.
    logger.info("Pass 2: Merging arrays per fold")
    n_total, folds_stats = _merge_folds(
        input_paths=input_paths,
        fold_read_assignments=fold_read_assignments,
        output_dir=output_dir,
        label_overrides=_multiclass_label_overrides(input_paths, labels, label_to_int),
        source_group_overrides={p: _source_group_from_path(p) for p in input_paths},
    )

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
