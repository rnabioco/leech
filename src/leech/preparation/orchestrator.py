"""
Read-level splitting and writing for a prepared corpus.

Chunk extraction itself goes through the parallel dispatcher
(``preparation/parallel.py``) at every worker count, including 1 -- the
separate sequential pipeline that used to live here (``prepare_training_data``,
``prepare_training_data_with_split``) was retired in issue #275. What remains
is the read-level split rule and the one place a spooled corpus is written out
by split.
"""

import logging
import random
from pathlib import Path

import numpy as np

from leech.chunking import ChunkSpool
from leech.splitting.splitter import _SPLITS, _assign_splits, _split_codes

logger = logging.getLogger("leech.preparation.orchestrator")


def split_rows_by_read(
    read_ids: np.ndarray,
    train_frac: float,
    val_frac: float,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row indices per split, from the read-level assignment in ``leech.splitting``.

    Goes straight through :func:`leech.splitting.splitter._assign_splits`
    (partitions the unique read-id *set*) and
    :func:`leech.splitting.splitter._split_codes` (one vectorised pass
    assigning each row its split), rather than building one stand-in dict per
    row to hand :func:`~leech.splitting.splitter.split_chunks_by_read`: that
    intermediate was ~1.3 GB of transient Python dicts on the production
    corpus (6.7M chunks) purely to get back to a set of unique read ids
    (issue #275).

    Args:
        read_ids: One read id per chunk, in corpus order.
        train_frac: Fraction of reads for training.
        val_frac: Fraction of reads for validation.
        seed: Random seed. ``_assign_splits`` consumes the module-level
            ``random`` stream, which must already be seeded -- pass the
            *resolved* (non-``None``) seed when the caller has already called
            ``setup_random_seed``, or a seed here to have this function seed
            it itself.

    Returns:
        ``(train_rows, val_rows, test_rows)`` as int64 index arrays.
    """
    if seed is not None:
        random.seed(seed)

    unique_read_ids: set[str] = (
        set(read_ids.tolist())
        if read_ids.dtype.kind == "U"
        else {str(r) for r in read_ids.tolist()}
    )
    splits = _assign_splits(unique_read_ids, train_frac, val_frac)
    codes = _split_codes(read_ids, splits)

    # `_split_codes` numbers a row by the position of its split's set in
    # `splits`' iteration order. `_assign_splits` returns its keys in
    # `_SPLITS` order today, but both are private names in `splitter.py` with
    # no documented guarantee of that from its own side -- asserted here
    # rather than assumed, so a future reordering fails loudly at the source
    # of the coupling instead of silently mislabeling every split.
    assert tuple(splits) == _SPLITS, (
        f"leech.splitting.splitter._assign_splits key order changed: "
        f"expected {_SPLITS}, got {tuple(splits)}"
    )
    train_rows = np.nonzero(codes == 0)[0]
    val_rows = np.nonzero(codes == 1)[0]
    test_rows = np.nonzero(codes == 2)[0]
    return train_rows, val_rows, test_rows


def write_splits(
    spool: ChunkSpool,
    output_dir: Path,
    train_frac: float,
    val_frac: float,
    seed: int | None,
) -> dict[str, np.ndarray]:
    """Split a spooled corpus by read and write ``{split}.npz`` files.

    The one write step both prepare paths (``workers == 1`` and
    ``workers > 1`` go through the same dispatcher now, so there is only one
    caller) use -- previously duplicated between ``handle_prepare`` and the
    now-retired ``prepare_training_data_with_split`` (issue #275). An empty
    split is skipped rather than written as an empty file, matching prior
    behavior.

    Returns:
        ``{split_name: row_indices}`` for each split actually written.
    """
    train_rows, val_rows, test_rows = split_rows_by_read(
        spool.read_ids(), train_frac, val_frac, seed
    )

    written: dict[str, np.ndarray] = {}
    for split, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        if len(rows) == 0:
            continue
        split_file = output_dir / f"{split}.npz"
        spool.write_npz(split_file, rows=rows)
        written[split] = rows
        logger.info(f"Saved {len(rows)} {split} chunks to {split_file}")

    return written
