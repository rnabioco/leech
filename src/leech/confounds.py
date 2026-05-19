"""Confound-map helpers for adversarial training.

Adversarial training penalises the encoder for encoding information that
predicts a *confound* signal in addition to the target label. This module
provides domain-neutral plumbing for two confound styles:

- ``label_map``: a precomputed mapping ``{label_int -> confound_class_int}``
  loaded from a JSON sidecar produced by the pipeline. Use this when the
  confound is a fixed function of the class label (e.g. a per-class
  sequence-derived feature).
- ``ref_map``: a per-chunk mapping ``{reference_name -> confound_class_int}``
  computed at training time from the unique reference strings observed in
  the data. Use this when the confound is per-instance metadata (e.g. the
  full reference identity).

Neither style hardcodes any biology — the pipeline is responsible for
defining what the classes mean.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("leech.confounds")


def load_label_class_map(data_dir: Path) -> dict[str, int] | None:
    """Load a ``label_class_map.json`` sidecar from a data directory.

    Looks in ``data_dir`` and ``data_dir/..`` (for k-fold subdirectories).

    The expected JSON schema is either a flat ``{label_name: class_int}``
    mapping or a nested object with a ``"label_to_class"`` key holding the
    same mapping (the nested form leaves room for sibling provenance fields
    like ``"class_to_seq"`` and ``"num_classes"``).

    Returns:
        ``{label_name: class_int}`` mapping, or None if not found.
    """
    for parent in [data_dir, data_dir.parent]:
        path = parent / "label_class_map.json"
        if path.exists():
            with open(path) as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "label_to_class" in payload:
                return payload["label_to_class"]
            return payload
    return None


def build_label_confound_map(
    label_map: dict[str, int],
    class_map: dict[str, int] | None,
) -> dict[int, int]:
    """Compose ``{label_int -> confound_class_int}`` from two name-keyed maps.

    Args:
        label_map: ``{label_name: label_int}`` from ``label_map.json``.
        class_map: ``{label_name: confound_class_int}`` from
            ``label_class_map.json``. Required — no hardcoded defaults.

    Returns:
        ``{label_int: confound_class_int}`` for labels present in both maps.
        Labels missing from ``class_map`` are omitted; the adversarial loss
        ignores them via ``ignore_index=-1``.

    Raises:
        ValueError: If ``class_map`` is None.
    """
    if class_map is None:
        raise ValueError(
            "class_map is required. Generate label_class_map.json from the "
            "pipeline-side extraction script before training with --confound label_map."
        )

    confound_map: dict[int, int] = {}
    for name, label_int in label_map.items():
        cls = class_map.get(name)
        if cls is not None:
            confound_map[label_int] = int(cls)

    return confound_map


def build_string_id_map(
    reference_names: list[str],
) -> tuple[dict[str, int], int]:
    """Assign a contiguous integer class to each unique string.

    Used to build the per-chunk ``ref_map`` confound from a list of
    reference-name strings (one per chunk). Each unique string becomes a
    distinct class. Useful when you want the adversary to penalise
    encoding of *any* per-reference information.

    Args:
        reference_names: Array or list of per-chunk reference-name strings,
            typically ``npz["reference_names"]``.

    Returns:
        ``(name_to_int, num_classes)`` where ``name_to_int`` maps each
        unique string to a contiguous integer class, and ``num_classes``
        is the total number of unique strings.
    """
    import numpy as np

    unique = sorted(set(np.asarray(reference_names).flat))
    name_to_int = {name: i for i, name in enumerate(unique)}
    logger.info("ref_map confound: %d unique reference strings", len(name_to_int))
    for name, idx in name_to_int.items():
        logger.debug("  %s -> %d", name, idx)
    return name_to_int, len(name_to_int)
