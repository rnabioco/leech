"""Confound mappings for adversarial training.

The discriminator base (position 73 in Sprinzl numbering) is the nucleotide
immediately 5' of the CCA tail.  It varies between tRNA species and creates a
sequence-level confound that can bias signal-based classifiers.

The disc_base_map is derived at pipeline time from the reference FASTA (see
``extract_disc_bases`` in the Snakemake rules).  The library never hardcodes
organism-specific base assignments.
"""

import json
import logging
from pathlib import Path

DISC_BASE_TO_INT: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}

NUM_DISC_BASES = 4

logger = logging.getLogger("leech.confounds")


def load_disc_base_map(data_dir: Path) -> dict[str, str] | None:
    """Load disc_base_map.json sidecar from a data directory.

    Looks in ``data_dir`` and ``data_dir/..`` (for k-fold subdirectories).

    Returns:
        ``{amino_acid: base}`` mapping, or None if not found.
    """
    for parent in [data_dir, data_dir.parent]:
        path = parent / "disc_base_map.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def build_confound_map(
    label_map: dict[str, int],
    disc_base_map: dict[str, str] | None = None,
) -> dict[int, int]:
    """Build mapping from label_int to confound class int.

    Args:
        label_map: ``{amino_acid_name: label_int}`` from ``label_map.json``.
        disc_base_map: ``{amino_acid: base}`` mapping.  Required — no
            hardcoded defaults.

    Returns:
        ``{label_int: confound_class_int}`` for amino acids with known
        discriminator bases.  Classes not found in *disc_base_map*
        (e.g. ``"uncharged"``) are omitted — the adversarial loss will
        ignore them via ``ignore_index=-1``.

    Raises:
        ValueError: If *disc_base_map* is None.
    """
    if disc_base_map is None:
        raise ValueError(
            "disc_base_map is required. Generate disc_base_map.json from the "
            "reference FASTA using the extract_disc_bases Snakemake rule."
        )

    confound_map: dict[int, int] = {}
    for aa_name, label_int in label_map.items():
        base = disc_base_map.get(aa_name)
        if base is not None:
            confound_map[label_int] = DISC_BASE_TO_INT[base]

    return confound_map


def build_trna_identity_map(
    reference_names: list[str],
) -> tuple[dict[str, int], int]:
    """Build a per-isoacceptor confound map from chunk reference names.

    Each unique tRNA reference name (e.g. ``tRNA-Ser-CGA-1-1``) becomes
    a distinct confound class. This is a much stronger adversarial target
    than the 4-class discriminator base, because the model is penalised
    for encoding *any* tRNA-body information — not just the single
    nucleotide at Sprinzl position 73.

    Args:
        reference_names: Array or list of per-chunk reference name strings.
            Typically read from ``npz["reference_names"]``.

    Returns:
        (ref_to_int, num_classes) where ``ref_to_int`` maps each unique
        reference name to a contiguous integer class, and ``num_classes``
        is the total number of unique tRNAs.
    """
    import numpy as np

    unique = sorted(set(np.asarray(reference_names).flat))
    ref_to_int = {name: i for i, name in enumerate(unique)}
    logger.info(f"tRNA identity confound: {len(ref_to_int)} unique isoacceptors")
    for name, idx in ref_to_int.items():
        logger.debug(f"  {name} -> {idx}")
    return ref_to_int, len(ref_to_int)
