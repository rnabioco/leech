"""Confound mappings for adversarial training.

The discriminator base (position 73 in Sprinzl numbering) is the nucleotide
immediately 5' of the CCA tail.  It varies between tRNA species and creates a
sequence-level confound that can bias signal-based classifiers.

The disc_base_map is derived at pipeline time from the reference FASTA (see
``extract_disc_bases`` in the Snakemake rules and ``extract_disc_bases_from_fasta``
below).  The library never hardcodes organism-specific base assignments.
"""

import json
import logging
from collections import Counter
from pathlib import Path

DISC_BASE_TO_INT: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}

NUM_DISC_BASES = 4

logger = logging.getLogger("leech.confounds")


def extract_disc_bases_from_fasta(
    fasta_path: str | Path,
    motif: str = "CCAGGC",
) -> dict[str, str]:
    """Extract the discriminator base for each tRNA from a reference FASTA.

    The discriminator base is the nucleotide immediately 5' of the motif
    (typically CCAGGC, i.e. the base before the CCA tail + adaptor).

    For amino acids with multiple isoacceptors, the majority base is used.

    Args:
        fasta_path: Path to adapted reference FASTA with tRNA sequences.
        motif: Motif string to locate (default CCAGGC).

    Returns:
        ``{amino_acid: base}`` mapping (e.g. ``{"Ala": "A", "His": "C", ...}``).
    """
    import pysam

    fa = pysam.FastaFile(str(fasta_path))
    aa_bases: dict[str, list[str]] = {}

    for name in fa.references:
        seq = fa.fetch(name)
        idx = seq.find(motif)
        if idx < 1:
            continue
        # Parse AA from tRNA name: tRNA-{AA}-{anticodon}-...
        parts = name.split("-")
        if len(parts) < 2:
            continue
        aa = parts[1]
        disc = seq[idx - 1]
        aa_bases.setdefault(aa, []).append(disc)

    fa.close()

    # Majority vote per AA
    disc_base_map: dict[str, str] = {}
    for aa, bases in sorted(aa_bases.items()):
        counts = Counter(bases)
        majority_base, majority_count = counts.most_common(1)[0]
        disc_base_map[aa] = majority_base
        if len(counts) > 1:
            logger.info(
                f"{aa}: mixed disc bases {dict(counts)}, "
                f"using majority={majority_base} ({majority_count}/{len(bases)})"
            )

    return disc_base_map


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
            "reference FASTA using extract_disc_bases_from_fasta() or the "
            "extract_disc_bases Snakemake rule."
        )

    confound_map: dict[int, int] = {}
    for aa_name, label_int in label_map.items():
        base = disc_base_map.get(aa_name)
        if base is not None:
            confound_map[label_int] = DISC_BASE_TO_INT[base]

    return confound_map
