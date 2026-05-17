"""Bundled data files for leech (kmer level tables, etc.)."""

import hashlib
from importlib import resources
from pathlib import Path


def get_kmer_table(name: str = "rna004_9mer_levels_v1.txt.gz") -> Path:
    """Return the path to a bundled kmer level table.

    Args:
        name: Filename of the kmer table (default: rna004_9mer_levels_v1.txt.gz)

    Returns:
        Path to the kmer table file
    """
    ref = resources.files("leech.data").joinpath(name)
    # resources.files returns a Traversable; as_posix works for installed packages
    # For development (editable install), this is already a real path
    return Path(str(ref))


def compute_kmer_table_sha256(path: Path) -> str:
    """SHA256 of a kmer-level-table file, used as a provenance fingerprint.

    Persisted in model config at train/bundle time so inference can detect
    when a leech upgrade (or a custom --kmer-table override) silently
    swapped the table out from under a trained model.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()
