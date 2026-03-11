"""Bundled data files for leech (kmer level tables, etc.)."""

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
