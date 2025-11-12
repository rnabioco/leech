"""
Reference sequence management utilities.

Provides functions for loading and managing reference sequences from
BAM headers or FASTA files.
"""

import logging
from pathlib import Path

import pysam

logger = logging.getLogger("leech.io.reference")


def extract_reference_from_bam(bam_path: Path) -> dict[str, str]:
    """
    Extract reference sequences from BAM header @SQ records.

    Args:
        bam_path: Path to BAM file

    Returns:
        Dictionary mapping reference name to sequence.
        Empty dict if no sequences in header.

    Example:
        >>> refs = extract_reference_from_bam(Path("alignments.bam"))
        >>> if refs:
        ...     print(f"Found {len(refs)} reference sequences")
        ...     for name, seq in refs.items():
        ...         print(f"{name}: {len(seq)} bp")
    """
    references = {}

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        # Check if reference sequences are in header
        if hasattr(bam.header, "to_dict"):
            header_dict = bam.header.to_dict()
            for sq in header_dict.get("SQ", []):
                ref_name = sq.get("SN")
                ref_seq = sq.get("SQ", None)  # SQ tag contains sequence
                if ref_name and ref_seq:
                    references[ref_name] = ref_seq

    if references:
        logger.info(f"Extracted {len(references)} reference sequences from BAM header")
    else:
        logger.warning("No reference sequences found in BAM @SQ header")

    return references


def load_reference_fasta(fasta_path: Path) -> dict[str, str]:
    """
    Load reference sequences from FASTA file.

    Args:
        fasta_path: Path to FASTA file

    Returns:
        Dictionary mapping reference name to sequence

    Raises:
        FileNotFoundError: If FASTA file not found
        ValueError: If FASTA file is malformed

    Example:
        >>> refs = load_reference_fasta(Path("reference.fasta"))
        >>> print(f"Loaded {len(refs)} sequences")
        >>> for name, seq in refs.items():
        ...     print(f"{name}: {len(seq)} bp")
    """
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    references = {}

    with pysam.FastaFile(str(fasta_path)) as fasta:
        for ref_name in fasta.references:
            references[ref_name] = fasta.fetch(ref_name)

    logger.info(f"Loaded {len(references)} reference sequences from {fasta_path}")
    return references


def get_reference_sequences(bam_path: Path, fasta_path: Path | None = None) -> dict[str, str]:
    """
    Get reference sequences, trying BAM header first, then FASTA.

    This is the recommended function for loading reference sequences,
    as it automatically falls back to external FASTA if needed.

    Args:
        bam_path: Path to BAM file
        fasta_path: Optional path to external FASTA file

    Returns:
        Dictionary mapping reference name to sequence

    Raises:
        ValueError: If no reference sequences available from either source

    Example:
        >>> # Try BAM first, fall back to FASTA
        >>> refs = get_reference_sequences(
        ...     bam_path=Path("alignments.bam"),
        ...     fasta_path=Path("reference.fasta")
        ... )
        >>> print(f"Loaded {len(refs)} reference sequences")
    """
    # Try BAM header first
    references = extract_reference_from_bam(bam_path)

    # Fall back to FASTA if provided and BAM had no sequences
    if not references and fasta_path:
        logger.info("Falling back to external FASTA file")
        references = load_reference_fasta(fasta_path)

    # Error if still no references
    if not references:
        raise ValueError(
            "No reference sequences found. BAM file must contain @SQ sequences "
            "or provide --reference-fasta path."
        )

    return references


class ReferenceManager:
    """
    Manager for reference sequences with lazy loading and caching.

    Example:
        >>> manager = ReferenceManager(
        ...     bam_path=Path("alignments.bam"),
        ...     fasta_path=Path("reference.fasta")
        ... )
        >>> seq = manager.get_sequence("chr1")
        >>> print(f"chr1: {len(seq)} bp")
    """

    def __init__(self, bam_path: Path, fasta_path: Path | None = None):
        """
        Initialize reference manager.

        Args:
            bam_path: Path to BAM file
            fasta_path: Optional path to external FASTA file
        """
        self.bam_path = bam_path
        self.fasta_path = fasta_path
        self._sequences: dict[str, str] | None = None

    def load(self) -> None:
        """
        Load reference sequences.

        This is called automatically when accessing sequences,
        but can be called explicitly for eager loading.
        """
        if self._sequences is None:
            self._sequences = get_reference_sequences(self.bam_path, self.fasta_path)

    def get_sequence(self, ref_name: str) -> str:
        """
        Get sequence for a specific reference.

        Args:
            ref_name: Reference name

        Returns:
            Reference sequence

        Raises:
            KeyError: If reference name not found
        """
        if self._sequences is None:
            self.load()

        if self._sequences is None or ref_name not in self._sequences:
            raise KeyError(f"Reference '{ref_name}' not found")

        return self._sequences[ref_name]

    def get_all_sequences(self) -> dict[str, str]:
        """
        Get all reference sequences.

        Returns:
            Dictionary mapping reference names to sequences
        """
        if self._sequences is None:
            self.load()

        return self._sequences if self._sequences is not None else {}

    def has_reference(self, ref_name: str) -> bool:
        """
        Check if reference exists.

        Args:
            ref_name: Reference name

        Returns:
            True if reference exists
        """
        if self._sequences is None:
            self.load()

        return self._sequences is not None and ref_name in self._sequences

    @property
    def reference_names(self) -> list[str]:
        """
        Get list of all reference names.

        Returns:
            List of reference names
        """
        if self._sequences is None:
            self.load()

        return list(self._sequences.keys()) if self._sequences is not None else []

    def __len__(self) -> int:
        """Return number of references."""
        if self._sequences is None:
            self.load()

        return len(self._sequences) if self._sequences is not None else 0

    def __contains__(self, ref_name: str) -> bool:
        """Check if reference exists."""
        return self.has_reference(ref_name)
