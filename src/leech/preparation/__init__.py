"""
Data preparation pipeline for training data extraction.

This module provides high-level orchestration functions for preparing
training data from POD5 and BAM files, including parallel processing,
read building, and sequence encoding utilities.

Extraction goes through the parallel dispatcher
(``prepare_training_data_parallel``) at every worker count, including 1; the
separate sequential pipeline this package used to also export
(``prepare_training_data``, ``prepare_training_data_with_split``,
``iter_bam_with_pod5``) was retired in issue #275.
"""

from leech.preparation.encoding import encode_kmer
from leech.preparation.orchestrator import split_rows_by_read, write_splits
from leech.preparation.parallel import prepare_training_data_parallel
from leech.preparation.reader import build_leech_read, read_pod5_signal

__all__ = [
    # Orchestration
    "prepare_training_data_parallel",
    "split_rows_by_read",
    "write_splits",
    # Reading
    "build_leech_read",
    "read_pod5_signal",
    # Encoding
    "encode_kmer",
]
