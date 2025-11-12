"""
Data preparation pipeline for training data extraction.

This module provides high-level orchestration functions for preparing
training data from POD5 and BAM files, including parallel processing,
read iteration, and sequence encoding utilities.
"""

from leech.preparation.encoding import encode_kmer, int_to_seq, one_hot_encode_sequence, seq_to_int
from leech.preparation.orchestrator import (
    prepare_training_data,
    prepare_training_data_with_split,
)
from leech.preparation.parallel import prepare_training_data_parallel
from leech.preparation.reader import iter_bam_with_pod5, read_pod5_signal

__all__ = [
    # Orchestration
    "prepare_training_data",
    "prepare_training_data_with_split",
    "prepare_training_data_parallel",
    # Reading
    "iter_bam_with_pod5",
    "read_pod5_signal",
    # Encoding
    "encode_kmer",
    "seq_to_int",
    "int_to_seq",
    "one_hot_encode_sequence",
]
