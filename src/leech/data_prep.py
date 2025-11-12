"""
Data preparation: high-level orchestration for reading POD5 and BAM files.

This module re-exports functionality from refactored submodules:
- leech.preparation: Pipeline orchestration and encoding
- leech.io: BAM/POD5 reading, reference sequences, motif search
- leech.chunking: Chunk extraction and serialization
- leech.splitting: Read-level splitting

You can import from this module or directly from the submodules.
"""

import logging

# Re-export from chunking
from leech.chunking import LeechRead, extract_training_chunks, load_chunks, save_chunks

# Re-export from io
from leech.io import get_reference_sequences

# Re-export from preparation
from leech.preparation import (
    encode_kmer,
    int_to_seq,
    iter_bam_with_pod5,
    one_hot_encode_sequence,
    prepare_training_data,
    prepare_training_data_parallel,
    prepare_training_data_with_split,
    read_pod5_signal,
    seq_to_int,
)

# Re-export from splitting
from leech.splitting import (
    merge_and_split_chunks,
    parse_comparison_spec,
    process_comparison_spec,
    split_chunks_by_read,
)

logger = logging.getLogger("leech.data_prep")

# Public API - maintains backward compatibility
__all__ = [
    # From chunking
    "LeechRead",
    "extract_training_chunks",
    "save_chunks",
    "load_chunks",
    # From splitting
    "split_chunks_by_read",
    "merge_and_split_chunks",
    "parse_comparison_spec",
    "process_comparison_spec",
    # From io
    "get_reference_sequences",
    # From preparation
    "iter_bam_with_pod5",
    "read_pod5_signal",
    "prepare_training_data",
    "prepare_training_data_parallel",
    "prepare_training_data_with_split",
    # Encoding utilities
    "encode_kmer",
    "seq_to_int",
    "int_to_seq",
    "one_hot_encode_sequence",
]
