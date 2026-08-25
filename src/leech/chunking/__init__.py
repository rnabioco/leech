"""
Chunk extraction and serialization utilities.

This module provides functionality for extracting training chunks from
processed reads and saving/loading them to disk.
"""

from leech.chunking.extractor import (
    LeechRead,
    extract_training_chunks,
    extraction_sequence,
    find_focus_bases,
    resolve_feature_window,
)
from leech.chunking.serialization import (
    DEFERRABLE_FIELDS,
    csr_from_object_rows,
    csr_gather_index,
    csr_offsets_from_lens,
    get_chunk_statistics,
    iter_npz_row_blocks,
    load_chunks,
    load_seq_to_sig_csr,
    npz_array_members,
    npz_member_names,
    save_chunks,
)

__all__ = [
    # Extraction
    "LeechRead",
    "extract_training_chunks",
    "extraction_sequence",
    "find_focus_bases",
    "resolve_feature_window",
    # Serialization
    "save_chunks",
    "load_chunks",
    "get_chunk_statistics",
    "DEFERRABLE_FIELDS",
    "iter_npz_row_blocks",
    "load_seq_to_sig_csr",
    "npz_array_members",
    "npz_member_names",
    "csr_from_object_rows",
    "csr_gather_index",
    "csr_offsets_from_lens",
]
