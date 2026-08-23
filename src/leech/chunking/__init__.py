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
from leech.chunking.serialization import get_chunk_statistics, load_chunks, save_chunks

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
]
