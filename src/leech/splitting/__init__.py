"""
Read-level data splitting to prevent data leakage.

This module provides functions for splitting training data at the read level
(not chunk level) and processing comparison specifications for pairwise
classification tasks.
"""

from leech.splitting.splitter import (
    merge_and_split_chunks,
    parse_comparison_spec,
    process_comparison_spec,
    split_chunks_by_read,
)

__all__ = [
    "split_chunks_by_read",
    "merge_and_split_chunks",
    "parse_comparison_spec",
    "process_comparison_spec",
]
