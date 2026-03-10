"""
Command handlers for the leech CLI.

This module contains the business logic for each CLI command, separated from
the Click interface layer. Each command module exports handler functions that
can be called from cli.py or other interfaces (API, Snakemake, etc.).
"""

from leech.commands.merge_split import handle_merge_and_split, handle_merge_and_split_kfold
from leech.commands.prepare import handle_prepare

__all__ = [
    "handle_prepare",
    "handle_merge_and_split",
    "handle_merge_and_split_kfold",
]
