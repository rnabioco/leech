"""
Command handlers for the leech CLI.

This module contains the business logic for each CLI command, separated from
the Click interface layer. Each command module exports handler functions that
can be called from cli.py or other interfaces (API, Snakemake, etc.).
"""

from leech.commands.bundle import handle_bundle, handle_bundle_info, handle_export, pick_best_fold
from leech.commands.calibrate import handle_calibrate
from leech.commands.eval import handle_test
from leech.commands.merge_split import handle_merge_and_split, handle_merge_and_split_kfold
from leech.commands.optimize import handle_optimize
from leech.commands.predict import handle_predict, validate_predict_args
from leech.commands.prepare import handle_prepare
from leech.commands.train import handle_train

__all__ = [
    "handle_bundle",
    "handle_bundle_info",
    "handle_calibrate",
    "handle_export",
    "handle_merge_and_split",
    "handle_merge_and_split_kfold",
    "handle_optimize",
    "handle_predict",
    "handle_prepare",
    "handle_test",
    "handle_train",
    "pick_best_fold",
    "validate_predict_args",
]
