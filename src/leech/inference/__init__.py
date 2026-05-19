"""
Inference engine for running predictions on new data.

Reads POD5 and BAM files, extracts features, runs model predictions,
and writes modification probabilities to output BAM files.

Supports:
- Leech native models (checkpoint directories)
- Remora TorchScript models (.pt files) via auto-detection
- Parallel chunk extraction with multiprocessing
- Signal-level kmer encoding (signal_kmer) and base-level one-hot (base_onehot)
"""

from leech.inference.aggregation import (
    aggregate_one_vs_all,
    aggregate_pairwise,
    aggregate_pairwise_tournament,
    aggregate_pairwise_weighted,
)
from leech.inference.bundle import run_bundle_inference
from leech.inference.helpers import (
    InferenceConfigError,
    _check_config_consistency,
    _encode_sequence_for_inference,
    _run_batch,
    _run_batch_multiclass,
    _warn_if_kmer_table_drifted,
    _write_mega_batch_predictions,
    _write_prediction_tags,
    load_model_auto,
    validate_inference_shapes,
)
from leech.inference.single import run_inference

__all__ = [
    "aggregate_one_vs_all",
    "aggregate_pairwise",
    "aggregate_pairwise_tournament",
    "aggregate_pairwise_weighted",
    "run_bundle_inference",
    "run_inference",
    "InferenceConfigError",
    "_check_config_consistency",
    "_encode_sequence_for_inference",
    "_run_batch",
    "_run_batch_multiclass",
    "_warn_if_kmer_table_drifted",
    "_write_mega_batch_predictions",
    "_write_prediction_tags",
    "load_model_auto",
    "validate_inference_shapes",
]
