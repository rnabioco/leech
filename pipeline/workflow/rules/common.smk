"""
Common variables and functions for the leech pipeline.
"""

import itertools
from pathlib import Path


# Extract sample names and amino acid pairs
SAMPLES = list(config["samples"].keys())
AMINO_ACIDS = config.get("amino_acids", [])
AA_PAIRS = [f"{aa1}_vs_{aa2}"
            for aa1, aa2 in itertools.combinations(sorted(AMINO_ACIDS), 2)]

# Output directories
CHUNKS_DIR = config.get("chunks_dir", "results/chunks")
MODELS_DIR = config.get("models_dir", "results/models")
INFER_DIR = config.get("inference_dir", "results/inference")
METRICS_DIR = config.get("metrics_dir", "results/metrics")

# Model architectures
COMPARE_MODELS = config.get("compare_models", False)
if COMPARE_MODELS:
    MODEL_ARCHITECTURES = config.get("models_to_compare", ["ConvLSTMDwell"])
else:
    MODEL_ARCHITECTURES = [config.get("model", "ConvLSTMDwell")]


def get_charged_samples():
    """Get samples labeled as charged or uncharged."""
    return [s for s in SAMPLES
            if config["samples"][s].get("label") in ["charged", "uncharged"]]


def get_samples_for_aa_pair(pair):
    """Get samples for a specific amino acid pair."""
    aa1, aa2 = pair.split("_vs_")
    return [s for s in SAMPLES
            if config["samples"][s].get("amino_acid") in [aa1, aa2]]
