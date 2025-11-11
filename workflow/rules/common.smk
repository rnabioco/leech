"""
Common variables and functions for the leech pipeline.
"""

import itertools
from pathlib import Path


def get_project_path(base_dir):
    """Insert project_name into directory path if configured.

    This function allows for project-specific subdirectories to organize
    multiple workflows/projects under the same leech installation.

    Examples:
        Without project_name:
            "/scratch/alpine/user/leech/pod5" -> "/scratch/alpine/user/leech/pod5"

        With project_name: "synthetic-trna"
            "/scratch/alpine/user/leech/pod5" -> "/scratch/alpine/user/leech/synthetic-trna/pod5"

    Args:
        base_dir: Base directory path (may be absolute or relative)

    Returns:
        Directory path with project_name inserted after "leech" if configured,
        otherwise returns the original base_dir unchanged.
    """
    project_name = config.get("project_name")
    if not project_name:
        return base_dir

    # Split path and insert project_name after "leech" directory
    path = Path(base_dir)
    parts = list(path.parts)

    # Try to find "leech" in the path and insert project_name after it
    try:
        leech_idx = parts.index("leech")
        # Insert project_name after "leech"
        parts.insert(leech_idx + 1, project_name)
        return str(Path(*parts))
    except ValueError:
        # "leech" not found in path (e.g., relative path like "results/chunks")
        # Insert project_name before the last component
        if len(parts) > 1:
            return str(path.parent / project_name / path.name)
        else:
            # Single-component path, prepend project_name
            return str(Path(project_name) / path.name)


# Extract sample names
SAMPLES = list(config["samples"].keys())

# Output directories (with optional project_name support)
CHUNKS_DIR = get_project_path(config.get("chunks_dir", "results/chunks"))
MODELS_DIR = get_project_path(config.get("models_dir", "results/models"))
INFER_DIR = get_project_path(config.get("inference_dir", "results/inference"))
METRICS_DIR = get_project_path(config.get("metrics_dir", "results/metrics"))

# Model architectures
COMPARE_MODELS = config.get("compare_models", False)
if COMPARE_MODELS:
    MODEL_ARCHITECTURES = config.get("models_to_compare", ["ConvLSTMDwell"])
else:
    MODEL_ARCHITECTURES = [config.get("model", "ConvLSTMDwell")]


def get_charged_samples():
    """Get samples labeled as charged or uncharged (deprecated)."""
    return [
        s
        for s in SAMPLES
        if config["samples"][s].get("label") in ["charged", "uncharged"]
    ]
