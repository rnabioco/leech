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

# Pairwise amino acid comparisons from TSV spec file
AA_PAIRS = []
AA_PAIR_TO_LABELS = {}  # Maps pair name to (labels_group1, labels_group2)

comparison_spec_file = config.get("comparison_spec_file")
if comparison_spec_file:
    # Parse TSV comparison spec to extract pairwise comparisons
    # TSV format: meta_label1 \t label_set1 \t meta_label2 \t label_set2
    # Example: "Ala\tAla\tArg\tArg" creates comparison "Ala_Arg"
    from pathlib import Path

    spec_path = Path(comparison_spec_file)
    if spec_path.exists():
        with open(spec_path) as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")
                if len(parts) != 4:
                    continue

                meta1, labels1_str, meta2, labels2_str = parts
                labels1 = [l.strip() for l in labels1_str.split(",")]
                labels2 = [l.strip() for l in labels2_str.split(",")]

                # Create pair name: meta1_meta2 (e.g., "Ala_Arg", "basic_acidic")
                pair_name = f"{meta1}_{meta2}"
                AA_PAIRS.append(pair_name)
                AA_PAIR_TO_LABELS[pair_name] = (labels1, labels2)


def get_charged_samples():
    """Get samples labeled as charged or uncharged (deprecated)."""
    return [
        s
        for s in SAMPLES
        if config["samples"][s].get("label") in ["charged", "uncharged"]
    ]


def get_samples_for_aa_pair(pair):
    """Get samples for a specific amino acid pair.

    Supports both TSV-based multi-label comparisons and legacy single-label pairs.

    Args:
        pair: Pair name (e.g., "Ala_Arg", "basic_acidic")

    Returns:
        List of sample names that have labels matching either group in the pair
    """
    if pair in AA_PAIR_TO_LABELS:
        # TSV-based comparison with potentially multiple labels per group
        labels1, labels2 = AA_PAIR_TO_LABELS[pair]
        all_labels = labels1 + labels2
    else:
        # Legacy format: pair name like "Ala_vs_Arg" or "Ala_Arg"
        if "_vs_" in pair:
            aa1, aa2 = pair.split("_vs_")
        else:
            aa1, aa2 = pair.split("_", 1)
        all_labels = [aa1, aa2]

    return [s for s in SAMPLES if config["samples"][s].get("label") in all_labels]


def build_merge_input_args(pair, chunk_files):
    """Build -i label=file arguments for merge-and-split command.

    Args:
        pair: Pair name (e.g., "Ala_Arg", "basic_acidic")
        chunk_files: List of chunk file paths

    Returns:
        String of -i arguments like "-i Ala=file1.npz -i Gly=file2.npz"
    """
    if pair in AA_PAIR_TO_LABELS:
        # TSV-based comparison: map samples to meta-labels
        labels1, labels2 = AA_PAIR_TO_LABELS[pair]
        meta1, meta2 = pair.split("_", 1)

        # Build mapping of sample -> meta-label
        sample_to_meta = {}
        for sample in SAMPLES:
            sample_label = config["samples"][sample].get("label")
            if sample_label in labels1:
                sample_to_meta[sample] = meta1
            elif sample_label in labels2:
                sample_to_meta[sample] = meta2

        # Build -i arguments
        args = []
        for chunk_file in chunk_files:
            # Extract sample name from path (e.g., chunks/ala_synthetic/all.npz -> ala_synthetic)
            sample = Path(chunk_file).parent.name
            if sample in sample_to_meta:
                meta_label = sample_to_meta[sample]
                args.append(f"-i {meta_label}={chunk_file}")

        return " ".join(args)
    else:
        # Simple pairwise: use the pair labels directly
        if "_vs_" in pair:
            label1, label2 = pair.split("_vs_")
        else:
            label1, label2 = pair.split("_", 1)

        # Build mapping of sample -> label
        sample_to_label = {}
        for sample in SAMPLES:
            sample_label = config["samples"][sample].get("label")
            if sample_label == label1:
                sample_to_label[sample] = label1
            elif sample_label == label2:
                sample_to_label[sample] = label2

        # Build -i arguments
        args = []
        for chunk_file in chunk_files:
            sample = Path(chunk_file).parent.name
            if sample in sample_to_label:
                label = sample_to_label[sample]
                args.append(f"-i {label}={chunk_file}")

        return " ".join(args)
