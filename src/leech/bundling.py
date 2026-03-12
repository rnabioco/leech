"""
Bundling utilities for leech.

Functions for creating, loading, and inspecting multi-model bundles.
"""

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn as nn

from leech.export import (
    deserialize_traced_model,
    serialize_traced_model,
    trace_model,
)
from leech.util import (
    _architecture_config,
    _instantiate_model,
)

logger = logging.getLogger("leech.bundling")


def create_bundle(
    model_dirs: dict[str, Path],
    output_path: Path,
    comparison_type: str,
    version: str,
) -> Path:
    """
    Bundle multiple trained models into a single versioned file.

    Args:
        model_dirs: Mapping of pair name -> model directory (with config.json and model_best.pt)
        output_path: Output .pt file path
        comparison_type: "pairwise" or "one_vs_all"
        version: Semantic version string (e.g., "0.1.0-alpha.1")

    Returns:
        Path to the saved bundle file

    Raises:
        FileNotFoundError: If config.json or model_best.pt missing in any dir
        ValueError: If architecture configs don't match across models
    """
    if not model_dirs:
        raise ValueError("model_dirs must not be empty")

    pairs = sorted(model_dirs.keys())

    # Load first model's config as reference
    first_dir = model_dirs[pairs[0]]
    ref_config_path = first_dir / "config.json"
    if not ref_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {ref_config_path}")
    with open(ref_config_path) as f:
        ref_full_config = json.load(f)
    ref_arch_config = _architecture_config(ref_full_config)

    architecture = ref_full_config["model_name"]

    models_dict: dict[str, dict] = {}
    for pair in pairs:
        model_dir = Path(model_dirs[pair])

        # Validate config matches
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            pair_config = json.load(f)
        pair_arch_config = _architecture_config(pair_config)

        if pair_arch_config != ref_arch_config:
            raise ValueError(
                f"Architecture config mismatch for {pair}. "
                f"Expected {ref_arch_config}, got {pair_arch_config}"
            )

        # Load checkpoint
        checkpoint_path = model_dir / "model_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model_entry = {
            "state_dict": checkpoint["model_state_dict"],
            "best_val_acc": checkpoint.get("best_val_acc"),
            "best_epoch": checkpoint.get("best_epoch"),
        }

        # Include Platt scaling if calibrated
        platt_path = model_dir / "platt.json"
        if platt_path.exists():
            with open(platt_path) as f:
                platt_data = json.load(f)
            model_entry["platt_a"] = platt_data["platt_a"]
            model_entry["platt_b"] = platt_data["platt_b"]
            logger.info(
                f"{pair}: platt a={platt_data['platt_a']:.4f}, b={platt_data['platt_b']:.4f}"
            )

        models_dict[pair] = model_entry

    bundle = {
        "metadata": {
            "format_version": 1,
            "bundle_version": version,
            "architecture": architecture,
            "comparison_type": comparison_type,
            "num_models": len(models_dict),
            "pairs": pairs,
            "created_at": datetime.now(UTC).isoformat(),
        },
        "config": ref_arch_config,
        "models": models_dict,
    }

    # Atomic save: write to temp file then rename
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".pt.tmp")
    try:
        os.close(fd)
        torch.save(bundle, tmp_path)
        os.rename(tmp_path, output_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    logger.info(f"Bundle saved to {output_path} ({len(models_dict)} models, version {version})")
    return output_path


def create_torchscript_bundle(
    model_dirs: dict[str, Path],
    output_path: Path,
    comparison_type: str,
    version: str,
) -> Path:
    """Bundle multiple trained models as TorchScript into a single versioned file.

    Like :func:`create_bundle` but stores traced model bytes instead of raw
    state dicts.  The resulting bundle is loadable without the leech model
    registry — each pair is a self-contained TorchScript graph.

    Args:
        model_dirs: Mapping of pair name -> model directory
        output_path: Output .pt file path
        comparison_type: "pairwise" or "one_vs_all"
        version: Semantic version string

    Returns:
        Path to the saved bundle file
    """
    from leech.models.inference_wrapper import ModelInferenceWrapper

    if not model_dirs:
        raise ValueError("model_dirs must not be empty")

    pairs = sorted(model_dirs.keys())

    # Load first model's config as reference
    first_dir = model_dirs[pairs[0]]
    ref_config_path = first_dir / "config.json"
    if not ref_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {ref_config_path}")
    with open(ref_config_path) as f:
        ref_full_config = json.load(f)
    ref_arch_config = _architecture_config(ref_full_config)

    architecture = ref_full_config["model_name"]
    requires_features = architecture in ModelInferenceWrapper.FEATURE_MODELS

    models_dict: dict[str, dict] = {}
    for pair in pairs:
        model_dir = Path(model_dirs[pair])

        # Validate config matches
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            pair_config = json.load(f)
        pair_arch_config = _architecture_config(pair_config)

        if pair_arch_config != ref_arch_config:
            raise ValueError(
                f"Architecture config mismatch for {pair}. "
                f"Expected {ref_arch_config}, got {pair_arch_config}"
            )

        # Load checkpoint, instantiate, trace
        checkpoint_path = model_dir / "model_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Strip _orig_mod. prefix added by torch.compile()
        state_dict = checkpoint["model_state_dict"]
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

        model = _instantiate_model(ref_arch_config)
        model.load_state_dict(state_dict)
        traced = trace_model(model, ref_arch_config)
        traced_bytes = serialize_traced_model(traced)

        models_dict[pair] = {
            "traced_bytes": traced_bytes,
            "best_val_acc": checkpoint.get("best_val_acc"),
            "best_epoch": checkpoint.get("best_epoch"),
        }

    bundle = {
        "metadata": {
            "format_version": 2,
            "bundle_version": version,
            "architecture": architecture,
            "comparison_type": comparison_type,
            "num_models": len(models_dict),
            "pairs": pairs,
            "created_at": datetime.now(UTC).isoformat(),
            "torchscript": True,
            "requires_features": requires_features,
        },
        "config": ref_arch_config,
        "models": models_dict,
    }

    # Atomic save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".pt.tmp")
    try:
        os.close(fd)
        torch.save(bundle, tmp_path)
        os.rename(tmp_path, output_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    logger.info(
        f"TorchScript bundle saved to {output_path} ({len(models_dict)} models, version {version})"
    )
    return output_path


def load_model_from_bundle(
    bundle_path: Path,
    pair: str,
    device: str = "cuda",
) -> tuple[nn.Module, dict]:
    """
    Load a single model from a bundle file.

    Supports both format_version 1 (state_dict) and format_version 2
    (TorchScript) bundles transparently.

    Args:
        bundle_path: Path to bundle .pt file
        pair: Pair name to load (e.g., "Asn_Gln")
        device: Device to load model on

    Returns:
        Tuple of (model, config_dict) — same interface as load_model_from_checkpoint

    Raises:
        KeyError: If pair not found in bundle
    """
    bundle = torch.load(bundle_path, map_location=device)
    models = bundle["models"]

    if pair not in models:
        available = ", ".join(sorted(models.keys()))
        raise KeyError(f"Pair '{pair}' not in bundle. Available: {available}")

    config = bundle["config"]
    metadata = bundle.get("metadata", {})

    if metadata.get("torchscript", False):
        # TorchScript bundle (format_version 2)
        model = deserialize_traced_model(models[pair]["traced_bytes"], device=device)
        model.eval()
    else:
        # Legacy state_dict bundle (format_version 1)
        model = _instantiate_model(config)
        model.load_state_dict(models[pair]["state_dict"])
        model = model.to(device)
        model.eval()

    return model, config


def list_bundle_models(bundle_path: Path) -> dict:
    """
    List metadata from a bundle file.

    Args:
        bundle_path: Path to bundle .pt file

    Returns:
        Metadata dict with keys: format_version, bundle_version, architecture,
        comparison_type, num_models, pairs, created_at
    """
    bundle = torch.load(bundle_path, map_location="cpu")
    return bundle["metadata"]
