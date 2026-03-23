"""Model bundling: package multiple trained models into versioned .pt files."""

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn as nn

import leech
from leech.model_loading import (
    _architecture_config,
    _instantiate_model,
    _migrate_state_dict_keys,
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
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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
            "leech_version": leech.__version__,
            "git_commit": leech._get_git_revision(),
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


def create_multiclass_bundle(
    model_dir: Path,
    output_path: Path,
    version: str,
) -> Path:
    """Bundle a single multiclass model into a versioned .pt file.

    Args:
        model_dir: Directory with config.json and model_best.pt
        output_path: Output .pt file path
        version: Semantic version string

    Returns:
        Path to the saved bundle file
    """
    model_dir = Path(model_dir)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        full_config = json.load(f)

    arch_config = _architecture_config(full_config)
    architecture = full_config["model_name"]
    label_map = full_config.get("label_map", {})
    num_classes = full_config.get("num_out", len(label_map))

    checkpoint_path = model_dir / "model_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_entry: dict = {
        "state_dict": checkpoint["model_state_dict"],
        "best_val_acc": checkpoint.get("best_val_acc"),
        "best_epoch": checkpoint.get("best_epoch"),
    }

    # Include CL regression head if present
    if checkpoint.get("cl_regression_head_state_dict") is not None:
        model_entry["cl_regression_head_state_dict"] = checkpoint["cl_regression_head_state_dict"]
        logger.info("CL regression head included in multiclass bundle")

    # Include calibration only if it improved ECE
    from leech.calibration import load_calibration

    cal_params = load_calibration(model_dir)
    if cal_params is not None:
        ece_before = cal_params.get("ece_before", 1.0)
        ece_after = cal_params.get("ece_after", 0.0)
        if ece_after < ece_before:
            model_entry["calibration"] = cal_params
            cal_method = cal_params.get("method", "unknown")
            logger.info(f"Calibration ({cal_method}): ECE {ece_before:.4f} -> {ece_after:.4f}")
        else:
            cal_method = cal_params.get("method", "unknown")
            logger.info(
                f"Skipping calibration ({cal_method}, ECE {ece_before:.4f} -> {ece_after:.4f}, no improvement)"
            )

    bundle = {
        "metadata": {
            "format_version": 1,
            "bundle_version": version,
            "architecture": architecture,
            "comparison_type": "multiclass",
            "num_models": 1,
            "num_classes": num_classes,
            "label_map": label_map,
            "pairs": sorted(label_map.keys()),
            "created_at": datetime.now(UTC).isoformat(),
            "leech_version": leech.__version__,
            "git_commit": leech._get_git_revision(),
            "cl_regression": full_config.get("cl_regression", False),
        },
        "config": arch_config,
        "model": model_entry,
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
        f"Multiclass bundle saved to {output_path} ({num_classes} classes, version {version})"
    )
    return output_path


def load_model_from_multiclass_bundle(
    bundle_path: Path,
    device: str = "cuda",
    bundle_data: dict | None = None,
) -> tuple[nn.Module, dict]:
    """Load the multiclass model from a bundle file.

    Args:
        bundle_path: Path to multiclass bundle .pt file
        device: Device to load model on
        bundle_data: Pre-loaded bundle dict (avoids double torch.load)

    Returns:
        Tuple of (model, config_dict) with label_map/num_out merged into config
    """
    bundle = bundle_data or torch.load(bundle_path, map_location="cpu", weights_only=False)
    config = bundle["config"]
    metadata = bundle["metadata"]
    model_data = bundle["model"]

    # Merge multiclass metadata into config for run_inference
    config["label_map"] = metadata.get("label_map", {})
    config["num_out"] = metadata.get("num_classes", len(config["label_map"]))

    # Extract calibration params
    if "calibration" in model_data:
        config["calibration"] = model_data["calibration"]

    model = _instantiate_model(config)
    state_dict = model_data["state_dict"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    state_dict = _migrate_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Pass CL regression head state dict through config for inference setup
    if model_data.get("cl_regression_head_state_dict") is not None:
        config["cl_regression"] = True
        config["cl_regression_head_state_dict"] = model_data["cl_regression_head_state_dict"]

    return model, config


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
    from leech.model_export import export_model, serialize_exported_model
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

        # Load checkpoint, instantiate, export
        checkpoint_path = model_dir / "model_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Strip _orig_mod. prefix added by torch.compile()
        state_dict = checkpoint["model_state_dict"]
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        state_dict = _migrate_state_dict_keys(state_dict)

        model = _instantiate_model(ref_arch_config)
        model.load_state_dict(state_dict)
        ep = export_model(model, ref_arch_config)
        exported_bytes = serialize_exported_model(ep)

        models_dict[pair] = {
            "exported_bytes": exported_bytes,
            "best_val_acc": checkpoint.get("best_val_acc"),
            "best_epoch": checkpoint.get("best_epoch"),
        }

    bundle = {
        "metadata": {
            "format_version": 3,
            "bundle_version": version,
            "architecture": architecture,
            "comparison_type": comparison_type,
            "num_models": len(models_dict),
            "pairs": pairs,
            "created_at": datetime.now(UTC).isoformat(),
            "leech_version": leech.__version__,
            "git_commit": leech._get_git_revision(),
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


# Architectures known to be vmap-incompatible (contain nn.LSTM or nn.BatchNorm1d)
_VMAP_INCOMPATIBLE_ARCHITECTURES = {
    "ConvLSTMBase",
    "ConvLSTMBaseBN",
    "ConvLSTMBaseAttn",
    "ConvLSTMBaseBNAttn",
    "ConvLSTMDwell",
    "ConvLSTMDwellBN",
    "ConvLSTMDwellAttn",
    "ConvLSTMDwellBNAttn",
    "ConvLSTMDwellGNAttn",
    "ConvLSTMDwellLNAttn",
    "ConvLSTMRemora",
    "ConvLSTMRemoraBase",
    "ConvOnly",  # uses nn.BatchNorm1d
}


def create_vmap_bundle(
    model_dirs: dict[str, Path],
    output_path: Path,
    comparison_type: str,
    version: str,
) -> Path:
    """Bundle multiple trained models with pre-stacked parameters for vectorized inference.

    Uses ``torch.func.stack_module_state`` to stack all model parameters into
    tensors with a leading model dimension, enabling ``torch.vmap`` to run all
    N models in a single vectorized forward pass.

    Requires a vmap-compatible architecture (no ``nn.LSTM``, no ``nn.BatchNorm1d``).
    TCN variants with GroupNorm or LayerNorm are fully compatible.

    Args:
        model_dirs: Mapping of pair name -> model directory
        output_path: Output .pt file path
        comparison_type: "pairwise" or "one_vs_all"
        version: Semantic version string

    Returns:
        Path to the saved bundle file

    Raises:
        ValueError: If architecture is vmap-incompatible or configs don't match
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

    # Validate vmap compatibility
    if architecture in _VMAP_INCOMPATIBLE_ARCHITECTURES:
        raise ValueError(
            f"Architecture '{architecture}' is not vmap-compatible "
            f"(contains nn.LSTM or nn.BatchNorm1d). "
            f"Use create_bundle() or create_torchscript_bundle() instead, "
            f"or switch to a TCN variant with GroupNorm/LayerNorm."
        )

    # Instantiate all models and load state dicts
    models_list: list[nn.Module] = []
    platt_a_list: list[float] = []
    platt_b_list: list[float] = []

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
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Strip _orig_mod. prefix added by torch.compile()
        state_dict = checkpoint["model_state_dict"]
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        state_dict = _migrate_state_dict_keys(state_dict)

        model = _instantiate_model(ref_arch_config)
        model.load_state_dict(state_dict)
        model.eval()
        models_list.append(model)

        # Platt scaling
        platt_path = model_dir / "platt.json"
        if platt_path.exists():
            with open(platt_path) as f:
                platt_data = json.load(f)
            platt_a_list.append(platt_data["platt_a"])
            platt_b_list.append(platt_data["platt_b"])
            logger.info(
                f"{pair}: platt a={platt_data['platt_a']:.4f}, b={platt_data['platt_b']:.4f}"
            )
        else:
            platt_a_list.append(1.0)
            platt_b_list.append(0.0)

    # Stack model parameters for vmap
    stacked_params, stacked_buffers = torch.func.stack_module_state(models_list)

    bundle = {
        "metadata": {
            "format_version": 4,
            "bundle_version": version,
            "architecture": architecture,
            "comparison_type": comparison_type,
            "num_models": len(pairs),
            "pairs": pairs,
            "created_at": datetime.now(UTC).isoformat(),
            "leech_version": leech.__version__,
            "git_commit": leech._get_git_revision(),
            "vmap": True,
            "requires_features": requires_features,
        },
        "config": ref_arch_config,
        "stacked_params": stacked_params,
        "stacked_buffers": stacked_buffers,
        "platt_a": torch.tensor(platt_a_list, dtype=torch.float32),
        "platt_b": torch.tensor(platt_b_list, dtype=torch.float32),
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

    logger.info(f"vmap bundle saved to {output_path} ({len(pairs)} models, version {version})")
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
    from leech.model_export import deserialize_exported_model, deserialize_traced_model

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    config = bundle["config"]
    metadata = bundle.get("metadata", {})
    format_version = metadata.get("format_version", 1)

    if format_version == 4:
        # vmap bundle: extract single model from stacked params
        pairs = metadata["pairs"]
        if pair not in pairs:
            available = ", ".join(sorted(pairs))
            raise KeyError(f"Pair '{pair}' not in bundle. Available: {available}")
        pair_idx = pairs.index(pair)
        model = _instantiate_model(config)
        # Extract this pair's params from the stacked tensors
        single_params = {k: v[pair_idx] for k, v in bundle["stacked_params"].items()}
        single_buffers = {k: v[pair_idx] for k, v in bundle["stacked_buffers"].items()}
        model.load_state_dict({**single_params, **single_buffers})
        model = model.to(device)
        model.eval()
    else:
        models = bundle["models"]
        if pair not in models:
            available = ", ".join(sorted(models.keys()))
            raise KeyError(f"Pair '{pair}' not in bundle. Available: {available}")

        if format_version >= 3:
            # torch.export bundle (format_version 3+)
            model = deserialize_exported_model(models[pair]["exported_bytes"], device=device)
        elif metadata.get("torchscript", False):
            # Legacy TorchScript bundle (format_version 2)
            model = deserialize_traced_model(models[pair]["traced_bytes"], device=device)
            model.eval()
        else:
            # Legacy state_dict bundle (format_version 1)
            model = _instantiate_model(config)
            state_dict = _migrate_state_dict_keys(models[pair]["state_dict"])
            model.load_state_dict(state_dict)
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
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    return bundle["metadata"]
