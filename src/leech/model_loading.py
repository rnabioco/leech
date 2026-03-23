"""Model loading, instantiation, and checkpoint management."""

import inspect
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from leech.constants import generate_random_seed
from leech.models import MODEL_REGISTRY, get_model

logger = logging.getLogger("leech.model_loading")


def setup_random_seed(seed: int | None, output_dir: Path | None = None) -> int:
    """Setup random seed for reproducibility and optionally save to file.

    Args:
        seed: Random seed value, or None to generate one
        output_dir: Directory to save seed.txt file, or None to skip saving

    Returns:
        The seed value used
    """
    # Generate if needed
    if seed is None:
        seed = generate_random_seed()
        logger.info(f"Generated random seed: {seed}")
    else:
        logger.info(f"Using provided seed: {seed}")

    # Set for all libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Save if requested
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        seed_file = output_dir / "seed.txt"
        with open(seed_file, "w") as f:
            f.write(f"{seed}\n")
        logger.info(f"Saved seed to {seed_file}")

    return seed


def load_model_from_checkpoint(
    checkpoint_path: Path, device: str = "cuda", checkpoint_name: str = "model_best.pt"
) -> tuple[nn.Module, dict]:
    """
    Load a trained model from checkpoint directory.

    Args:
        checkpoint_path: Path to checkpoint directory (contains config.json and .pt files)
        device: Device to load model on
        checkpoint_name: Name of checkpoint file (default: model_best.pt)

    Returns:
        Tuple of (model, config_dict)

    Raises:
        FileNotFoundError: If config.json or checkpoint file not found
        ValueError: If config is invalid
    """
    checkpoint_path = Path(checkpoint_path)

    # Load config
    config_file = checkpoint_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file) as f:
        config = json.load(f)

    # Create model from config (filters training params and validates constructor args)
    model = _instantiate_model(config)

    # Load checkpoint
    checkpoint_file = checkpoint_path / checkpoint_name
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

    # Strip _orig_mod. prefix added by torch.compile()
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    state_dict = _migrate_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Pass CL regression head state dict through config for inference setup
    if checkpoint.get("cl_regression_head_state_dict") is not None:
        config["cl_regression"] = True
        config["cl_regression_head_state_dict"] = checkpoint["cl_regression_head_state_dict"]

    return model, config


# Training-specific parameters that should NOT be in bundle configs
_TRAINING_PARAMS = {
    "epochs",
    "batch_size",
    "learning_rate",
    "device",
    "seed",
    "val_split",
    "patience",
    "min_delta",
    "save_dir",
    "log_dir",
    "num_workers",
    "pin_memory",
    "prefetch_factor",
    "use_class_weights",
    "pos_weight",
    "scheduler",
    "scheduler_patience",
    "scheduler_factor",
    "max_grad_norm",
    "weight_decay",
    "mixed_precision",
    "warmup_epochs",
    "loss_type",
    "focal_gamma",
    "augment_jitter",
    "augment_scale_min",
    "augment_scale_max",
    "resume_from",
}


def _migrate_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap legacy state_dict keys from old naming conventions."""
    migrated = {}
    did_migrate = False
    for key, value in state_dict.items():
        new_key = key.replace(".bn1.", ".norm1.").replace(".bn2.", ".norm2.")
        if new_key != key:
            did_migrate = True
        migrated[new_key] = value
    if did_migrate:
        logger.warning(
            "Migrated legacy state_dict keys: bn1/bn2 -> norm1/norm2. "
            "Re-save the checkpoint to silence this warning."
        )
    return migrated


def _architecture_config(config: dict) -> dict:
    """Extract architecture-only parameters from a full training config."""
    arch = {k: v for k, v in config.items() if k not in _TRAINING_PARAMS}
    # Normalize optional keys added after initial training runs
    arch.setdefault("signal_in_channels", 1)
    return arch


def _instantiate_model(config: dict) -> nn.Module:
    """Instantiate a model from a config dict, filtering to valid constructor params.

    Strips training-specific keys and any kwargs not accepted by the model class,
    so it works with both full training configs and bundle architecture configs.
    """
    model_name = config["model_name"]
    signal_len = config["signal_len"]
    kmer_len = config["kmer_len"]

    model_kwargs = {
        k: v
        for k, v in config.items()
        if k not in {"model_name", "signal_len", "kmer_len"} and k not in _TRAINING_PARAMS
    }

    model_class = MODEL_REGISTRY[model_name]
    sig = inspect.signature(model_class)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_keyword:
        # Constructor accepts **kwargs (e.g. TCNDwellGN) — resolve params from
        # the parent class that actually defines the signature
        for base in model_class.__mro__[1:]:
            base_sig = inspect.signature(base)
            if not any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in base_sig.parameters.values()
            ):
                valid_params = set(base_sig.parameters.keys()) - {"self"}
                break
        else:
            valid_params = set(sig.parameters.keys()) - {"self"}
        filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in valid_params}
    else:
        valid_params = set(sig.parameters.keys()) - {"self"}
        filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in valid_params}

    return get_model(model_name, signal_len=signal_len, kmer_len=kmer_len, **filtered_kwargs)
