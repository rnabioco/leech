"""Model loading, instantiation, and checkpoint management."""

import inspect
import json
import logging
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from leech.models import MODEL_REGISTRY, get_model

# Re-exported for backward compatibility: this used to be defined here, but
# lived in the one module that made merely importing it cost a torch import
# (this file does `import torch` above) -- a problem for CPU-only callers
# like `data prepare` that fork a worker pool downstream. See
# `leech.seeding` for why and `tests/test_data_prep.py` /
# `-X importtime` for the regression guard. Import it from either module.
from leech.seeding import setup_random_seed  # noqa: F401

logger = logging.getLogger("leech.model_loading")


def resolve_dwell_template_path(config: Mapping, base_dir: Path | None) -> Path | None:
    """Resolve a checkpoint's recorded ``dwell_template_table`` to a real path.

    ``train_model`` (issue #369) copies the TSV into ``<output_dir>/
    dwell_template_table.tsv`` and records that bare name in ``config.json``,
    so the checkpoint directory is the first place to look -- not the
    process's current working directory, which has no reason to relate to
    wherever training happened to run. A checkpoint trained before #369
    still records whatever string the CLI got (often a repo-relative path,
    e.g. ``report/data/legacy/aa_transit_table.tsv``); that only ever
    resolved against the CWD, so it stays supported as a fallback.

    Args:
        config: A checkpoint's ``config.json``, or a bundle's ``config``.
        base_dir: The checkpoint (or bundle) directory to resolve a relative
            path against. ``None`` skips straight to the CWD-relative
            fallback -- e.g. a caller with no directory of its own, such as
            a bundle that embeds the table and never needs this at all.

    Returns:
        ``None`` when the config has no ``dwell_template_table``. An
        absolute recorded path, returned as-is (no existence check -- the
        caller finds out when it tries to open it). Otherwise
        ``base_dir / dwell_template_table`` when that exists; failing that,
        the bare CWD-relative path when THAT exists (with a one-line
        warning naming both, since this is exactly the legacy shape #369
        fixes for new checkpoints).

    Raises:
        FileNotFoundError: The config records a relative
            ``dwell_template_table`` and neither candidate exists.
    """
    raw = config.get("dwell_template_table")
    if not raw:
        return None
    raw_path = Path(raw)
    if raw_path.is_absolute():
        return raw_path

    model_dir_candidate = (Path(base_dir) / raw_path) if base_dir is not None else None
    if model_dir_candidate is not None and model_dir_candidate.exists():
        return model_dir_candidate

    cwd_candidate = raw_path
    if cwd_candidate.exists():
        if model_dir_candidate is not None:
            logger.warning(
                "dwell_template_table %r not found at %s; falling back to the "
                "CWD-relative path %s (resolved: %s). This checkpoint predates "
                "issue #369's copy-into-model-dir behavior -- re-train, or copy "
                "the table into %s, to make this model directory self-contained "
                "and silence this warning.",
                raw,
                model_dir_candidate,
                cwd_candidate,
                cwd_candidate.resolve(),
                base_dir,
            )
        return cwd_candidate

    candidates = [str(c) for c in (model_dir_candidate, cwd_candidate) if c is not None]
    raise FileNotFoundError(
        f"dwell_template_table {raw!r} recorded in config not found. Tried: "
        + ", ".join(candidates)
    )


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


# Training-specific parameters that should NOT be in bundle configs. This
# denylist is exhaustive against what config.json actually writes -- verified
# by grepping every downstream `config.get(...)`/`config[...]` read in
# inference/single.py and inference/bundle.py (#270): every key below is
# absent from that read set, and every key present in it (motif, dwell_offset,
# num_out, label_map, cl_regression, dwell_template_table, the refine_* and
# feature_*/dwell_margin_* provenance fields, ...) is deliberately NOT here.
# `_instantiate_model`'s own constructor-signature filter is what ultimately
# protects model construction (it drops anything the class doesn't accept,
# training params included) -- this set exists only to keep training-only
# noise out of `_architecture_config`'s bundle-comparison and stored output.
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
    "quantile_grad_clip",
    "grad_accum_split",
    "save_optim_every",
    "weight_decay",
    "mixed_precision",
    "warmup_epochs",
    "loss_type",
    "focal_gamma",
    "focal_neg_gamma",
    "label_smoothing",
    "augment_jitter",
    "augment_scale_min",
    "augment_scale_max",
    "augment_scale_range",
    "augment_time_mask_bases",
    "augment_time_mask_count",
    "augment_shift_max_bases",
    "augment_feature_noise_scale",
    "augment_time_stretch_min",
    "augment_time_stretch_max",
    "balance_groups",
    "oversample_minority",
    "adversarial_lambda",
    "adversarial_anneal_epochs",
    "confound",
    "label_noise_rates",
    "cl_lambda",
    "checkpoint_metric",
    "signal_mode",
    "gpus",
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
        # Constructor accepts **kwargs (e.g. TransformerDwellResidual) — resolve params from
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
