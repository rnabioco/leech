"""
Post-hoc calibration for model outputs.

Binary models: Platt scaling — sigmoid(a * logit + b) — 2 parameters.
Multiclass models: Temperature, Matrix, or Dirichlet scaling.

- Temperature scaling (Guo et al. 2017): softmax(logits / T) — 1 param
- Matrix scaling (Guo et al. 2017): softmax(logits @ W^T + b) — K²+K params
- Dirichlet scaling (Kull et al. 2019): softmax(alpha * log_softmax(logits) + beta) — 2K params

References:
  Platt, "Probabilistic Outputs for SVMs" (1999)
  Guo et al., "On Calibration of Modern Neural Networks" (2017)
  Kull et al., "Beyond temperature scaling" (2019)
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import LBFGS
from torch.utils.data import DataLoader

from leech.dataset import LeechDataset, collate_fn, resolve_val_dataloader_workers
from leech.models import get_model
from leech.models.inference_wrapper import ModelInferenceWrapper

logger = logging.getLogger("leech.calibration")


def collect_logits(
    model_wrapper: ModelInferenceWrapper,
    val_loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect raw logits and labels from validation set."""
    all_logits = []
    all_labels = []

    model_wrapper.model.eval()
    with torch.no_grad():
        for batch in val_loader:
            logits = model_wrapper.forward_batch(batch, device)
            labels = batch["label"].float().to(device)
            all_logits.append(logits.flatten())
            all_labels.append(labels.flatten())

    return torch.cat(all_logits), torch.cat(all_labels)


def learn_platt(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
) -> tuple[float, float]:
    """
    Learn Platt scaling parameters (a, b) via NLL minimization.

    Optimizes: a, b = argmin BCE(sigmoid(a * logit + b), labels)
    using L-BFGS. Initialized at a=1.0, b=0.0 (identity transform).

    Args:
        logits: Raw model logits from validation set [N]
        labels: Binary labels [N]
        max_iter: Maximum L-BFGS iterations

    Returns:
        Tuple of (a, b) — scale and bias parameters
    """
    a = nn.Parameter(torch.ones(1, device=logits.device))
    b = nn.Parameter(torch.zeros(1, device=logits.device))
    criterion = nn.BCEWithLogitsLoss()

    optimizer = LBFGS([a, b], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = a * logits + b
        loss = criterion(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    a_val = a.item()
    b_val = b.item()

    # a must be positive (monotonicity); clamp to reasonable range
    a_val = max(a_val, 0.01)
    if a_val > 10.0:
        logger.warning(
            f"Platt a={a_val:.4f} exceeds 10.0 — clamping. "
            "This may indicate overfitting or degenerate logits."
        )
        a_val = 10.0
    if abs(b_val) < 0.01:
        logger.warning(
            f"Platt b={b_val:.4f} is near zero — possible degenerate fit. "
            "Check that validation set has sufficient class diversity."
        )

    logger.info(f"Learned Platt params: a={a_val:.4f}, b={b_val:.4f}")
    return a_val, b_val


def calibrate_model(
    model_dir: Path,
    val_data_path: Path,
    output_path: Path | None = None,
    device: str = "cpu",
    batch_size: int = 1024,
    num_workers: int = 0,
) -> tuple[float, float]:
    """
    Learn Platt scaling for a trained model.

    Loads the model and validation data, collects logits, optimizes (a, b),
    and saves to platt.json in the model directory.

    Args:
        model_dir: Directory with model_best.pt and config.json
        val_data_path: Path to validation data (.npz)
        output_path: Output JSON path (default: model_dir/platt.json)
        device: Device for inference
        batch_size: Batch size for validation pass
        num_workers: DataLoader workers

    Returns:
        Tuple of (a, b) — Platt scaling parameters
    """
    if output_path is None:
        output_path = model_dir / "platt.json"

    # Load model config
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    model_name = config["model_name"]
    signal_len = config["signal_len"]
    kmer_len = config["kmer_len"]
    seq_encoding = config.get("seq_encoding", "base_onehot")
    signal_kmer_context = tuple(config.get("signal_kmer_context", (4, 4)))

    # Build model init kwargs
    signal_in_channels = config.get("signal_in_channels", 1)
    model_kwargs: dict = {
        "signal_len": signal_len,
        "kmer_len": kmer_len,
        "seq_encoding": seq_encoding,
        "signal_kmer_context": signal_kmer_context,
        "signal_in_channels": signal_in_channels,
    }
    if "num_features" in config:
        model_kwargs["num_features"] = config["num_features"]

    model = get_model(model_name, **model_kwargs)

    # Load weights (always load to CPU first, then move to device)
    checkpoint_path = model_dir / "model_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    from leech.model_loading import _migrate_state_dict_keys

    state_dict = _migrate_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    wrapper = ModelInferenceWrapper(model, model_name)

    # Load validation data. Forward dwell_template_table from training config
    # so the dataset builds a feature tensor with the same channel count the
    # model was trained on — otherwise the Conv1d in the feature branch
    # crashes with "expected N channels, but got M channels".
    # chunk_path (not pre-loaded chunks) so the dataset streams the arrays out
    # of the npz instead of holding a second full copy of them (#211).
    dwell_template_table = config.get("dwell_template_table") or None
    val_dataset = LeechDataset(
        chunk_path=val_data_path,
        model_type=model_name,
        signal_len=signal_len,
        kmer_len=kmer_len,
        seq_encoding=seq_encoding,
        signal_kmer_context=signal_kmer_context,
        signal_mode=config.get("signal_mode", "both"),
        dwell_template_table=dwell_template_table,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        # `num_workers=0` means AUTO, not "no workers" -- on CUDA feeding the
        # forward pass from the main process alone leaves the GPU idle (#205,
        # #207). The resolver caps by the Slurm cpuset, returns 0 in daemonic
        # workers, and keeps 0 when the dataset fell back to per-chunk lists.
        num_workers=resolve_val_dataloader_workers(val_dataset, num_workers, device),
    )

    logger.info(f"Collecting logits from {len(val_dataset)} validation samples...")
    logits, labels = collect_logits(wrapper, val_loader, device)
    logits = logits.cpu()
    labels = labels.cpu()

    # Compute pre-calibration metrics
    pre_probs = torch.sigmoid(logits)
    pre_ece = _expected_calibration_error(pre_probs, labels)

    a, b = learn_platt(logits, labels)

    # Compute post-calibration metrics
    post_probs = torch.sigmoid(a * logits + b)
    post_ece = _expected_calibration_error(post_probs, labels)

    if post_ece >= pre_ece:
        logger.warning(
            f"Calibration did not improve ECE: {pre_ece:.4f} -> {post_ece:.4f}. "
            "Platt params may be unreliable for this model."
        )
    logger.info(f"ECE: {pre_ece:.4f} -> {post_ece:.4f} (a={a:.4f}, b={b:.4f})")

    result = {
        "platt_a": a,
        "platt_b": b,
        "ece_before": pre_ece,
        "ece_after": post_ece,
        "n_val_samples": len(val_dataset),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Platt scaling saved to {output_path}")
    return a, b


def collect_logits_multiclass(
    model_wrapper: ModelInferenceWrapper,
    val_loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect raw logits (N, K) and integer labels (N,) from validation set."""
    all_logits = []
    all_labels = []

    model_wrapper.model.eval()
    with torch.no_grad():
        for batch in val_loader:
            logits = model_wrapper.forward_batch(batch, device)
            labels = batch["label"].long().to(device)
            all_logits.append(logits)
            all_labels.append(labels.flatten())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels)


def learn_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
) -> float:
    """
    Learn temperature scaling parameter T via NLL minimization.

    Optimizes: T = argmin CrossEntropy(logits / T, labels) using L-BFGS.
    Initialized at T=1.5 (slight over-confidence is typical for neural nets).

    Args:
        logits: Raw model logits from validation set (N, K)
        labels: Integer class labels (N,)
        max_iter: Maximum L-BFGS iterations

    Returns:
        Learned temperature T
    """
    # Use log(T) as the parameter to keep T positive without clamping during optimization
    log_t = nn.Parameter(torch.tensor([0.4055], device=logits.device))  # log(1.5)
    criterion = nn.CrossEntropyLoss()

    optimizer = LBFGS([log_t], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        t = log_t.exp()
        scaled = logits / t
        loss = criterion(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    t_val = log_t.exp().item()

    # Clamp to reasonable range
    t_val = max(t_val, 0.1)
    if t_val > 10.0:
        logger.warning(
            f"Temperature T={t_val:.4f} exceeds 10.0 — clamping. "
            "This may indicate degenerate logits."
        )
        t_val = 10.0

    logger.info(f"Learned temperature: T={t_val:.4f}")
    return t_val


def learn_matrix_scaling(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reg_lambda: float = 0.01,
    max_iter: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Learn matrix scaling parameters (W, b) via regularized NLL minimization.

    Optimizes: W, b = argmin CE(logits @ W^T + b, labels) + reg * (||W-I||²_F + ||b||²)
    using L-BFGS. W is K×K initialized to identity; b is K initialized to zero.

    Args:
        logits: Raw model logits from validation set (N, K)
        labels: Integer class labels (N,)
        reg_lambda: L2 regularization strength toward identity
        max_iter: Maximum L-BFGS iterations

    Returns:
        Tuple of (W, b) — detached CPU tensors
    """
    K = logits.shape[1]
    W = nn.Parameter(torch.eye(K, device=logits.device))
    b = nn.Parameter(torch.zeros(K, device=logits.device))
    criterion = nn.CrossEntropyLoss()

    optimizer = LBFGS([W, b], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = logits @ W.T + b
        loss = criterion(scaled, labels)
        # Regularize toward identity transform
        loss = loss + reg_lambda * (
            (W - torch.eye(K, device=W.device)).pow(2).sum() + b.pow(2).sum()
        )
        loss.backward()
        return loss

    optimizer.step(closure)

    W_out = W.detach().cpu()
    b_out = b.detach().cpu()
    logger.info(
        f"Learned matrix scaling: W diag mean={W_out.diag().mean():.4f}, b norm={b_out.norm():.4f}"
    )
    return W_out, b_out


def learn_dirichlet_scaling(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reg_lambda: float = 0.01,
    max_iter: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Learn Dirichlet (Dir-L diagonal) scaling parameters via regularized NLL.

    Optimizes: alpha, beta = argmin CE(alpha * log_softmax(logits) + beta, labels)
                             + reg * (||alpha-1||² + ||beta||²)
    using L-BFGS. alpha is log-parameterized for positivity.

    Args:
        logits: Raw model logits from validation set (N, K)
        labels: Integer class labels (N,)
        reg_lambda: L2 regularization strength toward identity
        max_iter: Maximum L-BFGS iterations

    Returns:
        Tuple of (alpha, beta) — detached CPU tensors, alpha already exponentiated
    """
    K = logits.shape[1]
    # Parameterize in log-space so alpha = exp(log_alpha) > 0
    log_alpha = nn.Parameter(torch.zeros(K, device=logits.device))  # exp(0) = 1
    beta = nn.Parameter(torch.zeros(K, device=logits.device))
    criterion = nn.CrossEntropyLoss()

    log_probs = torch.log_softmax(logits, dim=-1).detach()

    optimizer = LBFGS([log_alpha, beta], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        alpha = log_alpha.exp()
        scaled = alpha * log_probs + beta
        loss = criterion(scaled, labels)
        # Regularize toward identity (alpha=1, beta=0)
        loss = loss + reg_lambda * ((alpha - 1).pow(2).sum() + beta.pow(2).sum())
        loss.backward()
        return loss

    optimizer.step(closure)

    alpha_out = log_alpha.exp().detach().cpu()
    beta_out = beta.detach().cpu()
    logger.info(
        f"Learned Dirichlet scaling: alpha mean={alpha_out.mean():.4f}, beta norm={beta_out.norm():.4f}"
    )
    return alpha_out, beta_out


def apply_calibration(logits: torch.Tensor, cal_params: dict) -> torch.Tensor:
    """Apply calibration transform to raw logits.

    Dispatches on cal_params["method"]:
      - temperature: logits / T
      - matrix: logits @ W^T + b
      - dirichlet: alpha * log_softmax(logits) + beta

    Returns calibrated logits (caller applies softmax).
    """
    method = cal_params["method"]
    if method == "temperature":
        T = cal_params["temperature"]
        return logits / T
    elif method == "matrix":
        W = torch.tensor(cal_params["W"], dtype=logits.dtype, device=logits.device)
        b = torch.tensor(cal_params["b"], dtype=logits.dtype, device=logits.device)
        return logits @ W.T + b
    elif method == "dirichlet":
        alpha = torch.tensor(cal_params["alpha"], dtype=logits.dtype, device=logits.device)
        beta = torch.tensor(cal_params["beta"], dtype=logits.dtype, device=logits.device)
        return alpha * torch.log_softmax(logits, dim=-1) + beta
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def load_calibration(model_dir: Path) -> dict | None:
    """Load calibration parameters from a model directory.

    Reads calibration.json. Returns dict with "method" key or None.
    """
    model_dir = Path(model_dir)

    cal_path = model_dir / "calibration.json"
    if cal_path.exists():
        with open(cal_path) as f:
            return json.load(f)

    return None


def calibrate_model_multiclass(
    model_dir: Path,
    val_data_path: Path,
    method: str = "temperature",
    reg_lambda: float = 0.01,
    output_path: Path | None = None,
    device: str = "cpu",
    batch_size: int = 1024,
    num_workers: int = 0,
) -> dict:
    """
    Unified multiclass calibration pipeline.

    Supports temperature, matrix, and dirichlet scaling methods. Saves
    results to calibration.json (and legacy temperature.json for backward
    compat when method is temperature).

    Args:
        model_dir: Directory with model_best.pt and config.json
        val_data_path: Path to validation data (.npz)
        method: Calibration method ("temperature", "matrix", "dirichlet")
        reg_lambda: L2 regularization for matrix/dirichlet
        output_path: Output JSON path (default: model_dir/calibration.json)
        device: Device for inference
        batch_size: Batch size for validation pass
        num_workers: DataLoader workers

    Returns:
        Calibration result dict with method-specific params
    """
    if output_path is None:
        output_path = model_dir / "calibration.json"

    # Load model config
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    model_name = config["model_name"]
    signal_len = config["signal_len"]
    kmer_len = config["kmer_len"]
    num_out = config.get("num_out", 1)
    seq_encoding = config.get("seq_encoding", "base_onehot")
    signal_kmer_context = tuple(config.get("signal_kmer_context", (4, 4)))

    # Build model init kwargs
    signal_in_channels = config.get("signal_in_channels", 1)
    model_kwargs: dict = {
        "signal_len": signal_len,
        "kmer_len": kmer_len,
        "seq_encoding": seq_encoding,
        "signal_kmer_context": signal_kmer_context,
        "signal_in_channels": signal_in_channels,
    }
    if "num_features" in config:
        model_kwargs["num_features"] = config["num_features"]
    model_kwargs["num_out"] = num_out

    model = get_model(model_name, **model_kwargs)

    # Load weights
    checkpoint_path = model_dir / "model_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    from leech.model_loading import _migrate_state_dict_keys

    state_dict = _migrate_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    wrapper = ModelInferenceWrapper(model, model_name)

    # Load validation data. Forward dwell_template_table from training config
    # so the dataset builds a feature tensor with the same channel count the
    # model was trained on — otherwise the Conv1d in the feature branch
    # crashes with "expected N channels, but got M channels".
    # chunk_path (not pre-loaded chunks) so the dataset streams the arrays out
    # of the npz instead of holding a second full copy of them (#211).
    dwell_template_table = config.get("dwell_template_table") or None
    val_dataset = LeechDataset(
        chunk_path=val_data_path,
        model_type=model_name,
        signal_len=signal_len,
        kmer_len=kmer_len,
        seq_encoding=seq_encoding,
        signal_kmer_context=signal_kmer_context,
        signal_mode=config.get("signal_mode", "both"),
        dwell_template_table=dwell_template_table,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=resolve_val_dataloader_workers(val_dataset, num_workers, device),
    )

    logger.info(f"Collecting logits from {len(val_dataset)} validation samples...")
    logits, labels = collect_logits_multiclass(wrapper, val_loader, device)
    logits = logits.cpu()
    labels = labels.cpu()

    # Pre-calibration ECE
    pre_probs = torch.softmax(logits, dim=-1)
    pre_ece = _expected_calibration_error_multiclass(pre_probs, labels)

    # Dispatch to learn function
    if method == "temperature":
        t = learn_temperature(logits, labels)
        cal_logits = logits / t
        result: dict = {
            "method": "temperature",
            "temperature": t,
        }
    elif method == "matrix":
        W, b = learn_matrix_scaling(logits, labels, reg_lambda=reg_lambda)
        cal_logits = apply_calibration(
            logits, {"method": "matrix", "W": W.tolist(), "b": b.tolist()}
        )
        result = {
            "method": "matrix",
            "W": W.tolist(),
            "b": b.tolist(),
            "reg_lambda": reg_lambda,
        }
    elif method == "dirichlet":
        alpha, beta = learn_dirichlet_scaling(logits, labels, reg_lambda=reg_lambda)
        cal_logits = apply_calibration(
            logits, {"method": "dirichlet", "alpha": alpha.tolist(), "beta": beta.tolist()}
        )
        result = {
            "method": "dirichlet",
            "alpha": alpha.tolist(),
            "beta": beta.tolist(),
            "reg_lambda": reg_lambda,
        }
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    # Post-calibration ECE
    post_probs = torch.softmax(cal_logits, dim=-1)
    post_ece = _expected_calibration_error_multiclass(post_probs, labels)

    if post_ece >= pre_ece:
        logger.warning(
            f"Calibration ({method}) did not improve ECE: {pre_ece:.4f} -> {post_ece:.4f}."
        )
    logger.info(f"ECE: {pre_ece:.4f} -> {post_ece:.4f} (method={method})")

    result["ece_before"] = pre_ece
    result["ece_after"] = post_ece
    result["n_val_samples"] = len(val_dataset)
    result["num_classes"] = num_out

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Calibration ({method}) saved to {output_path}")

    return result


def _expected_calibration_error(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> float:
    """Compute Expected Calibration Error (ECE) for binary predictions."""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean().item()
        bin_acc = labels[mask].mean().item()
        ece += mask.sum().item() / len(probs) * abs(bin_conf - bin_acc)
    return ece


def _expected_calibration_error_multiclass(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> float:
    """Compute top-1 confidence ECE for multiclass predictions.

    Bins samples by max(softmax), measures |mean_confidence - accuracy| per bin.
    """
    confidences, preds = probs.max(dim=-1)
    accuracies = (preds == labels).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean().item()
        bin_acc = accuracies[mask].mean().item()
        ece += mask.sum().item() / n * abs(bin_conf - bin_acc)
    return ece
