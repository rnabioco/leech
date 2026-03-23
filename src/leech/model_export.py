"""Model export and serialization (torch.export and legacy TorchScript)."""

import io
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger("leech.model_export")


def _build_example_inputs(
    model: nn.Module, config: dict, batch_size: int = 1
) -> tuple[torch.Tensor, ...]:
    """Build example input tensors for model export/tracing.

    Args:
        model: The model (used to read dwell_margin for wide-feature models)
        config: Model config dict with signal_len, kmer_len, etc.
        batch_size: Batch size for example inputs (use 2 for torch.export with
            dynamic batch dims to avoid specialization to batch=1)

    Returns:
        Tuple of example input tensors
    """
    from leech.models.inference_wrapper import ModelInferenceWrapper

    signal_len = config["signal_len"]
    kmer_len = config["kmer_len"]
    seq_encoding = config.get("seq_encoding", "base_onehot")
    signal_kmer_context = config.get("signal_kmer_context", [4, 4])

    if seq_encoding == "signal_kmer":
        seq_channels = sum(signal_kmer_context) * 4 + 4
        seq_len = signal_len
    else:
        seq_channels = 4
        seq_len = kmer_len

    signal_in_channels = config.get("signal_in_channels", 1)
    if signal_in_channels > 1:
        signal = torch.randn(batch_size, signal_in_channels, signal_len)
    else:
        signal = torch.randn(batch_size, signal_len)
    sequence = torch.randn(batch_size, seq_channels, seq_len)

    model_name = config.get("model_name", "")
    requires_features = model_name in ModelInferenceWrapper.FEATURE_MODELS

    if requires_features:
        num_features = config.get("num_features", 5)
        wide_features = model_name in ModelInferenceWrapper.WIDE_FEATURE_MODELS
        if wide_features:
            # Compute feature width from config (feature_end - feature_start + 1)
            _fs = config.get("feature_start")
            _fe = config.get("feature_end")
            if _fs is not None and _fe is not None:
                feat_len = _fe - _fs + 1
            else:
                # Fallback for old configs: use model's dwell_margin
                dwell_margin = getattr(model, "dwell_margin", 0)
                feat_len = kmer_len + 2 * dwell_margin
        else:
            feat_len = kmer_len
        features = torch.randn(batch_size, num_features, feat_len)
        return (signal, sequence, features)

    return (signal, sequence)


def export_model(model: nn.Module, config: dict) -> "torch.export.ExportedProgram":
    """Export a leech model using torch.export (PyTorch 2+ export API).

    The batch dimension is marked dynamic so the exported model accepts
    any batch size at inference time.

    Args:
        model: PyTorch model (will be set to eval mode)
        config: Model config dict with signal_len, kmer_len, etc.

    Returns:
        ExportedProgram
    """
    model.eval()
    # Use batch_size=2 to avoid torch.export specializing batch dim to 1
    example_inputs = _build_example_inputs(model, config, batch_size=2)

    # Mark batch dimension (dim 0) as dynamic for all inputs
    batch_dim = torch.export.Dim("batch", min=1, max=2**15)
    dynamic_shapes = tuple({0: batch_dim} for _ in example_inputs)

    with torch.no_grad():
        ep = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)

    return ep


def serialize_exported_model(ep: "torch.export.ExportedProgram") -> bytes:
    """Serialize an exported model to bytes."""
    buf = io.BytesIO()
    torch.export.save(ep, buf)
    return buf.getvalue()


def deserialize_exported_model(data: bytes, device: str = "cpu") -> nn.Module:
    """Deserialize an exported model from bytes.

    Returns the underlying nn.Module (moved to device), not the ExportedProgram.
    """
    buf = io.BytesIO(data)
    ep = torch.export.load(buf)
    return ep.module().to(device)


def trace_model(model: nn.Module, config: dict) -> torch.jit.ScriptModule:
    """Trace a leech model into a TorchScript ScriptModule.

    .. deprecated::
        Use :func:`export_model` instead (torch.export API).
        Retained for loading legacy TorchScript bundles/exports.

    Args:
        model: PyTorch model (must be in eval mode or will be set to eval)
        config: Model config dict with signal_len, kmer_len, etc.

    Returns:
        Traced ScriptModule
    """
    model.eval()
    example_inputs = _build_example_inputs(model, config)

    with torch.no_grad():
        traced = torch.jit.trace(model, example_inputs)

    return traced


def serialize_traced_model(traced: torch.jit.ScriptModule) -> bytes:
    """Serialize a traced TorchScript model to bytes.

    .. deprecated::
        Use :func:`serialize_exported_model` instead.
    """
    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    return buf.getvalue()


def deserialize_traced_model(data: bytes, device: str = "cpu") -> torch.jit.ScriptModule:
    """Deserialize a traced TorchScript model from bytes.

    .. deprecated::
        Use :func:`deserialize_exported_model` instead.
    """
    buf = io.BytesIO(data)
    return torch.jit.load(buf, map_location=device)


def export_single_model(model_dir: Path, output_path: Path) -> Path:
    """Export a single trained model as a standalone .pt file.

    Uses ``torch.export`` (PyTorch 2+ API).  The exported file is loadable
    with ``torch.export.load()`` — no leech required.  Model config is
    embedded via ``extra_files``.

    Args:
        model_dir: Directory with config.json and model_best.pt
        output_path: Where to write the exported .pt file

    Returns:
        Path to the saved file
    """
    from leech.model_loading import load_model_from_checkpoint

    model, config = load_model_from_checkpoint(model_dir, device="cpu")
    ep = export_model(model, config)

    meta = json.dumps(config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, str(output_path), extra_files={"leech_meta.txt": meta})

    logger.info(f"Exported model to {output_path}")
    return output_path
