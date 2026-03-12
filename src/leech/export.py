"""
TorchScript export helpers for leech models.

Functions for tracing, serializing, and exporting trained models as standalone
TorchScript modules.
"""

import io
import json
import logging
from pathlib import Path

import torch

from leech.models.inference_wrapper import ModelInferenceWrapper
from leech.util import load_model_from_checkpoint

logger = logging.getLogger("leech.export")


def trace_model(model: torch.nn.Module, config: dict) -> torch.jit.ScriptModule:
    """Trace a leech model into a TorchScript ScriptModule.

    Args:
        model: PyTorch model (must be in eval mode or will be set to eval)
        config: Model config dict with signal_len, kmer_len, etc.

    Returns:
        Traced ScriptModule
    """
    model.eval()

    signal_len = config["signal_len"]
    kmer_len = config["kmer_len"]
    seq_encoding = config.get("seq_encoding", "base_onehot")
    signal_kmer_context = config.get("signal_kmer_context", [4, 4])

    # Determine sequence tensor shape based on encoding
    if seq_encoding == "signal_kmer":
        seq_channels = sum(signal_kmer_context) * 4 + 4  # 36 for default (4,4)
        seq_len = signal_len
    else:
        seq_channels = 4
        seq_len = kmer_len

    # Build example inputs
    signal = torch.randn(1, signal_len)
    sequence = torch.randn(1, seq_channels, seq_len)

    model_name = config.get("model_name", "")
    requires_features = model_name in ModelInferenceWrapper.FEATURE_MODELS

    if requires_features:
        num_features = config.get("num_features", 5)
        features = torch.randn(1, num_features, kmer_len)
        example_inputs = (signal, sequence, features)
    else:
        example_inputs = (signal, sequence)

    with torch.no_grad():
        traced = torch.jit.trace(model, example_inputs)

    return traced


def serialize_traced_model(traced: torch.jit.ScriptModule) -> bytes:
    """Serialize a traced model to bytes."""
    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    return buf.getvalue()


def deserialize_traced_model(data: bytes, device: str = "cpu") -> torch.jit.ScriptModule:
    """Deserialize a traced model from bytes."""
    buf = io.BytesIO(data)
    return torch.jit.load(buf, map_location=device)


def export_single_model(model_dir: Path, output_path: Path) -> Path:
    """Export a single trained model as a standalone TorchScript .pt file.

    The exported file is loadable with just ``torch.jit.load()`` — no leech
    required.  Model config is embedded via ``_extra_files``.

    Args:
        model_dir: Directory with config.json and model_best.pt
        output_path: Where to write the TorchScript .pt file

    Returns:
        Path to the saved file
    """
    model, config = load_model_from_checkpoint(model_dir, device="cpu")
    traced = trace_model(model, config)

    meta = json.dumps(config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(output_path), _extra_files={"leech_meta.txt": meta})

    logger.info(f"Exported TorchScript model to {output_path}")
    return output_path
