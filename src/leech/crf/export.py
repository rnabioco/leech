"""Export a trained CRF **encoder** to ONNX, for native (Rust) inference.

The split is not arbitrary. The encoder — convolutions, LSTMs, a linear and the
blank splice — is ordinary torch and exports cleanly; it emits the transition
scores. The **decode is what standard ONNX ops cannot express**, which is why
the consumer owns it: `escapepod-demux`'s `crf/lattice.rs` runs a two-pass
Viterbi over 256 states x 5 transitions, with AVX2/AVX-512 and CUDA backends.
So the export target is the encoder alone.

The contract
------------
::

    input   signal  [batch, 1, chunk]          float32, BATCH-major, standardised raw pA
    output  scores  [chunk // stride, batch, n_score]  float32, TIME-major

The output being time-major is the axis-order trap: the boundary CNN in the same
stack is batch-major ``[B, 2, L]``, so a consumer that reuses that assumption
silently transposes every call instead of failing. It needs its own load-time
shape probe.

The sidecar exists because the constants live nowhere else
----------------------------------------------------------
``metadata.json`` carries the standardisation mean and stdev. Those are **not**
in the architecture config — whose ``[standardisation]`` block, where one exists
upstream, holds values for a different signal region entirely — and **not** in
the checkpoint: :class:`~leech.crf.training.CrfTrainer` derives them from the
corpus. A consumer holding only weights and a config cannot standardise
correctly and gets a silently degraded decode, so shipping them alongside is
what makes the artifact self-contained.

The emitted references are the other half. ``state_len`` means the model emits
``target[state_len:]``, so a consumer matching decodes against full-length
targets calls the same sequence but inflates every edit distance and compresses
the confidence margin ranking depends on. Pass ``references=`` and this writes
what the model actually emits, computed once from the ``state_len`` the encoder
declares, so no caller can supply the wrong thing by hand.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

__all__ = ["export_crf_onnx", "load_training_sidecar"]

logger = logging.getLogger("leech.crf.export")


def load_training_sidecar(path: str | Path) -> dict[str, Any]:
    """Read a ``model.json`` written by :class:`~leech.crf.training.CrfTrainer`.

    Preferred over passing standardisation by hand: the trainer records what it
    actually derived alongside the geometry it trained at, so reading them
    together is the only way an export cannot drift from its weights.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "model.json"
    if not path.is_file():
        raise FileNotFoundError(f"training sidecar not found: {path}")
    return json.loads(path.read_text())


def export_crf_onnx(
    checkpoint: str | Path,
    out_dir: str | Path,
    *,
    sidecar: dict[str, Any] | str | Path | None = None,
    arch_config: dict | None = None,
    mean: float | None = None,
    std: float | None = None,
    chunk: int | None = None,
    references: dict[str, str] | None = None,
    verify: bool = True,
    opset: int | None = None,
) -> Path:
    """Export ``checkpoint`` to ``<out_dir>/crf_encoder.onnx`` plus ``metadata.json``.

    Args:
        checkpoint: a ``model.pt`` (or the directory holding one).
        out_dir: written to; created if absent.
        sidecar: the trainer's ``model.json``, or the directory holding it.
            Supplies standardisation and geometry — strongly preferred over
            passing them separately.
        arch_config: architecture config; defaults to the packaged one.
        mean: standardisation mean; overrides the sidecar. Required if absent.
        std: standardisation stdev; overrides the sidecar. Required if absent.
        chunk: window width in samples; overrides the sidecar.
        references: ``{name: full_length_target}``. Written as what the model
            *emits* (``target[state_len:]``), never as given.
        verify: run the graph against torch and record the difference.
        opset: ONNX opset; defaults to :data:`leech.onnx_export.OPSET`.

    Returns:
        Path to the written ``crf_encoder.onnx``.
    """
    import torch

    from leech.onnx_export import OPSET, export_onnx, verify_onnx

    from .config import load_config
    from .encoder import CrfEncoder, encoder_config_from_toml, load_crf_state_dict

    opset = opset or OPSET
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.pt"
    out_dir = Path(out_dir)

    if sidecar is not None and not isinstance(sidecar, dict):
        sidecar = load_training_sidecar(sidecar)
    side: dict[str, Any] = sidecar or {}

    mean = mean if mean is not None else side.get("mean")
    std = std if std is not None else side.get("std")
    if mean is None or std is None:
        raise ValueError(
            "standardisation (mean, std) is required and was not supplied. It is "
            "in neither the architecture config nor the checkpoint — the trainer "
            "derives it from the corpus and records it in model.json. Pass "
            "sidecar=<run dir>, or mean= and std= explicitly. A consumer without "
            "them cannot standardise and decodes silently worse."
        )

    cfg = encoder_config_from_toml(arch_config if arch_config is not None else load_config())
    chunk = chunk or side.get("chunk") or cfg.chunk
    from dataclasses import replace

    cfg = replace(cfg, chunk=chunk)

    model = CrfEncoder(cfg)
    load_crf_state_dict(model, torch.load(checkpoint, map_location="cpu"))
    model.eval()

    example = (torch.randn(1, 1, cfg.chunk),)
    onnx_path = out_dir / "crf_encoder.onnx"
    export_onnx(
        model,
        example,
        onnx_path,
        input_names=["signal"],
        output_names=["scores"],
        opset=opset,
        # Batch is axis 1 of the OUTPUT, not axis 0: the graph is time-major.
        dynamic_axes={"signal": {0: "batch"}, "scores": {1: "batch"}},
    )

    max_diff = None
    if verify:
        max_diff = verify_onnx(onnx_path, model, example, input_names=["signal"])

    state_len = cfg.state_len
    emitted = (
        {name: target[state_len:] for name, target in references.items()} if references else None
    )

    metadata: dict[str, Any] = {
        "format": "onnx",
        "opset": opset,
        "signal": {
            "chunk": cfg.chunk,
            "stride": cfg.stride,
            "window": "[anchor_end - chunk, anchor_end]",
            "layout": "[batch, 1, chunk] float32, batch-major, standardised raw pA",
        },
        "standardisation": {
            "mean": float(mean),
            "stdev": float(std),
            "source": "derived from the training corpus; in neither the config nor "
            "the checkpoint, so a consumer must read it here",
        },
        "crf": {
            "state_len": state_len,
            "n_base": cfg.n_base,
            "n_states": cfg.n_states,
            "n_score": cfg.n_score,
            "scores_per_state": cfg.n_base + 1,
            "blank_score": cfg.blank_score,
            "blank_index": 0,
            "score_index": "state * (n_base + 1) + label, label 0 = stay",
            "output_layout": "[chunk // stride, batch, n_score] float32, TIME-major "
            "— the opposite of the boundary CNN's batch-major [B, 2, L] in the same "
            "stack, so this needs its own load-time shape probe",
            "decode": "two-pass: log-semiring posteriors, then max-semiring over "
            "log(posteriors + 1e-8). Not expressible in standard ONNX ops, so it "
            "stays in the consumer. A one-pass Viterbi over these scores is a "
            "different and worse decode.",
            "emits": f"target[{state_len}:] — the first {state_len} bases only fix "
            f"the initial state and are never emitted, at any window width",
        },
    }
    if emitted is not None:
        metadata["references"] = emitted
        metadata["references_are"] = (
            f"what the model EMITS (target[{state_len}:]), not the full-length "
            f"targets. Matching full-length calls the same sequence but inflates "
            f"every edit distance and compresses the confidence margin."
        )
    if max_diff is not None:
        metadata["verification"] = {
            "onnxruntime_vs_torch_max_abs_diff": max_diff,
            "float32_eps": 1.1920929e-07,
        }
    for key in ("selected_epoch", "selected_loss", "seed", "corpus", "target_len"):
        if key in side:
            metadata.setdefault("training", {})[key] = side[key]

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("wrote %s and metadata.json", onnx_path)
    return onnx_path
