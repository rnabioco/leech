"""ONNX export, shared by the classifier arms and the CRF encoder.

`torch.export` makes a trained model loadable by anything with PyTorch, and by
nothing else. A runtime that consumes ONNX — which is what `escapepod-rs` runs
in production — cannot load one at all, and that gap has been recorded
downstream as though "not a shippable format" were a property of the models.
It is not: the graphs convert, and agree with torch to within float32 rounding.

**Use the dynamo exporter.** ``dynamo=False`` — the legacy TorchScript path, and
the obvious first thing to try — fails on leech's architectures with

    SymbolicValueError: Unsupported: ONNX export of operator
    adaptive_avg_pool1d, output size that are not factor of input size

and, for the TCN family, an ``input size not accessible`` variant of the same.
That is an exporter limitation, not a model problem, but the message reads like
one and will send whoever hits it looking in the wrong place. ``dynamo=True`` at
opset 18 exports the same modules cleanly.

Re-measured 2026-08-30, on torch 2.12, because
:class:`leech.models.components.AdaptiveAvgPool1d` no longer emits an aten
adaptive pool and that could have retired the reason. **It did not.** Both
``nn.AdaptiveAvgPool1d(7)`` and leech's replacement, on a length-100 input,
still fail ``dynamo=False`` with the message above — the replacement because
``torch.jit.trace`` turns ``.shape[-1]`` into a Tensor, which takes its
dynamic-length fallback straight back to the aten op. The paragraph stands as
written; it is now measured rather than inherited.

**But dynamo alone is not sufficient**, which is the half that was missing and
cost `charging_tcn_rna004@v0.1.0` a release nothing could run
(rnabioco/escapepod-models#96). Two of its graphs' properties were unloadable
by tract, the ONNX runtime `escpod` links statically:

* ``adaptive_avg_pool1d`` with a non-dividing output size, which dynamo
  open-codes as a rank-8 ``GatherND``. Fixed in the model, by
  :class:`leech.models.components.AdaptiveAvgPool1d`.
* the ``value_info`` dynamo writes for every intermediate, carrying the batch
  axis as a symbol. Fixed here, by :func:`strip_value_info`, which
  :func:`export_onnx` now always calls.

Neither is fixable in the consumer, and neither shows up in
:func:`verify_onnx` — onnxruntime loads both graphs happily. "It exports and
round-trips" is a weaker claim than "a runtime can load it", and only the
second one ships.

What a graph cannot carry
-------------------------
Two things a consumer needs and cannot recover from the ONNX file, so both are
written beside it:

* **Which input is which.** Arity and channel counts are visible; roles are not,
  and ``seq_channels = sum(signal_kmer_context) * 4 + 4`` is not something a
  consumer should have to re-derive.
* **What the output means.** leech classifiers emit a single BCE logit ``(N, 1)``,
  not a two-class softmax. Read as the latter, every call is wrong and nothing
  errors.

And one thing the graph does not contain at all: for ``seq_encoding="signal_kmer"``
the 36-channel sequence input is :func:`leech.features.encode_signal_kmer`
output, computed in the *dataset* from the base-to-signal map. A non-Python
runtime must produce it before it can call the model — see
:func:`describe_inputs`, which names the parameters it needs rather than
leaving them to be guessed.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "OPSET",
    "InputSpec",
    "contract",
    "describe_inputs",
    "export_onnx",
    "strip_value_info",
    "verify_onnx",
]

logger = logging.getLogger("leech.onnx_export")

#: Opset the dynamo exporter is known to handle for these architectures.
OPSET = 18


@dataclass(frozen=True)
class InputSpec:
    """One graph input: what it is called, what shape it takes, and what it *is*."""

    name: str
    shape: list[Any]
    dtype: str
    role: str
    produced_by: str | None = None
    """How a consumer obtains this input, when it is not raw signal. ``None``
    means "the caller already has it"."""


def describe_inputs(config: dict, example_inputs: tuple) -> list[InputSpec]:
    """Name and explain each input of a classifier graph.

    Shapes come from the tensors actually exported, so they cannot drift from
    the graph; the *roles* come from the config, which is the only place they
    exist.
    """
    seq_encoding = config.get("seq_encoding", "base_onehot")
    kmer_context = config.get("signal_kmer_context", [4, 4])

    if seq_encoding == "signal_kmer":
        produced_by = (
            "leech.features.encode_signal_kmer(signal, seq_to_sig_map, sequence, "
            f"context={list(kmer_context)}) — a scatter of the one-hot k-mer context "
            "along the signal axis. It lives in the dataset, not the model, so it is "
            "NOT in this graph — a consumer must produce it before calling the "
            "model. The rule is owned by escapepod-signal "
            "(`escapepod_signal::seq_encoding::encode_signal_kmer`), which leech "
            "calls rather than duplicates; a Rust consumer should link that crate, "
            "and a Python one can call leech-core's `_rs_encode_signal_kmer`, which "
            "is the same code. Do not reimplement it — two definitions of this rule "
            "diverge silently."
        )
    else:
        produced_by = "one-hot encoding of the k-mer context, 4 channels"

    roles = [
        ("signal", "normalised raw signal", None),
        ("sequence", f"sequence context, seq_encoding={seq_encoding!r}", produced_by),
        ("features", "per-base dwell and level statistics", None),
    ]
    specs = []
    for tensor, (name, role, made) in zip(example_inputs, roles, strict=False):
        specs.append(
            InputSpec(
                name=name,
                shape=["batch", *list(tensor.shape[1:])],
                dtype=str(tensor.dtype).removeprefix("torch."),
                role=role,
                produced_by=made,
            )
        )
    return specs


def export_onnx(
    model,
    example_inputs: tuple,
    path: str | Path,
    *,
    input_names: list[str],
    output_names: list[str],
    opset: int = OPSET,
    dynamic_axes: dict | None = None,
) -> Path:
    """Write ``model`` to ``path`` as ONNX. Batch is dynamic by default.

    Always uses the dynamo exporter — see the module docstring for why the
    legacy path is not an option here, and why its error message is misleading.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if dynamic_axes is None:
        dynamic_axes = {n: {0: "batch"} for n in input_names}
        dynamic_axes |= {n: {0: "batch"} for n in output_names}

    model.eval()
    torch.onnx.export(
        model,
        example_inputs,
        str(path),
        dynamo=True,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    dropped = strip_value_info(path)
    logger.info(
        "wrote %s (%.2f MB, opset %d, %d value_info entries dropped)",
        path,
        path.stat().st_size / 1e6,
        opset,
        dropped,
    )
    return path


def strip_value_info(path: str | Path) -> int:
    """Drop the graph's inferred intermediate shapes. Returns how many went.

    ``value_info`` is optional: it records the shapes the *exporter* inferred
    for intermediate tensors, and every runtime re-infers them anyway. The
    dynamo exporter writes one entry per intermediate — 667 of them for the
    charging TCN — and writes the batch axis into them as the **symbol**
    ``batch``, because that is what ``dynamic_axes`` asked for.

    That is a contradiction waiting for a consumer that pins the batch.
    ``escapepod_classify`` loads every graph with
    ``with_input_fact(0, f32::fact([1, ...]))``, and tract then has to unify a
    declared ``Sym(batch)`` with the pinned ``Val(1)``, which it cannot:

        Failed analyse for node "node_conv1d" ConvHir: Unifying shapes
        batch,64,390 and 1,64,390: Impossible to unify Sym(batch) with Val(1)

    — at the *first convolution*, nowhere near anything interesting. Removing
    the entries lets tract infer from the pinned input and the graph loads.
    Nothing needs them: onnxruntime re-infers, ``onnx.checker`` is satisfied,
    and the legacy TorchScript exporter never wrote them in the first place,
    which is why the graphs it produced (``charging_feature_nn_rna004@v0.1.0``)
    always loaded. See rnabioco/escapepod-models#96.

    Initializers are left exactly as they are, external-data references
    included: the proto is read without resolving them and written straight
    back, so an ``.onnx.data`` sidecar keeps working.
    """
    import onnx

    path = Path(path)
    proto = onnx.load(str(path), load_external_data=False)
    n = len(proto.graph.value_info)
    del proto.graph.value_info[:]
    onnx.save(proto, str(path))
    return n


def verify_onnx(
    path: str | Path,
    model,
    example_inputs: tuple,
    *,
    input_names: list[str] | None = None,
    check_model: bool = True,
) -> float:
    """Run the exported graph against torch and return the max absolute difference.

    Exporting across a serialization boundary and comparing is what catches
    what an in-process assert cannot. Both production arms measured here land
    at 4.77e-07 and 1.19e-06 against a float32 eps of 1.19e-07 — rounding, not
    disagreement.
    """
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    path = Path(path)
    if check_model:
        onnx.checker.check_model(onnx.load(str(path)))

    # Pin the thread pool. onnxruntime's default affinity call fails inside a
    # cgroup-restricted allocation (any SLURM job) and floods stderr without
    # affecting results.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    names = input_names or [i.name for i in session.get_inputs()]
    feed = {n: t.detach().cpu().numpy() for n, t in zip(names, example_inputs, strict=True)}
    got = session.run(None, feed)[0]

    model.eval()
    with torch.no_grad():
        want = model(*example_inputs)
    if isinstance(want, tuple):
        want = want[0]

    diff = float(np.abs(np.asarray(got) - want.detach().cpu().numpy()).max())
    logger.info("onnxruntime vs torch: max abs diff %.3e", diff)
    return diff


def contract(
    inputs: list[InputSpec],
    outputs: list[dict[str, Any]],
    *,
    opset: int = OPSET,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The sidecar a consumer reads to call the graph correctly."""
    return {
        "format": "onnx",
        "opset": opset,
        "inputs": [asdict(spec) for spec in inputs],
        "outputs": outputs,
        **(extra or {}),
    }
