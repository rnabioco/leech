"""Config-driven neural-network layer registry for leech.

Modeled on Oxford Nanopore's bonito (``bonito/nn.py``): every layer type is
registered in a name -> class table, layers know how to serialize themselves
via ``to_dict()``, and ``from_dict()`` rebuilds an architecture from a plain
nested dict (which is what a TOML config parses to).

Composition primitives
----------------------
``Serial``
    Strictly sequential container (bonito's ``Serial``).
``Stack``
    ``Serial`` built by repeating one layer ``depth`` times.
``Parallel``
    Fans a single input out to a set of *named* branches and concatenates
    their outputs.  This is the self-contained multi-branch primitive.
``Graph``
    Flat named dataflow container.  Unlike ``Parallel`` it accepts several
    distinct model inputs (``signal``/``sequence``/``features``), routes each
    to the branches that need it, and — crucially — installs every node as a
    *direct attribute of the root module*.  That keeps ``state_dict()`` keys
    identical to the hand-written classes it replaces (``signal_branch.
    conv_layers.0.weight``, ``lstm.weight_ih_l0``, ``fc.1.weight``, ...), so
    existing checkpoints load unchanged.

Every leaf layer here either subclasses the ``torch.nn`` module it wraps or
subclasses the corresponding ``leech.models.components`` branch, so wrapping
never introduces an extra level of nesting in the state dict.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from leech.models import components

# ── Registry ────────────────────────────────────────────────────────────────

layers: dict[str, type] = {}


def register[LayerT: type](layer: LayerT) -> LayerT:
    """Register a layer class under its lower-cased class name."""
    layer.name = layer.__name__.lower()  # ty: ignore[unresolved-attribute]
    layers[layer.name] = layer  # ty: ignore[unresolved-attribute]
    return layer


def to_dict(layer: Any, include_weights: bool = False) -> dict:
    """Serialize a registered layer to a plain dict."""
    if hasattr(layer, "to_dict"):
        return {"type": layer.name, **layer.to_dict(include_weights)}
    return {"type": layer.name}


def from_dict(model_dict: Any, layer_types: dict[str, type] | None = None) -> Any:
    """Rebuild a layer (or a whole architecture) from a plain dict."""
    if not isinstance(model_dict, dict):
        # allow concrete objects to be embedded directly
        return model_dict
    model_dict = dict(model_dict)
    if layer_types is None:
        layer_types = layers
    type_name = model_dict.pop("type")
    if type_name not in layer_types:
        raise KeyError(f"Unknown layer type {type_name!r}. Known: {sorted(layer_types)}")
    typ = layer_types[type_name]
    if hasattr(typ, "from_dict"):
        return typ.from_dict(model_dict, layer_types)
    if "sublayers" in model_dict:
        sublayers = model_dict["sublayers"]
        model_dict["sublayers"] = (
            [from_dict(x, layer_types) for x in sublayers]
            if isinstance(sublayers, list)
            else from_dict(sublayers, layer_types)
        )
    try:
        return typ(**model_dict)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to build layer {type_name!r} with args {model_dict}") from exc


class _RecordsKwargs:
    """Mixin that remembers the kwargs a layer was built with, for ``to_dict``."""

    def _record(self, kwargs: dict) -> None:
        self._config_kwargs = dict(kwargs)

    def to_dict(self, include_weights: bool = False) -> dict:
        if include_weights:
            raise NotImplementedError(f"{type(self).__name__} cannot serialize weights")
        return dict(getattr(self, "_config_kwargs", {}))


# ── Conv branches (leech components) ────────────────────────────────────────


@register
class SignalBranch(components.SignalBranch, _RecordsKwargs):
    """Registered wrapper around :class:`leech.models.components.SignalBranch`."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._record(kwargs)


@register
class SequenceBranch(components.SequenceBranch, _RecordsKwargs):
    """Registered wrapper around :class:`leech.models.components.SequenceBranch`."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._record(kwargs)


@register
class FeatureBranch(components.FeatureBranch, _RecordsKwargs):
    """Registered wrapper around :class:`leech.models.components.FeatureBranch`."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._record(kwargs)


# ── Leaf layers (thin subclasses so state-dict keys stay flat) ──────────────


@register
class Linear(nn.Linear):
    def to_dict(self, include_weights: bool = False) -> dict:
        if include_weights:
            raise NotImplementedError
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias is not None,
        }


@register
class LayerNorm(nn.LayerNorm):
    def to_dict(self, include_weights: bool = False) -> dict:
        if include_weights:
            raise NotImplementedError
        return {"normalized_shape": list(self.normalized_shape), "eps": self.eps}


@register
class AdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):
    def to_dict(self, include_weights: bool = False) -> dict:
        return {"output_size": self.output_size}


@register
class BiLSTM(nn.LSTM):
    """``nn.LSTM`` that returns only the output sequence.

    Subclassing (rather than wrapping) keeps parameter names as
    ``<attr>.weight_ih_l0`` instead of ``<attr>.lstm.weight_ih_l0``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-return-type]
        out, _ = super().forward(x)
        return out

    def to_dict(self, include_weights: bool = False) -> dict:
        if include_weights:
            raise NotImplementedError
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "batch_first": self.batch_first,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
        }


@register
class CrossAttention(nn.MultiheadAttention):
    """``nn.MultiheadAttention`` returning only the attended output.

    ``need_weights=False`` enables the fused SDPA kernel.
    """

    def forward(  # ty: ignore[invalid-return-type]
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        out, _ = super().forward(query, key, value, need_weights=False)
        return out

    def to_dict(self, include_weights: bool = False) -> dict:
        if include_weights:
            raise NotImplementedError
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "kdim": self.kdim,
            "vdim": self.vdim,
            "dropout": self.dropout,
            "batch_first": self.batch_first,
        }


@register
class MLPHead(nn.Sequential):
    """Dropout -> Linear -> ReLU -> Dropout -> Linear classification head.

    Layout matches the hand-written ``fc``/``classifier`` heads exactly, so
    parameter keys stay ``fc.1.weight`` / ``fc.4.weight``.
    """

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, dropout: float = 0.0
    ) -> None:
        super().__init__(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
        )
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.dropout_p = dropout

    def to_dict(self, include_weights: bool = False) -> dict:
        if include_weights:
            raise NotImplementedError
        return {
            "in_features": self.in_features,
            "hidden_features": self.hidden_features,
            "out_features": self.out_features,
            "dropout": self.dropout_p,
        }


# ── Parameter-free plumbing ops ────────────────────────────────────────────


@register
class Concat(nn.Module):
    """Concatenate several tensors along ``dim``."""

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        return torch.cat(list(tensors), dim=self.dim)

    def to_dict(self, include_weights: bool = False) -> dict:
        return {"dim": self.dim}

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


@register
class Transpose(nn.Module):
    """Swap two tensor dimensions (e.g. ``(B, C, T) <-> (B, T, C)``)."""

    def __init__(self, dim0: int = 1, dim1: int = 2) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)

    def to_dict(self, include_weights: bool = False) -> dict:
        return {"dim0": self.dim0, "dim1": self.dim1}

    def extra_repr(self) -> str:
        return f"dim0={self.dim0}, dim1={self.dim1}"


@register
class Add(nn.Module):
    """Elementwise sum of two tensors (residual connection)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    def to_dict(self, include_weights: bool = False) -> dict:
        return {}


@register
class Index(nn.Module):
    """Pick a single position along ``dim`` (e.g. the centre base of a k-mer)."""

    def __init__(self, index: int, dim: int = 1) -> None:
        super().__init__()
        self.index = index
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.select(self.dim, self.index)

    def to_dict(self, include_weights: bool = False) -> dict:
        return {"index": self.index, "dim": self.dim}

    def extra_repr(self) -> str:
        return f"index={self.index}, dim={self.dim}"


@register
class AttentionPool(nn.Module):
    """Softmax-weighted pooling of ``values`` using per-position ``scores``.

    ``scores`` come from a separate ``Linear`` node so that the linear layer
    keeps its own top-level attribute name (``pool_linear``/``attn_linear``).
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, scores: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(scores, dim=self.dim)
        return (weights * values).sum(dim=self.dim)

    def to_dict(self, include_weights: bool = False) -> dict:
        return {"dim": self.dim}


# ── Composition primitives ─────────────────────────────────────────────────


@register
class Serial(nn.Sequential):
    """Strictly sequential container (bonito-compatible)."""

    def __init__(self, sublayers: list[nn.Module]) -> None:
        super().__init__(*sublayers)

    def to_dict(self, include_weights: bool = False) -> dict:
        return {"sublayers": [to_dict(layer, include_weights) for layer in self._modules.values()]}

    def __repr__(self) -> str:
        return nn.ModuleList.__repr__(self)  # ty: ignore[invalid-argument-type]


@register
class Stack(Serial):
    """``Serial`` formed by repeating one layer ``depth`` times."""

    @classmethod
    def from_dict(cls, model_dict: dict, layer_types: dict[str, type] | None = None) -> Stack:
        return cls(
            [from_dict(model_dict["layer"], layer_types) for _ in range(model_dict["depth"])]
        )

    def to_dict(self, include_weights: bool = False) -> dict:
        if include_weights:
            raise NotImplementedError
        layer_dicts = [to_dict(layer) for layer in self]
        for layer_dict in layer_dicts[1:]:
            assert layer_dict == layer_dicts[0], "all layers in a Stack should be identical"
        return {"layer": layer_dicts[0], "depth": len(self)}


@register
class Parallel(nn.Module):
    """Fan one input out to named branches and merge their outputs.

    This is the self-contained multi-branch primitive: it nests, so it can
    appear anywhere a layer can.  Note that nesting means branch parameters
    live under ``<parallel_attr>.<branch_name>...``; use :class:`Graph` when
    flat (checkpoint-compatible) parameter names are required.

    Args:
        sublayers: Mapping of branch name -> layer.
        mode: ``"cat"`` (default) or ``"sum"``.
        dim: Concatenation dimension when ``mode="cat"``.
    """

    def __init__(self, sublayers: dict[str, nn.Module], mode: str = "cat", dim: int = 1) -> None:
        super().__init__()
        if mode not in ("cat", "sum"):
            raise ValueError(f"Unknown Parallel mode {mode!r}; use 'cat' or 'sum'")
        self.mode = mode
        self.dim = dim
        self.branches = nn.ModuleDict(OrderedDict(sublayers))

    @classmethod
    def from_dict(cls, model_dict: dict, layer_types: dict[str, type] | None = None) -> Parallel:
        model_dict = dict(model_dict)
        sublayers = {k: from_dict(v, layer_types) for k, v in model_dict.pop("sublayers").items()}
        return cls(sublayers, **model_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches.values()]
        if self.mode == "sum":
            return torch.stack(outs, dim=0).sum(dim=0)
        return torch.cat(outs, dim=self.dim)

    def to_dict(self, include_weights: bool = False) -> dict:
        return {
            "sublayers": {k: to_dict(v, include_weights) for k, v in self.branches.items()},
            "mode": self.mode,
            "dim": self.dim,
        }


@register
class Graph(components.BaseModel):
    """Flat named dataflow container for multi-input, multi-branch models.

    Each node is ``{"name", "layer", "inputs", "out"}``:

    * ``name`` — the attribute the layer is installed under (this is what
      shows up in ``state_dict()`` keys, so it is chosen to match the
      hand-written classes being replaced).
    * ``inputs`` — names of values in the environment to feed the layer.
    * ``out`` — name to bind the result to (defaults to ``name``).  Rebinding
      an existing name (``seq_feat -> seq_feat``) makes optional nodes such as
      ``seq_pool`` drop out cleanly when their ``when`` condition is false.

    Nodes execute in declaration order, so declaration order must respect
    dependencies.  It is otherwise free, which lets configs reproduce the
    module-construction order of the original classes (and hence identical
    random initialisation for a given seed).
    """

    def __init__(
        self,
        nodes: list[dict],
        inputs: list[str] | None = None,
        output: str = "logits",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.graph_inputs = list(inputs or ["signal", "sequence", "features"])
        self.graph_output = output

        # Plain (non-module) attributes: signal_len, kmer_len, num_features, ...
        for key, value in (attrs or {}).items():
            setattr(self, key, value)

        plan: list[tuple[str, tuple[str, ...], str]] = []
        for node in nodes:
            name = node["name"]
            layer = node["layer"]
            if not isinstance(layer, nn.Module):
                raise TypeError(f"Graph node {name!r} layer must be an nn.Module, got {layer!r}")
            if name in self._modules:
                raise ValueError(f"Duplicate Graph node name {name!r}")
            setattr(self, name, layer)
            plan.append((name, tuple(node.get("inputs", [])), node.get("out", name)))
        self._plan = plan

    @classmethod
    def from_dict(cls, model_dict: dict, layer_types: dict[str, type] | None = None) -> Graph:
        model_dict = dict(model_dict)
        nodes = [
            {**node, "layer": from_dict(node["layer"], layer_types)}
            for node in model_dict.pop("nodes")
        ]
        return cls(nodes, **model_dict)

    def to_dict(self, include_weights: bool = False) -> dict:
        if include_weights:
            raise NotImplementedError
        nodes = []
        for name, in_names, out_name in self._plan:
            nodes.append(
                {
                    "name": name,
                    "inputs": list(in_names),
                    "out": out_name,
                    "layer": to_dict(getattr(self, name)),
                }
            )
        return {
            "nodes": nodes,
            "inputs": list(self.graph_inputs),
            "output": self.graph_output,
        }

    def forward(
        self,
        signal: torch.Tensor,
        sequence: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        env: dict[str, Any] = {"signal": signal, "sequence": sequence, "features": features}
        for name, in_names, out_name in self._plan:
            layer = getattr(self, name)
            env[out_name] = layer(*[env[n] for n in in_names])
        return env[self.graph_output]
