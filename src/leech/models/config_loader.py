"""Load model architectures declared in TOML config files.

A config declares one fully-declarative architecture — a list of nodes wired
into a :class:`leech.models.nn.Graph` — plus a ``[[variants]]`` block per
registry name it serves.  Variants pin parameters that are not user-facing
(``has_features``, ``norm_type``, ...), which is what removes the
normalisation/pooling subclass explosion: adding a variant is a config entry,
not a class.

:func:`build_model_class` turns a declaration into a real ``type``, so
``MODEL_REGISTRY[name]`` still yields a class: ``isinstance()`` works and
``inspect.signature()`` returns the declared parameters (needed by
``leech.model_loading._instantiate_model``).

This module is deliberately torch-free at import time.  Discovering and
parsing configs (used to render CLI ``--model`` choices) touches only
``tomllib`` and ``leech.constants``; ``torch`` is imported lazily inside
:func:`build_model_class`.
"""

from __future__ import annotations

import inspect
import re
import tomllib
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

from leech import constants

CONFIG_DIR = Path(__file__).parent / "configs"

# ``"${expr}"`` -> evaluate ``expr``;  ``"$name"`` -> look up ``name``.
_EXPR_RE = re.compile(r"^\$\{(?P<expr>.*)\}$", re.DOTALL)
_NAME_RE = re.compile(r"^\$(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")

# Constants exposed to config expressions (DEFAULT_SIGNAL_LEN, ...).
_CONSTANTS = {k: v for k, v in vars(constants).items() if k.isupper()}

_SAFE_BUILTINS: dict[str, Any] = {
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "len": len,
    "list": list,
    "sum": sum,
    "abs": abs,
    "range": range,
}


# ── Expression resolution ──────────────────────────────────────────────────


def _eval(expr: str, env: dict[str, Any]) -> Any:
    namespace = {**_CONSTANTS, **_SAFE_BUILTINS, **env}
    try:
        return eval(expr, {"__builtins__": {}}, namespace)  # noqa: S307
    except Exception as exc:
        raise ValueError(f"Failed to evaluate config expression {expr!r}: {exc}") from exc


def resolve(value: Any, env: dict[str, Any]) -> Any:
    """Recursively resolve ``${...}`` / ``$name`` placeholders in a config value."""
    if isinstance(value, str):
        match = _EXPR_RE.match(value)
        if match:
            return _eval(match.group("expr"), env)
        match = _NAME_RE.match(value)
        if match:
            return _eval(match.group("name"), env)
        return value
    if isinstance(value, list):
        return [resolve(v, env) for v in value]
    if isinstance(value, dict):
        return {k: resolve(v, env) for k, v in value.items()}
    return value


# ── Config discovery ───────────────────────────────────────────────────────


@cache
def _parse(path: str) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


@lru_cache(maxsize=1)
def discover_configs() -> dict[str, tuple[str, str | None]]:
    """Map model name -> (config path, variant name).

    Torch-free: only reads and parses TOML.
    """
    found: dict[str, tuple[str, str | None]] = {}
    if not CONFIG_DIR.is_dir():
        return found
    for path in sorted(CONFIG_DIR.glob("*.toml")):
        doc = _parse(str(path))
        variants = doc.get("variants")
        if variants:
            for variant in variants:
                found[variant["name"]] = (str(path), variant["name"])
        elif "name" in doc:
            found[doc["name"]] = (str(path), None)
        else:
            raise ValueError(f"Config {path} declares neither 'name' nor '[[variants]]'")
    return found


def _variant_doc(path: str, variant_name: str | None) -> tuple[dict, dict, str]:
    """Return ``(doc, variant_params, docstring)`` for a config entry."""
    doc = _parse(path)
    if variant_name is None:
        return doc, {}, doc.get("doc", "")
    for variant in doc.get("variants", []):
        if variant["name"] == variant_name:
            return doc, dict(variant.get("params", {})), variant.get("doc", doc.get("doc", ""))
    raise KeyError(f"Variant {variant_name!r} not found in {path}")


# ── Parameter resolution ───────────────────────────────────────────────────


def resolve_params(doc: dict, fixed: dict, overrides: dict) -> dict:
    """Resolve ``[params]`` then ``[derived]`` into a concrete environment.

    ``fixed`` holds variant-level params (not user-overridable); ``overrides``
    holds user kwargs passed to ``get_model()``.
    """
    declared = doc.get("params", {})
    unknown = set(overrides) - set(declared)
    if unknown:
        raise TypeError(
            f"{doc.get('name', 'model')} got unexpected keyword argument(s): "
            f"{', '.join(sorted(unknown))}"
        )

    env: dict[str, Any] = {}
    for key, default in declared.items():
        if key in overrides:
            env[key] = overrides[key]
        elif key in fixed:
            env[key] = resolve(fixed[key], env)
        else:
            env[key] = resolve(default, env)
    # Variant params that are not user-facing (has_features, norm_type, ...)
    for key, value in fixed.items():
        if key not in declared:
            env[key] = resolve(value, env)
    for key, expr in doc.get("derived", {}).items():
        env[key] = resolve(expr, env)
    return env


def _signature(doc: dict, fixed: dict) -> inspect.Signature:
    """Build an ``inspect.Signature`` from the config's user-facing params."""
    env = resolve_params(doc, fixed, {})
    params = [
        inspect.Parameter(
            name, inspect.Parameter.KEYWORD_ONLY, default=env.get(name), annotation=Any
        )
        for name in doc.get("params", {})
    ]
    return inspect.Signature(
        [inspect.Parameter("self", inspect.Parameter.POSITIONAL_ONLY), *params]
    )


# ── Class construction ─────────────────────────────────────────────────────


def _build_graph_class(name: str, doc: dict, fixed: dict, docstring: str) -> type:
    from leech.models import nn as leech_nn  # torch import happens here

    def __init__(self, **kwargs: Any) -> None:  # noqa: N807
        env = resolve_params(doc, fixed, kwargs)
        specs = [
            spec
            for spec in doc.get("nodes", [])
            if spec.get("when") is None or resolve(spec["when"], env)
        ]
        by_name = {spec["name"]: spec for spec in specs}

        # Layers are *constructed* in build_order (default: declaration order),
        # which is what fixes both the RNG draw order and the state_dict key
        # order.  Execution order stays declaration order.
        declared_order = doc.get("build_order")
        build_order = (
            [node for node in declared_order if node in by_name]
            if declared_order
            else list(by_name)
        )
        missing = sorted(set(by_name) - set(build_order))
        if missing:
            raise ValueError(f"build_order for {name} omits node(s): {missing}")
        layers = {
            node_name: leech_nn.from_dict(
                {
                    "type": by_name[node_name]["layer"],
                    **resolve(by_name[node_name].get("args", {}), env),
                }
            )
            for node_name in build_order
        }

        nodes = [
            {
                "name": spec["name"],
                "layer": layers[spec["name"]],
                "inputs": resolve(spec.get("inputs", []), env),
                "out": spec.get("out", spec["name"]),
            }
            for spec in specs
        ]
        attrs = {k: env[k] for k in doc.get("params", {})}
        attrs.update({k: env[k] for k in doc.get("expose", [])})
        leech_nn.Graph.__init__(
            self,
            nodes=nodes,
            inputs=doc.get("inputs", ["signal", "sequence", "features"]),
            output=doc.get("output", "logits"),
            attrs=attrs,
            build_order=build_order if declared_order else None,
        )

    namespace: dict[str, Any] = {
        "__init__": __init__,
        "__doc__": docstring or f"{name} (config-driven architecture).",
        "__module__": "leech.models",
        "__signature__": _signature(doc, fixed),
        "_leech_config": (doc, fixed),
    }
    return type(name, (leech_nn.Graph,), namespace)


@cache
def build_model_class(name: str) -> type:
    """Build (and cache) the model class for a TOML-declared architecture."""
    specs = discover_configs()
    if name not in specs:
        raise KeyError(name)
    path, variant_name = specs[name]
    doc, fixed, docstring = _variant_doc(path, variant_name)
    kind = doc.get("kind", "graph")
    if kind != "graph":
        raise ValueError(f"Unknown config kind {kind!r} in {path}")
    return _build_graph_class(name, doc, fixed, docstring)
