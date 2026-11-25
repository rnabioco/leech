"""Label-divergence guard (issue #402, Part A).

In a library whose classes differ by chemistry (two adapters that share a
common arm and then diverge), the base sequence past the divergence *is* the
label. A model whose inputs reach that far can read the class straight off the
reference slice, and scores as if it had learned something about charging.

The user declares where the divergence starts, ``P``, counted from the first
base of ``--motif`` (0-based). Leech derives the focus-relative position
``D = P - motif_offset`` -- the first label-defining base, in the same
coordinates as ``--feature-end`` -- and refuses any window in which a present
input can carry the identity of a base at offset ``>= D``.

Never hardcode ``P``: it is a property of the library, declared per experiment,
and a focus-relative number silently shifts by one base when ``--motif-offset``
changes, which is why the declaration is motif-relative.

Reach table (the definition; ``L`` is the k-mer lookahead, read from the
refiner and never hardcoded):

========================  =================================================
input                     reach (largest focus-relative offset it can carry)
========================  =================================================
``features`` (non k-mer)  ``feature_end`` as resolved
``kmer_features``         ``feature_end + L`` (channel stored, table present)
``signal``                ``R`` exactly for ``--signal-context-bases``;
                          ``ceil(theta + R / s) - 1`` for sample windows
``signal_residual``       ``signal + L`` (refinement on)
``sequence``              ``kmer_context`` (0 under ``--mask-seq-side right``)
``sequence_kmer``         ``signal + signal_kmer right context`` (0 when masked)
========================  =================================================

The boundary is exclusive: an input reaching ``D - 1`` passes.

This module is imported by the Snakefile at parse time, so it must stay
torch-free: stdlib, :mod:`leech.constants`, :mod:`leech.context_grid` and
:func:`leech.chunking.resolve_feature_window` only. It is deliberately not in
the package's lazy exports (docs CI collects statically).
"""

from __future__ import annotations

import functools
import gzip
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from leech.constants import (
    ALL_FEATURE_CHANNELS,
    DEFAULT_FEATURE_CHANNELS,
    DEFAULT_KMER_CONTEXT,
    DEFAULT_MIN_SAMPLES_PER_BASE,
    DEFAULT_SIGNAL_CONTEXT,
    DEFAULT_SIGNAL_KMER_CONTEXT,
    KMER_RESIDUAL_FEATURE_CHANNELS,
)

STATUS_CLEAR = "clear"
STATUS_OVERRIDE = "override"
STATUS_UNDECLARED = "undeclared"

#: Fraction of a base the window origin sits from the focus base's start, by
#: ``--base-justify``: the focus base's own signal falls ``theta`` of the way
#: into the window.
_JUSTIFY_THETA = {"start": Fraction(0), "center": Fraction(1, 2), "end": Fraction(1)}

#: Order the inputs are reported in.
REACH_KEYS = (
    "features",
    "kmer_features",
    "signal",
    "signal_residual",
    "sequence",
    "sequence_kmer",
)

_OVERRIDE_FLAG = "--allow-window-past-divergence"
_POSITION_FLAG = "--label-divergence-position"


def focus_offset(position: int, motif_offset: int) -> int:
    """Focus-relative first label-defining base, ``D = P - motif_offset``."""
    return int(position) - int(motif_offset)


def table_kmer_len(table_path: Path) -> int:
    """k-mer length of a level table, read from its first data row."""
    opener = gzip.open if str(table_path).endswith(".gz") else open
    with opener(table_path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 2:
                continue
            try:
                float(parts[1])
            except ValueError:
                continue  # header row
            return len(parts[0])
    raise ValueError(f"no k-mer rows found in {table_path}")


def refiner_lookahead(kmer_len: int, center_idx: int = -1) -> int:
    """Bases past a position that its expected level still depends on.

    ``kmer_len - 1 - center_idx``, with the same ``center_idx < 0`` default as
    ``SigMapRefiner`` (``kmer_len // 2``).
    """
    if center_idx < 0:
        center_idx = kmer_len // 2
    return kmer_len - 1 - center_idx


@functools.lru_cache(maxsize=1)
def bundled_lookahead() -> int:
    """Lookahead ``L`` of the bundled k-mer table (4 for the 9-mer table)."""
    from leech.data import get_kmer_table

    return refiner_lookahead(table_kmer_len(get_kmer_table()))


def validate_declaration(
    position: int,
    *,
    motif: str | None,
    motif_offset: int,
    focus_tsv: Any = None,
) -> int:
    """Check a declaration is well-formed and return its focus-relative ``D``."""
    if focus_tsv is not None:
        raise ValueError(
            f"{_POSITION_FLAG} cannot be combined with --focus-tsv: a focus TSV anchors "
            "chunks at signal samples, not at a motif, so the declared position has "
            "nothing to be relative to"
        )
    if not motif:
        raise ValueError(
            f"{_POSITION_FLAG} is counted from the first base of --motif; pass --motif"
        )
    d = focus_offset(position, motif_offset)
    if d < 1:
        raise ValueError(
            f"{_POSITION_FLAG} {position} with motif {motif!r} and motif offset "
            f"{motif_offset} puts the first label-defining base at focus-relative "
            f"{d:+d}; it must be >= +1 (downstream of the focus base). The position is "
            "0-based from the first base of the motif."
        )
    return d


def resolve_feature_channels(channels: Sequence[str] | None) -> list[str]:
    """The channel list a prepare run stores: ``None`` is the default twelve."""
    if not channels:
        return list(DEFAULT_FEATURE_CHANNELS)
    if len(channels) == 1 and str(channels[0]).lower() == "all":
        return list(ALL_FEATURE_CHANNELS)
    return [str(c) for c in channels]


def signal_reach(
    *,
    base_justify: str = "center",
    signal_context: Sequence[int] = DEFAULT_SIGNAL_CONTEXT,
    signal_context_bases: Sequence[int] | None = None,
    rate_floor: float = DEFAULT_MIN_SAMPLES_PER_BASE,
) -> int:
    """Largest focus-relative base whose signal the raw-signal window can cover."""
    if signal_context_bases is not None:
        return int(signal_context_bases[1])
    if rate_floor <= 0:
        raise ValueError(f"rate_floor must be positive, got {rate_floor}")
    theta = _JUSTIFY_THETA[base_justify]
    return math.ceil(theta + Fraction(int(signal_context[1])) / Fraction(rate_floor)) - 1


def compute_reach(
    *,
    feature_start: int | None = None,
    feature_end: int | None = None,
    kmer_context: int = DEFAULT_KMER_CONTEXT,
    base_justify: str = "center",
    signal_context: Sequence[int] = DEFAULT_SIGNAL_CONTEXT,
    signal_context_bases: Sequence[int] | None = None,
    lookahead: int | None = None,
    feature_channels: Sequence[str] | None = None,
    mask_seq_side: str | None = None,
    rate_floor: float = DEFAULT_MIN_SAMPLES_PER_BASE,
    signal_kmer_right: int = DEFAULT_SIGNAL_KMER_CONTEXT[1],
) -> dict[str, int]:
    """Reach of every input present in a corpus prepared with these settings.

    ``lookahead`` is the refiner's ``L``, or ``None`` when no refiner (k-mer
    table) is configured -- in which case the k-mer feature channels are not
    stored and there is no ``signal_residual``. Keys absent from the result are
    inputs that are not present.
    """
    from leech.chunking import resolve_feature_window

    _, feat_end, _ = resolve_feature_window(feature_start, feature_end, kmer_context)
    channels = resolve_feature_channels(feature_channels)
    sig = signal_reach(
        base_justify=base_justify,
        signal_context=signal_context,
        signal_context_bases=signal_context_bases,
        rate_floor=rate_floor,
    )
    reach: dict[str, int] = {}
    if any(c not in KMER_RESIDUAL_FEATURE_CHANNELS for c in channels):
        reach["features"] = feat_end
    if lookahead is not None and any(c in KMER_RESIDUAL_FEATURE_CHANNELS for c in channels):
        reach["kmer_features"] = feat_end + lookahead
    reach["signal"] = sig
    if lookahead is not None:
        reach["signal_residual"] = sig + lookahead
    # Masking the 3' side blanks the sequence branches downstream of the focus
    # base; it does not touch the k-mer feature channels or the residual,
    # which are computed from the same reference slice.
    masked = mask_seq_side == "right"
    reach["sequence"] = 0 if masked else kmer_context
    reach["sequence_kmer"] = 0 if masked else sig + signal_kmer_right
    return {k: reach[k] for k in REACH_KEYS if k in reach}


def format_refusal(
    *,
    position: int,
    motif: str | None,
    motif_offset: int,
    d: int,
    reach: Mapping[str, int],
    offenders: Mapping[str, int],
    rate_floor: float,
    sample_window: bool,
    prefix: str = "",
) -> str:
    """The message naming P, motif, offset, D, each offender, and both ways out."""
    lines = [
        f"{prefix}Window reaches the label-divergence point: declared position {position} "
        f"from the start of motif {motif!r} at motif offset {motif_offset} puts the first "
        f"label-defining base at focus-relative +{d}, and these inputs reach it:",
    ]
    for key, value in offenders.items():
        lines.append(f"  - {key}: reach +{value} (>= +{d})")
    if sample_window:
        lines.append(
            f"The signal window is sample-defined, so its reach is bounded with a rate floor "
            f"of {rate_floor:g} samples/base (a read no faster than that); use "
            "--signal-context-bases for an exact bound."
        )
    lines.append(
        "A model that reads these bases can score by recognising the adapter, not the "
        "chemistry it is meant to classify. Either shorten the window so every input reaches "
        f"at most +{d - 1} (for example a smaller --feature-end or signal right context, or "
        f"exclude the k-mer feature channels), or pass {_OVERRIDE_FLAG} to proceed knowingly; "
        "the corpus is then recorded as status 'override'."
    )
    return "\n".join(lines)


def evaluate(
    position: int,
    *,
    motif: str | None,
    motif_offset: int = 0,
    allow_override: bool = False,
    focus_tsv: Any = None,
    rate_floor: float = DEFAULT_MIN_SAMPLES_PER_BASE,
    error_prefix: str = "",
    **window: Any,
) -> dict[str, Any]:
    """Check a window against a declaration and return the sidecar block.

    ``window`` is whatever :func:`compute_reach` takes. Raises ``ValueError``
    when the declaration is malformed or an input reaches ``D`` and the
    override was not given.
    """
    d = validate_declaration(position, motif=motif, motif_offset=motif_offset, focus_tsv=focus_tsv)
    reach = compute_reach(rate_floor=rate_floor, **window)
    offenders = {k: v for k, v in reach.items() if v >= d}
    status = STATUS_CLEAR
    if offenders:
        if not allow_override:
            raise ValueError(
                format_refusal(
                    position=position,
                    motif=motif,
                    motif_offset=motif_offset,
                    d=d,
                    reach=reach,
                    offenders=offenders,
                    rate_floor=rate_floor,
                    sample_window=window.get("signal_context_bases") is None,
                    prefix=error_prefix,
                )
            )
        status = STATUS_OVERRIDE
    return {
        "position": int(position),
        "focus_offset": d,
        "motif": motif,
        "motif_offset": int(motif_offset),
        "rate_floor": float(rate_floor),
        "reach": reach,
        "status": status,
    }


def leakage_summary(block: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """``{status, position, focus_offset}`` for ``metrics["leakage"]``, or None."""
    if not block:
        return None
    return {
        "status": block.get("status", STATUS_CLEAR),
        "position": block.get("position"),
        "focus_offset": block.get("focus_offset"),
    }


# ---------------------------------------------------------------------------
# Pipeline pre-flight
# ---------------------------------------------------------------------------

CONFIG_POSITION_KEY = "label_divergence_position"
CONFIG_OVERRIDE_KEY = "allow_window_past_divergence"


def _as_int_list(value: Any) -> list[int]:
    from leech.context_grid import parse_values

    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return parse_values(str(value))


def grid_right_contexts(config: Mapping[str, Any]) -> list[int]:
    """Right contexts the pipeline's grid search will sweep, as the rules pass them."""
    grid = config.get("grid_search") or {}
    if not isinstance(grid, Mapping):
        grid = {}
    if grid.get("right_contexts") is not None:
        return _as_int_list(grid["right_contexts"])
    from leech.context_grid import parse_context_grid

    spec = grid.get("context_grid", "200:1000:200")
    if isinstance(spec, (list, tuple)):
        spec = ",".join(str(v) for v in spec)
    _, rights = parse_context_grid(str(spec))
    return rights


def pipeline_position(config: Mapping[str, Any]) -> int | None:
    """The declared position, ``None`` meaning "labels are not chemistry-defined".

    The key is required: absent raises, because a pipeline that never asks the
    question is the one that ships the leak.
    """
    if CONFIG_POSITION_KEY not in config:
        raise ValueError(
            f"Required config key '{CONFIG_POSITION_KEY}' is not set. Declare where the "
            "label-defining sequence starts, counted from the first base of `motif` "
            "(0-based), or set it to null if the labels are not defined by chemistry "
            "in this experiment (rnabioco/leech#402)."
        )
    value = config[CONFIG_POSITION_KEY]
    return None if value is None else int(value)


def preflight(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Evaluate the guard over what the pipeline's prepare rules will forward.

    Covers ``chunk_context`` (or the default) and, with ``use_grid_search``, the
    widest right context the search can select. Raises ``ValueError`` before
    any job exists. Returns the sidecar blocks that were evaluated (empty when
    the declaration is null).
    """
    position = pipeline_position(config)
    if position is None:
        return []
    motif = config.get("motif")
    motif_offset = int(config.get("motif_offset", 0) or 0)
    override = bool(config.get(CONFIG_OVERRIDE_KEY, False))
    channels = config.get("feature_channels")
    if isinstance(channels, str):
        channels = [c.strip() for c in channels.split(",") if c.strip()]
    lookahead = bundled_lookahead()

    contexts: list[tuple[str, tuple[int, int]]] = []
    chunk_context = config.get("chunk_context") or DEFAULT_SIGNAL_CONTEXT
    contexts.append(("chunk_context", (int(chunk_context[0]), int(chunk_context[1]))))
    if config.get("use_grid_search", False):
        rights = grid_right_contexts(config)
        if rights:
            contexts.append(("grid_search right context", (int(chunk_context[0]), max(rights))))

    blocks = []
    for name, ctx in contexts:
        blocks.append(
            evaluate(
                position,
                motif=motif,
                motif_offset=motif_offset,
                allow_override=override,
                lookahead=lookahead,
                feature_channels=channels,
                signal_context=ctx,
                error_prefix=f"[{name} = {list(ctx)}] ",
            )
        )
    return blocks


def pipeline_divergence_arg(config: Mapping[str, Any]) -> str:
    """Render the prepare flags for the declaration, or "" when it is null."""
    position = pipeline_position(config)
    if position is None:
        return ""
    arg = f"{_POSITION_FLAG} {position}"
    if config.get(CONFIG_OVERRIDE_KEY, False):
        arg += f" {_OVERRIDE_FLAG}"
    return arg


def merge_blocks(blocks: Sequence[Mapping[str, Any] | None]) -> dict[str, Any] | None:
    """Combine the declarations of merged inputs, refusing disagreement.

    All-undeclared returns ``None`` (silently). Some declared and some not, or
    declarations that disagree on position or focus offset, raise
    ``ValueError``: a merged corpus has one declaration or none.
    """
    declared = [b for b in blocks if b]
    if not declared:
        return None
    if len(declared) != len(blocks):
        missing = [i for i, b in enumerate(blocks) if not b]
        raise ValueError(
            f"inputs disagree on the label-divergence declaration: inputs {missing} are "
            "undeclared while others declare one. Merging them would give a corpus whose "
            f"guard covers only part of it -- re-prepare the undeclared inputs (input "
            f"index {missing}) with {_POSITION_FLAG}."
        )
    first = declared[0]
    for i, block in enumerate(declared[1:], start=1):
        for key in ("position", "focus_offset"):
            if block.get(key) != first.get(key):
                raise ValueError(
                    f"inputs disagree on the label-divergence declaration '{key}': input 0 "
                    f"has {first.get(key)!r}, input {i} has {block.get(key)!r}. Re-prepare "
                    f"input {i} with the same {_POSITION_FLAG} (and --motif-offset) before "
                    "merging."
                )
    merged = dict(first)
    reaches = [dict(b.get("reach") or {}) for b in declared]
    keys = {k for r in reaches for k in r}
    merged["reach"] = {k: max(r[k] for r in reaches if k in r) for k in REACH_KEYS if k in keys}
    if any(b.get("status") == STATUS_OVERRIDE for b in declared):
        merged["status"] = STATUS_OVERRIDE
    return merged
