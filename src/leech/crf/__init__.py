"""CTC-CRF sequence models: encoder, training objective, and decode.

A second task alongside leech's chunk classifiers. Where a classifier maps a
signal window to a label, this maps one to a *sequence*: a CRF over
``n_base ** state_len`` states whose Viterbi traceback emits one base per move.
It is what the barcode basecallers in escapepod-models are trained with, and
what ``escapepod-demux``'s Rust decoder runs in production.

The encoder, the objective and the decode are all here, so a model can be
trained, evaluated and exported from this package alone. ``encoder.py`` records
where the architecture and the formulation come from.

**This subpackage imports only torch and numpy**, and only when a symbol that
needs them is actually touched. Not pysam, not polars, not escapepod, ever. Both
halves of that are load-bearing:

* escapepod-models installs leech into a conda-forge pixi environment with
  ``--no-deps`` precisely so the solver is never asked to reconcile leech's
  POD5/BAM stack against conda's pytorch, and the CRF path is the only part of
  leech it needs.
* The config path is resolved eagerly and everything else lazily (PEP 562, as in
  :mod:`leech.models`), so ``from leech.crf import DEFAULT_CONFIG`` costs
  nothing. escapepod-models' ``ldxlib`` exposes it as a module constant and is
  imported by two dozen scripts that want only edit distances and panel
  lookups; the CLI will want the same for a ``--config`` default without paying
  the torch import at parse time.

``tests/test_crf_package.py`` fails if either property regresses.

Three invariants the rest of this package assumes, each of which has already
cost a debugging session somewhere in this stack:

* **The model cannot emit the first ``state_len`` bases of its target.** They fix
  the initial state and nothing else, so a ``target_len``-base target decodes to
  ``target_len - state_len`` bases *at any window width* — widening the window
  recovers nothing. Size targets so the sacrificial bases come from a constant
  prefix, and match decodes against ``target[state_len:]``.
* **Blank is entry 0 of each state's group**, giving a score width of
  ``n_states * (n_base + 1)`` = 1280, not the linear layer's 1024.
* **The decode is two passes** — log-semiring posteriors, then max-semiring over
  ``log(post + 1e-8)``. A one-pass Viterbi over the raw scores is a different and
  worse decode, and is the obvious thing to simplify away.
"""

from __future__ import annotations

import importlib
from typing import Any

# Torch-free, so it stays eager: this is the half consumers reach for without
# wanting a model.
from .config import CONFIG_DIR, DEFAULT_CONFIG, load_config

#: Symbol -> module, resolved on first access. Plain strings that no import
#: statement checks, so `tests/test_crf_package.py` walks the table.
_LAZY: dict[str, str] = {
    "ALPHABET": "leech.crf.decode",
    "best_path": "leech.crf.decode",
    "decode_batch": "leech.crf.decode",
    "BLANK_SCORE": "leech.crf.encoder",
    "CrfEncoder": "leech.crf.encoder",
    "EncoderConfig": "leech.crf.encoder",
    "encoder_config_from_toml": "leech.crf.encoder",
    "load_crf_state_dict": "leech.crf.encoder",
    "CtcCrfLoss": "leech.crf.loss",
    "CorpusPlan": "leech.crf.corpus",
    "build_corpus": "leech.crf.corpus",
    "load_corpus": "leech.crf.corpus",
    "load_corpus_meta": "leech.crf.corpus",
    "plan_corpus": "leech.crf.corpus",
    "Call": "leech.crf.evaluate",
    "balanced_recall": "leech.crf.evaluate",
    "call_references": "leech.crf.evaluate",
    "decode_corpus": "leech.crf.evaluate",
    "emitted_references": "leech.crf.evaluate",
    "encode_references": "leech.crf.evaluate",
    "lev": "leech.crf.evaluate",
    "lev_vs_refs": "leech.crf.evaluate",
    "CrfTrainConfig": "leech.crf.training",
    "CrfTrainer": "leech.crf.training",
    "EpochStats": "leech.crf.training",
    "apply_quality_gate": "leech.crf.training",
    "compute_standardisation": "leech.crf.training",
    "encode_targets": "leech.crf.training",
    "resolve_split": "leech.crf.training",
    "select_checkpoint": "leech.crf.training",
    "train_crf": "leech.crf.training",
    "CrfManifest": "leech.crf.manifest",
    "OPTIONAL_COLUMNS": "leech.crf.manifest",
    "REQUIRED_COLUMNS": "leech.crf.manifest",
    "check_geometry": "leech.crf.manifest",
    "emitted_target": "leech.crf.manifest",
    "load_manifest": "leech.crf.manifest",
}

__all__ = [
    "ALPHABET",
    "BLANK_SCORE",
    "CONFIG_DIR",
    "Call",
    "DEFAULT_CONFIG",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "CorpusPlan",
    "CrfEncoder",
    "CrfTrainConfig",
    "CrfTrainer",
    "CtcCrfLoss",
    "EncoderConfig",
    "EpochStats",
    "CrfManifest",
    "apply_quality_gate",
    "balanced_recall",
    "best_path",
    "build_corpus",
    "compute_standardisation",
    "call_references",
    "check_geometry",
    "decode_batch",
    "decode_corpus",
    "emitted_references",
    "emitted_target",
    "encode_references",
    "encode_targets",
    "encoder_config_from_toml",
    "lev",
    "lev_vs_refs",
    "load_config",
    "load_corpus",
    "load_corpus_meta",
    "load_crf_state_dict",
    "load_manifest",
    "plan_corpus",
    "resolve_split",
    "select_checkpoint",
    "train_crf",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])
