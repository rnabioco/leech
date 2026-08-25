"""Environment switches for the CRF path, and why each answers to two names.

This package was ported from ``escapepod_models.crf``, where the switches were
spelled ``ESCAPEPOD_*``. escapepod-models' scripts, Snakemake rules and — most
importantly — the bonito equivalence checks (``scripts/ldx/analysis/verify_crf_*.py``)
set them by that name, and keep doing so after the move. Reading both spellings
is what lets one implementation serve both callers: a rename only leech knew
about would silently stop honouring ``ESCAPEPOD_NO_TRITON=1`` in the very
scripts whose job is to prove this code correct, and they would report a green
comparison of the Triton kernel against itself.

Set to any non-empty value to enable; unset or empty is off, which is the
semantics the original ``os.environ.get(...)`` truthiness test had.
"""

from __future__ import annotations

import os

__all__ = ["flag"]


def flag(name: str) -> bool:
    """True when ``LEECH_<name>`` or ``ESCAPEPOD_<name>`` is set non-empty."""
    return bool(os.environ.get(f"LEECH_{name}") or os.environ.get(f"ESCAPEPOD_{name}"))
