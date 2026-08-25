"""Contracts on the ``leech.crf`` package itself, rather than on its arithmetic.

Both of these look like housekeeping and are not. The import-weight guard is
what lets escapepod-models install leech into a conda-forge pixi environment
with ``--no-deps``; the environment-variable aliasing is what keeps that repo's
bonito equivalence checks honest after the move.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: Modules `leech.crf` must not pull in. Nothing here is exotic — they are the
#: rest of leech's runtime stack, and every one of them is a conda/pip package
#: escapepod-models' pixi solver would otherwise have to reconcile against its
#: own pytorch.
FORBIDDEN = ("pysam", "polars", "escapepod", "sklearn", "click", "rich", "rich_click")


def test_crf_imports_nothing_but_torch_and_numpy():
    """`import leech.crf` must stay installable next to a conda pytorch.

    escapepod-models needs the CRF path and none of leech's POD5/BAM stack, and
    installs leech `--no-deps` for exactly that reason (its own pyproject
    carries the same note about not fighting the conda solver). An innocuous
    `from leech.io import ...` at the top of one of these modules turns that
    install into a broken one at first use, which is a bad place to find out.
    """
    probe = (
        "import sys, leech.crf; "
        f"print(','.join(sorted(m for m in {FORBIDDEN!r} if m in sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    pulled = result.stdout.strip()
    assert not pulled, f"leech.crf pulled in: {pulled}"


@pytest.mark.parametrize("prefix", ["LEECH", "ESCAPEPOD"])
@pytest.mark.parametrize("name", ["COMPILE", "NO_COMPILE", "NO_TRITON"])
def test_switches_answer_to_both_prefixes(monkeypatch, prefix: str, name: str):
    """`ESCAPEPOD_*` keeps working, so escapepod-models' scripts keep working.

    `verify_crf_triton.py` there sets `ESCAPEPOD_NO_TRITON=1` to compare the
    kernel against the PyTorch reference. Honouring only the new spelling would
    leave that script comparing the kernel against itself and reporting green.
    """
    from leech.crf._flags import flag

    monkeypatch.delenv(f"LEECH_{name}", raising=False)
    monkeypatch.delenv(f"ESCAPEPOD_{name}", raising=False)
    assert not flag(name)

    monkeypatch.setenv(f"{prefix}_{name}", "1")
    assert flag(name)


def test_empty_is_off_not_on(monkeypatch):
    """An exported-but-empty variable must not silently enable a code path."""
    from leech.crf._flags import flag

    monkeypatch.setenv("LEECH_NO_TRITON", "")
    monkeypatch.delenv("ESCAPEPOD_NO_TRITON", raising=False)
    assert not flag("NO_TRITON")


def test_public_api_is_reachable():
    """Everything in `__all__` resolves — both tables are hand-maintained.

    `_LAZY` maps names to module paths as plain strings that no import
    statement checks, so a renamed or split module keeps importing cleanly and
    fails only at attribute access. leech has been bitten by exactly that at the
    top level (`leech.load_model_from_checkpoint`, after `util.py` was split).
    """
    import leech.crf as crf

    missing = [name for name in crf.__all__ if not hasattr(crf, name)]
    assert not missing, f"unreachable __all__ entries: {missing}"


def test_every_public_name_is_in_all():
    """A lazily-exported name absent from `__all__` is invisible to docs."""
    import leech.crf as crf

    assert set(crf._LAZY) <= set(crf.__all__)


def test_unknown_attribute_raises_attribute_error():
    import leech.crf as crf

    with pytest.raises(AttributeError, match="no attribute"):
        _ = crf.definitely_not_a_real_symbol


def test_config_path_costs_no_torch_import():
    """`from leech.crf import DEFAULT_CONFIG` must not pull torch.

    escapepod-models' `ldxlib` exposes this as a module-level constant and is
    imported by two dozen scripts that want only edit distances and panel
    lookups — none of which should pay for torch. Its `load_crf` already takes
    care to import torch in the function body for the same reason, so an eager
    torch import here would quietly undo that.
    """
    probe = (
        "import sys; from leech.crf import DEFAULT_CONFIG; "
        "assert DEFAULT_CONFIG.is_file(), DEFAULT_CONFIG; "
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False", "importing DEFAULT_CONFIG pulled torch"


def test_model_symbols_still_resolve_through_the_lazy_table():
    """The laziness must not turn a real export into an AttributeError."""
    from leech.crf import CrfEncoder, CtcCrfLoss, decode_batch

    assert all(callable(obj) for obj in (CrfEncoder, CtcCrfLoss, decode_batch))
