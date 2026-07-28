"""Guards on the top-level ``leech`` package API.

``leech/__init__.py`` resolves its public symbols through a PEP 562
``__getattr__`` backed by the ``_LAZY_IMPORTS`` table, so the module paths in
that table are plain strings that no import statement ever checks. When a module
is renamed or split, a stale entry keeps importing cleanly and only fails at
attribute access -- ``leech.load_model_from_checkpoint`` stayed broken this way
after ``leech/util.py`` was split into ``bundling``/``model_loading``.
"""

import pytest

import leech


@pytest.mark.parametrize("name", sorted(leech._LAZY_IMPORTS))
def test_lazy_import_resolves(name: str):
    """Every _LAZY_IMPORTS entry points at a module that actually exports it."""
    assert callable(getattr(leech, name))


def test_all_exports_are_reachable():
    """Everything advertised in __all__ can actually be accessed."""
    unreachable = []
    for name in leech.__all__:
        try:
            getattr(leech, name)
        except AttributeError as exc:
            unreachable.append(f"{name}: {exc}")
    assert not unreachable, "unreachable __all__ entries: " + "; ".join(unreachable)


def test_unknown_attribute_still_raises_attribute_error():
    # Bound to a name so ruff sees an assignment: bare attribute access trips
    # B018 (useless expression) and getattr() trips B009.
    with pytest.raises(AttributeError, match="no attribute"):
        _ = leech.definitely_not_a_real_symbol
