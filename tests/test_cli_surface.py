"""Regression test: every registered CLI command's handler must import cleanly.

`leech eval compare`, `leech eval importance` and `leech eval ablation` were
registered in `cli.py` and wired to handlers in
`src/leech/commands/analyze.py` that imported `leech.analysis` -- a package
that was never committed (#262). Every command in this codebase lazily
imports its handler inside the callback body (see the docstring convention
in `cli.py`), and the handler functions themselves often do the same one
level deeper (`commands/analyze.py`'s module-level imports were all stdlib;
the missing `leech.analysis.*` imports lived inside `handle_compare` etc.).
So the module itself imported fine, `leech --help` and `leech eval --help`
worked, and nothing short of actually *calling* one of the three commands
raised `ModuleNotFoundError`.

This test walks that whole lazy-import chain statically: for every
registered command, find every `import leech...` / `from leech... import
...` anywhere in its callback's source, resolve it, and -- if the resolved
name is itself a function -- recurse into its source too. A command whose
handler (at any depth) references a module that does not exist can never be
registered again without this test failing.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap

import click
import pytest

from leech.cli import cli


def _iter_commands(group: click.Group, prefix: str = "") -> list[tuple[str, click.Command]]:
    """Recursively collect every leaf (non-group) command reachable from `group`."""
    found: list[tuple[str, click.Command]] = []
    for name, cmd in group.commands.items():
        full_name = f"{prefix}{name}"
        if isinstance(cmd, click.Group):
            found.extend(_iter_commands(cmd, prefix=f"{full_name} "))
        else:
            found.append((full_name, cmd))
    return found


_COMMANDS = _iter_commands(cli)


def _check_lazy_imports(func, path: str, visited: set[int], errors: list[str]) -> None:
    """Resolve every `leech` import inside `func`'s source, recursing into functions.

    Mutates `errors` with one message per broken import found anywhere in the
    chain. `visited` (keyed by `id(func)`) stops re-walking a function reached
    by more than one path and guards against import cycles.
    """
    if id(func) in visited:
        return
    visited.add(id(func))

    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError):
        # Built-in / C-implemented / otherwise source-less callable: nothing to walk.
        return

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("leech"):
            module_name = node.module
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                errors.append(f"{path}: `from {module_name} import ...` fails: {exc}")
                continue
            for alias in node.names:
                attr = alias.name
                if not hasattr(module, attr):
                    errors.append(f"{path}: {module_name!r} has no attribute {attr!r}")
                    continue
                value = getattr(module, attr)
                if inspect.isfunction(value):
                    _check_lazy_imports(value, f"{path} -> {module_name}.{attr}", visited, errors)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("leech"):
                    try:
                        importlib.import_module(alias.name)
                    except ImportError as exc:
                        errors.append(f"{path}: `import {alias.name}` fails: {exc}")


def test_cli_tree_walk_found_commands():
    """Sanity check that the walk above is not silently empty."""
    names = {name for name, _ in _COMMANDS}
    assert "data prepare" in names
    assert "model train" in names
    assert "eval test" in names
    assert "predict" in names
    assert len(_COMMANDS) > 10


@pytest.mark.parametrize("name,cmd", _COMMANDS, ids=[name for name, _ in _COMMANDS])
def test_command_handler_imports_cleanly(name: str, cmd: click.Command):
    """A registered command's lazily-imported handler chain must actually exist."""
    callback = inspect.unwrap(cmd.callback)
    errors: list[str] = []
    _check_lazy_imports(callback, f"leech {name}", visited=set(), errors=errors)
    assert not errors, "\n".join(errors)
