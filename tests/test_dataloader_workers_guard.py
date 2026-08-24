"""`num_workers` must never be set to a literal in this package.

The same bug shipped three times, each invisible to the check that caught the
one before:

  #205  `eval test` fed a GPU from one process        -> 8% GPU utilisation
  #206  added `resolve_dataloader_workers`, fixed eval only
  #207  the in-training VALIDATION loader still set 0 -> ~5 min idle GPU at
        every epoch boundary, ~75 min per 15-epoch run; `calibration.py` was
        passing a literal 0 through on CUDA at the same time

`resolve_dataloader_workers`'s docstring claims "Every caller that builds a
loader goes through this function". Nothing enforced it. This does.

The check is deliberately on the VALUE, not on `DataLoader(...)` calls: #207
lived in a `val_loader_kwargs` dict that reached the loader via `**kwargs`, and
an earlier version of this guard that looked at DataLoader call sites missed it
because the enclosing function resolved a *different* loader. Mutation-tested
below against exactly that reintroduction.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "leech"

#: A literal must be annotated at its call site with this marker plus a reason.
MARKER = "dataloader-workers: unresolved"

RESOLVERS = {"resolve_dataloader_workers", "resolve_val_dataloader_workers"}


def _is_resolved(value: ast.AST) -> bool:
    """A resolver call, or a name/attribute that carries a resolved count."""
    if isinstance(value, ast.Call):
        fn = value.func
        nm = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        return nm in RESOLVERS
    if isinstance(value, ast.Name):
        return "worker" in value.id.lower()
    if isinstance(value, ast.Attribute):
        return "worker" in value.attr.lower()
    return False


def _literal_num_workers():
    """Every place `num_workers` is bound to a constant, with its context."""
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text()
        lines = text.splitlines()
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            pairs = []
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values, strict=False):
                    if isinstance(k, ast.Constant) and k.value == "num_workers":
                        pairs.append(v)
            elif isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "num_workers":
                        pairs.append(kw.value)
            for v in pairs:
                if _is_resolved(v):
                    continue
                if not isinstance(v, ast.Constant):
                    continue  # an expression we cannot judge; not a literal
                lo = max(0, getattr(v, "lineno", 1) - 5)
                hi = getattr(node, "end_lineno", v.lineno) or v.lineno
                span = "\n".join(lines[lo:hi])
                yield str(path.relative_to(SRC)), v.lineno, v.value, MARKER in span


def test_num_workers_is_never_a_bare_literal():
    offenders = [
        f"{rel}:{ln} num_workers={val!r} is a literal, not a resolved count"
        for rel, ln, val, annotated in _literal_num_workers()
        if not annotated
    ]
    assert not offenders, (
        "num_workers must come from resolve_dataloader_workers / "
        "resolve_val_dataloader_workers, or carry the marker "
        f"{MARKER!r} with a reason:\n  " + "\n  ".join(offenders)
    )


def test_the_guard_rejects_a_literal():
    """A guard that cannot fail is not a guard."""
    tree = ast.parse('kw = {"num_workers": 0}')
    d = next(n for n in ast.walk(tree) if isinstance(n, ast.Dict))
    assert not _is_resolved(d.values[0])


def test_the_guard_accepts_a_resolver_call():
    tree = ast.parse("DataLoader(ds, num_workers=resolve_val_dataloader_workers(v, n, d))")
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
    kw = next(k for k in call.keywords if k.arg == "num_workers")
    assert _is_resolved(kw.value)


def test_the_guard_accepts_a_resolved_local():
    tree = ast.parse("DataLoader(ds, num_workers=effective_workers)")
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
    kw = next(k for k in call.keywords if k.arg == "num_workers")
    assert _is_resolved(kw.value)
