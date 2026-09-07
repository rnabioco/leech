#!/usr/bin/env python3
"""The two escapepod pins move together, or leech drives two escapepods.

`rust/Cargo.toml` pins the `escapepod-signal` crate by git tag and
`pyproject.toml` pins the `escapepod` PyPI package by floor. They are the same
upstream, released in lockstep from rnabioco/escapepod-rs, and leech drives
*both* -- `signal_refine.py` through the Python binding, `refinement.rs`
through the crate. When they disagree, the two prepare backends can compute
different dwells and different level features from the same read, which is
issue #193: it went unnoticed for four releases because nothing compares a
version to a version, and the arrays it produces have the right shape and
plausible contents either way.

Nothing enforced this. Dependabot's cargo ecosystem bumps the tag-pinned git
dependency on its own schedule (PR #236 took the crate from v0.16.1 to
v0.18.1) and has no way to know a PyPI package in a different manifest has to
move with it, so merging it leaves main skewed.

Used by `.github/workflows/escapepod-sync.yml`:

    escapepod_sync.py --check          report; exit 1 when the pins disagree
    escapepod_sync.py --fix            rewrite both to the higher version

Standard library only: it runs before any dependency is installed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RUST_MANIFEST = "rust/Cargo.toml"
PYTHON_MANIFEST = "pyproject.toml"

# escapepod-signal = { git = "...", tag = "v0.21.0" }
_RUST_PIN = re.compile(
    r"""(?P<head>^\s*escapepod-signal\s*=\s*\{[^}]*?\btag\s*=\s*")(?P<version>v[^"]+)(?P<tail>")""",
    re.MULTILINE,
)
#   "escapepod>=0.21.0",
_PYTHON_PIN = re.compile(
    r"""(?P<head>"escapepod\s*>=\s*)(?P<version>[0-9][^",]*?)(?P<tail>\s*")""",
)


class PinError(RuntimeError):
    """A manifest did not contain the pin this check exists to compare."""


def parse_version(text: str) -> tuple[int, ...]:
    """``"v0.21.0"`` and ``"0.21.0"`` both to ``(0, 21, 0)``.

    Only the numeric release segments are compared. A pin carrying a
    pre-release or local suffix is not something this check should silently
    reorder, so it is rejected rather than truncated.
    """
    core = text.strip().removeprefix("v")
    if not re.fullmatch(r"\d+(\.\d+)*", core):
        raise PinError(f"cannot compare a non-release version: {text!r}")
    return tuple(int(part) for part in core.split("."))


def _pin(root: Path, relative: str, pattern: re.Pattern[str]) -> str:
    path = root / relative
    match = pattern.search(path.read_text())
    if match is None:
        raise PinError(f"no escapepod pin found in {relative}")
    return match.group("version")


def read_pins(root: Path = REPO_ROOT) -> dict[str, str]:
    """The two pinned versions, keyed by the manifest they came from."""
    return {
        RUST_MANIFEST: _pin(root, RUST_MANIFEST, _RUST_PIN),
        PYTHON_MANIFEST: _pin(root, PYTHON_MANIFEST, _PYTHON_PIN),
    }


def in_sync(pins: dict[str, str]) -> bool:
    return len({parse_version(v) for v in pins.values()}) == 1


def target_version(pins: dict[str, str]) -> str:
    """The higher of the two, as a bare ``X.Y.Z``.

    Aligning upward is the only safe direction: the lower pin is the one that
    has not been exercised against the other's code, and downgrading a
    dependency someone deliberately raised would undo their work.
    """
    highest = max(parse_version(v) for v in pins.values())
    return ".".join(str(part) for part in highest)


def _write_pin(root: Path, relative: str, pattern: re.Pattern[str], version: str) -> bool:
    path = root / relative
    text = path.read_text()
    updated, count = pattern.subn(rf"\g<head>{version}\g<tail>", text)
    if count != 1:
        raise PinError(f"expected exactly one escapepod pin in {relative}, rewrote {count}")
    if updated == text:
        return False
    path.write_text(updated)
    return True


def apply_version(version: str, root: Path = REPO_ROOT) -> list[str]:
    """Point both manifests at ``version``. Returns the files actually changed.

    `rust/Cargo.lock` is left to `cargo update`, which rewrites the eight lines
    it owns and nothing else. `uv.lock` is *not* left to `uv lock`: see
    :func:`apply_uv_lock`.
    """
    changed = []
    if _write_pin(root, RUST_MANIFEST, _RUST_PIN, f"v{version}"):
        changed.append(RUST_MANIFEST)
    if _write_pin(root, PYTHON_MANIFEST, _PYTHON_PIN, version):
        changed.append(PYTHON_MANIFEST)
    return changed


def pypi_release(version: str) -> dict:
    """PyPI's own record of one escapepod release: the sdist and every wheel."""
    url = f"https://pypi.org/pypi/escapepod/{version}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _lock_artifact(file: dict, indent: str = "    ") -> str:
    return (
        f'{indent}{{ url = "{file["url"]}", hash = "sha256:{file["digests"]["sha256"]}", '
        f'size = {file["size"]}, upload-time = "{file["upload_time_iso_8601"]}" }},'
    )


def apply_uv_lock(version: str, root: Path = REPO_ROOT, release: dict | None = None) -> bool:
    """Rewrite `uv.lock`'s escapepod entry in place. True when it changed.

    Not `uv lock`, and not `uv lock --upgrade-package escapepod` either: both
    re-serialize the whole file against the running uv's conventions, which on
    this lock means adding `sys_platform != 'emscripten'` markers to ~300
    unrelated packages. Measured at 670 changed lines to move one version --
    the real edit is unreviewable inside it, and every run would flip the
    markers back and forth depending on whose uv ran last.

    The hashes come from PyPI's own release record, which is where uv would
    read them. Wheels are ordered by filename, which is the order uv writes
    them in, so the diff stays confined to the entry itself.
    """
    path = root / "uv.lock"
    text = path.read_text()

    block = re.search(
        r'\[\[package\]\]\nname = "escapepod"\n.*?\n\n(?=\[\[package\]\])', text, re.S
    )
    if block is None:
        raise PinError("no escapepod package entry found in uv.lock")

    data = release if release is not None else pypi_release(version)
    files = data["urls"]
    sdist = next(f for f in files if f["packagetype"] == "sdist")
    wheels = sorted(
        (f for f in files if f["packagetype"] == "bdist_wheel"),
        key=lambda f: f["filename"],
    )

    entry = (
        f'[[package]]\nname = "escapepod"\nversion = "{version}"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f"sdist = {_lock_artifact(sdist, indent='').rstrip(',')}\n"
        "wheels = [\n" + "\n".join(_lock_artifact(w) for w in wheels) + "\n]\n\n"
    )

    updated = text.replace(block.group(0), entry, 1)
    updated, count = re.subn(
        r'(\{ name = "escapepod", specifier = ">=)[^"]+(" \})',
        rf"\g<1>{version}\g<2>",
        updated,
    )
    if count != 1:
        raise PinError(f"expected one escapepod specifier in uv.lock, rewrote {count}")

    if updated == text:
        return False
    path.write_text(updated)
    return True


def released_upstream(version: str) -> tuple[bool, bool]:
    """Whether ``version`` exists as a PyPI release and as a git tag.

    A skew can only be closed by moving to something that exists on *both*
    sides; proposing otherwise produces a branch that cannot build. Network
    failure is reported as "present" so a PyPI outage does not look like a
    missing release.
    """

    def reachable(url: str) -> bool:
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return 200 <= response.status < 300
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            return True
        except OSError:
            return True

    on_pypi = reachable(f"https://pypi.org/pypi/escapepod/{version}/json")
    tagged = reachable(
        f"https://api.github.com/repos/rnabioco/escapepod-rs/git/ref/tags/v{version}"
    )
    return on_pypi, tagged


def _emit(name: str, value: str) -> None:
    """Publish a step output when running under Actions; print either way."""
    print(f"{name}={value}")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report; exit 1 when skewed")
    mode.add_argument("--fix", action="store_true", help="rewrite both to the higher pin")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    pins = read_pins(args.root)
    for manifest, version in pins.items():
        print(f"{manifest}: escapepod {version}")

    if in_sync(pins):
        _emit("skewed", "false")
        print("pins agree")
        return 0

    target = target_version(pins)
    _emit("skewed", "true")
    _emit("target", target)

    if args.check:
        on_pypi, tagged = released_upstream(target)
        _emit("actionable", "true" if (on_pypi and tagged) else "false")
        if not (on_pypi and tagged):
            print(
                f"::warning::escapepod pins disagree but {target} is not released on "
                f"both sides (pypi={on_pypi}, git tag={tagged}); not proposing a bump",
                file=sys.stderr,
            )
            return 1
        print(f"::error::escapepod pins disagree; both should be {target}", file=sys.stderr)
        return 1

    changed = apply_version(target, args.root)
    if apply_uv_lock(target, args.root):
        changed.append("uv.lock")
    _emit("changed", json.dumps(changed))
    print(f"rewrote {len(changed)} file(s) to escapepod {target}")
    print("run `cargo update -p escapepod-signal` and `uv lock --check` to finish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
