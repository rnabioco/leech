"""The daily escapepod pin check (`.github/scripts/escapepod_sync.py`).

These test the *tool*, not today's versions. Whether the two pins currently
agree is the daily workflow's business -- asserting it here would turn every
dependabot PR red, which is the thing the workflow exists to avoid.

The load-bearing test is `test_the_real_manifests_are_still_parseable`: the
regexes read two files this script does not own, and if either changes shape
the check starts reporting "pins agree" about a pin it never found.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "escapepod_sync.py"


def _load():
    spec = importlib.util.spec_from_file_location("escapepod_sync", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sync = _load()


@pytest.fixture
def manifests(tmp_path: Path) -> Path:
    """A miniature repo carrying only the two lines the check reads."""
    (tmp_path / "rust").mkdir()
    (tmp_path / "rust" / "Cargo.toml").write_text(
        "[dependencies]\n"
        'pyo3 = { version = "0.28", features = ["extension-module"] }\n'
        'escapepod-signal = { git = "https://github.com/rnabioco/escapepod-rs", '
        'tag = "v0.18.1" }\n'
        'rayon = "1"\n'
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\ndependencies = [\n"
        '  "numpy>=1.5",\n'
        '  "escapepod>=0.16.0",\n'
        '  "pysam>=0.22",\n'
        "]\n"
    )
    return tmp_path


def test_the_real_manifests_are_still_parseable():
    """Both pins must be found in the files this repo actually ships.

    Not their values -- only that the regexes still locate them. A manifest
    reformat that made either pattern miss would leave the check silently
    reporting agreement between two things it did not read.
    """
    pins = sync.read_pins(REPO_ROOT)

    assert set(pins) == {sync.RUST_MANIFEST, sync.PYTHON_MANIFEST}
    for manifest, version in pins.items():
        assert sync.parse_version(version), f"{manifest} pin unparseable: {version!r}"


def test_a_skew_is_detected_and_resolved_upward(manifests):
    pins = sync.read_pins(manifests)
    assert pins[sync.RUST_MANIFEST] == "v0.18.1"
    assert pins[sync.PYTHON_MANIFEST] == "0.16.0"

    assert not sync.in_sync(pins)
    assert sync.target_version(pins) == "0.18.1", "the lower pin is the untested one"


def test_the_fix_moves_both_and_is_idempotent(manifests):
    assert sorted(sync.apply_version("0.21.0", manifests)) == [
        sync.PYTHON_MANIFEST,
        sync.RUST_MANIFEST,
    ]

    pins = sync.read_pins(manifests)
    assert pins == {sync.RUST_MANIFEST: "v0.21.0", sync.PYTHON_MANIFEST: "0.21.0"}
    assert sync.in_sync(pins)
    assert sync.apply_version("0.21.0", manifests) == [], "a no-op must report no change"


def test_the_v_prefix_belongs_to_the_tag_only(manifests):
    """The crate is tagged ``v0.21.0``; the PyPI floor is ``0.21.0``.

    Carrying the ``v`` into pyproject produces a specifier no resolver accepts,
    and dropping it from the tag produces a git ref that does not exist.
    """
    sync.apply_version("0.21.0", manifests)

    assert 'tag = "v0.21.0"' in (manifests / "rust" / "Cargo.toml").read_text()
    assert '"escapepod>=0.21.0"' in (manifests / "pyproject.toml").read_text()


def test_upward_resolution_works_from_either_side(manifests):
    """Whichever manifest is ahead, the other one follows it."""
    (manifests / "pyproject.toml").write_text(
        '[project]\ndependencies = [\n  "escapepod>=0.30.0",\n]\n'
    )

    assert sync.target_version(sync.read_pins(manifests)) == "0.30.0"


def test_versions_compare_numerically_not_lexically():
    assert sync.parse_version("v0.21.0") > sync.parse_version("0.9.0")
    assert sync.parse_version("0.16.0") < sync.parse_version("0.16.1")


@pytest.mark.parametrize("version", ["0.21.0rc1", "0.21.0.dev3", "latest", "0.21.0+local"])
def test_a_non_release_pin_refuses_to_be_ordered(version):
    """Truncating a pre-release to its numeric head would order it wrongly.

    Better to fail the daily check loudly than to open a PR that quietly
    downgrades a deliberately pinned pre-release.
    """
    with pytest.raises(sync.PinError):
        sync.parse_version(version)


def test_a_missing_pin_is_an_error_not_a_silent_pass(tmp_path):
    (tmp_path / "rust").mkdir()
    (tmp_path / "rust" / "Cargo.toml").write_text('[dependencies]\nrayon = "1"\n')
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["escapepod>=0.21.0"]\n')

    with pytest.raises(sync.PinError, match="rust/Cargo.toml"):
        sync.read_pins(tmp_path)


def test_check_mode_exits_nonzero_on_skew(manifests, capsys, monkeypatch):
    # Stubbed: whether 0.18.1 is on PyPI today is not what this asserts, and a
    # unit test that reaches the network fails for reasons unrelated to it.
    monkeypatch.setattr(sync, "released_upstream", lambda version: (True, True))

    assert sync.main(["--check", "--root", str(manifests)]) == 1

    sync.apply_version("0.21.0", manifests)
    assert sync.main(["--check", "--root", str(manifests)]) == 0
    assert "pins agree" in capsys.readouterr().out


def _release(version: str) -> dict:
    """A PyPI release record shaped like the one uv.lock is built from."""

    def artifact(name: str, kind: str) -> dict:
        return {
            "filename": name,
            "packagetype": kind,
            "url": f"https://files.pythonhosted.org/packages/ab/cd/{name}",
            "digests": {"sha256": "0" * 64},
            "size": 1234,
            "upload_time_iso_8601": "2026-09-06T11:55:40.788012Z",
        }

    # Deliberately out of order: uv writes wheels sorted by filename.
    return {
        "urls": [
            artifact(f"escapepod-{version}-cp39-abi3-musllinux_1_2_x86_64.whl", "bdist_wheel"),
            artifact(f"escapepod-{version}.tar.gz", "sdist"),
            artifact(f"escapepod-{version}-cp39-abi3-macosx_11_0_arm64.whl", "bdist_wheel"),
        ]
    }


@pytest.fixture
def lockfile(tmp_path: Path) -> Path:
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "escapepod"\nversion = "0.16.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        'sdist = { url = "https://example/escapepod-0.16.0.tar.gz", hash = "sha256:dead", '
        'size = 1, upload-time = "2026-08-25T22:41:46.226Z" }\n'
        "wheels = [\n"
        '    { url = "https://example/old.whl", hash = "sha256:beef", size = 2, '
        'upload-time = "2026-08-25T22:41:36.705Z" },\n'
        "]\n\n"
        '[[package]]\nname = "execnet"\nversion = "2.1.2"\n\n'
        "[package.metadata]\nrequires-dist = [\n"
        '    { name = "escapepod", specifier = ">=0.16.0" },\n'
        '    { name = "numpy", specifier = ">=1.5" },\n'
        "]\n"
    )
    return tmp_path


def test_the_lock_entry_is_rewritten_surgically(lockfile):
    """A bare ``uv lock`` re-serializes the whole file; this touches one entry.

    Measured on the real lock: 670 changed lines against 18, because ~300
    unrelated packages gain ``sys_platform != 'emscripten'`` markers. The one
    line worth reviewing is invisible inside that.
    """
    assert sync.apply_uv_lock("0.21.0", lockfile, release=_release("0.21.0"))

    text = (lockfile / "uv.lock").read_text()
    assert 'name = "escapepod"\nversion = "0.21.0"' in text
    assert '{ name = "escapepod", specifier = ">=0.21.0" }' in text
    assert "0.16.0" not in text, "no trace of the old release may survive"
    # Neighbours are untouched.
    assert '[[package]]\nname = "execnet"\nversion = "2.1.2"' in text
    assert '{ name = "numpy", specifier = ">=1.5" }' in text


def test_lock_wheels_are_written_in_uv_order(lockfile):
    """uv writes wheels sorted by filename; PyPI returns them in upload order.

    Emitting PyPI's order would rewrite every wheel line on a lock uv later
    regenerates, turning a 9-line diff back into a noisy one.
    """
    sync.apply_uv_lock("0.21.0", lockfile, release=_release("0.21.0"))

    urls = re.findall(r"escapepod-0\.21\.0-\S+?\.whl", (lockfile / "uv.lock").read_text())
    assert urls == sorted(urls)


def test_a_lock_without_the_entry_is_an_error(tmp_path):
    (tmp_path / "uv.lock").write_text('[[package]]\nname = "numpy"\nversion = "1.5"\n\n')

    with pytest.raises(sync.PinError, match="uv.lock"):
        sync.apply_uv_lock("0.21.0", tmp_path, release=_release("0.21.0"))


def test_an_unreleased_target_is_reported_but_not_proposed(manifests, monkeypatch):
    """A version missing on either side cannot close the skew.

    Bumping to it would produce a branch that does not build, so the check
    still fails -- it just does not hand the workflow something to open.
    """
    monkeypatch.setattr(sync, "released_upstream", lambda version: (False, True))

    assert sync.main(["--check", "--root", str(manifests)]) == 1
