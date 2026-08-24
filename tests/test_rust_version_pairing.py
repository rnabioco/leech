"""`leech` and `leech_core` are released together and must report the same version.

They are separate distributions built from one repository, so an extension
compiled at one revision can sit alongside a `leech` from another. That pairing
does not raise — it produces different numbers. It is how #176 stayed hidden
(new Rust, old serial driver), and how a stale `uv` archive-cache entry
silently reinstated pre-#188 chunk behaviour over a current build: uv keys its
cache on the version string, and `leech_core` sat at `0.3.0` from leech v0.3.1
to v0.6.4 while the Rust changed underneath it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from leech._rust_accel import HAS_RUST, rust_version_mismatch

REPO_ROOT = Path(__file__).resolve().parent.parent


def _version(path: Path, *keys: str) -> str:
    data = tomllib.loads(path.read_text())
    for key in keys:
        data = data[key]
    return data


class TestDeclaredVersionsAgree:
    """Checked from the source tree, so it holds without an install."""

    def test_leech_core_tracks_leech(self):
        leech_version = _version(REPO_ROOT / "pyproject.toml", "project", "version")
        core_version = _version(REPO_ROOT / "rust" / "Cargo.toml", "package", "version")
        assert core_version == leech_version, (
            f"rust/Cargo.toml is {core_version} but pyproject.toml is "
            f"{leech_version}. They are released together and uv keys its "
            f"extension cache on the leech_core version -- if it does not move, "
            f"`uv sync` can restore a stale compiled extension over a current "
            f"build. Bump both; see .claude/commands/release.md."
        )

    def test_wheel_version_is_not_pinned_separately(self):
        """`rust/pyproject.toml` must defer to Cargo.toml, not carry a third copy."""
        rust_pyproject = tomllib.loads((REPO_ROOT / "rust" / "pyproject.toml").read_text())
        project = rust_pyproject["project"]
        assert "version" not in project, (
            "rust/pyproject.toml pins its own version; it should declare "
            'dynamic = ["version"] so rust/Cargo.toml stays the single source.'
        )
        assert "version" in project.get("dynamic", [])


@pytest.mark.skipif(not HAS_RUST, reason="leech_core not installed")
class TestInstalledVersionsAgree:
    def test_extension_exports_its_version(self):
        import leech_core

        assert getattr(leech_core, "__version__", None), (
            "leech_core exports no __version__, so a stale pairing cannot be "
            "detected. It is added in rust/src/lib.rs via env!(CARGO_PKG_VERSION)."
        )

    def test_no_mismatch_in_this_environment(self):
        mismatch = rust_version_mismatch()
        assert mismatch is None, (
            f"leech {mismatch[0]} is paired with leech_core {mismatch[1]}. "
            f"Rebuild with `bash rust/build.sh`; if that does not clear it, "
            f"reinstall leech (`uv pip install -e .`)."
        )

    def test_installed_extension_matches_the_source_tree(self):
        import leech_core

        declared = _version(REPO_ROOT / "rust" / "Cargo.toml", "package", "version")
        assert leech_core.__version__ == declared, (
            f"installed leech_core is {leech_core.__version__} but the source "
            f"tree declares {declared} -- the build is stale. This is the exact "
            f"shape of the uv-cache hazard: run `bash rust/build.sh`."
        )
