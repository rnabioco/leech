"""
Thin wrapper around the ``gh`` CLI for GitHub release operations.

All GitHub interactions go through subprocess calls to ``gh``, avoiding
a PyGitHub dependency.  The wrapper raises clear errors when ``gh`` is
missing or unauthenticated.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("leech.github")


class GitHubCLIError(RuntimeError):
    """Raised when a ``gh`` command fails."""


def _run_gh(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a ``gh`` sub-command and return the result."""
    cmd = ["gh", *args]
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise GitHubCLIError(
            f"`gh {' '.join(args)}` failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )
    return result


# ------------------------------------------------------------------
# Availability / auth helpers
# ------------------------------------------------------------------


def check_gh_available() -> bool:
    """Return True if ``gh`` is on PATH."""
    try:
        subprocess.run(["gh", "--version"], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        return False
    return True


def check_gh_authenticated() -> bool:
    """Return True if ``gh`` is authenticated with GitHub."""
    result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    return result.returncode == 0


def get_repo_slug() -> str:
    """Derive ``owner/repo`` from the git remote origin URL."""
    result = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True)
    if result.returncode != 0:
        raise GitHubCLIError("Could not determine git remote origin URL")
    url = result.stdout.strip()
    # Handle SSH: git@github.com:owner/repo.git
    if url.startswith("git@"):
        slug = url.split(":")[-1]
    # Handle HTTPS: https://github.com/owner/repo.git
    elif "github.com" in url:
        slug = "/".join(url.rstrip("/").split("/")[-2:])
    else:
        raise GitHubCLIError(f"Cannot parse repo slug from remote URL: {url}")
    return slug.removesuffix(".git")


# ------------------------------------------------------------------
# Release operations
# ------------------------------------------------------------------


def create_release(
    tag: str,
    title: str,
    body: str,
    files: list[Path],
    *,
    prerelease: bool = False,
    repo: str | None = None,
    dry_run: bool = False,
) -> str | None:
    """Create a GitHub release and upload asset files.

    Returns the release URL, or *None* when *dry_run* is True.
    """
    if dry_run:
        logger.info("[dry-run] Would create release %s on %s", tag, repo or "origin")
        return None

    args = ["release", "create", tag, "--title", title, "--notes", body]
    if prerelease:
        args.append("--prerelease")
    if repo:
        args.extend(["--repo", repo])
    for f in files:
        args.append(str(f))
    result = _run_gh(args)
    url = result.stdout.strip()
    return url


def list_releases(
    pattern: str | None = None,
    repo: str | None = None,
    limit: int = 30,
) -> list[dict]:
    """List releases, optionally filtered by tag prefix.

    Returns a list of dicts with keys: ``tag``, ``name``, ``published``,
    ``prerelease``.
    """
    args = [
        "release",
        "list",
        "--json",
        "tagName,name,publishedAt,isPrerelease,isDraft",
        "--limit",
        str(limit),
    ]
    if repo:
        args.extend(["--repo", repo])
    result = _run_gh(args)
    releases: list[dict] = json.loads(result.stdout)
    if pattern:
        releases = [r for r in releases if r["tagName"].startswith(pattern)]
    return releases


def download_release_asset(
    tag: str,
    asset_pattern: str,
    output_dir: Path,
    *,
    repo: str | None = None,
) -> Path:
    """Download asset(s) matching *asset_pattern* from a release.

    Returns the directory where files were downloaded.
    """
    args = ["release", "download", tag, "--pattern", asset_pattern, "--dir", str(output_dir)]
    if repo:
        args.extend(["--repo", repo])
    _run_gh(args)
    return output_dir


def get_release_body(tag: str, *, repo: str | None = None) -> str:
    """Fetch the release notes body for a given tag."""
    args = ["release", "view", tag, "--json", "body"]
    if repo:
        args.extend(["--repo", repo])
    result = _run_gh(args)
    data = json.loads(result.stdout)
    return data.get("body", "")


def delete_release(tag: str, *, repo: str | None = None) -> None:
    """Delete a release (and its tag) by tag name."""
    args = ["release", "delete", tag, "--yes", "--cleanup-tag"]
    if repo:
        args.extend(["--repo", repo])
    _run_gh(args)
