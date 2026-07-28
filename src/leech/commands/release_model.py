"""
Handlers for ``leech model release``, ``leech model list``, and ``leech model fetch``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.table import Table

from leech.cli_config import make_console

logger = logging.getLogger("leech.commands.release_model")
console = make_console()


# ------------------------------------------------------------------
# release
# ------------------------------------------------------------------


def handle_release(
    spec_path: Path,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
    repo: str | None = None,
) -> str | None:
    """Publish a model bundle to GitHub Releases.

    Returns the release URL, or None for dry-run.
    """
    from leech.release import (
        check_gh_authenticated,
        check_gh_available,
        create_release,
        delete_release,
        get_repo_slug,
        load_release_spec,
        render_release_notes,
        validate_spec,
    )

    # --- Pre-flight ---
    if not check_gh_available():
        raise RuntimeError(
            "GitHub CLI (gh) is not installed. "
            "Install it from https://cli.github.com/ to use model releases."
        )
    if not dry_run and not check_gh_authenticated():
        raise RuntimeError("GitHub CLI is not authenticated. Run `gh auth login` first.")

    repo = repo or get_repo_slug()

    # --- Parse spec ---
    spec = load_release_spec(spec_path)
    warnings = validate_spec(spec)
    for w in warnings:
        console.print(f"[yellow]Warning:[/yellow] {w}")

    bundle_path = Path(spec.bundle)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    # --- Load bundle metadata ---
    bundle_metadata: dict | None = None
    bundle_size_mb: float | None = None
    try:
        from leech.bundling import list_bundle_models

        bundle_metadata = list_bundle_models(bundle_path)
        bundle_size_mb = bundle_path.stat().st_size / (1024 * 1024)
    except Exception:
        logger.warning("Could not load bundle metadata; release notes will omit bundle details")

    # --- Render release notes ---
    notes = render_release_notes(spec, bundle_metadata, bundle_size_mb)

    if dry_run:
        console.print(f"\n[bold]Tag:[/bold] {spec.tag}")
        console.print(f"[bold]Title:[/bold] {spec.name} v{spec.version}")
        console.print(f"[bold]Repo:[/bold] {repo}")
        console.print(f"[bold]Prerelease:[/bold] {spec.prerelease}")
        console.print(f"[bold]Assets:[/bold] {bundle_path}")
        for a in spec.extra_assets:
            console.print(f"         {a}")
        console.rule("Release Notes Preview")
        console.print(notes)
        return None

    # --- Handle overwrite ---
    if overwrite:
        try:
            delete_release(spec.tag, repo=repo)
            console.print(f"[yellow]Deleted existing release {spec.tag}[/yellow]")
        except Exception:
            pass  # release didn't exist — fine

    # --- Create release ---
    assets = [bundle_path] + [Path(a) for a in spec.extra_assets]
    title = f"{spec.name} v{spec.version}"

    url = create_release(
        tag=spec.tag,
        title=title,
        body=notes,
        files=assets,
        prerelease=spec.prerelease,
        repo=repo,
    )

    console.print(f"[bold green]Released {spec.tag}[/bold green]")
    if url:
        console.print(f"  {url}")
    return url


# ------------------------------------------------------------------
# list
# ------------------------------------------------------------------


def handle_list(*, repo: str | None = None, limit: int = 30) -> None:
    """List available model releases on GitHub."""
    from leech.release import check_gh_available, get_repo_slug, list_releases

    if not check_gh_available():
        raise RuntimeError("GitHub CLI (gh) is not installed.")

    repo = repo or get_repo_slug()
    releases = list_releases(pattern="model-", repo=repo, limit=limit)

    if not releases:
        console.print("[dim]No model releases found.[/dim]")
        return

    table = Table(title="Model Releases", show_header=True, header_style="bold magenta")
    table.add_column("Tag", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Published", style="dim")
    table.add_column("Pre", style="yellow")

    for r in releases:
        tag = r.get("tagName", "")
        name = r.get("name", "")
        published = r.get("publishedAt", "")[:10]
        pre = "yes" if r.get("isPrerelease") else ""
        table.add_row(tag, name, published, pre)

    console.print(table)


# ------------------------------------------------------------------
# fetch
# ------------------------------------------------------------------


def handle_fetch(
    *,
    name: str | None = None,
    version: str | None = None,
    tag: str | None = None,
    output_dir: Path = Path("."),
    repo: str | None = None,
) -> Path:
    """Download a model bundle from a GitHub release.

    Specify either (name + version) or tag directly.  Returns the
    path to the downloaded bundle.
    """
    from leech.release import (
        check_gh_available,
        download_release_asset,
        get_repo_slug,
    )

    if not check_gh_available():
        raise RuntimeError("GitHub CLI (gh) is not installed.")

    if tag is None:
        if name is None or version is None:
            raise ValueError("Provide either --tag or both --name and --version")
        tag = f"model-{name}-v{version}"

    repo = repo or get_repo_slug()
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"Downloading assets from [cyan]{tag}[/cyan] ...")
    download_release_asset(tag, "*.pt", output_dir, repo=repo)

    # Find downloaded bundle
    bundles = sorted(output_dir.glob("*.pt"))
    if not bundles:
        raise FileNotFoundError(f"No .pt file found after downloading release {tag}")

    bundle_path = bundles[0]
    size_mb = bundle_path.stat().st_size / (1024 * 1024)
    console.print(f"[bold green]Downloaded:[/bold green] {bundle_path} ({size_mb:.1f} MB)")

    # Quick verification
    try:
        from leech.bundling import list_bundle_models

        meta = list_bundle_models(bundle_path)
        console.print(
            f"  {meta.get('num_models', '?')} models, "
            f"arch={meta.get('architecture', '?')}, "
            f"type={meta.get('comparison_type', '?')}"
        )
    except Exception:
        console.print("[yellow]Warning: could not verify bundle metadata[/yellow]")

    return bundle_path
