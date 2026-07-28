"""Model release workflow: YAML spec parsing, release notes, GitHub integration."""

from leech.release.github import (
    GitHubCLIError,
    check_gh_authenticated,
    check_gh_available,
    create_release,
    delete_release,
    download_release_asset,
    get_release_body,
    get_repo_slug,
    list_releases,
)
from leech.release.notes import render_release_notes
from leech.release.spec import (
    MetricsSummary,
    ReleaseSpec,
    TrainingProvenance,
    load_release_spec,
    validate_spec,
)

__all__ = [
    "GitHubCLIError",
    "MetricsSummary",
    "ReleaseSpec",
    "TrainingProvenance",
    "check_gh_authenticated",
    "check_gh_available",
    "create_release",
    "delete_release",
    "download_release_asset",
    "get_release_body",
    "get_repo_slug",
    "list_releases",
    "load_release_spec",
    "render_release_notes",
    "validate_spec",
]
