"""GitHub Actions workflows: parse, and reference files that exist.

Nothing else in this repo reads these. A malformed workflow is reported only by
GitHub, only after a push, and a scheduled one that never fires looks exactly
like a scheduled one with nothing to report -- so it fails silently in the
direction that matters.

Written after shipping invalid YAML: a heredoc inside a `run: |` block has to
be indented to stay inside the block scalar, and at column 1 it terminates it.
The file looked fine and parsed as something else entirely.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))


def test_there_are_workflows_to_check():
    """Guards the glob: an empty list would make every test below vacuous."""
    assert WORKFLOWS, "no workflows found — has the directory moved?"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_parses_and_has_jobs(path: Path):
    document = yaml.safe_load(path.read_text())

    assert isinstance(document, dict), f"{path.name} is not a mapping"
    assert document.get("jobs"), f"{path.name} declares no jobs"
    for name, job in document["jobs"].items():
        assert job.get("steps") or job.get("uses"), f"{path.name}:{name} does nothing"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_run_steps_are_strings(path: Path):
    """A block scalar that swallowed the next key yields a dict, not a string.

    That is precisely the failure this file exists for, and it is invisible in
    a diff: the YAML is still valid, it just means something else.
    """
    document = yaml.safe_load(path.read_text())
    for name, job in document["jobs"].items():
        for index, step in enumerate(job.get("steps", [])):
            if "run" in step:
                assert isinstance(step["run"], str), f"{path.name}:{name} step {index}"


def test_the_escapepod_sync_workflow_references_files_that_exist():
    """A rename of the script or the PR body would break it only at 06:17 UTC."""
    workflow = REPO_ROOT / ".github" / "workflows" / "escapepod-sync.yml"
    text = workflow.read_text()

    for referenced in (
        ".github/scripts/escapepod_sync.py",
        ".github/escapepod-sync-pr-body.md",
    ):
        assert referenced in text, f"{referenced} no longer referenced"
        assert (REPO_ROOT / referenced).is_file(), f"{referenced} does not exist"
