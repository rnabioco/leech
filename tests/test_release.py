"""Tests for the model release workflow (spec parsing, notes rendering, gh wrapper)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from leech.release import (
    MetricsSummary,
    ReleaseSpec,
    TrainingProvenance,
    load_release_spec,
    render_release_notes,
    validate_spec,
)

# ======================================================================
# Fixtures
# ======================================================================

SAMPLE_SPEC_YAML = """\
name: ivc-pairwise-v1
version: 1.0.0
description: |
  Pairwise classification models for 12 amino acids.
bundle: {bundle_path}

organism: "E. coli"
experiment: "in vitro charging (IVC)"
data_version: "2026-03"

training:
  pipeline_commit: "abc1234"
  leech_version: "0.3.1"
  config_file: "config/config.yaml"
  date: "2026-03-20"

metrics:
  mean_accuracy: 0.95
  mean_f1: 0.93
  mean_auroc: 0.97

prerelease: false
extra_assets: []
"""


@pytest.fixture
def spec_file(tmp_path: Path) -> Path:
    """Write a sample YAML spec and a dummy bundle file."""
    bundle = tmp_path / "model.pt"
    bundle.write_bytes(b"dummy")
    spec_path = tmp_path / "release.yaml"
    spec_path.write_text(SAMPLE_SPEC_YAML.format(bundle_path=str(bundle)))
    return spec_path


@pytest.fixture
def sample_spec(tmp_path: Path) -> ReleaseSpec:
    """Return a fully-populated ReleaseSpec."""
    bundle = tmp_path / "model.pt"
    bundle.write_bytes(b"dummy")
    return ReleaseSpec(
        name="ivc-pairwise-v1",
        version="1.0.0",
        description="Pairwise classification models for 12 amino acids.",
        bundle=str(bundle),
        organism="E. coli",
        experiment="in vitro charging (IVC)",
        data_version="2026-03",
        training=TrainingProvenance(
            pipeline_commit="abc1234",
            leech_version="0.3.1",
            config_file="config/config.yaml",
            date="2026-03-20",
        ),
        metrics=MetricsSummary(
            mean_accuracy=0.95,
            mean_f1=0.93,
            mean_auroc=0.97,
        ),
    )


SAMPLE_BUNDLE_META = {
    "architecture": "ConvLSTMDwell",
    "comparison_type": "pairwise",
    "num_models": 66,
    "vmap": False,
    "torchscript": False,
    "pairs": ["Ala_vs_Gly", "Ala_vs_Lys"],
}


# ======================================================================
# TestReleaseSpec
# ======================================================================


class TestReleaseSpec:
    def test_load_spec(self, spec_file: Path):
        spec = load_release_spec(spec_file)
        assert spec.name == "ivc-pairwise-v1"
        assert spec.version == "1.0.0"
        assert spec.organism == "E. coli"
        assert spec.training.leech_version == "0.3.1"
        assert spec.metrics.mean_accuracy == 0.95

    def test_tag_property(self, sample_spec: ReleaseSpec):
        assert sample_spec.tag == "model-ivc-pairwise-v1-v1.0.0"

    def test_missing_required_field(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: foo\nversion: '1.0'\n")
        with pytest.raises(ValueError, match="Missing required field 'description'"):
            load_release_spec(bad)

    def test_validate_missing_bundle(self, sample_spec: ReleaseSpec):
        sample_spec.bundle = "/nonexistent/path.pt"
        warnings = validate_spec(sample_spec)
        assert any("not found" in w for w in warnings)

    def test_validate_clean(self, sample_spec: ReleaseSpec):
        warnings = validate_spec(sample_spec)
        assert warnings == []

    def test_extra_metrics(self, tmp_path: Path):
        bundle = tmp_path / "model.pt"
        bundle.write_bytes(b"dummy")
        yaml_text = f"""\
name: test
version: "0.1"
description: test
bundle: {bundle}
metrics:
  mean_accuracy: 0.9
  custom_metric: 0.88
"""
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml_text)
        spec = load_release_spec(spec_path)
        assert spec.metrics.extra["custom_metric"] == 0.88

    def test_defaults(self, tmp_path: Path):
        bundle = tmp_path / "model.pt"
        bundle.write_bytes(b"dummy")
        yaml_text = f"""\
name: minimal
version: "0.1"
description: bare minimum
bundle: {bundle}
"""
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml_text)
        spec = load_release_spec(spec_path)
        assert spec.prerelease is False
        assert spec.extra_assets == []
        assert spec.training.leech_version is None
        assert spec.metrics.mean_accuracy is None


# ======================================================================
# TestReleaseNotes
# ======================================================================


class TestReleaseNotes:
    def test_basic_render(self, sample_spec: ReleaseSpec):
        notes = render_release_notes(sample_spec, SAMPLE_BUNDLE_META, 42.5)
        assert "Pairwise classification" in notes
        assert "ConvLSTMDwell" in notes
        assert "42.5 MB" in notes
        assert "0.9500" in notes  # mean_accuracy
        assert sample_spec.tag in notes

    def test_render_without_metrics(self, sample_spec: ReleaseSpec):
        sample_spec.metrics = MetricsSummary()
        notes = render_release_notes(sample_spec, SAMPLE_BUNDLE_META, 10.0)
        assert "Performance" not in notes

    def test_render_without_bundle_meta(self, sample_spec: ReleaseSpec):
        notes = render_release_notes(sample_spec, None, None)
        assert "Bundle Details" not in notes
        assert "Download & Usage" in notes

    def test_large_pair_list(self, sample_spec: ReleaseSpec):
        meta = dict(SAMPLE_BUNDLE_META)
        meta["pairs"] = [f"pair_{i}" for i in range(50)]
        notes = render_release_notes(sample_spec, meta, 1.0)
        assert "50 total" in notes


# ======================================================================
# TestGitHubWrapper
# ======================================================================


def _mock_run(returncode=0, stdout="", stderr=""):
    """Build a mock CompletedProcess."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestGitHubWrapper:
    @patch("subprocess.run")
    def test_check_gh_available(self, mock_run):
        mock_run.return_value = _mock_run(stdout="gh version 2.0")
        from leech.release.github import check_gh_available

        assert check_gh_available() is True

    @patch("subprocess.run")
    def test_check_gh_not_available(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        from leech.release.github import check_gh_available

        assert check_gh_available() is False

    @patch("subprocess.run")
    def test_check_gh_authenticated(self, mock_run):
        mock_run.return_value = _mock_run()
        from leech.release.github import check_gh_authenticated

        assert check_gh_authenticated() is True

    @patch("subprocess.run")
    def test_check_gh_unauthenticated(self, mock_run):
        mock_run.return_value = _mock_run(returncode=1, stderr="not logged in")
        from leech.release.github import check_gh_authenticated

        assert check_gh_authenticated() is False

    @patch("subprocess.run")
    def test_get_repo_slug_https(self, mock_run):
        mock_run.return_value = _mock_run(stdout="https://github.com/owner/repo.git\n")
        from leech.release.github import get_repo_slug

        assert get_repo_slug() == "owner/repo"

    @patch("subprocess.run")
    def test_get_repo_slug_ssh(self, mock_run):
        mock_run.return_value = _mock_run(stdout="git@github.com:owner/repo.git\n")
        from leech.release.github import get_repo_slug

        assert get_repo_slug() == "owner/repo"

    @patch("leech.release.github._run_gh")
    def test_create_release(self, mock_gh):
        mock_gh.return_value = _mock_run(stdout="https://github.com/owner/repo/releases/tag/v1\n")
        from leech.release.github import create_release

        url = create_release(
            tag="model-test-v1.0",
            title="test v1.0",
            body="notes",
            files=[Path("bundle.pt")],
        )
        assert url == "https://github.com/owner/repo/releases/tag/v1"

    @patch("leech.release.github._run_gh")
    def test_create_release_dry_run(self, mock_gh):
        from leech.release.github import create_release

        url = create_release(
            tag="model-test-v1.0",
            title="test v1.0",
            body="notes",
            files=[],
            dry_run=True,
        )
        assert url is None
        mock_gh.assert_not_called()

    @patch("leech.release.github._run_gh")
    def test_list_releases(self, mock_gh):
        releases = [
            {"tagName": "model-foo-v1.0", "name": "foo", "publishedAt": "2026-01-01", "isPrerelease": False, "isDraft": False},
            {"tagName": "v0.3.1", "name": "code release", "publishedAt": "2026-01-01", "isPrerelease": False, "isDraft": False},
        ]
        mock_gh.return_value = _mock_run(stdout=json.dumps(releases))
        from leech.release.github import list_releases

        result = list_releases(pattern="model-")
        assert len(result) == 1
        assert result[0]["tagName"] == "model-foo-v1.0"

    @patch("leech.release.github._run_gh")
    def test_download_release_asset(self, mock_gh):
        mock_gh.return_value = _mock_run()
        from leech.release.github import download_release_asset

        out = download_release_asset("model-test-v1.0", "*.pt", Path("/tmp/dl"))
        assert out == Path("/tmp/dl")


# ======================================================================
# TestHandleRelease
# ======================================================================


class TestHandleRelease:
    def test_dry_run_skips_gh(self, spec_file: Path):
        """Dry-run should not call create_release."""
        with (
            patch("leech.release.check_gh_available", return_value=True),
            patch("leech.release.check_gh_authenticated", return_value=True),
            patch("leech.release.get_repo_slug", return_value="owner/repo"),
            patch("leech.release.create_release") as mock_create,
            patch("leech.util.list_bundle_models", return_value=SAMPLE_BUNDLE_META),
        ):
            from leech.commands.release_model import handle_release

            result = handle_release(spec_file, dry_run=True)
            assert result is None
            mock_create.assert_not_called()

    def test_overwrite_deletes_first(self, spec_file: Path):
        with (
            patch("leech.release.check_gh_available", return_value=True),
            patch("leech.release.check_gh_authenticated", return_value=True),
            patch("leech.release.get_repo_slug", return_value="owner/repo"),
            patch("leech.release.delete_release") as mock_delete,
            patch("leech.release.create_release", return_value="https://url"),
            patch("leech.util.list_bundle_models", return_value=SAMPLE_BUNDLE_META),
        ):
            from leech.commands.release_model import handle_release

            handle_release(spec_file, overwrite=True)
            mock_delete.assert_called_once()


# ======================================================================
# TestHandleFetch
# ======================================================================


class TestHandleFetch:
    def test_fetch_by_name_version(self, tmp_path: Path):
        """Constructs correct tag from name + version."""
        bundle = tmp_path / "model.pt"
        bundle.write_bytes(b"dummy")

        with (
            patch("leech.release.check_gh_available", return_value=True),
            patch("leech.release.get_repo_slug", return_value="owner/repo"),
            patch("leech.release.download_release_asset") as mock_dl,
            patch("leech.util.list_bundle_models", return_value=SAMPLE_BUNDLE_META),
        ):
            mock_dl.return_value = tmp_path
            from leech.commands.release_model import handle_fetch

            result = handle_fetch(name="foo", version="2.0", output_dir=tmp_path)
            assert result == bundle
            call_args = mock_dl.call_args
            assert call_args[0][0] == "model-foo-v2.0"

    def test_fetch_by_tag(self, tmp_path: Path):
        bundle = tmp_path / "model.pt"
        bundle.write_bytes(b"dummy")

        with (
            patch("leech.release.check_gh_available", return_value=True),
            patch("leech.release.get_repo_slug", return_value="owner/repo"),
            patch("leech.release.download_release_asset") as mock_dl,
            patch("leech.util.list_bundle_models", return_value=SAMPLE_BUNDLE_META),
        ):
            mock_dl.return_value = tmp_path
            from leech.commands.release_model import handle_fetch

            handle_fetch(tag="model-custom-v3.0", output_dir=tmp_path)
            call_args = mock_dl.call_args
            assert call_args[0][0] == "model-custom-v3.0"

    def test_fetch_missing_args(self):
        with (
            patch("leech.release.check_gh_available", return_value=True),
            patch("leech.release.get_repo_slug", return_value="owner/repo"),
        ):
            from leech.commands.release_model import handle_fetch

            with pytest.raises(ValueError, match="Provide either"):
                handle_fetch(name="foo")
