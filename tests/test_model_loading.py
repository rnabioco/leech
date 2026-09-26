"""Tests for leech.model_loading."""

import logging

import pytest

from leech.model_loading import resolve_dwell_template_path


class TestResolveDwellTemplatePath:
    """#369: a checkpoint's ``dwell_template_table`` must resolve against
    the checkpoint directory, not the process's current working directory,
    while staying loadable for a pre-existing checkpoint that recorded a
    CWD-relative path.
    """

    def test_returns_none_when_unset(self, tmp_path):
        assert resolve_dwell_template_path({}, tmp_path) is None
        assert resolve_dwell_template_path({"dwell_template_table": None}, tmp_path) is None
        assert resolve_dwell_template_path({"dwell_template_table": ""}, tmp_path) is None

    def test_absolute_path_returned_as_is(self, tmp_path):
        table_path = tmp_path / "somewhere" / "table.tsv"
        config = {"dwell_template_table": str(table_path)}

        resolved = resolve_dwell_template_path(config, tmp_path / "model_dir")

        assert resolved == table_path

    def test_relative_path_resolves_in_model_dir(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        table_path = model_dir / "dwell_template_table.tsv"
        table_path.write_text("aa\tposition\tdwell_mean\n")
        config = {"dwell_template_table": "dwell_template_table.tsv"}

        resolved = resolve_dwell_template_path(config, model_dir)

        assert resolved == table_path
        assert resolved.exists()

    def test_legacy_cwd_fallback_when_not_in_model_dir(self, tmp_path, monkeypatch, caplog):
        """A checkpoint trained before #369 recorded whatever string the CLI
        got -- often a repo-relative path -- which only ever resolved
        against the CWD. That must keep working while the file is still
        there, with a warning naming both candidates."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        cwd = tmp_path / "repo"
        cwd.mkdir()
        legacy_relative = "report/aa_transit_table.tsv"
        legacy_table = cwd / legacy_relative
        legacy_table.parent.mkdir(parents=True)
        legacy_table.write_text("aa\tposition\tdwell_mean\n")
        monkeypatch.chdir(cwd)

        config = {"dwell_template_table": legacy_relative}

        with caplog.at_level(logging.WARNING, logger="leech.model_loading"):
            resolved = resolve_dwell_template_path(config, model_dir)

        assert resolved.resolve() == legacy_table.resolve()
        assert any("not found" in r.getMessage() for r in caplog.records)

    def test_raises_when_neither_candidate_exists(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        empty_cwd = tmp_path / "elsewhere"
        empty_cwd.mkdir()
        monkeypatch.chdir(empty_cwd)

        config = {"dwell_template_table": "missing_table.tsv"}

        with pytest.raises(FileNotFoundError, match="missing_table.tsv"):
            resolve_dwell_template_path(config, model_dir)

    def test_base_dir_none_falls_back_to_cwd_only(self, tmp_path, monkeypatch):
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        table_path = cwd / "table.tsv"
        table_path.write_text("aa\tposition\tdwell_mean\n")
        monkeypatch.chdir(cwd)

        config = {"dwell_template_table": "table.tsv"}

        resolved = resolve_dwell_template_path(config, None)

        assert resolved.resolve() == table_path.resolve()
