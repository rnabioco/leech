"""Tests for evaluation helpers.

`_save_scores` is where the per-chunk scores are joined back to read ids, and
the join is positional -- so these tests are mostly about the failure mode that
join has, not about the happy path.

The rest covers how the eval DataLoader is sized: a worker-less loader on a GPU
is what left issue #205 running at 8% utilisation.
"""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import leech.dataset as dataset
from leech.constants import AUTO_DATALOADER_WORKERS
from leech.dataset import resolve_dataloader_workers
from leech.evaluation import _save_scores

READ_IDS = ["read-a", "read-b", "read-c", "read-d"]
LABELS = np.array([0, 1, 1, 0])
PROBS = np.array([0.1, 0.8, 0.9, 0.2])


def _test_npz(tmp_path, read_ids=READ_IDS):
    path = tmp_path / "test.npz"
    arrays = {"signals": np.zeros((len(LABELS), 4), dtype=np.float32)}
    if read_ids is not None:
        arrays["read_ids"] = np.array(read_ids, dtype=str)
    np.savez(path, **arrays)
    return path


class TestSaveScores:
    def test_scores_are_keyed_by_read_id(self, tmp_path):
        """Every score comes back attached to the read it came from."""
        out = tmp_path / "scores.npz"

        _save_scores(out, _test_npz(tmp_path), LABELS, PROBS)

        with np.load(out) as got:
            assert list(got["read_ids"]) == READ_IDS
            np.testing.assert_array_equal(got["labels"], LABELS)
            np.testing.assert_allclose(got["probs"], PROBS)

    def test_length_mismatch_raises(self, tmp_path):
        """A short score array must fail loudly, not shift the mapping.

        Row order is the only key. If the dataset ever filters or reorders
        chunks, every score after the first dropped one would be attributed to
        the wrong read -- and the file would look perfectly well-formed.
        """
        out = tmp_path / "scores.npz"

        with pytest.raises(ValueError, match="misattribute"):
            _save_scores(out, _test_npz(tmp_path), LABELS[:3], PROBS[:3])

        assert not out.exists()

    def test_missing_read_ids_degrades_to_positional(self, tmp_path):
        """A test set without read_ids still yields scores, minus the key."""
        out = tmp_path / "scores.npz"

        _save_scores(out, _test_npz(tmp_path, read_ids=None), LABELS, PROBS)

        with np.load(out) as got:
            assert "read_ids" not in got
            np.testing.assert_allclose(got["probs"], PROBS)

    def test_multiclass_probabilities_survive(self, tmp_path):
        """Multiclass scores are (N, C) and must not be flattened."""
        out = tmp_path / "scores.npz"
        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.5, 0.3, 0.2]])

        _save_scores(out, _test_npz(tmp_path), LABELS, probs)

        with np.load(out) as got:
            assert got["probs"].shape == (4, 3)

    def test_creates_parent_directory(self, tmp_path):
        """The caller should not have to mkdir first."""
        out = tmp_path / "nested" / "dir" / "scores.npz"

        _save_scores(out, _test_npz(tmp_path), LABELS, PROBS)

        assert out.exists()


class TestDataLoaderWorkers:
    """Eval must feed the GPU from more than one process (issue #205).

    ``eval test`` built its loader with ``num_workers`` pinned to 0, so collate,
    the host-to-device copy and the forward pass all ran serially in one Python
    process: 8% GPU utilisation on an A5000 while training on the same corpus
    and hardware ran at 98%.
    """

    def test_cuda_auto_gets_workers(self, monkeypatch):
        """0 means auto, and auto on a GPU is not zero."""
        monkeypatch.setattr(dataset, "_usable_cpus", lambda: 32)

        assert AUTO_DATALOADER_WORKERS > 0
        assert resolve_dataloader_workers(0, "cuda") == AUTO_DATALOADER_WORKERS

    def test_cpu_auto_stays_serial(self):
        """On CPU the workers would compete with the compute for the same cores."""
        assert resolve_dataloader_workers(0, "cpu") == 0

    def test_auto_fits_the_cpu_allocation(self, monkeypatch):
        """A GPU job given 2 cores must not fork 8 workers onto them."""
        monkeypatch.setattr(dataset, "_usable_cpus", lambda: 2)

        assert resolve_dataloader_workers(0, "cuda") == 1

    def test_auto_keeps_one_worker_on_a_single_core(self, monkeypatch):
        """Even one worker decouples collate and the H2D copy from the forward pass."""
        monkeypatch.setattr(dataset, "_usable_cpus", lambda: 1)

        assert resolve_dataloader_workers(0, "cuda") == 1

    def test_explicit_request_wins(self, monkeypatch):
        """Only auto is capped; a caller who asks for N gets N."""
        monkeypatch.setattr(dataset, "_usable_cpus", lambda: 2)

        assert resolve_dataloader_workers(3, "cuda") == 3
        assert resolve_dataloader_workers(3, "cpu") == 3

    def test_daemon_forces_zero(self, monkeypatch):
        """A pool worker (grid search) cannot spawn children; a loader with
        workers raises there, whatever the caller asked for."""
        import multiprocessing

        monkeypatch.setattr(
            multiprocessing, "current_process", lambda: SimpleNamespace(daemon=True)
        )

        assert resolve_dataloader_workers(8, "cuda") == 0

    def test_cli_forwards_num_workers(self, tmp_path):
        """--num-workers reaches evaluate_model, so a caller can fix this from
        outside even where the auto default is wrong."""
        from click.testing import CliRunner

        import leech.evaluation as evaluation
        from leech.cli import cli

        captured: dict = {}
        model = tmp_path / "model.pt"
        model.touch()
        test_data = tmp_path / "test.npz"
        test_data.touch()

        with mock.patch.object(evaluation, "evaluate_model", captured.update):
            result = CliRunner().invoke(
                cli,
                [
                    "eval",
                    "test",
                    "--model",
                    str(model),
                    "--test-data",
                    str(test_data),
                    "--output",
                    str(tmp_path / "metrics.json"),
                    "--num-workers",
                    "4",
                ],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured["num_workers"] == 4
