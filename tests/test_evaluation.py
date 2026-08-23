"""Tests for evaluation helpers.

`_save_scores` is where the per-chunk scores are joined back to read ids, and
the join is positional -- so these tests are mostly about the failure mode that
join has, not about the happy path.
"""

import numpy as np
import pytest

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
