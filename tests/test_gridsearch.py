"""Tests for grid search parsing utilities."""

import csv
import inspect

import numpy as np
import pytest

import leech.gridsearch
from leech.chunking import load_chunks, save_chunks
from leech.gridsearch import (
    GridSearchConfig,
    parse_context_grid,
    parse_values,
    run_grid_point,
    run_grid_search,
)


class TestParseValues:
    """Tests for parse_values."""

    def test_range_syntax(self):
        assert parse_values("200:1000:200") == [200, 400, 600, 800, 1000]

    def test_range_syntax_inclusive_stop(self):
        """Stop value is included when step divides evenly."""
        assert parse_values("0:1000:500") == [0, 500, 1000]

    def test_range_syntax_stop_not_evenly_divisible(self):
        """Stop value is excluded when step doesn't divide evenly."""
        assert parse_values("200:1000:300") == [200, 500, 800]

    def test_range_syntax_single_step(self):
        assert parse_values("100:100:50") == [100]

    def test_comma_separated(self):
        assert parse_values("200,500,1000") == [200, 500, 1000]

    def test_comma_separated_with_spaces(self):
        assert parse_values("200, 500, 1000") == [200, 500, 1000]

    def test_single_value(self):
        assert parse_values("500") == [500]

    def test_single_value_zero(self):
        assert parse_values("0") == [0]

    def test_range_invalid_parts(self):
        with pytest.raises(ValueError, match="must be start:stop:step"):
            parse_values("100:200")

    def test_range_four_parts(self):
        with pytest.raises(ValueError, match="must be start:stop:step"):
            parse_values("100:200:50:10")

    def test_range_non_integer(self):
        with pytest.raises(ValueError, match="non-integer"):
            parse_values("100:abc:50")

    def test_range_zero_step(self):
        with pytest.raises(ValueError, match="Step must be positive"):
            parse_values("100:200:0")

    def test_range_negative_step(self):
        with pytest.raises(ValueError, match="Step must be positive"):
            parse_values("100:200:-50")

    def test_range_start_greater_than_stop(self):
        with pytest.raises(ValueError, match="must be <= stop"):
            parse_values("1000:200:100")

    def test_invalid_single_value(self):
        with pytest.raises(ValueError):
            parse_values("abc")

    def test_whitespace_stripped(self):
        assert parse_values("  500  ") == [500]


class TestParseContextGrid:
    """Tests for parse_context_grid."""

    def test_symmetric_comma(self):
        left, right = parse_context_grid("200,500,1000")
        assert left == [200, 500, 1000]
        assert right == [200, 500, 1000]

    def test_symmetric_range(self):
        left, right = parse_context_grid("200:1000:200")
        assert left == [200, 400, 600, 800, 1000]
        assert right == [200, 400, 600, 800, 1000]

    def test_overrides(self):
        left, right = parse_context_grid(
            "200,500", left_contexts="100:300:100", right_contexts="400,500"
        )
        assert left == [100, 200, 300]
        assert right == [400, 500]

    def test_left_override_only(self):
        left, right = parse_context_grid("200,500", left_contexts="100:300:100")
        assert left == [100, 200, 300]
        assert right == [200, 500]

    def test_right_override_only(self):
        left, right = parse_context_grid("200,500", right_contexts="100:300:100")
        assert left == [200, 500]
        assert right == [100, 200, 300]


# ---------------------------------------------------------------------------
# Grid points must reach train_model by path, not as a pre-loaded corpus.
# ---------------------------------------------------------------------------


def _corpus(path, n=24):
    """A small corpus in the flat, row-streamable format."""
    rng = np.random.default_rng(3)
    chunks = []
    for i in range(n):
        chunks.append(
            {
                "signal": rng.standard_normal(400).astype(np.float32),
                "dwell": rng.integers(2, 12, 11).astype(np.float32),
                "features": rng.standard_normal((5, 11)).astype(np.float32),
                "sequence": "ACGTACGTACG",
                "label": "pos" if i % 3 else "neg",
                "label_int": 1 if i % 3 else 0,
                "read_id": f"read_{i:05d}",
                "base_idx": 10 + (i % 5),
                "source_group": f"grp{i % 2}",
                "feature_start": -5,
                "feature_end": 5,
                "seq_to_sig_map": np.linspace(0, 400, 12).astype(np.int64),
                "sequence_with_kmer_context": "ACGT" * 7,
                "focus_signal_pos": 200,
            }
        )
    save_chunks(chunks, path)
    return chunks


@pytest.fixture
def grid_corpus(tmp_path):
    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    _corpus(train)
    _corpus(val, n=12)
    return train, val


def _fake_history():
    """Everything run_grid_point reads out of a training history."""
    return {
        "train_loss": [0.5],
        "train_acc": [0.5],
        "val_loss": [0.5],
        "val_acc": [0.5],
        "val_auc": [0.5],
        "val_f1": [0.5],
    }


def _config(train, val, output_dir, **kwargs):
    return GridSearchConfig(
        train_data_path=train,
        val_data_path=val,
        model_name="ConvLSTMDwell",
        output_dir=output_dir,
        left_contexts=[200],
        right_contexts=[200],
        kmer_context=5,
        epochs=1,
        batch_size=8,
        learning_rate=0.001,
        device="cpu",
        seed=42,
        num_workers=0,
        motif="CCAGGC",
        **kwargs,
    )


class TestGridSearchStreamsTheCorpus:
    """The eager `chunks=` branch is the #211 memory profile, per worker."""

    def test_run_grid_point_takes_no_pre_loaded_chunks(self):
        """A grid point is described by paths; there is nothing to hand it.

        Accepting chunks is what routed LeechDataset down its eager branch:
        the whole numpy corpus resident alongside the tensors built from it.
        """
        parameters = inspect.signature(run_grid_point).parameters

        assert "train_chunks" not in parameters
        assert "val_chunks" not in parameters

    def test_grid_search_hands_train_model_a_path(self, grid_corpus, tmp_path, monkeypatch):
        """Every grid point reaches train_model by path, with no chunk list."""
        train, val = grid_corpus
        seen = []

        def spy(**kwargs):
            seen.append(kwargs)
            return _fake_history()

        monkeypatch.setattr(leech.gridsearch, "train_model", spy)
        run_grid_search(_config(train, val, tmp_path / "grid"))

        assert seen
        for call in seen:
            assert call["train_data_path"] == train
            assert call["val_data_path"] == val
            assert call.get("train_chunks") is None
            assert call.get("val_chunks") is None

    def test_grid_search_does_not_pre_load_the_corpus(self, grid_corpus, tmp_path, monkeypatch):
        """Nothing decompresses the per-chunk arrays before training starts."""
        train, val = grid_corpus
        loads = []

        original = leech.gridsearch.load_chunks

        def spy(path, *args, **kwargs):
            loads.append(path)
            return original(path, *args, **kwargs)

        monkeypatch.setattr(leech.gridsearch, "load_chunks", spy)
        monkeypatch.setattr(leech.gridsearch, "train_model", lambda **kwargs: _fake_history())
        run_grid_search(_config(train, val, tmp_path / "grid"))

        assert loads == [], f"grid search loaded whole corpora: {loads}"

    def test_every_sequential_grid_point_succeeds(self, grid_corpus, tmp_path):
        """Sharing one pre-loaded chunk list breaks every point after the first.

        LeechDataset nulls out each chunk's arrays once it has tensorized them,
        so the second grid point received chunks whose ``signal`` was None and
        died with ``'NoneType' object has no attribute 'dtype'``. Reading the
        corpus per grid point is what makes a multi-point run work at all.
        """
        train, val = grid_corpus
        config = _config(train, val, tmp_path / "grid")
        config.left_contexts = [200, 240]

        run_grid_search(config)

        with open(tmp_path / "grid" / "grid_summary.csv") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 2
        assert [row["status"] for row in rows] == ["success", "success"]

    def test_label_column_matches_load_chunks(self, grid_corpus):
        """The class weights are computed from the same labels as before."""
        from leech.gridsearch import _training_label_column

        train, _ = grid_corpus

        expected = [c["label_int"] for c in load_chunks(train) if c["label_int"] is not None]

        assert _training_label_column(train).tolist() == expected

    def test_pos_weight_reaches_every_grid_point(self, grid_corpus, tmp_path, monkeypatch):
        """pos_weight travelled in a pool global; it now travels in the args."""
        from leech.gridsearch import _training_label_column

        train, val = grid_corpus
        seen = []

        def spy(**kwargs):
            seen.append(kwargs.get("pos_weight"))
            return _fake_history()

        monkeypatch.setattr(leech.gridsearch, "train_model", spy)
        run_grid_search(_config(train, val, tmp_path / "grid"))

        labels = _training_label_column(train)
        expected = float((labels == 0).sum()) / float((labels == 1).sum())

        assert seen == [pytest.approx(expected)]
