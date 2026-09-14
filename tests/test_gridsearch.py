"""Tests for grid search parsing utilities."""

import csv
import inspect

import numpy as np
import pytest

import leech.gridsearch
from leech.chunking import load_chunks, save_chunks
from leech.configs import TrainConfig
from leech.gridsearch import (
    GridSearchConfig,
    _resolve_selection_metric,
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


def _config(train, val, output_dir, cfg=None, **kwargs):
    """Build a GridSearchConfig for tests.

    ``cfg`` overrides the default training recipe (TrainConfig); other
    keyword arguments apply to GridSearchConfig's own sweep/runtime fields
    (n_parallel, device, ...) -- the two are separate dataclasses since #270.
    """
    return GridSearchConfig(
        train_data_path=train,
        val_data_path=val,
        model_name="ConvLSTMDwell",
        output_dir=output_dir,
        left_contexts=[200],
        right_contexts=[200],
        cfg=cfg if cfg is not None else TrainConfig(epochs=1, batch_size=8, motif="CCAGGC"),
        kmer_context=5,
        device="cpu",
        seed=42,
        num_workers=0,
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
            seen.append(kwargs["cfg"].pos_weight)
            return _fake_history()

        monkeypatch.setattr(leech.gridsearch, "train_model", spy)
        run_grid_search(_config(train, val, tmp_path / "grid"))

        labels = _training_label_column(train)
        expected = float((labels == 0).sum()) / float((labels == 1).sum())

        assert seen == [pytest.approx(expected)]

    def test_grid_point_worker_and_sequential_path_pass_exactly_run_grid_points_keys(
        self, grid_corpus, tmp_path, monkeypatch
    ):
        """Both dispatch paths must pass run_grid_point exactly its parameters.

        `_grid_point_worker` used to hand-list ~35 of run_grid_point's
        parameters and simply missed `oversample_minority` (#270):
        `--parallel > 1` always trained with it off while `--parallel 1`
        (`run_grid_point(**args)` directly) honored it. A prior version of
        this test only checked that `_grid_point_worker`'s source contained
        the literal substring `"run_grid_point(**args)"` -- a source-text
        match that would stay green even if `grid_args` drifted out of sync
        with a signature change, since nothing it asserts is actually about
        keys. Issue #270's own acceptance criteria calls for comparing
        `inspect.signature(run_grid_point)` against the keys actually
        passed, so this does that directly: it records the kwargs
        run_grid_point is called with (in-process, so fork-based
        `--parallel` dispatch isn't exercised here -- both dispatchers draw
        from the identical `grid_args` list built once in run_grid_search,
        so proving that list's shape here covers both).
        """
        import leech.gridsearch as gs
        from leech.gridsearch import _grid_point_worker

        train, val = grid_corpus
        expected_params = set(inspect.signature(gs.run_grid_point).parameters.keys())

        monkeypatch.setattr(gs, "train_model", lambda **kwargs: _fake_history())

        seen_keys: list[set] = []
        original_run_grid_point = gs.run_grid_point

        def recording_run_grid_point(**kwargs):
            seen_keys.append(set(kwargs.keys()))
            return original_run_grid_point(**kwargs)

        monkeypatch.setattr(gs, "run_grid_point", recording_run_grid_point)

        run_grid_search(_config(train, val, tmp_path / "seq", n_parallel=1))

        assert seen_keys, "run_grid_point was never called"
        for keys in seen_keys:
            assert keys == expected_params, (
                f"grid_args keys {keys} != run_grid_point's parameters {expected_params}"
            )

        # _grid_point_worker's own forwarding, called directly (no fork, so
        # the same in-process recorder observes it) with a hand-built args
        # dict -- this is what --parallel > 1 actually dispatches, and must
        # forward the identical key set, not a re-listed subset.
        seen_keys.clear()
        worker_args = {
            "train_data_path": train,
            "val_data_path": val,
            "model_name": "ConvLSTMDwell",
            "output_dir": tmp_path / "worker",
            "left_context": 200,
            "right_context": 200,
            "kmer_len": 11,
            "device": "cpu",
            "seed": 42,
            "cfg": TrainConfig(epochs=1, batch_size=8, motif="CCAGGC"),
            "dwell_offset": 0,
            "pos_weight": None,
            "num_workers": 0,
            "selection_metric": "auto",
        }
        assert set(worker_args.keys()) == expected_params, (
            "this test's own args dict has drifted from run_grid_point's signature"
        )
        _grid_point_worker(worker_args)
        assert seen_keys == [expected_params]

    def test_oversample_minority_reaches_train_model_under_parallel(
        self, grid_corpus, tmp_path, monkeypatch
    ):
        """--parallel > 1 must train the same recipe as --parallel 1 (#270).

        n_parallel > 1 runs each grid point in a forked worker process, so a
        plain in-memory spy would silently lose its writes across the fork
        (copy-on-write memory, not shared) -- this records what it saw to a
        file instead, which is real, shared-filesystem I/O either way.
        """
        train, val = grid_corpus
        recipe = TrainConfig(epochs=1, batch_size=8, motif="CCAGGC", oversample_minority=True)

        def make_spy(record_path):
            def spy(**kwargs):
                with open(record_path, "a") as f:
                    f.write(f"{kwargs['cfg'].oversample_minority}\n")
                return _fake_history()

            return spy

        sequential_record = tmp_path / "sequential_seen.txt"
        monkeypatch.setattr(leech.gridsearch, "train_model", make_spy(sequential_record))
        run_grid_search(
            _config(
                train,
                val,
                tmp_path / "sequential",
                cfg=recipe,
                n_parallel=1,
            )
        )

        parallel_record = tmp_path / "parallel_seen.txt"
        monkeypatch.setattr(leech.gridsearch, "train_model", make_spy(parallel_record))
        run_grid_search(
            _config(
                train,
                val,
                tmp_path / "parallel",
                cfg=recipe,
                n_parallel=2,
            )
        )

        assert sequential_record.read_text().strip() == "True"
        assert parallel_record.read_text().strip() == "True"


class TestParametricSelectionMetric:
    """--selection-metric accepts tpr_at_fpr:<f> / callable_at_precision:<p>,
    the same parametric names Trainer's --checkpoint-metric does (#280)."""

    def test_resolve_passes_through_parametric_metric_for_binary(self):
        assert _resolve_selection_metric("tpr_at_fpr:0.0034", n_classes=2) == "tpr_at_fpr:0.0034"
        assert (
            _resolve_selection_metric("callable_at_precision:0.99", n_classes=2)
            == "callable_at_precision:0.99"
        )

    def test_resolve_refuses_parametric_metric_for_multiclass(self):
        with pytest.raises(ValueError, match="binary-only"):
            _resolve_selection_metric("tpr_at_fpr:0.1", n_classes=4)

    def test_resolve_refuses_unknown_metric(self):
        with pytest.raises(ValueError, match="selection_metric must be one of"):
            _resolve_selection_metric("not_a_real_metric", n_classes=2)

    def test_grid_search_ranks_on_the_parametric_metric(self, grid_corpus, tmp_path, monkeypatch):
        """run_grid_point/run_grid_search read best_epoch and
        best_val_selection off history["val_selection"] -- the only place a
        parametric metric's per-epoch value lives, since history has no key
        literally named "tpr_at_fpr:0.5"."""
        train, val = grid_corpus

        def fake_history_with_selection(values):
            return {
                "train_loss": [0.5] * len(values),
                "train_acc": [0.5] * len(values),
                "val_loss": [0.5] * len(values),
                "val_acc": [0.5] * len(values),
                "val_auc": [0.5] * len(values),
                "val_f1": [0.5] * len(values),
                "val_selection": values,
            }

        monkeypatch.setattr(
            leech.gridsearch,
            "train_model",
            lambda **kwargs: fake_history_with_selection([0.3, 0.7]),
        )

        config = _config(train, val, tmp_path / "grid", selection_metric="tpr_at_fpr:0.5")
        run_grid_search(config)

        with open(tmp_path / "grid" / "grid_summary.csv") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 1
        assert rows[0]["selection_metric"] == "tpr_at_fpr:0.5"
        assert float(rows[0]["best_val_selection"]) == pytest.approx(0.7)
        assert int(rows[0]["best_epoch"]) == 2
