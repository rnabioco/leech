"""Tests for evaluation helpers.

`_save_scores` is where the per-chunk scores are joined back to read ids, and
the join is positional -- so these tests are mostly about the failure mode that
join has, not about the happy path.

The rest covers how the eval DataLoader is sized: a worker-less loader on a GPU
is what left issue #205 running at 8% utilisation.
"""

import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import leech.dataset as dataset
from leech.constants import AUTO_DATALOADER_WORKERS
from leech.dataset import resolve_dataloader_workers, resolve_val_dataloader_workers
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


class TestFp32ProbabilitiesUnderSimulatedAMP:
    """`evaluate_model` must not inherit AMP's fp16 probability quantization
    (#264): under autocast the final Linear emits float16, and
    torch.sigmoid/torch.softmax are not on autocast's fp32 promotion list --
    computing them on the raw logits rounds the saved probability to ~3
    significant digits and saturates to exactly 1.0 above logit~11.

    This test machine has no CUDA, so real autocast can't be exercised.
    Instead the model's forward pass is patched to hand back a genuine
    float16 tensor, exactly what CUDA autocast's Linear layer would produce.
    """

    def _train_checkpoint(self, temp_chunks_file, tmp_path):
        from leech.training import train_model

        model_dir = tmp_path / "model"
        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=None,
            model_name="ConvLSTMDwell",
            output_dir=model_dir,
            epochs=1,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            seed=42,
        )
        return model_dir

    def test_emitted_scores_are_float32_even_with_fp16_model_output(
        self, temp_chunks_file, tmp_path, monkeypatch
    ):
        """The scores written to disk via --emit-scores are float32, with a
        resolution no worse than the underlying (already-fp16) logits --
        never further quantized by an un-cast sigmoid/softmax."""
        from leech.evaluation import evaluate_model
        from leech.models.inference_wrapper import ModelInferenceWrapper

        model_dir = self._train_checkpoint(temp_chunks_file, tmp_path)

        # Simulate exactly what CUDA autocast's final Linear hands back.
        original_forward = ModelInferenceWrapper.forward_batch
        monkeypatch.setattr(
            ModelInferenceWrapper,
            "forward_batch",
            lambda self, batch, device: original_forward(self, batch, device).half(),
        )

        scores_path = tmp_path / "scores.npz"
        evaluate_model(
            model_path=model_dir,
            test_data_path=temp_chunks_file,
            output_path=tmp_path / "metrics.json",
            device="cpu",
            batch_size=2,
            emit_scores=scores_path,
        )

        with np.load(scores_path) as data:
            probs = data["probs"]

        assert probs.dtype == np.float32
        # Casting to float32 before sigmoid can't create resolution the
        # (already fp16) logit didn't have -- but it must not lose any
        # either, which a raw fp16 sigmoid would (#264).
        assert len(np.unique(probs)) >= len(np.unique(probs.astype(np.float16)))

    def test_never_compiles_on_cpu(self, temp_chunks_file, tmp_path, monkeypatch):
        """torch.compile must never run on CPU, with or without --no-compile
        -- it previously ran unconditionally, mode=None included (#264)."""
        from leech.evaluation import evaluate_model

        model_dir = self._train_checkpoint(temp_chunks_file, tmp_path)

        calls: list = []
        monkeypatch.setattr(
            "leech.evaluation.torch.compile",
            lambda model, *a, **kw: calls.append(1) or model,
        )

        evaluate_model(
            model_path=model_dir,
            test_data_path=temp_chunks_file,
            output_path=tmp_path / "metrics.json",
            device="cpu",
            batch_size=2,
        )

        assert calls == []


class TestParametricCheckpointMetricReporting:
    """`eval test`'s output ("test_metrics.json") reports the same parametric
    checkpoint metric a run was trained with, so it can be compared against
    another run that used the same one after the fact (#280)."""

    def test_reports_the_training_run_s_parametric_metric(self, temp_chunks_file, tmp_path):
        from leech.evaluation import evaluate_model
        from leech.training import train_model

        model_dir = tmp_path / "model"
        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=None,
            model_name="ConvLSTMDwell",
            output_dir=model_dir,
            epochs=1,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            seed=42,
            checkpoint_metric="tpr_at_fpr:0.5",
        )

        with open(model_dir / "config.json") as f:
            config = json.load(f)
        assert config["checkpoint_metric"] == "tpr_at_fpr:0.5"

        metrics = evaluate_model(
            model_path=model_dir,
            test_data_path=temp_chunks_file,
            output_path=tmp_path / "test_metrics.json",
            device="cpu",
            batch_size=2,
        )

        assert "tpr_at_fpr:0.5" in metrics
        assert isinstance(metrics["tpr_at_fpr:0.5"], float)
        assert 0.0 <= metrics["tpr_at_fpr:0.5"] <= 1.0

        with open(tmp_path / "test_metrics.json") as f:
            saved = json.load(f)
        assert saved["tpr_at_fpr:0.5"] == metrics["tpr_at_fpr:0.5"]

    def test_plain_checkpoint_metric_reports_nothing_new(self, temp_chunks_file, tmp_path):
        """A run selected on val_auc (the default) adds no parametric key --
        non-goal: this must not change what a plain run reports."""
        from leech.evaluation import evaluate_model
        from leech.training import train_model

        model_dir = tmp_path / "model"
        train_model(
            train_data_path=temp_chunks_file,
            val_data_path=None,
            model_name="ConvLSTMDwell",
            output_dir=model_dir,
            epochs=1,
            batch_size=2,
            device="cpu",
            motif="CCAGGC",
            seed=42,
        )

        metrics = evaluate_model(
            model_path=model_dir,
            test_data_path=temp_chunks_file,
            output_path=tmp_path / "test_metrics.json",
            device="cpu",
            batch_size=2,
        )

        assert not any(
            k.startswith("tpr_at_fpr") or k.startswith("callable_at_precision") for k in metrics
        )


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


class _Stacked:
    """Stands in for a LeechDataset whose chunks stacked into one tensor."""

    _signals_tensor = object()


class _ListFallback:
    """Stands in for the `_try_stack` fallback: per-chunk lists, fork-unsafe."""

    _signals_tensor = None


class TestValLoaderWorkers:
    """The validation loader starved the GPU once per epoch (issue #207).

    #206 routed `eval test` through `resolve_dataloader_workers` but left the
    in-training validation pass hardcoded to 0, so a 1.18M-chunk val set spent
    ~5 minutes at near-idle GPU at every epoch boundary -- ~75 minutes across a
    15-epoch run.
    """

    def test_stacked_dataset_gets_workers_on_cuda(self, monkeypatch):
        """The normal case: contiguous buffers COW-share, so workers are safe."""
        monkeypatch.setattr(dataset, "_usable_cpus", lambda: 32)

        assert resolve_val_dataloader_workers(_Stacked(), 0, "cuda") > 0

    def test_val_matches_train_when_stacked(self, monkeypatch):
        """Validation should not be a special case just for being validation."""
        monkeypatch.setattr(dataset, "_usable_cpus", lambda: 32)

        assert resolve_val_dataloader_workers(_Stacked(), 0, "cuda") == (
            resolve_dataloader_workers(0, "cuda")
        )

    def test_list_fallback_stays_serial(self, monkeypatch):
        """The one real memory case: forking a list of tensors multiplies RSS.

        `LeechDataset` stacks precisely so a fork shares the buffers; when
        `_try_stack` could not, each worker faults N PyObject headers into
        private copies. This is the exception the old blanket 0 was protecting.
        """
        monkeypatch.setattr(dataset, "_usable_cpus", lambda: 32)

        assert resolve_val_dataloader_workers(_ListFallback(), 0, "cuda") == 0

    def test_cpu_stays_serial(self):
        """On CPU workers compete with compute, stacked or not."""
        assert resolve_val_dataloader_workers(_Stacked(), 0, "cpu") == 0

    def test_explicit_request_honoured_when_stacked(self, monkeypatch):
        """An explicit N is not capped, matching the train loader's contract."""
        monkeypatch.setattr(dataset, "_usable_cpus", lambda: 2)

        assert resolve_val_dataloader_workers(_Stacked(), 3, "cuda") == 3

    def test_explicit_request_still_loses_to_the_list_fallback(self, monkeypatch):
        """Memory safety wins over an explicit request -- OOM is not a tradeoff."""
        monkeypatch.setattr(dataset, "_usable_cpus", lambda: 32)

        assert resolve_val_dataloader_workers(_ListFallback(), 3, "cuda") == 0
