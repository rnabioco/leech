"""`leech model train-crf`.

Mostly a wiring test — the decisions live in `leech.crf.training` and are tested
there. What is worth pinning here is that every option actually reaches the
config (a click option that silently does not is the classic failure), and that
the command is discoverable in the group.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from click.testing import CliRunner

from leech.cli import cli
from leech.crf import CrfTrainConfig


@pytest.fixture
def corpus(tmp_path):
    n, chunk = 64, 200
    rng = np.random.default_rng(0)
    np.save(tmp_path / "c_X.npy", rng.normal(60, 10, size=(n, chunk)).astype(np.float32))
    np.savez(
        tmp_path / "c_meta.npz",
        y=np.array(["ACGTACGT"] * n),
        group=np.array(["g1"] * n),
        read_id=np.array([f"r{i}" for i in range(n)]),
        split=np.array(["train"] * 56 + ["test"] * 8),
        batch=np.array(["b1"] * n),
        gate_score=np.full(n, 70.0, np.float32),
        gate_margin=np.full(n, 9.0, np.float32),
        chunk=chunk,
        target_len=8,
        state_len=2,
    )
    return tmp_path / "c"


@pytest.fixture
def arch(tmp_path):
    path = tmp_path / "arch.toml"
    path.write_text(
        '[labels]\nlabels = ["N","A","C","G","T"]\n'
        "[input]\nfeatures = 1\n"
        "[global_norm]\nstate_len = 2\n"
        "[encoder]\nfeatures = 16\nwinlen = 5\nstride = 10\nscale = 5.0\n"
    )
    return path


def test_train_crf_is_listed_in_the_model_group():
    result = CliRunner().invoke(cli, ["model", "--help"])
    assert result.exit_code == 0
    assert "train-crf" in result.output
    # Scoped to the command list: the group's own docstring is "Train and
    # optimize models", so searching the whole output finds that "optimize"
    # first and compares against the wrong thing.
    commands = result.output[result.output.index("Commands") :]
    # Immediately after `train`: it is the same step, a different task.
    assert commands.index("train-crf") < commands.index("optimize")


def test_help_states_the_emission_rule():
    """The one thing a user will otherwise get wrong: widening the window does
    not lengthen the decode."""
    result = CliRunner().invoke(cli, ["model", "train-crf", "--help"])
    assert result.exit_code == 0
    assert "state_len" in result.output
    assert "Widening the window does not lengthen the decode" in " ".join(result.output.split())


def test_trains_and_writes_both_artifacts(tmp_path, corpus, arch):
    out = tmp_path / "run"
    result = CliRunner().invoke(
        cli,
        # fmt: off
        [
            "model",
            "train-crf",
            "--corpus",
            str(corpus),
            "--output-dir",
            str(out),
            "--config",
            str(arch),
            "--epochs",
            "2",
            "--batch-size",
            "16",
            "--device",
            "cpu",
        ],
        # fmt: on
    )
    assert result.exit_code == 0, result.output
    assert (out / "model.pt").exists()
    saved = json.loads((out / "model.json").read_text())
    assert saved["emits"] == 6  # target 8 - state_len 2
    assert "mean" in saved and "std" in saved


def test_output_dir_is_optional(tmp_path, corpus, arch):
    """Training without saving is legal — useful for a sweep that only wants the
    loss curve."""
    result = CliRunner().invoke(
        cli,
        # fmt: off
        [
            "model",
            "train-crf",
            "--corpus",
            str(corpus),
            "--config",
            str(arch),
            "--epochs",
            "1",
            "--batch-size",
            "16",
            "--device",
            "cpu",
        ],
        # fmt: on
    )
    assert result.exit_code == 0, result.output


def test_every_cli_option_reaches_the_config(tmp_path, corpus, arch, monkeypatch):
    """A click option that never reaches the config is silently ignored, which
    is exactly how a sweep ends up running the default every time."""
    seen = {}

    class Spy:
        def __init__(self, _corpus, *, config, arch_config=None, output_dir=None):
            seen.update(vars(config))

        def train(self):
            return {
                "chunk": 1,
                "target_len": 8,
                "state_len": 2,
                "emits": 6,
                "mean": 0.0,
                "std": 1.0,
                "split_source": "x",
                "n_train": 1,
                "n_test": 1,
                "quality_coverage": 1.0,
                "final_loss": 0.0,
                "selected_epoch": 1,
                "selected_loss": 0.0,
                "selected_because": "x",
                "history": [],
            }

    monkeypatch.setattr("leech.crf.CrfTrainer", Spy)
    result = CliRunner().invoke(
        cli,
        # fmt: off
        [
            "model",
            "train-crf",
            "--corpus",
            str(corpus),
            "--config",
            str(arch),
            "--epochs",
            "7",
            "--batch-size",
            "13",
            "--lr",
            "0.005",
            "--weight-decay",
            "0.002",
            "--max-grad-norm",
            "3.5",
            "--seed",
            "11",
            "--no-gate",
            "--min-score",
            "71",
            "--min-margin",
            "8",
            "--min-coverage",
            "0.5",
            "--test-frac",
            "0.25",
            "--resplit",
            "--holdout-batch",
            "b9",
            "--chunk",
            "150",
            "--target-len",
            "6",
            "--select-tol",
            "0.4",
            "--always-final",
            "--device",
            "cpu",
        ],
        # fmt: on
    )
    assert result.exit_code == 0, result.output

    expected = CrfTrainConfig(
        epochs=7,
        batch_size=13,
        lr=0.005,
        weight_decay=0.002,
        max_grad_norm=3.5,
        seed=11,
        gate=False,
        min_score=71,
        min_margin=8,
        min_coverage=0.5,
        test_frac=0.25,
        resplit=True,
        holdout_batch="b9",
        chunk=150,
        target_len=6,
        select_tol=0.4,
        always_final=True,
        device="cpu",
    )
    assert seen == vars(expected)


def test_defaults_match_the_config_dataclass():
    """The CLI's `show_default` text and the dataclass must not drift apart."""
    seen = {}

    class Spy:
        def __init__(self, _corpus, *, config, arch_config=None, output_dir=None):
            seen.update(vars(config))

        def train(self):
            raise SystemExit(0)

    import leech.crf

    original = leech.crf.CrfTrainer
    leech.crf.CrfTrainer = Spy
    try:
        CliRunner().invoke(cli, ["model", "train-crf", "--corpus", "x"])
    finally:
        leech.crf.CrfTrainer = original

    defaults = vars(CrfTrainConfig())
    for key in ("epochs", "batch_size", "lr", "seed", "select_tol", "min_score"):
        assert seen[key] == defaults[key], f"{key}: CLI {seen[key]} vs dataclass {defaults[key]}"
