"""CRF training: the decisions, and one end-to-end run.

The loop is mechanical; the decisions around it are where runs go wrong without
failing. Each is a plain function, tested here without a GPU: standardisation,
the label-quality gate, split resolution, target encoding and checkpoint
selection. The last test actually trains, on CPU, on a tiny synthetic corpus —
enough to prove the pieces fit together and the sidecar is written.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from leech.crf import (
    CrfTrainConfig,
    CrfTrainer,
    EpochStats,
    apply_quality_gate,
    compute_standardisation,
    encode_targets,
    epoch_order_rng,
    resolve_split,
    select_checkpoint,
)

ALPHABET = "NACGT"


# ── standardisation ────────────────────────────────────────────────────────


def test_standardisation_matches_numpy_over_the_window():
    rng = np.random.default_rng(0)
    signal = rng.normal(60.0, 10.0, size=(500, 300)).astype(np.float32)
    mean, std = compute_standardisation(signal, 200)
    window = signal[:, -200:].astype(np.float64)
    assert mean == pytest.approx(window.mean(), rel=1e-9)
    assert std == pytest.approx(window.std(), rel=1e-9)


def test_standardisation_is_blocked_but_block_size_does_not_change_it():
    """It is streamed so a corpus larger than RAM costs nothing to summarise —
    which is only true if the answer does not depend on the block size."""
    rng = np.random.default_rng(1)
    signal = rng.normal(60.0, 10.0, size=(997, 64)).astype(np.float32)
    a = compute_standardisation(signal, 64, block=10)
    b = compute_standardisation(signal, 64, block=1000)
    assert a == pytest.approx(b, rel=1e-12)


def test_standardisation_uses_only_the_trailing_window():
    signal = np.zeros((10, 100), dtype=np.float32)
    signal[:, :50] = 1000.0  # outside a 50-sample window
    mean, std = compute_standardisation(signal, 50)
    assert mean == 0.0 and std == 0.0


def test_empty_corpus_is_refused():
    with pytest.raises(ValueError, match="empty corpus"):
        compute_standardisation(np.zeros((0, 10), dtype=np.float32), 10)


# ── the label-quality gate ─────────────────────────────────────────────────


def test_gate_keeps_reads_above_both_thresholds():
    score = np.array([70.0, 70.0, 60.0, 70.0])
    margin = np.array([9.0, 2.0, 9.0, 9.0])
    keep, coverage = apply_quality_gate(score, margin, min_score=66, min_margin=5)
    assert keep.tolist() == [True, False, False, True]
    assert coverage == 1.0


def test_disabled_gate_keeps_everything_even_without_columns():
    keep, coverage = apply_quality_gate(None, None, enabled=False, n_reads=7)
    assert keep.all() and len(keep) == 7 and coverage == 1.0


def test_absent_quality_columns_refuse_rather_than_train_ungated():
    """Gating is what took one panel from 0.875 to 0.97, so silently skipping it
    is the expensive failure."""
    with pytest.raises(ValueError, match="no label-quality columns"):
        apply_quality_gate(None, None, n_reads=10)


def test_all_missing_scores_is_refused():
    nan = np.full(10, np.nan)
    with pytest.raises(ValueError, match="every value is missing"):
        apply_quality_gate(nan, nan)


def test_partial_coverage_is_refused_because_the_loss_is_silent():
    """An unscored read cannot pass, so it is dropped without a word — one
    corpus went from 56% usable to a non-random 13.5% this way."""
    score = np.array([70.0, np.nan, np.nan, np.nan])
    margin = np.array([9.0, np.nan, np.nan, np.nan])
    with pytest.raises(ValueError, match="covers only"):
        apply_quality_gate(score, margin, min_coverage=0.9)


def test_coverage_just_above_the_floor_is_allowed():
    score = np.array([70.0] * 9 + [np.nan])
    margin = np.array([9.0] * 9 + [np.nan])
    keep, coverage = apply_quality_gate(score, margin, min_coverage=0.9)
    assert coverage == pytest.approx(0.9)
    assert keep.sum() == 9


# ── split resolution ───────────────────────────────────────────────────────


def test_corpus_split_is_preferred_when_present():
    clean = np.ones(10, dtype=bool)
    split = np.array(["train"] * 7 + ["test"] * 3)
    train, test, why = resolve_split(clean, corpus_split=split)
    assert len(train) == 7 and len(test) == 3
    assert "corpus" in why


def test_resplit_overrides_the_corpus_split():
    clean = np.ones(100, dtype=bool)
    split = np.array(["train"] * 100)
    train, test, why = resolve_split(clean, corpus_split=split, resplit=True, test_frac=0.2)
    assert len(test) == 20
    assert "overridden" in why


def test_the_gate_is_applied_to_both_sides_of_the_split():
    clean = np.array([True] * 5 + [False] * 5)
    split = np.array(["train"] * 5 + ["test"] * 5)
    train, test, _ = resolve_split(clean, corpus_split=split)
    assert len(train) == 5 and len(test) == 0


def test_holdout_batch_takes_priority_and_is_prefix_matched():
    clean = np.ones(10, dtype=bool)
    batches = np.array(["LDX1-8"] * 6 + ["LDX9-16"] * 4)
    split = np.array(["train"] * 10)
    train, test, why = resolve_split(
        clean, corpus_split=split, batches=batches, holdout_batch="LDX9"
    )
    assert len(train) == 6 and len(test) == 4
    assert "held-out batch" in why


def test_holdout_batch_without_a_batch_column_is_an_error():
    with pytest.raises(ValueError, match="batch column"):
        resolve_split(np.ones(5, dtype=bool), holdout_batch="b1")


def test_holdout_batch_matching_nothing_is_an_error():
    batches = np.array(["b1"] * 5)
    with pytest.raises(ValueError, match="no reads from a batch"):
        resolve_split(np.ones(5, dtype=bool), batches=batches, holdout_batch="zzz")


def test_seeded_split_is_reproducible_and_seed_sensitive():
    clean = np.ones(200, dtype=bool)
    a = resolve_split(clean, test_frac=0.1, seed=3)[1]
    b = resolve_split(clean, test_frac=0.1, seed=3)[1]
    c = resolve_split(clean, test_frac=0.1, seed=4)[1]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_train_and_test_never_overlap():
    clean = np.ones(200, dtype=bool)
    train, test, _ = resolve_split(clean, test_frac=0.25, seed=1)
    assert not (set(train.tolist()) & set(test.tolist()))
    assert len(train) + len(test) == 200


# ── target encoding ────────────────────────────────────────────────────────


def test_targets_are_one_indexed_because_zero_is_the_stay_edge():
    out = encode_targets(["ACGT"], ALPHABET)
    assert out.tolist() == [[1, 2, 3, 4]]
    assert out.dtype == np.int64
    assert 0 not in out


def test_encoding_rejects_a_base_outside_the_alphabet():
    with pytest.raises(ValueError, match="not in alphabet"):
        encode_targets(["ACGX"], ALPHABET)


def test_encoding_is_one_array_for_the_whole_corpus():
    out = encode_targets(["ACGT"] * 100, ALPHABET)
    assert out.shape == (100, 4)


# ── checkpoint selection ───────────────────────────────────────────────────


def _hist(losses):
    return [
        EpochStats(i + 1, loss, loss, 1.0, 0, 0, 10, 1e-3, 1.0) for i, loss in enumerate(losses)
    ]


def test_the_last_epoch_ships_when_it_is_close_to_the_best():
    epoch, loss, why = select_checkpoint(_hist([0.5, 0.2, 0.21]))
    assert epoch == 3 and loss == 0.21
    assert "endpoint" in why


def test_a_diverged_run_ships_its_best_epoch_instead():
    """A run that reached 0.0047, diverged, and recovered only to 0.0072 has
    shipped weights it already beat by 53%."""
    epoch, loss, why = select_checkpoint(_hist([0.02, 0.0047, 0.4010, 0.0072]))
    assert epoch == 2 and loss == pytest.approx(0.0047)
    assert "over the 25% tolerance" in why


def test_always_final_disables_the_fallback():
    epoch, _, why = select_checkpoint(_hist([0.001, 0.5]), always_final=True)
    assert epoch == 2 and "endpoint" in why


def test_the_tolerance_is_loose_enough_not_to_chase_noise():
    """Training loss does NOT rank models at this scale — one seed reached
    0.0045 where another reached 0.0072 and measured worse on held-out recall."""
    epoch, _, _ = select_checkpoint(_hist([0.0060, 0.0072]))
    assert epoch == 2, "a 20% difference must not trigger the fallback"


def test_no_epochs_is_an_error():
    with pytest.raises(ValueError, match="no epochs"):
        select_checkpoint([])


# ── reporting ──────────────────────────────────────────────────────────────


def test_all_nonfinite_gradients_report_as_such_not_as_zero():
    """ "0.00" would read as "no gradient" when it means "no FINITE gradient
    seen all epoch" — the opposite diagnosis."""
    line = EpochStats(1, 0.1, 0.2, 0.0, 0, 10, 10, 1e-3, 5.0).render(32)
    assert "none-finite" in line and "NONFINITE-GRAD 10" in line


def test_a_healthy_epoch_reports_its_gradient_norm():
    line = EpochStats(1, 0.1, 0.2, 1.75, 0, 0, 10, 1e-3, 5.0).render(32)
    assert "|g|max 1.75" in line and "NONFINITE" not in line


# ── end to end ─────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_corpus(tmp_path):
    """A synthetic corpus small enough to train on CPU in seconds."""
    n, chunk, target_len = 64, 200, 8
    rng = np.random.default_rng(0)
    signal = rng.normal(60.0, 10.0, size=(n, chunk)).astype(np.float32)
    np.save(tmp_path / "c_X.npy", signal)
    np.savez(
        tmp_path / "c_meta.npz",
        y=np.array(["ACGTACGT"] * n),
        group=np.array(["g1"] * n),
        read_id=np.array([f"r{i}" for i in range(n)]),
        split=np.array(["train"] * 56 + ["test"] * 8),
        batch=np.array(["b1"] * n),
        gate_score=np.full(n, 70.0, dtype=np.float32),
        gate_margin=np.full(n, 9.0, dtype=np.float32),
        chunk=chunk,
        target_len=target_len,
        state_len=4,
    )
    return tmp_path / "c"


@pytest.fixture
def tiny_arch():
    """Same rules as the shipped geometry, small enough to train quickly."""
    return {
        "labels": {"labels": ["N", "A", "C", "G", "T"]},
        "global_norm": {"state_len": 2},
        "encoder": {"features": 16, "winlen": 5, "stride": 10, "scale": 5.0},
    }


def test_trains_end_to_end_and_writes_the_sidecar(tmp_path, tiny_corpus, tiny_arch):
    out = tmp_path / "run"
    trainer = CrfTrainer(
        tiny_corpus,
        config=CrfTrainConfig(epochs=2, batch_size=16, device="cpu", seed=0),
        arch_config=tiny_arch,
        output_dir=out,
    )
    result = trainer.train()

    assert (out / "model.pt").exists()
    saved = json.loads((out / "model.json").read_text())
    # The sidecar must carry standardisation: it is in neither the architecture
    # config nor the checkpoint, so weights alone cannot be used correctly.
    assert "mean" in saved and "std" in saved
    assert saved["chunk"] == 200 and saved["target_len"] == 8
    assert saved["emits"] == 6  # target_len - state_len
    assert saved["n_train"] == 56 and saved["n_test"] == 8
    assert len(saved["history"]) == 2
    assert result["selected_epoch"] in (1, 2)
    assert torch.load(out / "model.pt", map_location="cpu")


def test_the_loss_actually_moves(tmp_path, tiny_corpus, tiny_arch):
    """A loop that trains nothing still writes a checkpoint and reports numbers."""
    result = CrfTrainer(
        tiny_corpus,
        config=CrfTrainConfig(epochs=3, batch_size=16, device="cpu", seed=0),
        arch_config=tiny_arch,
    ).train()
    losses = [e["loss"] for e in result["history"]]
    assert losses[-1] < losses[0], f"loss did not fall: {losses}"


def test_emitted_length_is_the_state_len_rule(tiny_corpus, tiny_arch):
    trainer = CrfTrainer(tiny_corpus, arch_config=tiny_arch)
    assert trainer.emitted == trainer.target_len - trainer.encoder_cfg.state_len


def test_a_target_longer_than_the_corpus_is_refused(tiny_corpus, tiny_arch):
    with pytest.raises(ValueError, match="target_len"):
        CrfTrainer(tiny_corpus, config=CrfTrainConfig(target_len=99), arch_config=tiny_arch)


def test_a_chunk_wider_than_the_corpus_is_refused(tiny_corpus, tiny_arch):
    with pytest.raises(ValueError, match="chunk"):
        CrfTrainer(tiny_corpus, config=CrfTrainConfig(chunk=9999), arch_config=tiny_arch)


def test_fewer_training_reads_than_one_batch_is_refused(tiny_corpus, tiny_arch):
    """Otherwise the range loop simply never steps and the run reports a
    checkpoint trained on nothing."""
    with pytest.raises(ValueError, match="fewer than one batch"):
        CrfTrainer(
            tiny_corpus,
            config=CrfTrainConfig(batch_size=1024, device="cpu"),
            arch_config=tiny_arch,
        ).train()


class TestBatchOrderIsIndependentOfTheSplit:
    """The batch-order stream must not be the split's stream re-seeded.

    Both are seeded from the same ``seed``, and on the corpus-split and
    held-out-batch paths `resolve_split` shuffles an array of the *same length*
    the epoch loop shuffles. A plain ``default_rng(seed)`` in the loop replays
    the split's permutation, so epoch 1 trains on ``pi(pi(train))``. Nothing
    observable goes wrong when it does, which is exactly why it needs a test.
    """

    @staticmethod
    def _corpus_split_train_idx(n=512, seed=3):
        clean = np.ones(n, dtype=bool)
        split = np.array(["train"] * (n - n // 4) + ["test"] * (n // 4))
        train, _, why = resolve_split(clean, corpus_split=split, seed=seed)
        assert why == "the corpus's own split"
        return train

    def test_first_epoch_does_not_replay_the_split_permutation(self):
        seed = 3
        train = self._corpus_split_train_idx(seed=seed)

        replayed = train.copy()
        np.random.default_rng(seed).shuffle(replayed)  # what the bug produced

        actual = train.copy()
        epoch_order_rng(seed).shuffle(actual)

        assert not np.array_equal(actual, replayed)

    def test_the_stream_is_not_merely_offset_from_the_split_stream(self):
        """Skipping a single draw would fix epoch 1 and leave every later epoch
        as the split stream shifted by one, which is the same defect one epoch
        further in. Check several epochs deep, not just the first."""
        seed = 3
        train = self._corpus_split_train_idx(seed=seed)

        def draws(rng, n):
            out = []
            for _ in range(n):
                order = train.copy()
                rng.shuffle(order)
                out.append(order)
            return out

        split_stream = draws(np.random.default_rng(seed), 6)
        ours = draws(epoch_order_rng(seed), 5)

        for epoch, got in enumerate(ours, 1):
            assert not any(np.array_equal(got, s) for s in split_stream), (
                f"epoch {epoch} reuses a draw from the split's stream"
            )

    def test_it_is_still_reproducible_from_the_seed(self):
        train = self._corpus_split_train_idx()
        a, b = train.copy(), train.copy()
        epoch_order_rng(11).shuffle(a)
        epoch_order_rng(11).shuffle(b)
        assert np.array_equal(a, b)

        c = train.copy()
        epoch_order_rng(12).shuffle(c)
        assert not np.array_equal(a, c)
