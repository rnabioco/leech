"""Data-parallel training (``--gpus N``): the failures that do not announce themselves.

Every test here exists because the thing it checks fails *silently*. A run with
a broken shard converges, prints plausible metrics and writes a checkpoint; a
run whose checkpoint gained a ``module.`` prefix exports an untrained graph
under any ``strict=False`` loader. So these assert the mechanism, not the
outcome: a test that passed merely because both ranks converged would pass with
the sharding removed, which is the exact bug.

They run on gloo over CPU, so they run in CI, where the failures are as
detectable as they are on four A30s.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as td
import torch.nn as nn
from torch.utils.data import DistributedSampler, WeightedRandomSampler

from leech.distributed import (
    SINGLE,
    DistContext,
    DistributedEvalSampler,
    DistributedWeightedSampler,
    _free_port,
    backend_for,
    configure_rank_logging,
    device_for,
    validate_request,
)
from leech.training import Trainer, train_model

# --------------------------------------------------------------------------
# Sharding: the union of the ranks must be the epoch, exactly once
# --------------------------------------------------------------------------


def test_eval_shards_cover_the_epoch_exactly_once():
    """Validation must see every row once: duplicates land in the AUROC."""
    n, world = 97, 4
    shards = [list(DistributedEvalSampler(n, num_replicas=world, rank=r)) for r in range(world)]

    assert sorted(i for shard in shards for i in shard) == list(range(n))
    seen: set[int] = set()
    for shard in shards:
        assert not (seen & set(shard)), "ranks overlap"
        seen |= set(shard)


def test_eval_shards_are_not_padded():
    """``DistributedSampler`` would pad 97/4 up to 100 by repeating three rows."""
    n, world = 97, 4
    unpadded = sum(len(DistributedEvalSampler(n, num_replicas=world, rank=r)) for r in range(world))
    padded = sum(
        len(DistributedSampler(list(range(n)), num_replicas=world, rank=r, shuffle=False))
        for r in range(world)
    )
    assert unpadded == n
    assert padded == 100


def test_training_shards_are_equal_length_and_disjoint():
    """Unequal shards hang DDP: one rank leaves the epoch mid-allreduce."""
    world = 3
    dataset = list(range(100))
    shards = [
        list(DistributedSampler(dataset, num_replicas=world, rank=r, shuffle=False, drop_last=True))
        for r in range(world)
    ]

    assert len({len(shard) for shard in shards}) == 1
    union = [i for shard in shards for i in shard]
    assert len(union) == len(set(union)) == 99  # drop_last discards the remainder


# --------------------------------------------------------------------------
# The weighted sampler: a shard of one global draw, not a draw per rank
# --------------------------------------------------------------------------


def test_weighted_shards_interleave_to_the_single_global_draw():
    """The union of the shards IS the single-GPU epoch, not a resemblance of it."""
    weights = np.repeat([1.0, 9.0], 500)
    world, num_samples, seed, epoch = 4, 1000, 7, 3

    samplers = [
        DistributedWeightedSampler(weights, num_samples, num_replicas=world, rank=r, seed=seed)
        for r in range(world)
    ]
    for sampler in samplers:
        sampler.set_epoch(epoch)
    shards = [list(sampler) for sampler in samplers]

    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    reference = torch.multinomial(
        torch.as_tensor(weights, dtype=torch.double), num_samples, True, generator=generator
    ).tolist()

    interleaved = [shards[i % world][i // world] for i in range(num_samples)]
    assert interleaved == reference
    assert all(len(shard) == num_samples // world for shard in shards)


def test_weighted_shards_differ_between_ranks():
    """The failure this class exists to prevent, asserted from both sides.

    A ``WeightedRandomSampler`` handed to N ranks gives each of them the same
    indices -- the run then trains on ``world_size`` copies of one shard and
    reports metrics that look entirely normal.
    """
    weights = np.ones(200)

    torch.manual_seed(0)
    rank0 = list(WeightedRandomSampler(weights, num_samples=64, replacement=True))
    torch.manual_seed(0)
    rank1 = list(WeightedRandomSampler(weights, num_samples=64, replacement=True))
    assert rank0 == rank1, "the bug: two ranks, one draw"

    sharded = [
        list(DistributedWeightedSampler(weights, 64, num_replicas=2, rank=r, seed=0))
        for r in range(2)
    ]
    assert sharded[0] != sharded[1]


def test_weighted_shards_reproduce_the_single_gpu_class_ratio():
    """Oversampling has to still equalize the classes once the draw is split."""
    labels = np.array([0] * 900 + [1] * 100)
    counts = np.array([900, 100])
    weights = (1.0 / counts)[labels]
    num_samples, world = 8000, 4

    torch.manual_seed(11)
    single = np.array(list(WeightedRandomSampler(weights, num_samples, replacement=True)))
    shards = [
        list(DistributedWeightedSampler(weights, num_samples, num_replicas=world, rank=r, seed=11))
        for r in range(world)
    ]
    union = np.concatenate([np.asarray(shard) for shard in shards])

    assert abs(labels[union].mean() - 0.5) < 0.02
    assert abs(labels[union].mean() - labels[single].mean()) < 0.02
    for shard in shards:
        assert abs(labels[np.asarray(shard)].mean() - 0.5) < 0.05


def test_weighted_sampler_trims_to_a_multiple_of_the_world():
    """Every rank must yield the same count or the epoch deadlocks."""
    weights = np.ones(50)
    shards = [
        list(DistributedWeightedSampler(weights, 101, num_replicas=4, rank=r, seed=0))
        for r in range(4)
    ]
    assert {len(shard) for shard in shards} == {25}


def test_weighted_sampler_rejects_a_corpus_over_the_multinomial_ceiling():
    class _Huge:
        def __len__(self):
            return 2**24 + 1

    with pytest.raises(ValueError, match="torch.multinomial"):
        DistributedWeightedSampler(np.ones(2**24 + 1, dtype=np.float32), 10, num_replicas=2, rank=0)


# --------------------------------------------------------------------------
# Gradient accumulation: one allreduce per step, not one per micro-step
# --------------------------------------------------------------------------


class _SyncSpy(nn.Module):
    """Stands in for a DDP wrapper and records how often ``no_sync`` is entered."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = 0

    @contextlib.contextmanager
    def no_sync(self):
        self.entered += 1
        yield


def _trainer_stub(dist: DistContext, spy: nn.Module) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.dist = dist
    trainer._ddp_modules = [spy]
    return trainer


def test_no_sync_covers_every_micro_step_but_the_last():
    spy = _SyncSpy()
    trainer = _trainer_stub(DistContext(rank=0, local_rank=0, world_size=2, backend="gloo"), spy)

    for micro_step in range(4):
        with trainer._accumulating(micro_step == 3):
            pass

    assert spy.entered == 3, "the final micro-step must reduce, the others must not"


def test_no_sync_is_never_used_at_world_size_one():
    spy = _SyncSpy()
    trainer = _trainer_stub(SINGLE, spy)

    for micro_step in range(4):
        with trainer._accumulating(micro_step == 3):
            pass

    assert spy.entered == 0


# --------------------------------------------------------------------------
# Every module holding trainable parameters is wrapped
# --------------------------------------------------------------------------


class _StubModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Linear(8, 16)
        self.classifier = nn.Sequential(nn.Linear(16, 1))

    def forward(self, signal, sequence):  # pragma: no cover - never called here
        return self.classifier(self.body(signal))


def test_auxiliary_heads_are_wrapped_for_gradient_sync(monkeypatch):
    """The adversarial and CL heads sit outside the model but inside the optimizer.

    Unwrapped, their gradients are never allreduced: each rank keeps a private
    head while the backbone stays in sync, and nothing about the run says so.
    """
    wrapped: list[str] = []
    original = Trainer._wrap_ddp

    def spy(self, module, device):
        wrapped.append(type(module).__name__)
        return original(self, module, device)

    monkeypatch.setattr(Trainer, "_wrap_ddp", spy)
    Trainer(
        model=_StubModel(),
        model_type="Stub",
        train_loader=[],
        device="cpu",
        adversarial_lambda=0.5,
        cl_regression=True,
    )

    assert wrapped == ["_StubModel", "AdversarialHead", "RegressionHead"]


# --------------------------------------------------------------------------
# Numerical equivalence: two ranks at half the batch == one rank at the batch
# --------------------------------------------------------------------------


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fixed_batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(8, 6, generator=generator)
    y = (torch.rand(8, 1, generator=generator) > 0.5).float()
    return x, y


def _grad_worker(rank: int, world: int, port: int, out_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    td.init_process_group("gloo", rank=rank, world_size=world)
    try:
        torch.manual_seed(0)
        model = _TinyNet()
        ddp = nn.parallel.DistributedDataParallel(model)
        x, y = _fixed_batch()
        loss = nn.functional.binary_cross_entropy_with_logits(ddp(x[rank::world]), y[rank::world])
        loss.backward()
        if rank == 0:
            torch.save(
                {name: p.grad.clone() for name, p in model.named_parameters()},
                Path(out_dir) / "ddp_grads.pt",
            )
        td.barrier()
    finally:
        td.destroy_process_group()


def test_two_rank_gradients_match_the_single_rank_gradients(tmp_path):
    """The whole point, checked at the level where it is exact.

    A convergence comparison would need a corpus and hours; one step from an
    identical initialization is the same claim, sharply, in a second.
    """
    import torch.multiprocessing as mp

    torch.manual_seed(0)
    reference_model = _TinyNet()
    x, y = _fixed_batch()
    nn.functional.binary_cross_entropy_with_logits(reference_model(x), y).backward()
    reference = {name: p.grad.clone() for name, p in reference_model.named_parameters()}

    mp.spawn(_grad_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
    produced = torch.load(tmp_path / "ddp_grads.pt")

    for name, grad in reference.items():
        assert torch.allclose(produced[name], grad, atol=1e-6), name


# --------------------------------------------------------------------------
# The checkpoint a 2-rank run writes must be the one a 1-rank run writes
# --------------------------------------------------------------------------


def _train(chunks: Path, out_dir: Path, gpus: int) -> Path:
    train_model(
        train_data_path=chunks,
        val_data_path=chunks,
        model_name="ConvLSTMDwell",
        output_dir=out_dir,
        epochs=1,
        batch_size=4,
        device="cpu",
        motif="CCAGGC",
        seed=42,
        num_workers=0,
        gpus=gpus,
    )
    return out_dir


def test_two_rank_checkpoint_has_the_single_rank_keys(temp_chunks_file, tmp_path):
    """DDP prefixes ``state_dict`` keys with ``module.``; nothing downstream strips it.

    ``model_best.pt`` feeds ``leech model export`` and the escapepod-models
    registry chain, where a prefixed key set either fails to load or -- under
    ``strict=False`` -- loads nothing and exports an untrained graph.
    """
    one = _train(temp_chunks_file, tmp_path / "one_rank", gpus=1)
    two = _train(temp_chunks_file, tmp_path / "two_rank", gpus=2)

    keys_one = set(
        torch.load(one / "model_best.pt", map_location="cpu", weights_only=False)[
            "model_state_dict"
        ]
    )
    keys_two = set(
        torch.load(two / "model_best.pt", map_location="cpu", weights_only=False)[
            "model_state_dict"
        ]
    )

    assert keys_one == keys_two
    assert not any(key.startswith("module.") for key in keys_two)


def test_two_rank_run_records_its_world_size(temp_chunks_file, tmp_path):
    """A checkpoint that does not record how it was trained cannot be audited."""
    out = _train(temp_chunks_file, tmp_path / "two_rank", gpus=2)
    config = json.loads((out / "config.json").read_text())

    assert config["gpus"] == 2
    assert config["device"] == "cpu", "the rank's device must not leak into provenance"


def test_single_gpu_creates_no_process_group(temp_chunks_file, tmp_path):
    """world_size == 1 must be the old path: no group, no collectives, no spawn."""
    _train(temp_chunks_file, tmp_path / "one_rank", gpus=1)

    assert not td.is_initialized()


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_validate_request_is_a_no_op_for_one_gpu():
    validate_request(1, "cuda")  # no CUDA needed, nothing to check


def test_validate_request_rejects_more_gpus_than_exist(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(ValueError, match="only 2 CUDA device"):
        validate_request(4, "cuda")


def test_validate_request_rejects_inline_chunks():
    with pytest.raises(ValueError, match="on disk"):
        validate_request(2, "cpu", has_inline_chunks=True)


def test_backend_follows_the_device():
    assert backend_for("cuda") == "nccl"
    assert backend_for("cuda:2") == "nccl"
    assert backend_for("cpu") == "gloo"


def test_each_rank_gets_its_own_device():
    ctx = DistContext(rank=2, local_rank=2, world_size=4, backend="nccl")

    assert device_for("cuda", ctx) == "cuda:2"
    assert device_for("cpu", ctx) == "cpu"
    assert device_for("cuda", SINGLE) == "cuda", "world 1 must not be renamed"


# --------------------------------------------------------------------------
# A spawned rank has to configure the logging the CLI would have
# --------------------------------------------------------------------------


@pytest.fixture
def _restore_leech_logger():
    logger = logging.getLogger("leech")
    handlers, level = list(logger.handlers), logger.level
    yield
    logger.handlers = handlers
    logger.setLevel(level)


def test_spawned_ranks_get_the_logging_the_cli_would_have(_restore_leech_logger):
    """``setup_logging`` runs in the click entry point, which a rank never reaches.

    Without this the run is silent about its own configuration -- the effective
    batch, the sampler statistics, the encoding-fallback warning -- because the
    ``leech`` logger has no handler in the spawned process.
    """
    logger = logging.getLogger("leech")
    logger.handlers.clear()

    configure_rank_logging(DistContext(rank=0, local_rank=0, world_size=2, backend="gloo"))
    assert logger.handlers
    assert logger.level == logging.INFO

    configure_rank_logging(DistContext(rank=1, local_rank=1, world_size=2, backend="gloo"))
    assert logger.level == logging.WARNING, "off-rank chatter hides the line that differs"
    assert "[rank 1/2]" in logger.handlers[0].formatter._fmt
