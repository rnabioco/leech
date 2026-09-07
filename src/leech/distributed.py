"""Single-node data-parallel training for ``leech model train``.

The trainer is single-device by construction; this module is what lets one run
occupy more than one GPU of a node. It is *data* parallelism only — every rank
holds the whole model and a disjoint shard of each epoch — and it is opt-in via
``--gpus N``. At ``world_size == 1`` nothing here is constructed, no process
group exists, and the training path is byte-for-byte what it was.

Four things about this are load-bearing and none of them announce themselves
when they are wrong:

**``--batch-size`` is the GLOBAL batch, split across ranks.** This is not the
PyTorch convention, where the flag is per-rank and the effective batch grows
with the GPU count. It is the right rule here because a leech run is one arm of
a paired comparison: the same command line has to mean the same recipe at any
GPU count, or every existing number needs re-measuring. Splitting keeps the
optimizer-step count, the epoch-indexed LR schedule, ``ClipGrad``'s quantile
buffer and the ``grad_accum_split`` arithmetic exactly where they are, and DDP's
gradient average over equal-size shards is precisely the single-GPU batch mean.

**Every rank must draw a DISJOINT shard.** ``WeightedRandomSampler`` does not:
handed to N ranks it gives each the same oversampled draw, so the run converges,
reports plausible metrics, and has trained on ``world_size`` copies of one
shard. Nothing errors. :class:`DistributedWeightedSampler` draws the *same
global multinomial* on every rank — identical seed, identical generator — and
then takes ``draw[rank::world_size]``, so the shards are disjoint and their
union is exactly the single-GPU draw rather than merely resembling it.

**The multinomial is drawn per epoch, from a seed both ranks agree on.** If the
draw came from the global RNG it would diverge the moment any rank consumed a
random number the others did not — augmentation, dropout, worker seeding — and
the shards would silently start overlapping. Hence the private
:class:`torch.Generator` seeded ``seed + epoch``, and hence ``set_epoch()``
being mandatory rather than advisory.

**Validation shards without padding.** ``DistributedSampler`` pads the last
shard by repeating samples so all ranks see equal counts, which is right for
training (equal batch counts, no hang at the allreduce) and wrong for
evaluation, where those duplicates land in the AUROC. :class:`DistributedEvalSampler`
strides without padding; the ranks have unequal counts, which is harmless
because evaluation performs no collective inside the loop.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import socket
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as td
from torch.utils.data import Sampler

logger = logging.getLogger("leech.distributed")

# torch.multinomial refuses more categories than this, which caps the corpus a
# weighted sampler can serve. WeightedRandomSampler already lives under the same
# ceiling, so this is not a new limit — only a better error message.
_MAX_MULTINOMIAL_CATEGORIES = 2**24


@dataclass(frozen=True)
class DistContext:
    """Which rank this process is, and how it talks to the others."""

    rank: int
    local_rank: int
    world_size: int
    backend: str

    @property
    def is_main(self) -> bool:
        """Whether this rank owns the filesystem and the console."""
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        """Whether collectives are in play at all."""
        return self.world_size > 1


SINGLE = DistContext(rank=0, local_rank=0, world_size=1, backend="none")
"""The non-distributed context. ``enabled`` is False, so every guard is off."""


def backend_for(device: str) -> str:
    """NCCL on CUDA, gloo everywhere else.

    The gloo path is not a curiosity: it is what lets the sharding, checkpoint
    and gradient-equivalence tests run on CPU in CI, where the failures this
    module exists to prevent are just as detectable as they are on four A30s.
    """
    return "nccl" if device.startswith("cuda") else "gloo"


def context_from_env() -> DistContext | None:
    """Read a context from ``torchrun``'s environment, or None if unlaunched.

    Supporting both this and ``--gpus`` costs a dozen lines and means the CLI
    works unchanged under an external launcher.
    """
    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        return None
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size <= 1:
        return None
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = os.environ.get("LEECH_DIST_BACKEND", "nccl" if torch.cuda.is_available() else "gloo")
    return DistContext(rank=rank, local_rank=local_rank, world_size=world_size, backend=backend)


def device_for(device: str, ctx: DistContext) -> str:
    """Pin a rank to its own GPU.

    Every rank is handed ``"cuda"`` by the caller and has to become
    ``"cuda:<local_rank>"``, or all of them land on device 0 — which fits, runs,
    and delivers no parallelism whatsoever.
    """
    if not ctx.enabled or not device.startswith("cuda"):
        return device
    return f"cuda:{ctx.local_rank}"


def validate_request(gpus: int, device: str, *, has_inline_chunks: bool = False) -> None:
    """Reject a multi-GPU request that cannot work, before anything is loaded.

    Each of these fails late and confusingly otherwise: a daemonic process
    (a grid-search pool worker) cannot spawn children at all; inline chunks
    would be pickled to every rank; and asking for more GPUs than exist hangs
    at NCCL init rather than raising.
    """
    if gpus <= 1:
        return
    if device == "cpu":
        # Allowed, and only useful for tests — but say so, because a CPU
        # "multi-GPU" run is nobody's intent on a real corpus.
        logger.warning("gpus=%d requested on CPU: running %d gloo ranks", gpus, gpus)
    elif torch.cuda.is_available():
        available = torch.cuda.device_count()
        if gpus > available:
            raise ValueError(
                f"--gpus {gpus} requested but only {available} CUDA device(s) visible. "
                "Check the job's --gres=gpu:N."
            )
    else:
        raise ValueError(f"--gpus {gpus} requested but CUDA is not available")
    if multiprocessing.current_process().daemon:
        raise RuntimeError(
            "--gpus > 1 cannot run inside a daemonic process (a grid-search pool "
            "worker); daemonic processes may not spawn children."
        )
    if has_inline_chunks:
        raise ValueError(
            "--gpus > 1 requires train/val data on disk: pre-loaded chunks would be "
            "pickled to every rank."
        )


def configure_rank_logging(ctx: DistContext) -> None:
    """Give a spawned rank the logging the CLI would have given it.

    ``setup_logging`` runs once, in the click entry point. A spawned rank never
    passes through it, so the ``leech`` logger has no handler and *every* INFO
    line is dropped -- including rank 0's, which is where the effective batch,
    the sampler statistics and the encoding-fallback warning are reported. The
    run works and says nothing about itself, which is the worst of both.

    Non-main ranks are held at WARNING (four ranks reciting the same corpus
    statistics hides the one line that differs) and every line carries its rank,
    so a warning from rank 2 cannot be read as a warning about the run.
    """
    from leech.logging_config import setup_logging

    setup_logging(
        level=logging.INFO if ctx.is_main else logging.WARNING,
        format_string=(
            f"%(asctime)s - [rank {ctx.rank}/{ctx.world_size}] "
            "%(name)s - %(levelname)s - %(message)s"
        ),
    )


def init(ctx: DistContext) -> None:
    """Join the process group and pin this rank's device."""
    if not ctx.enabled:
        return
    if ctx.backend == "nccl":
        torch.cuda.set_device(ctx.local_rank)
    td.init_process_group(backend=ctx.backend, rank=ctx.rank, world_size=ctx.world_size)
    configure_rank_logging(ctx)
    logger.info("rank %d/%d initialized (%s)", ctx.rank, ctx.world_size, ctx.backend)


def shutdown() -> None:
    """Leave the process group if this process joined one."""
    if td.is_available() and td.is_initialized():
        td.destroy_process_group()


def barrier(ctx: DistContext) -> None:
    """Hold every rank until all of them arrive."""
    if ctx.enabled and td.is_initialized():
        td.barrier()


def _free_port() -> int:
    """Ask the kernel for an unused port for the rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def use_fork_for_dataloader_workers() -> None:
    """Put a spawned rank's child processes back on ``fork``.

    ``multiprocessing.spawn.prepare`` forces a spawned child's default start
    method to match how it was created, so inside a rank every DataLoader
    builds its workers by *spawn* -- which pickles the dataset's stacked
    tensors through ``/dev/shm`` instead of COW-sharing them. ``LeechDataset``
    stacks into contiguous buffers precisely so a fork shares them for free
    (see ``dataset.resolve_val_dataloader_workers``), and that invariant is
    silently void at ``--gpus > 1`` without this.

    It does not fail where it happens. The shm copies accumulate per worker per
    rank, and on the production corpus the run died only when the *fourth*
    rank's validation loader started, as ``No space left on device`` -- naming
    a tmpfs, not a start method. Peak RSS was 176 GiB against a 244 GiB
    allocation, so it does not look like a memory problem either.
    """
    if "fork" not in multiprocessing.get_all_start_methods():
        return
    if multiprocessing.get_start_method(allow_none=True) != "fork":
        multiprocessing.set_start_method("fork", force=True)


def _worker_entry(
    local_rank: int,
    entry: Callable[..., Any],
    kwargs: dict[str, Any],
    world_size: int,
    backend: str,
    port: int,
) -> None:
    """Body of one spawned rank: set up the environment, run, always tear down."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    ctx = DistContext(
        rank=local_rank, local_rank=local_rank, world_size=world_size, backend=backend
    )
    init(ctx)
    use_fork_for_dataloader_workers()
    try:
        entry(ctx, kwargs)
    finally:
        shutdown()


def spawn(
    entry: Callable[[DistContext, dict[str, Any]], Any],
    kwargs: dict[str, Any],
    *,
    nprocs: int,
    backend: str,
) -> None:
    """Run ``entry`` in ``nprocs`` processes and wait for all of them.

    ``torch.multiprocessing.spawn`` propagates a child's exception into the
    parent and terminates the siblings, so a failure on rank 2 does not leave
    three processes holding GPUs.
    """
    import torch.multiprocessing as mp

    mp.spawn(  # ty: ignore[no-matching-overload]
        _worker_entry,
        args=(entry, kwargs, nprocs, backend, _free_port()),
        nprocs=nprocs,
        join=True,
    )


def all_reduce_sum(values: dict[str, float], ctx: DistContext, device: str) -> dict[str, float]:
    """Sum a dict of scalars across ranks, preserving key order.

    One collective for the whole dict rather than one per key: the values are
    epoch tallies read once, and a per-key allreduce would add a round trip per
    metric at every epoch boundary.
    """
    if not ctx.enabled:
        return values
    keys = list(values)
    buf = torch.tensor([values[k] for k in keys], dtype=torch.float64, device=device)
    td.all_reduce(buf, op=td.ReduceOp.SUM)
    return dict(zip(keys, buf.tolist(), strict=True))


def all_reduce_tensor_sum(tensor: torch.Tensor, ctx: DistContext) -> torch.Tensor:
    """Sum one scalar tensor across ranks, in place.

    Unlike :func:`all_reduce_sum` this stays on the device and never reads the
    value back, because it runs in the training step rather than at an epoch
    boundary and ``.item()`` there would sync the host on every micro-step.
    """
    if not ctx.enabled:
        return tensor
    td.all_reduce(tensor, op=td.ReduceOp.SUM)
    return tensor


def gather_arrays(array: np.ndarray, ctx: DistContext) -> list[np.ndarray]:
    """Collect one array per rank onto every rank.

    Ranks hold different numbers of validation rows (the eval sampler does not
    pad), so this goes through ``all_gather_object`` rather than a fixed-size
    tensor gather. It runs once per epoch on a few MB.
    """
    if not ctx.enabled:
        return [array]
    gathered: list[Any] = [None] * ctx.world_size
    td.all_gather_object(gathered, array)
    return [np.asarray(part) for part in gathered]


class DistributedWeightedSampler(Sampler[int]):
    """A ``WeightedRandomSampler`` draw, split into disjoint per-rank shards.

    The draw is global and identical on every rank; the split is a stride. So
    the union of the ranks' indices *is* the single-GPU epoch — the same
    samples, in the same multiset — and any class ratio the weights produce on
    one GPU is reproduced on N by construction rather than by luck.

    ``num_samples`` is trimmed down to a multiple of ``num_replicas`` so every
    rank yields the same count. Unequal counts hang DDP: the rank with fewer
    batches leaves the epoch, and the others block forever in the allreduce of
    a step it will never take.
    """

    def __init__(
        self,
        weights: Sequence[float] | np.ndarray | torch.Tensor,
        num_samples: int,
        *,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        replacement: bool = True,
    ) -> None:
        weights_t = torch.as_tensor(np.asarray(weights, dtype=np.float64))
        if weights_t.numel() > _MAX_MULTINOMIAL_CATEGORIES:
            raise ValueError(
                f"weighted sampling supports at most {_MAX_MULTINOMIAL_CATEGORIES} chunks "
                f"(torch.multinomial's limit), got {weights_t.numel()}"
            )
        self.weights = weights_t
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.replacement = replacement
        self.total_size = (int(num_samples) // self.num_replicas) * self.num_replicas
        self.num_samples = self.total_size // self.num_replicas
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Re-seed the shared draw. Without this every epoch is the same one."""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        draw = torch.multinomial(
            self.weights, self.total_size, self.replacement, generator=generator
        )
        return iter(draw[self.rank :: self.num_replicas].tolist())

    def __len__(self) -> int:
        return self.num_samples


class DistributedEvalSampler(Sampler[int]):
    """Stride a dataset across ranks without padding.

    ``DistributedSampler`` repeats samples to equalize the shards. In training
    that is required; in evaluation those repeats are counted twice in every
    metric, so a 2-GPU AUROC would differ from the 1-GPU one for a reason that
    has nothing to do with the model.
    """

    def __init__(self, dataset_len: int, *, num_replicas: int, rank: int) -> None:
        self.dataset_len = int(dataset_len)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.dataset_len, self.num_replicas))

    def __len__(self) -> int:
        return len(range(self.rank, self.dataset_len, self.num_replicas))
