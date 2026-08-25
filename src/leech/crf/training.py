"""Train a CTC-CRF model on a corpus.

Separate from :class:`leech.training.Trainer`, which is classification-locked all
the way down — ``pos_weight``, ``num_out``, BCE/focal/CE, AUROC/F1 checkpointing,
and a wrapper that forwards three input branches. A sequence task shares none of
that, and forcing it through would put the production classifier path at risk to
save a few hundred lines.

What it *does* share is leech's conventions, and the decisions below are the
ones that are not obvious. Each is split out as a plain function so it can be
tested without a GPU: the loop is mechanical, the decisions are where things go
wrong quietly.

Five things that are load-bearing
---------------------------------
**The loss runs in fp32, outside autocast.** The lattice scan accumulates over
``chunk // stride`` timesteps and fp16 loses the tail of that sum. The encoder
still gets autocast — that is where the matmuls and the speed are.

**Standardisation is derived from the corpus and written to the sidecar.** It is
in neither the architecture config nor the checkpoint, so a consumer holding
only weights cannot reproduce it and decodes silently worse. It is computed in a
single streamed pass so a corpus larger than RAM costs nothing to summarise.

**The label-quality gate is applied here, not at extraction.** That is what
makes it sweepable: on one panel, gating moved accuracy from 0.875 to 0.97, and
baking a decision into the corpus would cost a re-extraction per threshold. An
*unscored* read cannot pass a gate, so partial coverage silently trains on a
small non-random subset — :func:`apply_quality_gate` refuses rather than let
that happen quietly.

**The last epoch is not automatically the best one.** A run that passes through
0.0047, diverges, and recovers only to 0.0072 has shipped weights it already
beat by 53%. :func:`select_checkpoint` falls back to the best epoch when the
last is worse by more than a tolerance — sized for blow-ups, not for ranking,
because training loss does *not* rank models at this scale.

**An epoch mean cannot distinguish one catastrophic batch from a thousand
mediocre ones.** The loop carries the worst single batch, the largest pre-clip
gradient norm, how many steps the ``GradScaler`` silently discarded, and how
many gradients were non-finite. A blown batch reads as max >> mean with a large
norm; a sustained excursion moves the mean instead.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CrfTrainConfig",
    "CrfTrainer",
    "EpochStats",
    "apply_quality_gate",
    "compute_standardisation",
    "encode_targets",
    "resolve_split",
    "select_checkpoint",
    "train_crf",
]

logger = logging.getLogger("leech.crf.training")


@dataclass
class CrfTrainConfig:
    """Everything the loop needs that is not the data itself."""

    epochs: int = 32
    """32, not 16. A longer target is under-trained at 16 and measures ~1.2pp
    worse than it is, which reads as a worse design rather than a shorter run."""

    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-5
    max_grad_norm: float = 2.0
    seed: int = 0

    gate: bool = True
    min_score: float = 66.0
    min_margin: float = 5.0
    min_coverage: float = 0.9

    test_frac: float = 0.1
    resplit: bool = False
    holdout_batch: str | None = None

    select_tol: float = 0.25
    always_final: bool = False

    chunk: int | None = None
    target_len: int | None = None
    device: str = "auto"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EpochStats:
    """One epoch's numbers. The extras are the ones that separate failure modes."""

    epoch: int
    loss: float
    worst_batch: float
    grad_max: float
    n_skipped: int
    n_nonfinite: int
    n_batches: int
    lr: float
    seconds: float

    def render(self, epochs: int) -> str:
        # "0.00" would read as "no gradient" when it means "no FINITE gradient
        # seen all epoch" — the opposite diagnosis.
        g = f"{self.grad_max:.2f}" if self.n_nonfinite < self.n_batches else "none-finite"
        extra = f" NONFINITE-GRAD {self.n_nonfinite}" if self.n_nonfinite else ""
        return (
            f"  epoch {self.epoch}/{epochs} loss {self.loss:.4f} max {self.worst_batch:.4f} "
            f"|g|max {g} lr {self.lr:.2e} skipped {self.n_skipped}{extra} "
            f"({self.seconds:.0f}s)"
        )


def compute_standardisation(signal, chunk: int, *, block: int = 20_000) -> tuple[float, float]:
    """Corpus mean and standard deviation over the trailing ``chunk`` samples.

    Streamed in blocks and accumulated in float64: the corpus is memory-mapped
    and may not fit in RAM, and a float32 sum over billions of samples loses its
    tail. Returns plain floats, because these travel in JSON.
    """
    n = 0
    total = 0.0
    total_sq = 0.0
    for start in range(0, len(signal), block):
        values = np.asarray(signal[start : start + block, -chunk:], dtype=np.float64)
        n += values.size
        total += values.sum()
        total_sq += (values * values).sum()
    if n == 0:
        raise ValueError("cannot standardise an empty corpus")
    mean = total / n
    variance = max(total_sq / n - mean * mean, 0.0)
    return float(mean), float(math.sqrt(variance))


def apply_quality_gate(
    score: np.ndarray | None,
    margin: np.ndarray | None,
    *,
    enabled: bool = True,
    min_score: float = 66.0,
    min_margin: float = 5.0,
    min_coverage: float = 0.9,
    n_reads: int | None = None,
) -> tuple[np.ndarray, float]:
    """Which reads are trustworthy enough to train on, and the score coverage.

    Raises rather than gating on a partially scored corpus. An unscored read
    cannot pass, so it is dropped *silently* — one corpus went from 56% usable
    to 13.5% that way, non-randomly, because the score table covered only the
    reads of an earlier extraction.
    """
    if n_reads is None:
        n_reads = len(score) if score is not None else 0
    if not enabled:
        logger.info("quality gate disabled; training on all %d reads", n_reads)
        return np.ones(n_reads, dtype=bool), 1.0

    if score is None or margin is None:
        raise ValueError(
            "the corpus carries no label-quality columns, so the gate cannot be "
            "applied. Gating is what took one panel from 0.875 to 0.97 — label "
            "noise, not capacity, was the ceiling — so this refuses rather than "
            "quietly training ungated. Rebuild the manifest with quality columns, "
            "or pass gate=False deliberately."
        )
    score = np.asarray(score, dtype=float)
    margin = np.asarray(margin, dtype=float)
    scored = ~np.isnan(margin) & ~np.isnan(score)
    coverage = float(scored.mean()) if len(scored) else 0.0

    if coverage == 0.0:
        raise ValueError(
            "the corpus carries label-quality columns but every value is missing. "
            "Re-score it, or pass gate=False deliberately."
        )
    if coverage < min_coverage:
        raise ValueError(
            f"label quality covers only {coverage:.3f} of {n_reads} reads "
            f"(need {min_coverage}). An unscored read cannot pass the gate, so it "
            f"is silently dropped — you would train on a small, non-random subset. "
            f"Re-score the corpus."
        )
    keep = scored & (margin > min_margin) & (score >= min_score)
    logger.info(
        "quality gate: %d/%d reads pass (%.3f), coverage %.3f",
        int(keep.sum()),
        n_reads,
        keep.mean(),
        coverage,
    )
    return keep, coverage


def resolve_split(
    clean: np.ndarray,
    *,
    corpus_split: np.ndarray | None = None,
    batches: np.ndarray | None = None,
    holdout_batch: str | None = None,
    test_frac: float = 0.1,
    resplit: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, str]:
    """``(train_idx, test_idx, provenance)`` — three sources, in priority order.

    A **held-out batch** is the honest generalisation number when classes are
    crossed with batch; a read-level split puts the same flowcell on both sides
    and reads optimistically. The **corpus's own split** comes next, because it
    was carved per class before capping, so every arm drawn from that corpus
    holds out the same reads by construction rather than by both happening to
    seed an RNG identically. Seeding one here is the fallback.
    """
    rng = np.random.default_rng(seed)

    if holdout_batch is not None:
        if batches is None:
            raise ValueError("holdout_batch needs a corpus carrying a batch column")
        held = np.char.startswith(batches.astype(str), holdout_batch)
        if not held.any():
            raise ValueError(
                f"no reads from a batch starting {holdout_batch!r}; "
                f"have {sorted(set(batches.astype(str)))}"
            )
        train = np.flatnonzero(clean & ~held)
        test = np.flatnonzero(clean & held)
        rng.shuffle(train)
        n_out = len(set(batches[held].astype(str)))
        return train, test, f"held-out batch {holdout_batch} ({n_out} batch(es) out)"

    if corpus_split is not None and not resplit:
        train = np.flatnonzero(clean & (corpus_split == "train"))
        test = np.flatnonzero(clean & (corpus_split == "test"))
        rng.shuffle(train)
        return train, test, "the corpus's own split"

    idx = np.flatnonzero(clean)
    rng.shuffle(idx)
    n_test = int(test_frac * len(idx))
    why = "seeded here" + ("; corpus split overridden" if corpus_split is not None else "")
    return idx[n_test:], idx[:n_test], why


def encode_targets(targets, alphabet: str) -> np.ndarray:
    """Targets as ``(N, L)`` int64, 1-indexed over ``alphabet[1:]``.

    Index 0 is the CRF's stay/blank edge and is never a target base. Encoded
    once for the whole corpus rather than per step: at the training shape that
    is 12,288 dict lookups on the critical path every step, 1.34 ms of a 39 ms
    step, against 0.06 ms to index a prebuilt array.
    """
    lookup = {base: i + 1 for i, base in enumerate(alphabet[1:])}
    try:
        return np.array([[lookup[b] for b in str(t)] for t in targets], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(
            f"target contains {exc.args[0]!r}, which is not in alphabet {alphabet!r}"
        ) from None


def select_checkpoint(
    history: list[EpochStats], *, select_tol: float = 0.25, always_final: bool = False
) -> tuple[int, float, str]:
    """Which epoch's weights to ship: ``(epoch, loss, why)``.

    The tolerance is sized to fire on a real divergence and nothing smaller. It
    is deliberately loose because training loss does **not** rank models at this
    scale — one seed reached 0.0045 where another reached 0.0072 on the same
    split and measured 0.2pp *worse* on held-out balanced recall. This is a
    divergence detector, not a ranking.
    """
    if not history:
        raise ValueError("no epochs were run")
    final = history[-1]
    best = min(history, key=lambda e: e.loss)
    if always_final or final.loss <= best.loss * (1 + select_tol):
        return final.epoch, final.loss, "the schedule's endpoint"
    return (
        best.epoch,
        best.loss,
        f"epoch {best.epoch} ({best.loss:.4f}); the last was {final.loss / best.loss:.2f}x "
        f"it, over the {select_tol:.0%} tolerance",
    )


class CrfTrainer:
    """Trains a :class:`~leech.crf.encoder.CrfEncoder` on a streamed corpus."""

    def __init__(
        self,
        corpus: str | Path,
        *,
        config: CrfTrainConfig | None = None,
        arch_config: dict | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        from .config import load_config
        from .corpus import load_corpus, load_corpus_meta
        from .encoder import encoder_config_from_toml

        self.cfg = config or CrfTrainConfig()
        self.corpus_path = Path(corpus)
        self.output_dir = Path(output_dir) if output_dir else None

        arch = arch_config if arch_config is not None else load_config()
        self.alphabet = "".join(arch["labels"]["labels"])
        self.encoder_cfg = encoder_config_from_toml(arch)

        signal, targets, groups, read_ids, corpus_split = load_corpus(self.corpus_path)
        meta = load_corpus_meta(self.corpus_path)

        chunk = self.cfg.chunk or int(signal.shape[1])
        target_len = self.cfg.target_len or len(str(targets[0]))
        if chunk > signal.shape[1]:
            raise ValueError(f"chunk {chunk} exceeds the extracted {signal.shape[1]}")
        if target_len > len(str(targets[0])):
            raise ValueError(
                f"target_len {target_len} exceeds the extracted {len(str(targets[0]))}"
            )

        self.signal = signal
        self.targets = np.array([str(t)[-target_len:] for t in targets])
        self.groups = groups
        self.read_ids = read_ids
        self.corpus_split = corpus_split
        self.meta = meta
        self.chunk = chunk
        self.target_len = target_len
        self.encoder_cfg = replace(self.encoder_cfg, chunk=chunk)

    @property
    def emitted(self) -> int:
        """Bases the model can emit: the first ``state_len`` only fix the state."""
        return self.target_len - self.encoder_cfg.state_len

    def prepare(self) -> dict[str, Any]:
        """Standardisation, gate and split — everything before the first step."""
        mean, std = compute_standardisation(self.signal, self.chunk)
        score = self.meta.get("gate_score")
        margin = self.meta.get("gate_margin")
        clean, coverage = apply_quality_gate(
            score,
            margin,
            enabled=self.cfg.gate,
            min_score=self.cfg.min_score,
            min_margin=self.cfg.min_margin,
            min_coverage=self.cfg.min_coverage,
            n_reads=len(self.targets),
        )
        batches = self.meta.get("batch", self.meta.get("run"))
        train_idx, test_idx, why = resolve_split(
            clean,
            corpus_split=self.corpus_split,
            batches=batches,
            holdout_batch=self.cfg.holdout_batch,
            test_frac=self.cfg.test_frac,
            resplit=self.cfg.resplit,
            seed=self.cfg.seed,
        )
        logger.info(
            "corpus %s -> window %d samples, target %d nt (emits %d)",
            self.signal.shape,
            self.chunk,
            self.target_len,
            self.emitted,
        )
        logger.info("standardisation: mean=%.3f std=%.3f", mean, std)
        logger.info("split: %s -> train %d, test %d", why, len(train_idx), len(test_idx))
        return {
            "mean": mean,
            "std": std,
            "coverage": coverage,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "split_source": why,
        }

    def train(self) -> dict[str, Any]:
        """Run the schedule and return the result, writing it if asked."""
        import torch
        from torch import nn

        from .encoder import CrfEncoder
        from .loss import CtcCrfLoss

        cfg = self.cfg
        device = cfg.resolved_device()
        prep = self.prepare()
        mean, std = prep["mean"], prep["std"]
        train_idx = prep["train_idx"]
        if len(train_idx) < cfg.batch_size:
            raise ValueError(
                f"{len(train_idx)} training reads is fewer than one batch of "
                f"{cfg.batch_size}; nothing would be stepped"
            )

        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        encoded = encode_targets(self.targets, self.alphabet)

        model = CrfEncoder(self.encoder_cfg).to(device)
        criterion = CtcCrfLoss(self.encoder_cfg.n_base, self.encoder_cfg.state_len).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        steps = cfg.epochs * (len(train_idx) // cfg.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps)
        scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

        history: list[EpochStats] = []
        best_state: dict[str, Any] | None = None
        best_loss = float("inf")

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            order = train_idx.copy()
            rng.shuffle(order)
            t0 = time.time()
            total = worst = grad_max = 0.0
            n_batches = n_skipped = n_nonfinite = 0

            for start in range(0, len(order) - cfg.batch_size + 1, cfg.batch_size):
                # Sorted: the signal is a memmap, and a sorted gather reads it
                # forwards instead of seeking per row.
                rows = np.sort(order[start : start + cfg.batch_size])
                window = np.asarray(self.signal[rows][:, -self.chunk :], dtype=np.float32)
                x = ((torch.from_numpy(window).to(device) - mean) / std).unsqueeze(1)
                tgt = torch.from_numpy(encoded[rows]).to(device)
                lengths = torch.full((len(rows),), tgt.shape[1], dtype=torch.long, device=device)

                with torch.amp.autocast(device, enabled=(device == "cuda")):
                    scores = model(x)
                # Outside autocast, in fp32: the lattice scan accumulates over
                # chunk/stride timesteps and fp16 loses the tail of that sum.
                loss = criterion(scores.float(), tgt, lengths)

                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gnorm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm))
                # A skipped step is invisible from outside: on non-finite
                # gradients scaler.step() silently does nothing and update()
                # halves the scale. The scale dropping IS the signal.
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                sched.step()
                n_skipped += scaler.get_scale() < scale_before
                if math.isfinite(gnorm):
                    grad_max = max(grad_max, gnorm)
                else:
                    n_nonfinite += 1
                value = float(loss.detach())
                worst = max(worst, value)
                total += value
                n_batches += 1

            stats = EpochStats(
                epoch=epoch,
                loss=total / max(n_batches, 1),
                worst_batch=worst,
                grad_max=grad_max,
                n_skipped=int(n_skipped),
                n_nonfinite=n_nonfinite,
                n_batches=n_batches,
                lr=float(sched.get_last_lr()[0]),
                seconds=time.time() - t0,
            )
            history.append(stats)
            logger.info("%s", stats.render(cfg.epochs))

            if stats.loss < best_loss:
                best_loss = stats.loss
                best_state = {
                    k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
                }

        epoch, loss, why = select_checkpoint(
            history, select_tol=cfg.select_tol, always_final=cfg.always_final
        )
        if epoch != history[-1].epoch and best_state is not None:
            logger.warning("shipping %s", why)
            model.load_state_dict(best_state)

        result = {
            "corpus": str(self.corpus_path),
            "chunk": self.chunk,
            "target_len": self.target_len,
            "state_len": self.encoder_cfg.state_len,
            "emits": self.emitted,
            "mean": mean,
            "std": std,
            "alphabet": self.alphabet,
            "epochs": cfg.epochs,
            "seed": cfg.seed,
            "n_train": int(len(train_idx)),
            "n_test": int(len(prep["test_idx"])),
            "test_idx": prep["test_idx"].tolist(),
            "split_source": prep["split_source"],
            "quality_coverage": prep["coverage"],
            "final_loss": history[-1].loss,
            "selected_epoch": epoch,
            "selected_loss": loss,
            "selected_because": why,
            "best_epoch": min(history, key=lambda e: e.loss).epoch,
            "best_loss": best_loss,
            "history": [asdict(e) for e in history],
            "config": asdict(cfg),
        }
        if self.output_dir is not None:
            self._write(model, result, self.output_dir)
        return result | {"model": model}

    def _write(self, model, result: dict[str, Any], output_dir: Path) -> None:
        import torch

        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / "model.pt")
        # The sidecar is not optional: standardisation lives in neither the
        # architecture config nor the checkpoint, so weights alone cannot be
        # used correctly.
        (output_dir / "model.json").write_text(json.dumps(result, indent=2))
        logger.info(
            "wrote %s/model.pt and model.json (weights from epoch %d)",
            output_dir,
            result["selected_epoch"],
        )


def train_crf(
    corpus: str | Path,
    output_dir: str | Path | None = None,
    *,
    config: CrfTrainConfig | None = None,
    arch_config: dict | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: build a :class:`CrfTrainer` and run it."""
    return CrfTrainer(corpus, config=config, arch_config=arch_config, output_dir=output_dir).train()
