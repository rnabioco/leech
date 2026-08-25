"""CTC-CRF training objective in plain PyTorch.

Two partition functions over the CRF lattice — one restricted to the target
path, one over every path — expressed as ordinary torch ops so autograd supplies
the backward pass. That matters more than it sounds: a hand-written backward for
a forward-backward algorithm is where numerical bugs hide, because a wrong
gradient does not crash, it just trains slightly worse.

(``_analytic`` then supersedes autograd on the hot path, computing the same
gradients directly from the beta scan — the trade ONT's koi makes too, whose
``LogZ.backward`` is one elementwise multiply. The plain scans here stay as the
readable reference the analytic version is tested against.)

The objective
-------------
A CRF over `n_base ** state_len` states. At each timestep a path either **stays**
in its state or **moves**, emitting one base. The loss is the negative
log-probability of the target path::

    loss = -(logZ(target lattice) - logZ(full lattice)) / target_length

The second term is the partition function over *all* paths. It is applied by
`normalise` as a per-read constant spread across timesteps, which is what makes
the scores comparable between reads rather than only within one.

State transitions
-----------------
A state is a `state_len`-mer, packed base-major: `s = b0*4^3 + b1*4^2 + b2*4 + b3`.
Moving emits a new base and shifts the window, so state `s` can only be reached
from a state whose last `state_len - 1` bases are `s`'s first `state_len - 1` —
that is, from `k * 4^(state_len-1) + s // n_base` for each `k`. Plus the stay
edge from `s` itself. Hence `n_base + 1` incoming edges per state, which is the
`(n_states, n_base + 1)` index this module builds and the same structure
escapepod-demux's Rust `CrfLayout` encodes for decoding.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._flags import flag

__all__ = ["CtcCrfLoss", "predecessor_index", "logZ_full", "logZ_target"]

#: Stand-in for "unreachable" in the target lattice.
#:
#: NOT `-inf`. `logaddexp(-inf, -inf)` is `-inf` in the forward pass but its
#: gradient is `exp(-inf - -inf)` = `exp(nan)` = nan, which then poisons every
#: upstream gradient — the loss looks perfect and training silently does nothing.
#: A large finite floor behaves identically in the forward direction (it
#: underflows to zero weight against any real path) while keeping the backward
#: pass finite. Chosen well below any achievable path score and well above
#: float32's limit, so adding a score to it cannot overflow.
_UNREACHABLE = -1e30


def predecessor_index(n_base: int, state_len: int, device=None) -> Tensor:
    """Incoming edges per state: `(n_states, n_base + 1)` of source-state ids.

    Column 0 is the stay edge (the state itself); columns `1..n_base` are the
    move edges, ordered by the base that falls out of the k-mer window. That
    ordering is not free choice — it has to match the score layout the encoder
    emits, `state * (n_base + 1) + label`, or the scores get paired with the
    wrong transitions and nothing complains.
    """
    n_states = n_base**state_len
    states = torch.arange(n_states, device=device)
    stay = states[:, None]
    # Predecessors share our first state_len-1 bases: shift right, then vary the
    # base that was dropped off the front.
    shifted = states // n_base
    moves = shifted[None, :] + (
        torch.arange(n_base, device=device)[:, None] * n_base ** (state_len - 1)
    )
    return torch.cat([stay, moves.T], dim=1).long()


def _incoming(alpha: Tensor, n_base: int, n_states: int) -> Tensor:
    """Line up each state's incoming alphas as `(N, n_states, n_base + 1)`.

    The obvious way is `alpha[:, idx]`, and it is 50x too slow: a 327k-element
    advanced-index gather, 300 times per forward, dominates everything else in
    training. The transition structure is regular enough not to need one.

    State `s` is reached from `s` (stay) and from `k * n_base^(state_len-1) +
    s // n_base` for each `k`. Writing `alpha` as `(N, n_base, n_states/n_base)`
    makes that second family a plain broadcast: the source for `(s, k)` is row
    `k`, column `s // n_base`, so expanding the column axis by `n_base`
    reproduces every destination without materialising an index.
    """
    stride = n_states // n_base
    moves = (
        alpha.view(-1, n_base, stride)  # [k, s // n_base]
        .unsqueeze(-1)
        .expand(-1, n_base, stride, n_base)  # repeat each column n_base times
        .reshape(-1, n_base, n_states)  # [k, s]
        .transpose(1, 2)  # [s, k]
    )
    return torch.cat([alpha.unsqueeze(-1), moves], dim=-1)


def _scan_full(scores: Tensor, n_base: int, n_states: int, n_edges: int) -> Tensor:
    """The forward scan itself, split out so it can be compiled as one graph."""
    t_len, batch, _ = scores.shape
    ms = scores.reshape(t_len, batch, n_states, n_edges)
    alpha = scores.new_zeros(batch, n_states)
    for t in range(t_len):
        alpha = torch.logsumexp(_incoming(alpha, n_base, n_states) + ms[t], dim=-1)
    return torch.logsumexp(alpha, dim=-1)


#: Cache for optionally-compiled scans.
#:
#: These plain scans are no longer on the loss path — `_analytic` supersedes
#: them — but they are kept as the readable reference the tests check the
#: analytic forward-backward against. Compiling the *step* did nothing (603 vs
#: 602 ms); compiling the whole scan gave 4.2x, which is why fusion across steps
#: is what matters here.
_COMPILED: dict[str, object] = {}


def _maybe_compiled(key: str, fn):
    """Compiled `fn`, built once. Falls back to eager if compilation is refused."""
    if flag("NO_COMPILE"):
        return fn
    cached = _COMPILED.get(key)
    if cached is None:
        try:
            cached = torch.compile(fn, dynamic=False)
        except Exception:  # noqa: BLE001 - torch.compile is best-effort
            cached = fn
        _COMPILED[key] = cached
    return cached


def logZ_full(scores: Tensor, idx: Tensor) -> Tensor:
    """Partition function over every path. `scores` is `(T, N, n_states*(n_base+1))`.

    Forward scan: `alpha[s] <- logsumexp_e(alpha[src(s,e)] + score[s,e])`. All
    states start allowed (`alpha_0 = 0`), because the CRF does not pin the first
    `state_len` bases — the same property that makes the decode `target[4:]`
    (#36).

    `idx` fixes the edge ordering and is used by the tests to check `_incoming`
    against an explicit gather; the scan itself derives the layout structurally.
    """
    from ._analytic import _FullLogZ

    n_states, n_edges = idx.shape
    return _FullLogZ.apply(scores, n_edges - 1, n_states, n_edges)


def _scan_target(stay: Tensor, move: Tensor) -> Tensor:
    """The target-chain forward scan, split out so it can be compiled as one graph."""
    t_len, batch, n = stay.shape
    alpha = stay.new_full((batch, n), _UNREACHABLE)
    alpha[:, 0] = 0.0
    for t in range(t_len):
        stayed = alpha + stay[t]
        moved = torch.nn.functional.pad(alpha[:, :-1] + move[t], (1, 0), value=_UNREACHABLE)
        alpha = torch.logaddexp(stayed, moved)
    return alpha


def logZ_target(stay: Tensor, move: Tensor, lengths: Tensor) -> Tensor:
    """Partition function restricted to the target path.

    A chain of `n` states where each step stays or advances by one. `stay` is
    `(T, N, n)` and `move` is `(T, N, n-1)`; the path starts at state 0 and must
    end at `lengths - 1`, so `lengths` is the number of target *states*, not the
    number of target bases.

    Deliberately NOT compiled, unlike the full-lattice scan. Measured at the
    training shape, compiling this one made the whole loss *slower* — 345 ms
    against 222 ms — because its chain is ~45 states rather than 256, so the
    per-call dispatch and guard overhead outweighs what fusion buys. Compiling
    more is not automatically better.
    """
    from ._analytic import _TargetLogZ

    return _TargetLogZ.apply(stay, move, lengths)


class CtcCrfLoss(torch.nn.Module):
    """Negative log-likelihood of a target sequence under the CRF.

    Targets are 1-indexed over the alphabet (`0` is reserved), matching the
    encoder's score layout where label 0 is the stay edge.

    **The first `state_len` target bases are never emitted.** They only fix the
    initial state, so a `target_len`-base target scores `target_len + 1 -
    state_len` states. Size targets accordingly — see issue #36, where a 40-nt
    target silently discarded the first four bases of a 27-nt barcode.
    """

    def __init__(self, n_base: int = 4, state_len: int = 4) -> None:
        super().__init__()
        self.n_base = n_base
        self.state_len = state_len
        self.register_buffer("idx", predecessor_index(n_base, state_len), persistent=False)

    @property
    def n_states(self) -> int:
        return self.n_base**self.state_len

    def normalise(self, scores: Tensor) -> Tensor:
        """Subtract the full-lattice logZ, spread evenly across timesteps."""
        return scores - logZ_full(scores, self.idx)[None, :, None] / scores.shape[0]

    def gather_target_scores(self, scores: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        """Pick out the stay/move scores along the target path.

        `targets` is `(N, L)` 1-indexed. Returns `(stay, move)` shaped
        `(T, N, n)` and `(T, N, n-1)` for `n = L + 1 - state_len` target states.
        """
        t_len = scores.shape[0]
        n_edges = self.n_base + 1
        tgt = (targets - 1).clamp(min=0)
        n = targets.shape[1] - (self.state_len - 1)
        # State id of each window of `state_len` consecutive target bases.
        state = sum(
            tgt[:, i : n + i] * self.n_base ** (self.state_len - i - 1)
            for i in range(self.state_len)
        )
        stay_at = state * n_edges
        # Moving into the next state emits the base leaving the window, which is
        # the target base `state_len` positions back — i.e. tgt[:, :n-1].
        move_at = stay_at[:, 1:] + tgt[:, : n - 1] + 1
        stay = scores.gather(2, stay_at.expand(t_len, -1, -1))
        move = scores.gather(2, move_at.expand(t_len, -1, -1))
        return stay, move

    def forward(
        self,
        scores: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
        *,
        normalise: bool = True,
        reduction: str = "mean",
    ) -> Tensor:
        """Negative log-likelihood of `targets`, normalised over all paths.

        Normalisation is applied **algebraically, not by materialising it.**
        `normalise()` subtracts `logZ_full / T` from every score, and every path
        through the target chain takes exactly one edge per timestep — so its
        partition function shifts by exactly `logZ_full`::

            logZ_target(normalised) == logZ_target(raw) - logZ_full

        Building the normalised tensor first cost 4.93 ms of a 14.68 ms loss at
        the training shape: 393 MB written and read back for a subtraction that
        cancels analytically. `normalise()` is kept because it is the readable
        statement of what this means, and the tests check the identity.
        """
        scores = scores.to(torch.float32)
        stay, move = self.gather_target_scores(scores, targets)
        logz = logZ_target(stay, move, target_lengths + 1 - self.state_len)
        if normalise:
            logz = logz - logZ_full(scores, self.idx)
        loss = -(logz / target_lengths)
        if reduction == "mean":
            return loss.mean()
        if reduction in ("none", None):
            return loss
        raise ValueError(f"unknown reduction {reduction!r}")
