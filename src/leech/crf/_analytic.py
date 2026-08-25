"""Analytic forward-backward for the CRF partition functions.

Autograd differentiates a 300-step scan by replaying 300 steps of graph. The
algorithm does not need that: for a CRF, `d logZ / d score[edge]` is exactly the
posterior probability of traversing that edge, which a second (beta) scan gives
directly. So the forward runs both scans and saves the posteriors, and the
backward is one elementwise multiply.

Measured at the training shape (T=300, N=256, 48-nt targets) before this change::

    full-lattice scan   fwd 4.0 ms   fwd+bwd 145.2 ms    -> ~141 ms of backward
    target scan         fwd 14.7     fwd+bwd  84.2       -> ~69 ms of backward

The forward roughly doubles (two scans instead of one) and the backward becomes
free, which is the trade koi makes — its `LogZ.backward` is `saved * grad`.

`torch.compile` is **off** here, and opt-in via `LEECH_COMPILE=1` (or the
`ESCAPEPOD_COMPILE=1` this package answered to before the move -- see
`_flags.py`). It helped
the plain autograd version (the 300-step alpha scan went 32.9 -> 4.0 ms), but a
fused forward+backward is ~600 unrolled steps plus the posterior arithmetic, and
inductor did not finish compiling that in 11 minutes even at T=40, N=4. A
one-time cost that large is worse than the throughput it buys.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._flags import flag

_UNREACHABLE = -1e30
_COMPILED: dict[str, object] = {}


def _compiled(key: str, fn):
    # Off by default: see the module docstring. Opt in with LEECH_COMPILE=1.
    if not flag("COMPILE"):
        return fn
    cached = _COMPILED.get(key)
    if cached is None:
        try:
            cached = torch.compile(fn, dynamic=False)
        except Exception:  # noqa: BLE001 - best effort
            cached = fn
        _COMPILED[key] = cached
    return cached


# --------------------------------------------------------------------------- #
# Full lattice
# --------------------------------------------------------------------------- #
def _incoming(alpha: Tensor, n_base: int, n_states: int) -> Tensor:
    """Each state's incoming alphas, `(N, S, n_base+1)`, without a gather.

    State `s` is reached from `s` (stay) and from `k * S/n_base + s // n_base`.
    Viewing alpha as `(k, s // n_base)` turns that into a broadcast.
    """
    stride = n_states // n_base
    moves = (
        alpha.view(-1, n_base, stride)
        .unsqueeze(-1)
        .expand(-1, n_base, stride, n_base)
        .reshape(-1, n_base, n_states)
        .transpose(1, 2)
    )
    return torch.cat([alpha.unsqueeze(-1), moves], dim=-1)


def _outgoing(x: Tensor, n_base: int, n_states: int) -> Tensor:
    """Transpose of `_incoming`: for each *source* state, its outgoing terms.

    `x[d, e]` is a quantity attached to the edge entering `d` on edge `e`. Source
    `s = k*stride + m` feeds destinations `m*n_base + b` on edge `1 + k`, so
    reshaping `x`'s move columns to `(m, b, k)` and permuting to `(k, m, b)`
    lines every source up with its four outgoing moves.
    """
    stride = n_states // n_base
    stay = x[..., :1]
    moves = (
        x[..., 1:]
        .reshape(-1, stride, n_base, n_base)  # [m, b, k]
        .permute(0, 3, 1, 2)  # [k, m, b]
        .reshape(-1, n_states, n_base)  # s = k*stride + m
    )
    return torch.cat([stay, moves], dim=-1)


def _full_fwd_bwd(ms: Tensor, n_base: int, n_states: int):
    """alpha, beta and edge posteriors for the full lattice.

    `ms` is `(T, N, S, E)`. Returns `(logZ, posteriors)` with posteriors the
    same shape as `ms`, summing to 1 over all edges at each timestep.
    """
    t_len, batch = ms.shape[0], ms.shape[1]
    alphas = ms.new_empty(t_len + 1, batch, n_states)
    alphas[0] = 0.0
    for t in range(t_len):
        alphas[t + 1] = torch.logsumexp(_incoming(alphas[t], n_base, n_states) + ms[t], dim=-1)
    logz = torch.logsumexp(alphas[t_len], dim=-1)

    # beta[t][s] = logsumexp over s's outgoing edges of (score + beta[t+1][dest])
    beta = ms.new_zeros(batch, n_states)
    post = ms.new_empty(ms.shape)
    for t in range(t_len - 1, -1, -1):
        joint = ms[t] + beta.unsqueeze(-1)  # [dest, edge]
        # edge posterior needs alpha at the SOURCE of each edge
        post[t] = _incoming(alphas[t], n_base, n_states) + joint
        beta = torch.logsumexp(_outgoing(joint, n_base, n_states), dim=-1)
    post = torch.exp(post - logz.reshape(1, batch, 1, 1))
    return logz, post


def _full_impl(ms: Tensor, n_base: int, n_states: int):
    """Triton where it is available and useful, PyTorch otherwise.

    The kernel is 38x on this scan but needs Triton and CUDA; the PyTorch path
    stays the reference implementation and the oracle the kernel is tested
    against, so falling back costs speed and nothing else.
    """
    if ms.is_cuda and not flag("NO_TRITON"):
        try:
            from ._triton import HAVE_TRITON, full_fwd_bwd

            if HAVE_TRITON:
                return full_fwd_bwd(ms, n_base, n_states)
        except Exception:  # noqa: BLE001 - never let the fast path break training
            pass
    return _compiled("full_fb", _full_fwd_bwd)(ms, n_base, n_states)


class _FullLogZ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: Tensor, n_base: int, n_states: int, n_edges: int):
        ms = scores.reshape(scores.shape[0], scores.shape[1], n_states, n_edges)
        logz, post = _full_impl(ms, n_base, n_states)
        ctx.save_for_backward(post)
        ctx.shape = scores.shape
        return logz

    @staticmethod
    def backward(ctx, grad: Tensor):
        (post,) = ctx.saved_tensors
        g = post * grad.reshape(1, -1, 1, 1)
        return g.reshape(ctx.shape), None, None, None


# --------------------------------------------------------------------------- #
# Target chain
# --------------------------------------------------------------------------- #
def _target_fwd_bwd(stay: Tensor, move: Tensor, lengths: Tensor):
    """alpha, beta and edge posteriors for the target chain."""
    t_len, batch, n = stay.shape
    alphas = stay.new_full((t_len + 1, batch, n), _UNREACHABLE)
    alphas[0, :, 0] = 0.0
    for t in range(t_len):
        stayed = alphas[t] + stay[t]
        moved = torch.nn.functional.pad(alphas[t][:, :-1] + move[t], (1, 0), value=_UNREACHABLE)
        alphas[t + 1] = torch.logaddexp(stayed, moved)

    last = (lengths - 1).clamp(min=0)
    logz = alphas[t_len].gather(1, last.unsqueeze(1)).squeeze(1)

    beta = stay.new_full((batch, n), _UNREACHABLE)
    beta.scatter_(1, last.unsqueeze(1), 0.0)
    stay_post = stay.new_empty(stay.shape)
    move_post = move.new_empty(move.shape)
    for t in range(t_len - 1, -1, -1):
        s_term = stay[t] + beta
        m_term = move[t] + beta[:, 1:]
        stay_post[t] = alphas[t] + s_term
        move_post[t] = alphas[t][:, :-1] + m_term
        beta = torch.logaddexp(s_term, torch.nn.functional.pad(m_term, (0, 1), value=_UNREACHABLE))
    z = logz.reshape(1, batch, 1)
    return logz, torch.exp(stay_post - z), torch.exp(move_post - z)


def _target_impl(stay: Tensor, move: Tensor, lengths: Tensor):
    """Triton where available, PyTorch otherwise — see `_full_impl`."""
    if stay.is_cuda and not flag("NO_TRITON"):
        try:
            from ._triton import HAVE_TRITON, target_fwd_bwd

            if HAVE_TRITON:
                return target_fwd_bwd(stay, move, lengths)
        except Exception:  # noqa: BLE001 - never let the fast path break training
            pass
    return _compiled("tgt_fb", _target_fwd_bwd)(stay, move, lengths)


class _TargetLogZ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, stay: Tensor, move: Tensor, lengths: Tensor):
        logz, sp, mp = _target_impl(stay, move, lengths)
        ctx.save_for_backward(sp, mp)
        return logz

    @staticmethod
    def backward(ctx, grad: Tensor):
        sp, mp = ctx.saved_tensors
        g = grad.reshape(1, -1, 1)
        return sp * g, mp * g, None
