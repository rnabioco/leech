"""CTC-CRF Viterbi decode in plain PyTorch.

Turns encoder scores into sequences. Structurally this reproduces bonito's
``SeqdistModel.decode_batch``, whose two-pass shape is load-bearing and is the
part worth reading before changing anything here.

Two passes, and both are needed
-------------------------------
`SeqdistModel.decode_batch` is **not** a Viterbi over the encoder's scores. It is:

1. **Log semiring.** Forward/backward over the lattice, then per-timestep edge
   posteriors — `softmax` over all `n_score` edges of
   ``alpha[t][source] + score + beta[t+1][dest]``. bonito obtains these as the
   *gradient* of ``logZ`` (``SequenceDist.posteriors`` is literally
   ``autograd.grad``), which is exactly what `_analytic._full_impl` already
   returns for the loss's backward — so pass 1 is free here, kernel and all.

2. **Max semiring** over ``log(posteriors + 1e-8)``. Forward/backward again, then
   the argmax edge at each timestep. Under the max semiring the gradient is a
   one-hot indicator of the best path, so bonito's ``posteriors(...).argmax(2)``
   and a traceback pick the same edge — which is why this returns the argmax
   directly and never walks a backpointer array.

Decoding the raw scores in one pass is a *different and worse* decode. The
two-pass structure is load-bearing; this note exists because it is the obvious
thing to simplify away.

The floor is applied to the probability, not to the log (``post + 1e-8`` then
``log``), so it is not a constant shift and cannot be folded into the softmax.

Which base a move emits
-----------------------
Edges are indexed ``(destination, dropped_base)``, not ``(source, emitted_base)``.
Destination ``c`` spells ``[c3 c2 c1 c0]``; its source under edge ``1+j`` spells
``[j c3 c2 c1]``. So a move drops ``j`` off the front and appends ``c0`` — the
base *emitted* is fixed by the destination alone (``c % n_base``) and the edge
index only says which base fell out. Getting that backwards decodes to
plausible-looking garbage, so `escapepod-demux`'s Rust port checks it against a
transcription of bonito's tensor and `test_crf_decode.py` checks this one.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ._analytic import _full_impl, _incoming, _outgoing
from ._flags import flag

#: bonito's `posteriors(x) + 1e-8`, floored on the probability before the log.
POSTERIOR_FLOOR = 1e-8

#: Index 0 is the blank/stay symbol and is never emitted.
ALPHABET = "NACGT"


def _best_edges_impl(ms: Tensor, n_base: int, n_states: int) -> Tensor:
    """Triton where available, PyTorch otherwise — mirrors `_analytic._full_impl`."""
    if ms.is_cuda and not flag("NO_TRITON"):
        try:
            from ._triton import HAVE_TRITON, max_best_edges

            if HAVE_TRITON:
                return max_best_edges(ms, n_base, n_states)
        except Exception:  # noqa: BLE001 - never let the fast path break a decode
            pass
    return _best_edges(ms, n_base, n_states)


def _best_edges(ms: Tensor, n_base: int, n_states: int) -> Tensor:
    """The max-semiring best edge at each timestep, as a flat `dest*E + edge`.

    Structurally identical to `_analytic._full_fwd_bwd` with `logsumexp` swapped
    for `max` — the same lattice, a different semiring. Kept separate rather than
    parameterised because the log-semiring version carries the autograd Function
    and this one does not.

    This is the reference the Triton kernel is tested against. It is also 15x
    slower: 600 Python-loop iterations launching a handful of tiny kernels each,
    which is dispatch-bound rather than compute-bound. Evaluation runs it on
    every read, so `_best_edges_impl` prefers the kernel.
    """
    t_len, batch = ms.shape[0], ms.shape[1]

    alphas = ms.new_empty(t_len + 1, batch, n_states)
    alphas[0] = 0.0
    for t in range(t_len):
        alphas[t + 1] = (_incoming(alphas[t], n_base, n_states) + ms[t]).amax(dim=-1)

    beta = ms.new_zeros(batch, n_states)
    best = torch.empty(t_len, batch, dtype=torch.long, device=ms.device)
    for t in range(t_len - 1, -1, -1):
        joint = ms[t] + beta.unsqueeze(-1)  # [dest, edge]
        full = _incoming(alphas[t], n_base, n_states) + joint
        best[t] = full.flatten(1).argmax(dim=1)
        beta = _outgoing(joint, n_base, n_states).amax(dim=-1)
    return best


def best_path(scores: Tensor, n_base: int = 4, state_len: int = 4) -> tuple[Tensor, Tensor]:
    """Per-timestep best edge, as `(destination_state, edge)` tensors `(T, N)`.

    `edge == 0` is the stay/blank edge and emits nothing. Exposed separately from
    `decode_batch` because the position profile analysis needs *where* bases were
    emitted, not just the string (`scripts/ldx/decode_position_profile.py`).
    """
    n_states, n_edges = n_base**state_len, n_base + 1
    t_len, batch, width = scores.shape
    if width != n_states * n_edges:
        raise ValueError(
            f"score width {width} != n_states*(n_base+1) = {n_states * n_edges}; "
            f"1024 here means the blank was never expanded per state"
        )

    ms = scores.to(torch.float32).reshape(t_len, batch, n_states, n_edges)
    _, post = _full_impl(ms, n_base, n_states)
    lp = torch.log(post + POSTERIOR_FLOOR)
    best = _best_edges_impl(lp, n_base, n_states)
    return best // n_edges, best % n_edges


@torch.no_grad()
def decode_batch(
    scores: Tensor,
    n_base: int = 4,
    state_len: int = 4,
    alphabet: str = ALPHABET,
) -> list[str]:
    """Scores `(T, N, n_states*(n_base+1))` -> one sequence per read.

    Drop-in for `bonito.crf.model.Model.decode_batch`, including its output type:
    a plain list of `str`, one per read, in batch order.

    The emitted sequence is `target[state_len:]` of whatever the model was
    trained on — the model cannot emit the first `state_len` bases at any window
    width (issue #36). Match decodes against `ldxlib.emitted_refs()`, not against
    the full-length target.
    """
    dest, edge = best_path(scores, n_base, state_len)
    # Emit `dest % n_base` on a move, blank (0) on a stay: see the module note on
    # why the base comes from the destination and not from the edge index.
    symbols = torch.where(edge > 0, dest % n_base + 1, torch.zeros_like(dest))
    out = symbols.t().to(torch.uint8).cpu().numpy()  # (N, T), batch-major

    # Byte lookup, then one `tobytes()` per READ. The obvious form,
    # `"".join(alphabet[s] for s in row if s)`, is one interpreter step per
    # SYMBOL — 76,800 of them for a 256-read batch at T=300, which profiled at
    # 3.79 ms, the second-largest cost in the whole decode behind the encoder.
    # This is the same computation with the per-symbol work pushed into numpy.
    lut = np.frombuffer(alphabet.encode(), dtype=np.uint8)
    chars = lut[out]
    blank = lut[0]
    return [row[row != blank].tobytes().decode() for row in chars]
