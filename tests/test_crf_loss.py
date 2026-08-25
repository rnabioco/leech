"""The bonito-free CTC-CRF loss must agree with the one that trained the models.

The equivalence check against bonito+koi itself needs CUDA and bonito, so it
lives in `scripts/ldx/analysis/verify_crf_loss.py` and is run by hand on a GPU
node. What is here runs anywhere: structural properties, and the invariants that
fail *silently* if broken.
"""

from __future__ import annotations

import pytest
import torch

from leech.crf.loss import (
    CtcCrfLoss,
    _incoming,
    logZ_full,
    logZ_target,
    predecessor_index,
)

N_BASE, STATE_LEN = 4, 4
N_STATES = N_BASE**STATE_LEN


def test_predecessor_index_shape_and_stay_edge():
    idx = predecessor_index(N_BASE, STATE_LEN)
    assert idx.shape == (N_STATES, N_BASE + 1)
    # Column 0 is the stay edge: every state is its own source.
    assert torch.equal(idx[:, 0], torch.arange(N_STATES))


def test_predecessors_share_the_kmer_overlap():
    """A move into state s must come from a state whose tail is s's head.

    This is the property that makes the lattice a k-mer lattice at all; get it
    wrong and the scan still runs and produces confident nonsense.
    """
    idx = predecessor_index(N_BASE, STATE_LEN)
    for s in range(0, N_STATES, 17):  # spot-check, not all 256
        head = s // N_BASE  # s's first state_len-1 bases
        for src in idx[s, 1:].tolist():
            assert src % (N_BASE ** (STATE_LEN - 1)) == head


def test_incoming_matches_an_explicit_gather():
    """The structured broadcast must equal `alpha[:, idx]`.

    `_incoming` avoids that gather because it is 50x too slow in the scan, so
    the cheap version has to be checked against the obvious one.
    """
    idx = predecessor_index(N_BASE, STATE_LEN)
    alpha = torch.randn(7, N_STATES)
    assert torch.equal(_incoming(alpha, N_BASE, N_STATES), alpha[:, idx])


def test_target_logZ_gradients_are_finite():
    """`-inf` initialisation gives NaN gradients; the finite floor must not.

    Caught for real: the forward pass and the loss value were both exactly
    right while every gradient was NaN, so training would have silently done
    nothing.
    """
    t_len, batch, n = 12, 3, 5
    stay = torch.randn(t_len, batch, n, requires_grad=True)
    move = torch.randn(t_len, batch, n - 1, requires_grad=True)
    lengths = torch.full((batch,), n, dtype=torch.long)
    logz = logZ_target(stay, move, lengths)
    logz.sum().backward()
    assert torch.isfinite(logz).all()
    assert torch.isfinite(stay.grad).all(), "NaN gradient from the unreachable floor"
    assert torch.isfinite(move.grad).all()


def test_logZ_full_is_a_proper_upper_bound_on_the_target_path():
    """Summing over all paths cannot be less than summing over one of them."""
    torch.manual_seed(0)
    loss = CtcCrfLoss(N_BASE, STATE_LEN)
    t_len, batch, tgt = 40, 2, 12
    scores = torch.randn(t_len, batch, N_STATES * (N_BASE + 1))
    targets = torch.randint(1, N_BASE + 1, (batch, tgt))
    lengths = torch.full((batch,), tgt, dtype=torch.long)

    z_all = logZ_full(scores, loss.idx)
    stay, move = loss.gather_target_scores(scores, targets)
    z_tgt = logZ_target(stay, move, lengths + 1 - STATE_LEN)
    assert torch.all(z_tgt <= z_all + 1e-4)


def test_loss_falls_when_the_targets_own_edges_are_rewarded():
    """Adding score to exactly the target path's edges must lower its loss.

    Scaling all scores does not qualify — that sharpens toward whichever path is
    already best, which need not be the target. This rewards the specific
    transitions the target traverses, which can only help it.
    """
    torch.manual_seed(0)
    loss = CtcCrfLoss(N_BASE, STATE_LEN)
    t_len, tgt = 60, 12
    targets = torch.randint(1, N_BASE + 1, (1, tgt))
    scores = torch.randn(t_len, 1, N_STATES * (N_BASE + 1)) * 0.1
    lengths = torch.tensor([tgt])

    base = loss(scores, targets, lengths).item()

    n = tgt - (STATE_LEN - 1)
    t = (targets - 1).clamp(min=0)
    state = sum(t[:, i : n + i] * N_BASE ** (STATE_LEN - i - 1) for i in range(STATE_LEN))
    stay_at = state * (N_BASE + 1)
    move_at = stay_at[:, 1:] + t[:, : n - 1] + 1

    boosted = scores.clone()
    for at in (stay_at, move_at):
        idx = at.unsqueeze(0).expand(t_len, -1, -1)
        boosted.scatter_add_(2, idx, torch.full(idx.shape, 3.0))

    assert loss(boosted, targets, lengths).item() < base


def test_reduction_none_gives_one_value_per_read():
    loss = CtcCrfLoss(N_BASE, STATE_LEN)
    batch, tgt = 4, 10
    scores = torch.randn(30, batch, N_STATES * (N_BASE + 1))
    targets = torch.randint(1, N_BASE + 1, (batch, tgt))
    lengths = torch.full((batch,), tgt, dtype=torch.long)
    per_read = loss(scores, targets, lengths, reduction="none")
    assert per_read.shape == (batch,)
    assert torch.allclose(per_read.mean(), loss(scores, targets, lengths))


def test_rejects_unknown_reduction():
    loss = CtcCrfLoss(N_BASE, STATE_LEN)
    scores = torch.randn(10, 1, N_STATES * (N_BASE + 1))
    targets = torch.randint(1, N_BASE + 1, (1, 8))
    with pytest.raises(ValueError, match="unknown reduction"):
        loss(scores, targets, torch.tensor([8]), reduction="sum")


def test_analytic_backward_matches_autograd():
    """The hand-written gradient must equal the one autograd derives.

    This is the check that makes a manual backward safe to ship. koi's gradient
    is analytic too, so without it we would be trusting two hand-derivations
    against each other; here the plain scans in `loss.py` are differentiated by
    autograd and used as the oracle, which needs no bonito and no GPU.
    """
    from leech.crf import _analytic
    from leech.crf.loss import _scan_full, _scan_target

    torch.manual_seed(0)
    t_len, batch, n = 9, 3, 6

    # --- target chain
    stay = torch.randn(t_len, batch, n, dtype=torch.double)
    move = torch.randn(t_len, batch, n - 1, dtype=torch.double)
    lengths = torch.full((batch,), n, dtype=torch.long)

    a = stay.clone().requires_grad_(True)
    b = move.clone().requires_grad_(True)
    ref = _scan_target(a, b).gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
    ref.sum().backward()

    c = stay.clone().requires_grad_(True)
    d = move.clone().requires_grad_(True)
    _analytic._TargetLogZ.apply(c, d, lengths).sum().backward()

    assert torch.allclose(a.grad, c.grad, atol=1e-8), "target stay gradient"
    assert torch.allclose(b.grad, d.grad, atol=1e-8), "target move gradient"

    # --- full lattice (small: 2 bases, state_len 2 -> 4 states, 3 edges)
    nb, sl = 2, 2
    ns = nb**sl
    scores = torch.randn(t_len, batch, ns * (nb + 1), dtype=torch.double)
    e = scores.clone().requires_grad_(True)
    _scan_full(e, nb, ns, nb + 1).sum().backward()
    f = scores.clone().requires_grad_(True)
    _analytic._FullLogZ.apply(f, nb, ns, nb + 1).sum().backward()
    assert torch.allclose(e.grad, f.grad, atol=1e-8), "full-lattice gradient"


def test_outgoing_is_the_transpose_of_incoming():
    """beta's scan needs each state's OUTGOING edges — the transpose of alpha's.

    Source `s = k*stride + m` feeds destinations `m*n_base + j`, and arrives
    there on edge `1 + k` — a different edge index from the one it leaves on,
    which is exactly the part that is easy to get subtly wrong while still
    producing plausible gradients.
    """
    from leech.crf._analytic import _outgoing

    nb, sl = 4, 2
    ns = nb**sl
    stride = ns // nb
    x = torch.randn(3, ns, nb + 1)
    got = _outgoing(x, nb, ns)

    want = torch.empty_like(got)
    for s in range(ns):
        k, m = divmod(s, stride)
        want[:, s, 0] = x[:, s, 0]  # stay edge stays put
        for j in range(nb):
            want[:, s, 1 + j] = x[:, m * nb + j, 1 + k]
    assert torch.allclose(got, want)


def test_algebraic_normalisation_equals_materialising_it():
    """`logZ_target(normalised) == logZ_target(raw) - logZ_full`.

    The loss relies on this identity to skip building a 393 MB tensor. Every
    target path takes exactly one edge per timestep, so subtracting `logZ/T`
    from every score shifts each path's total by exactly `logZ`. If that ever
    stopped holding the loss would be quietly wrong, so pin it.
    """
    torch.manual_seed(0)
    loss = CtcCrfLoss(N_BASE, STATE_LEN)
    t_len, batch, tgt = 25, 3, 11
    scores = torch.randn(t_len, batch, N_STATES * (N_BASE + 1), dtype=torch.double)
    targets = torch.randint(1, N_BASE + 1, (batch, tgt))
    lengths = torch.full((batch,), tgt, dtype=torch.long)

    normed = loss.normalise(scores)
    s_n, m_n = loss.gather_target_scores(normed, targets)
    materialised = logZ_target(s_n, m_n, lengths + 1 - STATE_LEN)

    s_r, m_r = loss.gather_target_scores(scores, targets)
    algebraic = logZ_target(s_r, m_r, lengths + 1 - STATE_LEN) - logZ_full(scores, loss.idx)
    assert torch.allclose(materialised, algebraic, atol=1e-8)
