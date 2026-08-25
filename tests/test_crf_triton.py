"""The Triton scan must agree with the PyTorch reference it replaces.

Skipped wholesale without Triton or a GPU — it ships with torch on Linux/CUDA
but is absent from the pixi `boundary` env, which is where CI runs. The
pure-PyTorch path in `_analytic` stays the reference, so a skip here costs
coverage of the fast path only, never of the maths.
"""

from __future__ import annotations

import pytest
import torch

from leech.crf._triton import HAVE_TRITON

pytestmark = pytest.mark.skipif(
    not (HAVE_TRITON and torch.cuda.is_available()),
    reason="needs triton and a CUDA device",
)


def _reference_alpha(ms, n_base, n_states):
    from leech.crf._analytic import _incoming

    t_len, batch = ms.shape[0], ms.shape[1]
    alphas = ms.new_empty(t_len + 1, batch, n_states)
    alphas[0] = 0.0
    for t in range(t_len):
        alphas[t + 1] = torch.logsumexp(_incoming(alphas[t], n_base, n_states) + ms[t], dim=-1)
    return torch.logsumexp(alphas[t_len], dim=-1), alphas


@pytest.mark.parametrize("t_len,batch", [(8, 3), (64, 16), (300, 32)])
def test_matches_pytorch_reference(t_len, batch):
    """Agreement has to hold at LENGTH, not just on a toy case.

    An earlier version kept `alpha` in global memory between timesteps and was
    exact at T=8 while diverging by ~5.0 at T=300 — a store is not guaranteed
    visible to the next load through L1. Short cases cannot catch that, hence
    the 300-step parameter.
    """
    from leech.crf._triton import full_alpha_scan

    torch.manual_seed(0)
    ms = torch.randn(t_len, batch, 256, 5, device="cuda") * 2
    with torch.no_grad():
        z_ref, a_ref = _reference_alpha(ms, 4, 256)
        z_tri, a_tri = full_alpha_scan(ms, 4, 256)

    # float32 accumulation over t_len steps; logZ is O(100s) so this is ~1e-6
    # relative, not a structural difference.
    assert torch.allclose(z_ref, z_tri, atol=1e-2), (z_ref - z_tri).abs().max()
    assert torch.allclose(a_ref, a_tri, atol=1e-2), (a_ref - a_tri).abs().max()


def test_alpha_zero_is_uniform_over_states():
    """All states start allowed — the property that makes the decode target[4:]."""
    from leech.crf._triton import full_alpha_scan

    ms = torch.zeros(3, 2, 256, 5, device="cuda")
    with torch.no_grad():
        _, alphas = full_alpha_scan(ms, 4, 256)
    assert torch.all(alphas[0] == 0.0)


@pytest.mark.parametrize("t_len,batch", [(8, 3), (64, 16), (300, 32)])
def test_fwd_bwd_matches_pytorch_reference(t_len, batch):
    """Posteriors are the gradient, so they must match, not merely correlate."""
    from leech.crf._analytic import _full_fwd_bwd
    from leech.crf._triton import full_fwd_bwd

    torch.manual_seed(0)
    ms = torch.randn(t_len, batch, 256, 5, device="cuda") * 2
    with torch.no_grad():
        z_ref, p_ref = _full_fwd_bwd(ms, 4, 256)
        z_tri, p_tri = full_fwd_bwd(ms, 4, 256)
    assert torch.allclose(z_ref, z_tri, atol=1e-2), (z_ref - z_tri).abs().max()
    assert torch.allclose(p_ref, p_tri, atol=1e-2), (p_ref - p_tri).abs().max()


def test_posteriors_are_a_distribution_over_edges():
    """They are probabilities: non-negative, and summing to 1 per timestep."""
    from leech.crf._triton import full_fwd_bwd

    torch.manual_seed(0)
    ms = torch.randn(30, 4, 256, 5, device="cuda") * 2
    with torch.no_grad():
        _, post = full_fwd_bwd(ms, 4, 256)
    assert torch.all(post >= 0)
    per_step = post.sum(dim=(-1, -2))  # over every edge of every state
    assert torch.allclose(per_step, torch.ones_like(per_step), atol=1e-3)


def test_loss_uses_triton_and_still_matches_the_pytorch_path():
    """The fast path must be switchable off and agree when it is on."""
    import os

    from leech.crf.loss import CtcCrfLoss

    torch.manual_seed(0)
    loss = CtcCrfLoss(4, 4).cuda()
    scores = torch.randn(40, 6, 1280, device="cuda") * 2
    targets = torch.randint(1, 5, (6, 14), device="cuda")
    lengths = torch.full((6,), 14, device="cuda", dtype=torch.long)

    a = scores.detach().requires_grad_(True)
    loss(a, targets, lengths).backward()
    os.environ["ESCAPEPOD_NO_TRITON"] = "1"
    try:
        b = scores.detach().requires_grad_(True)
        loss(b, targets, lengths).backward()
    finally:
        del os.environ["ESCAPEPOD_NO_TRITON"]
    assert torch.allclose(a.grad, b.grad, atol=1e-3), (a.grad - b.grad).abs().max()


@pytest.mark.parametrize("t_len,batch", [(8, 3), (64, 16), (300, 32)])
def test_max_semiring_kernel_matches_pytorch_reference(t_len, batch):
    """The decode's second pass, kernel vs reference — the EDGES must match.

    Not the values: this returns an argmax, so agreement is exact or it is a
    different decode. T=300 is here for the same reason as the log-semiring
    scan's — a loop-carried value round-tripped through global memory is right
    on short inputs and wrong on long ones.
    """
    from leech.crf._triton import max_best_edges
    from leech.crf.decode import _best_edges

    torch.manual_seed(0)
    # Log-posterior shaped, which is what the decode actually feeds it.
    ms = torch.log_softmax(
        torch.randn(t_len, batch, 256 * 5, device="cuda").flatten(2) * 2, dim=-1
    ).reshape(t_len, batch, 256, 5)
    with torch.no_grad():
        want = _best_edges(ms, 4, 256)
        got = max_best_edges(ms, 4, 256)
    assert torch.equal(want, got), (want != got).sum().item()


def test_decode_batch_agrees_with_and_without_the_kernel():
    """End to end: the decoded strings must be identical either way."""
    import os

    from leech.crf import decode_batch

    torch.manual_seed(1)
    scores = torch.randn(120, 8, 256 * 5, device="cuda") * 2
    with_kernel = decode_batch(scores)
    os.environ["ESCAPEPOD_NO_TRITON"] = "1"
    try:
        without = decode_batch(scores)
    finally:
        del os.environ["ESCAPEPOD_NO_TRITON"]
    assert with_kernel == without
