"""The decode must reproduce the reference rule, not merely resemble it.

Agreement with bonito itself needs CUDA and bonito, so it lives in
`scripts/ldx/analysis/verify_crf_decode.py` and runs by hand on a GPU node
against real weights. What is here runs anywhere: the structural properties, and
the two inversions that fail *silently* — emitting the wrong base for a move, and
collapsing the two passes into one.
"""

from __future__ import annotations

import pytest
import torch

from leech.crf import CrfEncoder, EncoderConfig, best_path, decode_batch
from leech.crf.decode import _best_edges

N_BASE, STATE_LEN = 4, 4
N_STATES, N_EDGES = N_BASE**STATE_LEN, N_BASE + 1


def test_rejects_unexpanded_score_width():
    """1024 is the linear's width; 1280 is the lattice's. Confusing them is easy."""
    scores = torch.randn(10, 2, N_STATES * N_BASE)
    with pytest.raises(ValueError, match="never expanded per state"):
        decode_batch(scores)


def test_emits_one_symbol_per_move_and_none_per_stay():
    torch.manual_seed(0)
    scores = torch.randn(30, 4, N_STATES * N_EDGES) * 2
    dest, edge = best_path(scores)
    seqs = decode_batch(scores)
    for i, s in enumerate(seqs):
        assert len(s) == int((edge[:, i] > 0).sum())


def test_decoded_length_never_exceeds_timesteps():
    """One base per move, at most one move per timestep."""
    torch.manual_seed(0)
    t_len = 25
    scores = torch.randn(t_len, 3, N_STATES * N_EDGES) * 2
    assert all(len(s) <= t_len for s in decode_batch(scores))


def test_move_emits_the_destinations_last_base_not_the_edge_index():
    """`base = dest % n_base`, NOT `edge - 1`.

    Edges are indexed (destination, dropped_base): the source of edge `1+j` into
    `c` spells `[j c3 c2 c1]`, so `j` is the base that fell OFF the front and
    `c % n_base` is the one appended. Swapping them still decodes, still yields
    ACGT strings of the right length, and calls the wrong barcode.
    """
    torch.manual_seed(0)
    scores = torch.randn(20, 3, N_STATES * N_EDGES) * 2
    dest, edge = best_path(scores)
    seqs = decode_batch(scores)

    for i, s in enumerate(seqs):
        want = ["ACGT"[int(dest[t, i]) % N_BASE] for t in range(scores.shape[0]) if edge[t, i] > 0]
        assert s == "".join(want)


def test_stay_edges_dominate_when_only_blanks_are_favoured():
    """Blank at index 0 of each state group — pin the layout end to end."""
    scores = torch.full((15, 2, N_STATES * N_EDGES), -10.0)
    grouped = scores.unflatten(-1, (N_STATES, N_EDGES))
    grouped[..., 0] = 10.0  # every stay edge
    assert decode_batch(grouped.reshape(15, 2, -1)) == ["", ""]


def test_a_move_only_lattice_emits_at_every_timestep():
    scores = torch.full((12, 2, N_STATES * N_EDGES), -10.0)
    grouped = scores.unflatten(-1, (N_STATES, N_EDGES))
    grouped[..., 1:] = 10.0  # every move edge
    seqs = decode_batch(grouped.reshape(12, 2, -1))
    assert all(len(s) == 12 for s in seqs)


def test_second_pass_is_a_different_decode_from_one_pass_viterbi():
    """The posterior pass is load-bearing, not a normalisation nicety.

    If the two structures agreed, the log-semiring pass would be dead code and
    someone would eventually delete it. They do not agree.

    Note what specifically makes them differ. A per-timestep softmax of the raw
    scores would NOT: `logsumexp` is constant within a timestep, so it shifts
    every path equally and leaves every argmax alone. The posteriors are a
    softmax of `alpha[source] + score + beta[dest]`, and it is the alpha/beta
    terms — global path context, not local normalisation — that change the
    decode.
    """
    from leech.crf._analytic import _full_impl

    torch.manual_seed(3)
    scores = torch.randn(60, 8, N_STATES * N_EDGES) * 2
    ms = scores.reshape(60, 8, N_STATES, N_EDGES)

    one_pass = _best_edges(ms, N_BASE, N_STATES)  # raw scores, max semiring only

    _, post = _full_impl(ms, N_BASE, N_STATES)  # the real pass 1
    two_pass = _best_edges(torch.log(post + 1e-8), N_BASE, N_STATES)

    assert not torch.equal(one_pass, two_pass)

    # ...and local normalisation alone really is a no-op, which is why the
    # posterior pass cannot be replaced by a cheaper softmax.
    local = torch.log_softmax(ms.flatten(2), dim=-1).reshape(ms.shape)
    assert torch.equal(one_pass, _best_edges(local, N_BASE, N_STATES))


def test_batch_order_is_preserved():
    """Decoding a batch must equal decoding its reads one at a time."""
    torch.manual_seed(1)
    scores = torch.randn(20, 5, N_STATES * N_EDGES) * 2
    together = decode_batch(scores)
    apart = [decode_batch(scores[:, i : i + 1])[0] for i in range(5)]
    assert together == apart


def test_untrained_encoder_output_decodes():
    """End to end on the real geometry: encoder -> decode, shapes must line up."""
    cfg = EncoderConfig(chunk=600)
    model = CrfEncoder(cfg).eval()
    with torch.no_grad():
        scores = model(torch.randn(3, 1, cfg.chunk))
    seqs = decode_batch(scores, cfg.n_base, cfg.state_len)
    assert len(seqs) == 3
    assert all(set(s) <= set("ACGT") for s in seqs)


def test_symbol_lookup_matches_the_naive_join():
    """The vectorised join must equal the per-symbol one it replaced.

    That form was 3.79 ms of a 25 ms decode -- one interpreter step per symbol,
    76,800 per batch. Correctness here is pure string handling, so it is worth
    pinning against the obvious implementation rather than trusting the LUT.
    """
    torch.manual_seed(4)
    scores = torch.randn(40, 6, N_STATES * N_EDGES) * 2
    dest, edge = best_path(scores)
    symbols = torch.where(edge > 0, dest % N_BASE + 1, torch.zeros_like(dest))
    naive = ["".join("NACGT"[s] for s in row if s) for row in symbols.t().cpu().numpy()]
    assert decode_batch(scores) == naive


def test_respects_a_custom_alphabet():
    """`alphabet` is a parameter, so the lookup table must come from it."""
    torch.manual_seed(5)
    scores = torch.randn(30, 3, N_STATES * N_EDGES) * 2
    default = decode_batch(scores)
    custom = decode_batch(scores, alphabet="-WXYZ")
    table = str.maketrans("ACGT", "WXYZ")
    assert custom == [s.translate(table) for s in default]
