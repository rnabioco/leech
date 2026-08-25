"""Triton kernels for the CRF full-lattice scans.

The PyTorch scans launch ~6 kernels per timestep and materialise a
`(N, 256, 5)` intermediate each time, 300 times per direction. That is
launch- and bandwidth-bound, which is why it degrades 1.75x under node
contention while koi's compiled kernel does not. These do the whole scan in one
launch, one program per read, with `alpha` never leaving the SM's caches.

Why no gather
-------------
The obvious formulation needs `alpha[src(s, e)]`, an arbitrary gather Triton
handles badly. The lattice is regular enough to avoid one. Writing a state as
`s = a * n_base + b`::

    stay term for s      alpha[s]                      -> layout [a, b]
    move term for (s, k) alpha[k * stride + a]         -> layout [a, k]

the move source does not depend on `b`, so it broadcasts across it. Both layouts
are strided reads of the same 256 floats, which stay in L1 across the scan.

Availability
------------
Triton ships with torch on Linux/CUDA but is absent from some environments --
macOS, CPU-only wheels, and the conda-forge pytorch in escapepod-models' pixi
`boundary` env, which has torch without it. So `HAVE_TRITON` gates use, and the
pure-PyTorch path in `_analytic` stays the reference implementation and the
correctness oracle. Set `LEECH_NO_TRITON=1` to force it.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - environment-dependent
    HAVE_TRITON = False

__all__ = [
    "HAVE_TRITON",
    "full_alpha_scan",
    "full_fwd_bwd",
    "max_best_edges",
    "target_fwd_bwd",
]


if HAVE_TRITON:

    @triton.jit
    def _alpha_kernel(
        ms_ptr,  # (T, N, S, E) float32
        alphas_ptr,  # (T+1, N, S) float32, written
        logz_ptr,  # (N,) float32, written
        t_len,
        n_states: tl.constexpr,
        n_base: tl.constexpr,
        stride_s: tl.constexpr,  # n_states // n_base
    ):
        """Forward scan. One program per read; `alpha` stays resident."""
        pid = tl.program_id(0)
        a = tl.arange(0, stride_s)[:, None, None]
        b = tl.arange(0, n_base)[None, :, None]
        k = tl.arange(0, n_base)[None, None, :]
        s_ab = a * n_base + b  # (stride, n_base, 1)
        flat2d = tl.reshape(s_ab, (stride_s, n_base))

        # alpha lives in REGISTERS for the whole scan. An earlier version kept it
        # in global memory and re-loaded each step; that was 41x faster than
        # PyTorch and quietly WRONG at T=300 (right at T=8), because a store is
        # not guaranteed visible to the next load through L1. Never round-trip a
        # loop-carried value.
        alpha = tl.zeros((n_states,), dtype=tl.float32)

        alpha_base = alphas_ptr + pid * n_states
        n_prog = tl.num_programs(0)
        tl.store(alpha_base + flat2d, tl.reshape(alpha, (stride_s, n_base)))

        for t in range(t_len):
            # Two views of the same 256 registers, no gather and no reload:
            #   [a, b] for the stay term      (flat = a*n_base + b)
            #   [a, k] for the move sources   (flat = k*stride + a), via transpose
            stay_a = tl.reshape(alpha, (stride_s, n_base))
            move_a = tl.trans(tl.reshape(alpha, (n_base, stride_s)))

            m_base = ms_ptr + (t * n_prog + pid) * n_states * (n_base + 1)
            stay_m = tl.load(m_base + flat2d * (n_base + 1))
            move_m = tl.load(m_base + s_ab * (n_base + 1) + 1 + k)

            stay = stay_a + stay_m  # (stride, n_base)
            move = move_a[:, None, :] + move_m  # (stride, n_base, n_base)

            # `move` is 3D and `mx` is 2D; Triton will not add the trailing axis.
            mx = tl.maximum(stay, tl.max(move, axis=2))
            acc = tl.exp(stay - mx) + tl.sum(tl.exp(move - mx[:, :, None]), axis=2)
            nxt = mx + tl.log(acc)

            tl.store(alpha_base + (t + 1) * (n_prog * n_states) + flat2d, nxt)
            alpha = tl.reshape(nxt, (n_states,))

        mx = tl.max(alpha, axis=0)
        z = mx + tl.log(tl.sum(tl.exp(alpha - mx), axis=0))
        tl.store(logz_ptr + pid, z)


def full_alpha_scan(ms: torch.Tensor, n_base: int, n_states: int):
    """`(logZ, alphas)` for the full lattice. `ms` is `(T, N, S, E)`."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    t_len, batch = ms.shape[0], ms.shape[1]
    ms = ms.contiguous()
    alphas = ms.new_empty(t_len + 1, batch, n_states)
    logz = ms.new_empty(batch)
    _alpha_kernel[(batch,)](
        ms,
        alphas,
        logz,
        t_len,
        n_states=n_states,
        n_base=n_base,
        stride_s=n_states // n_base,
    )
    return logz, alphas


if HAVE_TRITON:

    @triton.jit
    def _beta_kernel(
        ms_ptr,  # (T, N, S, E)
        alphas_ptr,  # (T+1, N, S), from the forward kernel
        logz_ptr,  # (N,)
        post_ptr,  # (T, N, S, E), written
        t_len,
        n_states: tl.constexpr,
        n_base: tl.constexpr,
        stride_s: tl.constexpr,
    ):
        """Reverse scan + edge posteriors. One program per read.

        Two decompositions of a state are in play and they are NOT the same one:

            destination   s = m * n_base + j     (a state and the base leaving it)
            source        s = k * stride + m     (which predecessor fed it)

        Both are reshapes of the same flat 256 registers, so the kernel moves
        between them with reshape/trans rather than any gather — the same trick
        the forward scan uses.
        """
        pid = tl.program_id(0)
        n_prog = tl.num_programs(0)
        m_ = tl.arange(0, stride_s)[:, None, None]
        j_ = tl.arange(0, n_base)[None, :, None]
        k_ = tl.arange(0, n_base)[None, None, :]
        dest2d = tl.reshape(m_ * n_base + j_, (stride_s, n_base))

        logz = tl.load(logz_ptr + pid)
        # beta_T = 0: every state is an allowed end, matching alpha_0.
        beta = tl.zeros((n_states,), dtype=tl.float32)

        for i in range(t_len):
            t = t_len - 1 - i
            m_base = ms_ptr + (t * n_prog + pid) * n_states * (n_base + 1)
            ms_stay = tl.load(m_base + dest2d * (n_base + 1))
            ms_move = tl.load(m_base + (m_ * n_base + j_) * (n_base + 1) + 1 + k_)

            beta_dest = tl.reshape(beta, (stride_s, n_base))  # [m, j]
            joint_stay = ms_stay + beta_dest
            joint_move = ms_move + beta_dest[:, :, None]

            # --- posteriors need alpha at each edge's SOURCE
            a_base = alphas_ptr + (t * n_prog + pid) * n_states
            a_flat = tl.load(a_base + tl.arange(0, n_states))
            a_stay = tl.reshape(a_flat, (stride_s, n_base))  # [m, j] = alpha[s]
            a_move = tl.trans(tl.reshape(a_flat, (n_base, stride_s)))  # [m, k]

            p_base = post_ptr + (t * n_prog + pid) * n_states * (n_base + 1)
            tl.store(p_base + dest2d * (n_base + 1), tl.exp(a_stay + joint_stay - logz))
            tl.store(
                p_base + (m_ * n_base + j_) * (n_base + 1) + 1 + k_,
                tl.exp(a_move[:, None, :] + joint_move - logz),
            )

            # --- beta[s] = logsumexp over s's OUTGOING edges
            # stay: the edge leaving s and re-entering s, already indexed by s
            stay_term = tl.reshape(joint_stay, (n_states,))
            # moves: source (k, m) feeds destinations (m, j) on edge 1+k, so
            # reduce over j, then move from [m, k] into [k, m] = flat source id.
            mx = tl.max(joint_move, axis=1)  # [m, k]
            lse = mx + tl.log(tl.sum(tl.exp(joint_move - mx[:, None, :]), axis=1))
            move_term = tl.reshape(tl.trans(lse), (n_states,))  # [k, m] -> flat

            top = tl.maximum(stay_term, move_term)
            beta = top + tl.log(tl.exp(stay_term - top) + tl.exp(move_term - top))


def full_fwd_bwd(ms: torch.Tensor, n_base: int, n_states: int):
    """`(logZ, posteriors)` for the full lattice, both scans on the GPU.

    Drop-in for `_analytic._full_fwd_bwd`. Two launches rather than one: the
    beta pass reads the alphas the forward pass wrote, and a kernel boundary is
    the cheap way to get that visibility guarantee — an in-kernel store is not
    reliably visible to a later load in the same program, which is exactly the
    bug the forward scan hit.
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    logz, alphas = full_alpha_scan(ms, n_base, n_states)
    post = torch.empty_like(ms)
    _beta_kernel[(ms.shape[1],)](
        ms.contiguous(),
        alphas,
        logz,
        post,
        ms.shape[0],
        n_states=n_states,
        n_base=n_base,
        stride_s=n_states // n_base,
    )
    return logz, post


# --------------------------------------------------------------------------- #
# Target chain
#
# The lattice scans move between state layouts with reshape/trans because its
# transitions are strided. The target chain's are not: `alpha[i-1]` is a shift by
# one, which no reshape expresses. A shift IS a permutation, though, and a
# permutation is a matmul — `shifted = alpha @ M` with `M[j, i] = (j == i-1)`.
# That is why these kernels take a BLOCK of reads per program rather than one:
# `tl.dot` needs rows to work with, and one read gives a single row.
# --------------------------------------------------------------------------- #
if HAVE_TRITON:

    @triton.jit
    def _target_kernel(
        stay_ptr,  # (T, N, n)
        move_ptr,  # (T, N, n-1)
        alphas_ptr,  # (T+1, N, n) written by the forward pass, read by the reverse
        logz_ptr,  # (N,)
        stay_post_ptr,  # (T, N, n)     written on the reverse pass
        move_post_ptr,  # (T, N, n-1)   written on the reverse pass
        lengths_ptr,  # (N,)
        t_len,
        batch,
        n,
        FORWARD: tl.constexpr,
        BB: tl.constexpr,  # reads per program
        NMAX: tl.constexpr,  # padded chain length
        NEG: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * BB + tl.arange(0, BB)
        rmask = rows < batch
        i = tl.arange(0, NMAX)
        imask = i < n
        ok = rmask[:, None] & imask[None, :]

        # Permutation matrices. `fwd` shifts up (alpha[i-1]), `bwd` shifts down
        # (beta[i+1]); ieee precision because a permutation must be exact.
        jj = tl.arange(0, NMAX)[:, None]
        ii = tl.arange(0, NMAX)[None, :]
        shift_up = (jj == ii - 1).to(tl.float32)
        shift_dn = (jj == ii + 1).to(tl.float32)

        if FORWARD:
            # alpha_0: only state 0 is reachable.
            v = tl.where(i[None, :] == 0, 0.0, NEG) + tl.zeros((BB, NMAX), tl.float32)
            tl.store(alphas_ptr + rows[:, None] * n + i[None, :], v, mask=ok)
            for t in range(t_len):
                base = (t * batch + rows[:, None]) * n + i[None, :]
                stay = tl.load(stay_ptr + base, mask=ok, other=NEG)
                mv = tl.load(
                    move_ptr + (t * batch + rows[:, None]) * (n - 1) + i[None, :] - 1,
                    mask=ok & (i[None, :] >= 1),
                    other=NEG,
                )
                prev = tl.dot(v, shift_up, input_precision="ieee")
                prev = tl.where(i[None, :] >= 1, prev, NEG)
                a = v + stay
                b = prev + mv
                mx = tl.maximum(a, b)
                v = mx + tl.log(tl.exp(a - mx) + tl.exp(b - mx))
                tl.store(
                    alphas_ptr + ((t + 1) * batch + rows[:, None]) * n + i[None, :],
                    v,
                    mask=ok,
                )
            last = tl.load(lengths_ptr + rows, mask=rmask, other=1) - 1
            z = tl.sum(tl.where(i[None, :] == last[:, None], v, 0.0), axis=1)
            tl.store(logz_ptr + rows, z, mask=rmask)
        else:
            last = tl.load(lengths_ptr + rows, mask=rmask, other=1) - 1
            logz = tl.load(logz_ptr + rows, mask=rmask, other=0.0)
            # beta_T: only the final target state is an accepting end.
            v = tl.where(i[None, :] == last[:, None], 0.0, NEG)
            for u in range(t_len):
                t = t_len - 1 - u
                base = (t * batch + rows[:, None]) * n + i[None, :]
                stay = tl.load(stay_ptr + base, mask=ok, other=NEG)
                mv = tl.load(
                    move_ptr + (t * batch + rows[:, None]) * (n - 1) + i[None, :],
                    mask=ok & (i[None, :] < n - 1),
                    other=NEG,
                )
                alpha_t = tl.load(alphas_ptr + base, mask=ok, other=NEG)
                nxt = tl.dot(v, shift_dn, input_precision="ieee")
                nxt = tl.where(i[None, :] < n - 1, nxt, NEG)

                tl.store(stay_post_ptr + base, tl.exp(alpha_t + stay + v - logz[:, None]), mask=ok)
                tl.store(
                    move_post_ptr + (t * batch + rows[:, None]) * (n - 1) + i[None, :],
                    tl.exp(alpha_t + mv + nxt - logz[:, None]),
                    mask=ok & (i[None, :] < n - 1),
                )

                a = stay + v
                b = mv + nxt
                mx = tl.maximum(a, b)
                v = mx + tl.log(tl.exp(a - mx) + tl.exp(b - mx))


def target_fwd_bwd(stay: torch.Tensor, move: torch.Tensor, lengths: torch.Tensor):
    """`(logZ, stay_post, move_post)` for the target chain.

    Drop-in for `_analytic._target_fwd_bwd`. Two launches, same reason as the
    lattice: the reverse pass reads alphas the forward pass wrote, and a kernel
    boundary is the cheap way to guarantee they are visible.
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    t_len, batch, n = stay.shape
    stay = stay.contiguous()
    move = move.contiguous()
    lengths = lengths.to(torch.int32).contiguous()
    nmax = max(16, triton.next_power_of_2(n))
    bb = 16
    grid = (triton.cdiv(batch, bb),)
    alphas = stay.new_empty(t_len + 1, batch, n)
    logz = stay.new_empty(batch)
    stay_post = torch.empty_like(stay)
    move_post = torch.empty_like(move)
    for forward in (True, False):
        _target_kernel[grid](
            stay,
            move,
            alphas,
            logz,
            stay_post,
            move_post,
            lengths,
            t_len,
            batch,
            n,
            FORWARD=forward,
            BB=bb,
            NMAX=nmax,
            NEG=-1e30,
        )
    return logz, stay_post, move_post


# --------------------------------------------------------------------------- #
# Max semiring — the decode's second pass
#
# Structurally the same lattice as the log-semiring scans above, with `logsumexp`
# replaced by `max`. It got a kernel late: the decode is its only consumer, so
# training never exposed it, and the PyTorch version ran 600 Python-loop
# iterations launching a handful of tiny kernels each — dispatch-bound, not
# compute-bound, which is exactly the shape a fused scan fixes.
#
# The backward pass emits the best EDGE per timestep rather than posteriors.
# Under the max semiring the gradient is a one-hot on the best path, so the
# per-timestep argmax of `alpha[source] + score + beta[dest]` IS that path and no
# traceback array is needed.
# --------------------------------------------------------------------------- #
if HAVE_TRITON:

    @triton.jit
    def _max_alpha_kernel(
        ms_ptr,
        alphas_ptr,
        t_len,
        n_states: tl.constexpr,
        n_base: tl.constexpr,
        stride_s: tl.constexpr,
    ):
        """Forward max scan. Mirrors `_alpha_kernel`; alpha stays in registers."""
        pid = tl.program_id(0)
        a = tl.arange(0, stride_s)[:, None, None]
        b = tl.arange(0, n_base)[None, :, None]
        k = tl.arange(0, n_base)[None, None, :]
        s_ab = a * n_base + b
        flat2d = tl.reshape(s_ab, (stride_s, n_base))

        alpha = tl.zeros((n_states,), dtype=tl.float32)
        alpha_base = alphas_ptr + pid * n_states
        n_prog = tl.num_programs(0)
        tl.store(alpha_base + flat2d, tl.reshape(alpha, (stride_s, n_base)))

        for t in range(t_len):
            stay_a = tl.reshape(alpha, (stride_s, n_base))
            move_a = tl.trans(tl.reshape(alpha, (n_base, stride_s)))

            m_base = ms_ptr + (t * n_prog + pid) * n_states * (n_base + 1)
            stay = stay_a + tl.load(m_base + flat2d * (n_base + 1))
            move = move_a[:, None, :] + tl.load(m_base + s_ab * (n_base + 1) + 1 + k)

            nxt = tl.maximum(stay, tl.max(move, axis=2))
            tl.store(alpha_base + (t + 1) * (n_prog * n_states) + flat2d, nxt)
            alpha = tl.reshape(nxt, (n_states,))

    @triton.jit
    def _max_beta_kernel(
        ms_ptr,
        alphas_ptr,
        best_ptr,
        t_len,
        n_states: tl.constexpr,
        n_base: tl.constexpr,
        stride_s: tl.constexpr,
    ):
        """Reverse max scan, writing the best edge as flat `dest*E + edge`.

        The argmax is taken in two pieces rather than over one `(S, E)` tile,
        because assembling that tile needs a per-column select Triton has no
        cheap form for. Stay edges are one 1-D tile of `n_states`; move edges are
        one of `n_states * n_base`, indexed `s * n_base + k`. Comparing the two
        maxima and decoding whichever wins costs two reductions instead of a
        materialised tile.
        """
        pid = tl.program_id(0)
        n_prog = tl.num_programs(0)
        m_ = tl.arange(0, stride_s)[:, None, None]
        j_ = tl.arange(0, n_base)[None, :, None]
        k_ = tl.arange(0, n_base)[None, None, :]
        dest2d = tl.reshape(m_ * n_base + j_, (stride_s, n_base))

        beta = tl.zeros((n_states,), dtype=tl.float32)

        for i in range(t_len):
            t = t_len - 1 - i
            m_base = ms_ptr + (t * n_prog + pid) * n_states * (n_base + 1)
            ms_stay = tl.load(m_base + dest2d * (n_base + 1))
            ms_move = tl.load(m_base + (m_ * n_base + j_) * (n_base + 1) + 1 + k_)

            beta_dest = tl.reshape(beta, (stride_s, n_base))
            joint_stay = ms_stay + beta_dest
            joint_move = ms_move + beta_dest[:, :, None]

            a_base = alphas_ptr + (t * n_prog + pid) * n_states
            a_flat = tl.load(a_base + tl.arange(0, n_states))
            a_stay = tl.reshape(a_flat, (stride_s, n_base))
            a_move = tl.trans(tl.reshape(a_flat, (n_base, stride_s)))

            # --- best edge this timestep
            stay_full = tl.reshape(a_stay + joint_stay, (n_states,))
            # [m, j, k] reshapes to [s, k] with s = m*n_base + j, row-major.
            move_full = tl.reshape(a_move[:, None, :] + joint_move, (n_states * n_base,))
            # Index by masked MIN over precomputed flat indices, not by
            # `tl.argmax`. Two reasons, and the second is the load-bearing one:
            #
            #  * `tl.argmax` on these reshaped tiles returned indices whose value
            #    was not the maximum at all (picking -48.59 where the max was
            #    -40.21), so it is not to be trusted on a tile built this way.
            #  * ties then break toward the LOWEST flat index, which is exactly
            #    what `full.flatten(1).argmax(dim=1)` does in the reference. Any
            #    other rule would make the kernel a different decode on ties
            #    rather than the same one.
            s_max = tl.max(stay_full, axis=0)
            m_max = tl.max(move_full, axis=0)
            gmax = tl.maximum(s_max, m_max)

            s_i = tl.arange(0, n_states)
            stay_idx = s_i * (n_base + 1)  # edge 0 of each state
            m_i = tl.arange(0, n_states * n_base)  # i = s * n_base + k
            move_idx = (m_i // n_base) * (n_base + 1) + 1 + (m_i % n_base)

            unreachable = n_states * (n_base + 1)
            stay_cand = tl.min(tl.where(stay_full == gmax, stay_idx, unreachable))
            move_cand = tl.min(tl.where(move_full == gmax, move_idx, unreachable))
            tl.store(best_ptr + t * n_prog + pid, tl.minimum(stay_cand, move_cand))

            # --- beta[s] = max over s's OUTGOING edges
            stay_term = tl.reshape(joint_stay, (n_states,))
            move_term = tl.reshape(tl.trans(tl.max(joint_move, axis=1)), (n_states,))
            beta = tl.maximum(stay_term, move_term)


def max_best_edges(ms: torch.Tensor, n_base: int, n_states: int) -> torch.Tensor:
    """Best edge per timestep, `(T, N)` int64 flat `dest * (n_base+1) + edge`.

    Drop-in for `decode._best_edges`. `ms` is `(T, N, S, E)` log-posteriors.

    Two launches, as with the log-semiring pair: the reverse pass reads the
    alphas the forward pass wrote, and a kernel boundary is the cheap way to get
    that visibility guarantee.
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    t_len, batch = ms.shape[0], ms.shape[1]
    ms = ms.contiguous()
    alphas = ms.new_empty(t_len + 1, batch, n_states)
    best = torch.empty(t_len, batch, dtype=torch.int32, device=ms.device)
    _max_alpha_kernel[(batch,)](
        ms,
        alphas,
        t_len,
        n_states=n_states,
        n_base=n_base,
        stride_s=n_states // n_base,
    )
    _max_beta_kernel[(batch,)](
        ms,
        alphas,
        best,
        t_len,
        n_states=n_states,
        n_base=n_base,
        stride_s=n_states // n_base,
    )
    return best.long()
