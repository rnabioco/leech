"""CTC-CRF barcode encoder in plain PyTorch — no bonito, no ont-koi.

Why this exists
---------------
The shipped barcode models are trained through ``bonito.crf.model``, which is
licensed for Research Purposes only (ONT Public License v1.0, §2.1), and calls
into ``ont-koi``, whose wheel ships no licence of its own. That is fine for
grant-funded research — the ONT licence explicitly carves it in — but it
forecloses any commercial route and leaves an unlicensed dependency in the
training path. See issue #40.

This module is the encoder half of removing that dependency. It is a *from-graph*
reimplementation: every layer, shape, padding and constant below was read off
**our own exported ONNX** (``crf_encoder.onnx``) and verified numerically against
**our own fixture** (``reference_io.npz``), not transcribed from bonito's Python.
The architecture hyperparameters themselves are the published SeqTagger values
(Genome Res 35:956), cited rather than copied — see ``scripts/ldx/crf_config.toml``.

Honesty about "clean room": this is an *informed* reimplementation, not a
clean-room one in the legal sense. Whoever wrote it had read bonito's source. If
the licence review needs the stricter standard, the implementer has to be someone
who has not — the graph and fixtures in this repo are enough to work from.

What this does NOT remove
-------------------------
Training also needs the CTC-CRF loss, which is where ``ont-koi``'s CUDA kernels
live (``logZ_cu`` for the target lattice, ``logZ_cu_sparse`` for the normaliser).
Those are a separate piece of work. This module makes the *forward pass*
independent; it does not yet make training independent.

Architecture
------------
Read off the graph::

    Conv1d(1,  4, k=5,  s=1,  p=2)  + SiLU
    Conv1d(4, 16, k=5,  s=1,  p=2)  + SiLU
    Conv1d(16,96, k=31, s=10, p=15) + SiLU
    permute (N,C,T) -> (T,N,C)
    5x LSTM(96,96), alternating direction
    Linear(96, n_base**state_len * n_base)
    tanh, x scale
    insert a constant blank score per state -> (T, N, n_states*(n_base+1))

The trailing blank insertion is the part worth reading twice: the linear emits
only the *move* scores (4^4 states x 4 bases = 1024), and a constant
``blank_score`` is spliced in as the first entry of each state's group, giving
5*4^4 = 1280. That ordering — blank at index 0 of each group — is what makes
``score_index = state * (n_base + 1) + label`` with ``label == 0`` meaning stay,
which is the layout escapepod-demux's Rust decoder assumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ._analytic import _compiled

#: Constant score assigned to the stay/blank transition, spliced in after the
#: linear. A constant rather than a learned output is the CRF formulation's
#: choice: only moves are scored, and staying is free at a fixed cost.
BLANK_SCORE = 2.0


@dataclass(frozen=True)
class EncoderConfig:
    """Geometry of the encoder. Defaults are the shipped barcode models'.

    ``chunk`` is not used to build the module — the convolutions are
    length-agnostic — but it records what the model was trained at, since the
    CRF's window and target are coupled (see issue #36).
    """

    n_base: int = 4
    state_len: int = 4
    features: int = 96
    winlen: int = 31
    stride: int = 10
    scale: float = 5.0
    blank_score: float = BLANK_SCORE
    n_layers: int = 5
    chunk: int = 3000

    @property
    def n_states(self) -> int:
        """Number of CRF states: ``n_base ** state_len``."""
        return self.n_base**self.state_len

    @property
    def n_score(self) -> int:
        """Width of the score output: one blank + one per base, per state."""
        return self.n_states * (self.n_base + 1)

    @property
    def t_len(self) -> int:
        """Output timesteps for a ``chunk``-sample window."""
        return self.chunk // self.stride


def _crf_tail(lin: Tensor, n_states: int, n_base: int, scale: float, blank_score: float) -> Tensor:
    """`tanh * scale`, then splice the constant blank in as entry 0 per state.

    A free function of tensors and plain scalars rather than a method, so
    `torch.compile` can fuse it — this is three passes over ~700 MB at the
    training shape (tanh, scale, and the pad's copy into a 393 MB output), and
    inductor collapses them into one: **3.46 -> 0.92 ms, bit-identical**
    (`max|diff|` exactly 0, so it is the same scores, not merely close).

    Only on CUDA — see `CrfEncoder.forward` for why the CPU path stays eager.

    Compilation is opt-in via `LEECH_COMPILE=1`, and costs ~60 s the first
    time on a machine before inductor's on-disk cache makes it ~free. That pays
    for itself over a training run (36,960 forward passes, ~94 s saved) and over
    repeated evaluation on the same machine; it does NOT pay for a single short
    job, where 60 s buys back ~0.2 s. Off by default for that reason.

    `unflatten` + `pad` beats a manual reshape/cat: it says what the trailing
    axis means, and pads exactly one column on the low side. Preallocating and
    slice-assigning instead measured identical (5.24 vs 5.23 ms) — `pad` was
    already doing the efficient thing, so the win here is fusion, not the copy.
    """
    t, n, _ = lin.shape
    scores = torch.tanh(lin) * scale
    scores = scores.unflatten(-1, (n_states, n_base))
    scores = nn.functional.pad(scores, (1, 0), value=blank_score)
    return scores.reshape(t, n, n_states * (n_base + 1))


class _ReversibleLSTM(nn.Module):
    """A unidirectional LSTM that optionally runs backwards over time.

    Alternating direction down the stack gives every timestep access to context
    from both sides without the parameter cost of a bidirectional layer, which
    would double the width at each level. Implemented as flip-run-flip rather
    than ``bidirectional=True`` because those are different models: bidirectional
    concatenates two directions at one level, this alternates one direction
    across levels.
    """

    def __init__(self, size: int, reverse: bool) -> None:
        super().__init__()
        self.rnn = nn.LSTM(size, size, num_layers=1, bidirectional=False)
        self.reverse = reverse

    def forward(self, x: Tensor) -> Tensor:  # (T, N, C) -> (T, N, C)
        if self.reverse:
            x = x.flip(0)
        y, _ = self.rnn(x)
        return y.flip(0) if self.reverse else y


class CrfEncoder(nn.Module):
    """Raw signal -> CRF transition scores.

    Input  ``(N, 1, T_samples)`` standardised raw pA, batch-major.
    Output ``(T_samples // stride, N, n_score)`` float32, **time-major**.

    Time-major output is not incidental: it is the layout the decoder consumes,
    and the boundary CNN's batch-major ``[B,2,L]`` contract is a different one.
    A consumer that assumes batch-major will silently transpose barcode calls.
    """

    def __init__(self, cfg: EncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        c = self.cfg

        # SiLU is swish: x * sigmoid(x). torch has had it natively since 1.7, so
        # there is no reason to carry a custom activation.
        self.conv = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.SiLU(),
            nn.Conv1d(4, 16, kernel_size=5, stride=1, padding=2),
            nn.SiLU(),
            nn.Conv1d(16, c.features, kernel_size=c.winlen, stride=c.stride, padding=c.winlen // 2),
            nn.SiLU(),
        )
        # Layer i runs backwards when (n_layers - i) is odd, so the last layer
        # is always reversed. Matching this exactly matters: get the parity
        # wrong and the weights still load, the shapes still work, and the
        # output is quietly wrong.
        self.rnns = nn.ModuleList(
            _ReversibleLSTM(c.features, reverse=(c.n_layers - i) % 2 == 1)
            for i in range(c.n_layers)
        )
        self.linear = nn.Linear(c.features, c.n_states * c.n_base)

    def forward(self, x: Tensor) -> Tensor:
        c = self.cfg
        x = self.conv(x)  # (N, features, T)
        x = x.permute(2, 0, 1)  # -> (T, N, features)
        for rnn in self.rnns:
            x = rnn(x)
        lin = self.linear(x)
        # CUDA only. Inductor's CPU codegen for `tanh` does NOT round-trip
        # bit-exactly (measured 4.77e-07 against ATen), while its CUDA codegen
        # does (exactly 0). CPU is where the bit-exact-vs-shipped-ONNX test runs
        # and where the ONNX export traces, so the one place the fusion is not
        # provably exact is the one place exactness is asserted. Fuse where it is
        # free, stay eager where it is not.
        tail = _compiled("crf_tail", _crf_tail) if lin.is_cuda else _crf_tail
        return tail(lin, c.n_states, c.n_base, c.scale, c.blank_score)


#: Maps a bonito ``Serial`` checkpoint's positional names onto this module's.
#: Bonito indexes layers by position in a flat container (``encoder.4.rnn.*``);
#: naming them makes the checkpoint readable at the cost of needing this table.
def _legacy_key_map(n_layers: int = 5) -> dict[str, str]:
    m = {
        "encoder.0.conv.weight": "conv.0.weight",
        "encoder.0.conv.bias": "conv.0.bias",
        "encoder.1.conv.weight": "conv.2.weight",
        "encoder.1.conv.bias": "conv.2.bias",
        "encoder.2.conv.weight": "conv.4.weight",
        "encoder.2.conv.bias": "conv.4.bias",
        "encoder.9.linear.weight": "linear.weight",
        "encoder.9.linear.bias": "linear.bias",
    }
    # LSTMs sit at bonito indices 4..8 (3 convs + a permute occupy 0..3).
    for i in range(n_layers):
        for p in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
            m[f"encoder.{4 + i}.rnn.{p}"] = f"rnns.{i}.rnn.{p}"
    return m


def load_crf_state_dict(
    model: CrfEncoder, state_dict: dict[str, Tensor], *, strict: bool = True
) -> CrfEncoder:
    """Load a checkpoint into this module, in either naming.

    Accepts both because both exist and will keep existing: every model shipped
    before the trainer moved off bonito carries bonito's positional names
    (``encoder.4.rnn.*``), and everything trained since carries this module's
    (``rnns.0.rnn.*``). The parameters are identical — same layers, same shapes —
    so the only difference is the key, and refusing one naming would strand
    either the shipped models or the new ones.

    Raises if the checkpoint contains anything unmapped, rather than silently
    leaving a layer at its random initialisation.
    """
    mapping = _legacy_key_map(model.cfg.n_layers)
    # Which naming, decided by prefix rather than by whether the keys happen to
    # map. bonito's live under its flat container (`encoder.<i>.…`); this
    # module's are `conv.*`, `rnns.*`, `linear.*`. Asking "do any keys map?"
    # instead would classify an unmapped bonito key as native naming and report
    # the mismatch as torch's generic missing-keys error rather than as the
    # unknown-layer error that says what actually went wrong.
    if not any(k.startswith("encoder.") for k in state_dict):
        model.load_state_dict(state_dict, strict=strict)
        return model

    unmapped = sorted(set(state_dict) - set(mapping))
    if unmapped and strict:
        raise KeyError(
            f"checkpoint has {len(unmapped)} key(s) this module does not know: "
            f"{unmapped[:5]}{'...' if len(unmapped) > 5 else ''}"
        )
    renamed = {mapping[k]: v for k, v in state_dict.items() if k in mapping}
    model.load_state_dict(renamed, strict=strict)
    return model


def encoder_config_from_toml(cfg: dict) -> EncoderConfig:
    """Build an `EncoderConfig` from the architecture TOML.

    One reader for the config, so the trainer, the exporter and every evaluation
    script derive geometry the same way instead of each reaching into the dict
    for the keys it happens to need. `n_base` is the alphabet minus the CTC
    blank; it is not a key of its own, and reading it as one is a `KeyError`
    that only fires on the config's first use.
    """
    enc = cfg.get("encoder", {})
    return EncoderConfig(
        n_base=len(cfg["labels"]["labels"]) - 1,
        state_len=cfg["global_norm"]["state_len"],
        features=enc.get("features", 96),
        winlen=enc.get("winlen", 31),
        stride=enc.get("stride", 10),
        scale=enc.get("scale", 5.0),
        blank_score=enc.get("blank_score", BLANK_SCORE),
    )
