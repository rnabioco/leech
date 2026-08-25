"""Regenerate ``crf_encoder_reference.npz``, the encoder's architecture golden.

    uv run python tests/fixtures/gen_crf_encoder_reference.py

Run this **only** when the architecture is meant to change, and say so in the
commit message. Regenerating it to make a red test go green destroys the only
thing it guards: that `CrfEncoder`'s graph still computes what the shipped
checkpoints' weights mean.

The weights are not stored — 520k parameters is 2 MB — they are filled
deterministically from a seeded generator, so the fixture pins the parameter
*names and shapes* as well as the arithmetic. That is deliberate: a renamed or
resized layer fails at the fill rather than surviving into a comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from leech.crf import CrfEncoder, EncoderConfig

#: Small enough to commit (T=20 x 1 read x 1280 scores), real geometry otherwise
#: — state_len 4, features 96, stride 10, so the 1280-wide blank splice and the
#: five alternating LSTMs are all exercised.
CONFIG = EncoderConfig(chunk=200)
SEED = 0
OUT = Path(__file__).parent / "crf_encoder_reference.npz"


def seeded_weights(model: CrfEncoder, seed: int = SEED) -> None:
    """Fill every parameter from one generator, in sorted key order."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, param in sorted(model.state_dict().items()):
            param.copy_(torch.empty_like(param).uniform_(-0.1, 0.1, generator=gen))


def reference_io(cfg: EncoderConfig = CONFIG) -> tuple[np.ndarray, np.ndarray]:
    """``(signal, scores)`` for a seeded encoder at ``cfg``."""
    model = CrfEncoder(cfg)
    seeded_weights(model)
    model.eval()
    gen = torch.Generator().manual_seed(SEED + 1)
    signal = torch.empty(1, 1, cfg.chunk).uniform_(-3.0, 3.0, generator=gen)
    with torch.no_grad():
        return signal.numpy(), model(signal).numpy()


def main() -> None:
    signal, scores = reference_io()
    np.savez_compressed(OUT, signal=signal, scores=scores)
    print(f"wrote {OUT} — signal {signal.shape}, scores {scores.shape}")


if __name__ == "__main__":
    main()
