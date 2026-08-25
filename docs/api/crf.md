# CRF Module

CTC-CRF sequence models: encoder, training objective, and decode.

## Overview

A second task alongside leech's chunk classifiers. Where a classifier maps a
signal window to a label, `leech.crf` maps one to a **sequence**: a CRF over
`n_base ** state_len` states whose Viterbi traceback emits one base per move.
It is what the barcode basecallers in
[escapepod-models](https://github.com/rnabioco/escapepod-models) are trained
with, and what `escapepod-demux`'s Rust decoder runs in production.

```python
import torch
from leech.crf import CrfEncoder, CtcCrfLoss, decode_batch, encoder_config_from_toml, load_config

cfg = encoder_config_from_toml(load_config())      # packaged default geometry
model = CrfEncoder(cfg)
criterion = CtcCrfLoss(cfg.n_base, cfg.state_len)

scores = model(signal)                              # (N, 1, chunk) -> (T, N, n_score)
loss = criterion(scores.float(), targets, target_lengths)
sequences = decode_batch(scores, cfg.n_base, cfg.state_len)
```

The subpackage imports **only torch and numpy** — not pysam, polars, or
escapepod. That is deliberate: escapepod-models installs leech into a
conda-forge environment with `--no-deps` and needs the CRF path alone.

## Four things to know before using it

### The model cannot emit the first `state_len` bases of its target

They fix the initial state and nothing else, so a `target_len`-base target
decodes to `target_len - state_len` bases — **at any window width**. Widening
the signal window does not lengthen the decode. Size targets so the sacrificial
bases come from a constant prefix, and match decodes against
`target[state_len:]`, never the full-length target. Matching against the full
target still calls the right sequence, but inflates every edit distance and
compresses the confidence margin that ranking depends on.

### Blank is entry 0 of each state's group

`score_index = state * (n_base + 1) + label`, with `label == 0` meaning stay.
The score width is therefore `n_states * (n_base + 1)` = **1280** for the
default geometry, not the linear layer's 1024. This is the layout
`escapepod-demux`'s Rust decoder assumes; moving the blank to the end of each
group keeps every shape valid and makes every call wrong.

### Output is time-major

`(T, N, n_score)`, not `(N, T, n_score)`. The boundary CNN in the same stack is
batch-major `[B, 2, L]`, so the two contracts sit next to each other and a
consumer that assumes the wrong one silently transposes rather than failing.

### The loss runs in fp32, outside autocast

The lattice scan accumulates over `chunk // stride` timesteps and fp16 loses the
tail of that sum. Autocast the encoder — that is where the matmuls and the speed
are — and cast the scores back before the loss.

## Encoder

::: leech.crf.encoder.CrfEncoder
    options:
      show_root_heading: true
      show_source: true

::: leech.crf.encoder.EncoderConfig
    options:
      show_root_heading: true

::: leech.crf.encoder.encoder_config_from_toml
    options:
      show_root_heading: true

::: leech.crf.encoder.load_crf_state_dict
    options:
      show_root_heading: true

## Loss

::: leech.crf.loss.CtcCrfLoss
    options:
      show_root_heading: true
      show_source: true

::: leech.crf.loss.predecessor_index
    options:
      show_root_heading: true

## Decode

::: leech.crf.decode.decode_batch
    options:
      show_root_heading: true
      show_source: true

::: leech.crf.decode.best_path
    options:
      show_root_heading: true

## Configuration

::: leech.crf.config.load_config
    options:
      show_root_heading: true

The packaged default is `leech/crf/configs/crf_ctc.toml`. It travels with the
package rather than beside a corpus: an architecture config kept only in scratch
means a purge leaves trained weights nobody can load.

## Acceleration

Two optional fast paths, both gated and both falling back to the PyTorch
reference implementation — which stays the correctness oracle.

| Switch | Effect |
|---|---|
| `LEECH_COMPILE=1` | `torch.compile` the CRF tail and the forward-backward scans. CUDA only; the CPU path stays eager because inductor's CPU `tanh` is not bit-exact. |
| `LEECH_NO_TRITON=1` | Disable the Triton lattice kernels and use the PyTorch scans. |
| `LEECH_NO_COMPILE=1` | Disable compilation of the reference scans in `loss.py`. |

Each also answers to the `ESCAPEPOD_` prefix, which is what escapepod-models'
equivalence checks set.
