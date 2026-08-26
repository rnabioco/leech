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

Importing the subpackage costs **nothing but torch and numpy**, and only when a
symbol that needs them is touched — never pysam or escapepod. That is
deliberate: escapepod-models installs leech into a conda-forge environment with
`--no-deps` and needs the CRF path alone. (`load_manifest` pulls polars, but
only when you actually read a manifest.)

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

## The manifest seam

`leech.crf` cuts a corpus from a **manifest**: one row per read, naming what
leech needs and nothing about where it came from.

| column | required | meaning |
|---|---|---|
| `read_id` | yes | the read, as POD5 and BAM both name it |
| `pod5` | yes | file or directory holding that read's signal |
| `anchor_end` | yes | signal index the window ends at (exclusive) |
| `target` | yes | the **resolved** CRF target sequence |
| `label` | no | class name, for evaluation and reporting |
| `group` | no | reporting/balancing bucket (defaults to `label`) |
| `batch` | no | acquisition batch, for leave-one-batch-out holdout |
| `quality_score` / `quality_margin` | no | label quality, gated at *training* time |
| `split` | no | `train`/`test`, when the producer carved one |

```python
from leech.crf import load_manifest, check_geometry

man = load_manifest("manifest.parquet", require=("batch",))
check_geometry(window=3000, target_len=48, samples_per_base=56.0)
print(len(man), man.batches(), man.quality_coverage())
```

Everything above the manifest is vocabulary — panels, codes, oligos, gates —
and belongs to whatever project defines those. Everything below is signal ML.
There is deliberately **no `keep` boolean**: quality travels as numbers so the
gate stays sweepable without re-cutting the corpus.

## Building a corpus

`plan_corpus` decides *which reads and in which split*, touching no POD5;
`build_corpus` then extracts their signal, streaming it to a memory-mappable
`<out>_X.npy` beside a `<out>_meta.npz`. The split is deliberately separate: it
is where every subtle rule lives, and it is testable without a gigabyte of
fixture.

```python
from leech.crf import load_manifest, plan_corpus, build_corpus, load_corpus

plan = plan_corpus(load_manifest("manifest.parquet"), chunk=3000, per_group="auto")
print(len(plan), plan.cap, plan.counts_by_split())
build_corpus(plan, "corpus")
signal, targets, groups, read_ids, split = load_corpus("corpus")   # signal is mmap'd
```

Four rules it enforces, each of which fails silently when broken:

- **A cap only caps if every class can reach it.** `per_group="auto"` uses the
  rarest class's *trainable* depth (the test fraction is reserved first), which
  is the only value that actually balances. A larger explicit cap warns and
  de-balances the corpus it was meant to balance.
- **The split is carved before capping, ranked per class and globally across
  batches.** Ranking per `(batch, class)` multiplies the cap by the number of
  batches whenever classes are crossed with batch.
- **Batches are interleaved, not concatenated.** Otherwise the whole test set
  comes from whichever batch sorts first and the headline number measures batch.
- **Shard after planning.** The plan is deterministic in `(manifest, seed)`, so
  each shard keeps its share of one global split; filtering first would give
  each shard its own test set drawn from one batch.

## Training

```python
from leech.crf import CrfTrainer, CrfTrainConfig

result = CrfTrainer(
    "corpus",
    config=CrfTrainConfig(epochs=32, batch_size=256, lr=2e-3, seed=0),
    output_dir="run/",
).train()
```

Writes `run/model.pt` and `run/model.json`. **The sidecar is not optional**: the
standardisation constants live in neither the architecture config nor the
checkpoint, so a consumer holding only weights cannot reproduce them and decodes
silently worse.

This is a separate trainer from `leech.training.Trainer`, which is
classification-locked through `pos_weight`, `num_out`, BCE/focal/CE and
AUROC/F1 checkpointing. Five things it does that are easy to get wrong:

- **The loss runs in fp32, outside autocast.** The lattice scan accumulates over
  `chunk // stride` timesteps and fp16 loses the tail. The encoder still gets
  autocast — that is where the matmuls are.
- **Standardisation is streamed** over the corpus in float64, so a corpus larger
  than RAM costs nothing to summarise.
- **The quality gate is applied here, not at extraction**, which is what keeps
  it sweepable. Partial score coverage is refused rather than silently training
  on a non-random subset.
- **The last epoch is not automatically shipped.** `select_checkpoint` falls
  back to the best epoch when the last is worse by more than `select_tol`
  (default 25%) — a divergence detector, not a ranking, because training loss
  does not rank models at this scale.
- **Per-epoch stats separate failure modes**: the worst single batch, the
  largest pre-clip gradient norm, discarded `GradScaler` steps, and non-finite
  gradient counts. An epoch mean alone cannot tell one blown batch from a
  thousand mediocre ones.

## Exporting for a native runtime

`export_crf_onnx` writes `crf_encoder.onnx` plus a `metadata.json` contract.
The **encoder only** — the decode is not expressible in standard ONNX ops, which
is why `escapepod-demux` owns it.

```python
from leech.crf.export import export_crf_onnx

export_crf_onnx("run/", "export/", sidecar="run/", references={"code01": target})
```

```
input   signal  [batch, 1, chunk]                 float32, BATCH-major
output  scores  [chunk // stride, batch, n_score] float32, TIME-major
```

Time-major output is the trap: the boundary CNN in the same stack is batch-major
`[B, 2, L]`, so a consumer reusing that assumption silently transposes rather
than failing, and needs its own load-time shape probe.

The sidecar is not decoration. **Standardisation is in neither the architecture
config nor the checkpoint** — the trainer derives it from the corpus — so a
consumer holding only weights cannot standardise and decodes silently worse.
Passing `references=` writes what the model *emits* (`target[state_len:]`),
computed once from the `state_len` the encoder declares, so no caller can supply
full-length targets by hand and inflate every edit distance.

Requires the `onnx` extra: `uv sync --extra onnx`.

## Evaluating

`leech.crf.evaluate` is the generic half of scoring: decode a corpus, match each
decode to its nearest reference, report per group. What a *panel* is — which
classes exist, which share a flowcell — stays with whatever defines the panel.

```python
from leech.crf import (decode_corpus, emitted_references, call_references,
                       balanced_recall)

refs = emitted_references(targets, state_len=4)      # target[state_len:]
decodes = decode_corpus(model, signal, test_idx, mean=..., std=..., chunk=3000)
calls = call_references(decodes, refs, candidates=classes_in_this_group)
report = balanced_recall(truth, calls, groups)
```

Three rules it holds:

- **Match against what the model emits.** Scoring against full-length targets
  forces `state_len` leading deletions into every alignment — inflating every
  distance *and compressing the margin*, since an aligner places those deletions
  where they help most, discounting wrong references more than the right one.
- **Report per group, never one pooled table.** When classes are crossed with
  batch, a pooled table measures batch. The grouping is an argument because only
  the caller knows whether their classes are confounded; the refusal to pool is
  here, and an empty grouping raises rather than reporting `null`.
- **Balanced, not raw, recall.** A pooled accuracy over unbalanced classes is
  dominated by the deepest class.

`lev_vs_refs` scores one decode against the whole reference set at once — the
shape of every evaluation loop. It recovers the serial insertion term exactly as
`j + cummin(tmp[k] - k)`, and that identity is asserted against the scalar
implementation rather than assumed. edlib is used when importable; the pure
Python fallback stays named so the two can be compared on machines that have
both.

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

## Manifest

::: leech.crf.manifest.load_manifest
    options:
      show_root_heading: true

::: leech.crf.manifest.CrfManifest
    options:
      show_root_heading: true

::: leech.crf.manifest.check_geometry
    options:
      show_root_heading: true

::: leech.crf.manifest.emitted_target
    options:
      show_root_heading: true

## Corpus

::: leech.crf.corpus.plan_corpus
    options:
      show_root_heading: true

::: leech.crf.corpus.CorpusPlan
    options:
      show_root_heading: true

::: leech.crf.corpus.build_corpus
    options:
      show_root_heading: true

::: leech.crf.corpus.load_corpus
    options:
      show_root_heading: true

::: leech.crf.corpus.load_corpus_meta
    options:
      show_root_heading: true

## Training API

::: leech.crf.training.CrfTrainer
    options:
      show_root_heading: true

::: leech.crf.training.CrfTrainConfig
    options:
      show_root_heading: true

::: leech.crf.training.compute_standardisation
    options:
      show_root_heading: true

::: leech.crf.training.apply_quality_gate
    options:
      show_root_heading: true

::: leech.crf.training.resolve_split
    options:
      show_root_heading: true

::: leech.crf.training.select_checkpoint
    options:
      show_root_heading: true

## Export API

::: leech.crf.export.export_crf_onnx
    options:
      show_root_heading: true

::: leech.crf.export.load_training_sidecar
    options:
      show_root_heading: true

## Evaluation API

::: leech.crf.evaluate.decode_corpus
    options:
      show_root_heading: true

::: leech.crf.evaluate.emitted_references
    options:
      show_root_heading: true

::: leech.crf.evaluate.call_references
    options:
      show_root_heading: true

::: leech.crf.evaluate.balanced_recall
    options:
      show_root_heading: true

::: leech.crf.evaluate.lev_vs_refs
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
