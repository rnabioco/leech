# Training Module

Model training orchestration and utilities.

## Overview

The training module provides the Trainer class for training leech models.

## Trainer Class

::: leech.training.Trainer
    options:
      show_root_heading: true
      show_source: true

## Training Functions

::: leech.training.train_model
    options:
      show_root_heading: true
      show_source: true

## Example Usage

```python title="Python" linenums="1"
from leech.training import Trainer
from leech.models import get_model
from leech.dataset import LeechDataset
from torch.utils.data import DataLoader

# Prepare data
train_dataset = LeechDataset("train_chunks.json")
val_dataset = LeechDataset("val_chunks.json")

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# Get model
model = get_model("ConvLSTMDwell")

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=0.001,
    weight_decay=0.0001,
    device="cuda",
    output_dir="models/"
)

# Train
history = trainer.train(
    epochs=50,
    early_stopping_patience=5
)

# Access training history
print(f"Best validation loss: {min(history['val_loss'])}")
```

## Multi-GPU training

`leech model train --gpus N` runs N data-parallel ranks on one node. It is
opt-in and single-node; `--gpus 1` (the default) is the single-device path
unchanged.

```bash title="Shell"
# 4 A30s of one node; --gres=gpu:4 and ~4x the memory of a single-GPU run
uv run leech model train --train-data chunks/train.npz --val-data chunks/val.npz \
  --model TCNDwellResidualLN --output-dir models/ \
  --batch-size 1024 --gpus 4
```

Two things to know before using it:

- **`--batch-size` is the global batch and is split across ranks.** With
  `--batch-size 1024 --gpus 4` each rank sees 256. The optimizer-step count, the
  LR schedule and the gradient-accumulation arithmetic are therefore identical
  at any `--gpus`, so a multi-GPU run stays comparable with arms already
  measured. (This is the opposite of the usual PyTorch convention, where the
  flag is per-rank and the effective batch grows with the GPU count.)
- **Memory scales with the rank count.** Each rank loads its own copy of the
  corpus, so a run whose single-GPU peak is ~40 GiB needs ~160 GiB at
  `--gpus 4`. Size the job's `--mem` accordingly; the second rank is
  OOM-killed during its load otherwise.

The checkpoint a multi-GPU run writes is byte-compatible with a single-GPU one
(same keys, no `module.` prefix), so `leech model export`, bundling and
inference are unaffected.

## Label-noise-aware loss

`--loss noise_corrected_bce` is for labels that are enrichments with a known,
measured impurity rather than per-read ground truth -- e.g. a positive block
that is only ~90% pure because the enrichment chemistry (or an upstream
barcode/demux step) mislabels a known fraction of reads. It applies a forward
correction (Patrini et al. 2017) so the model is fit against the corrected
probability of the *observed* label rather than the raw noisy target, using a
per-sample flip rate looked up from each chunk's `source_group` via
`--label-noise-rate group=rate[,group=rate,...]` (e.g.
`--label-noise-rate gold=0.09,enzymatic=0.17`); groups not named get rate 0
(plain BCE). Reach for it when a purity estimate exists for some or all of the
training data and soft targets have already been tried and measured worse --
soft-labeling the noisy fraction is not equivalent to this correction and
underperforms it. Because the correction trades bias for variance in the
noisy regime, judge it by **callable yield at a fixed precision floor**
(e.g. AUROC-adjacent metrics can look flat or even move against it while the
fraction of confidently-called reads at 99% precision improves) rather than by
AUROC alone.

### Multiclass (`--num-out > 1`)

The same `--loss noise_corrected_bce` value trains a multiclass forward
correction at `--num-out N` for `N > 1`, for a corpus where every class's
label noise flows to one designated **sink** class -- e.g. a 20+1-class
charge-aware classifier (20 amino acids + `uncharged`) where a known fraction
of each amino acid's labeled chunks are secretly `uncharged`. Name the sink
with `--noise-sink-class uncharged` (a class label resolved against the
corpus's `label_map.json`, or a raw index); `--label-noise-rate` keys are then
class labels rather than arbitrary `source_group` values (e.g.
`--label-noise-rate Gln=0.041,Thr=0.948`), one purity per non-sink class. As
in the binary case, a class not named gets rate 0 and an all-zero
`--label-noise-rate` (or none at all) reproduces `--loss cross_entropy`
bit-for-bit.

The two losses differ in what the correction needs: the binary loss looks up
one flip rate per *sample* from its `source_group`, but the multiclass sink
structure needs the *global* per-class rate vector to correctly weigh how
much of a sink-labeled chunk's evidence should be credited to each other
class -- so rates are resolved once from `label_map`, not read per chunk.
Only samples observed as the sink actually get a different gradient from
plain cross-entropy; a non-sink-observed sample's correction is a per-class
constant that does not move the gradient at all, since the sink is the only
class more than one other class can leak into. See
`leech.losses.NoiseCorrectedCrossEntropyLoss` for the full derivation.
