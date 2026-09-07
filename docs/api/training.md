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
