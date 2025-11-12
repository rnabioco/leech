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

```python
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
