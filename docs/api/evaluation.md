# Evaluation Module

Model evaluation and testing utilities.

## Overview

The evaluation module provides functions for evaluating trained models.

::: leech.evaluation.evaluate_model
    options:
      show_root_heading: true
      show_source: true

## Example Usage

```python
from leech.evaluation import evaluate_model
from torch.utils.data import DataLoader
from pathlib import Path

# Evaluate model
metrics = evaluate_model(
    model_path=Path("models/model_best.pt"),
    test_data=[Path("chunks/test.json")],
    batch_size=128,
    device="cuda"
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
print(f"AUC-ROC: {metrics['auc']:.4f}")
```
