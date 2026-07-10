# Utilities Module

Helper functions and utilities for leech.

## Overview

The util module provides various helper functions for model loading, metrics computation, and more.

## Model Loading

::: leech.model_loading.load_model_from_checkpoint
    options:
      show_root_heading: true
      show_source: true

## Metrics Computation

::: leech.metrics.compute_metrics
    options:
      show_root_heading: true
      show_source: true

::: leech.metrics.save_metrics
    options:
      show_root_heading: true
      show_source: true

::: leech.metrics.print_metrics
    options:
      show_root_heading: true
      show_source: true

## Reproducibility

::: leech.model_loading.setup_random_seed
    options:
      show_root_heading: true
      show_source: true

## Example Usage

```python title="Python" linenums="1"
from leech.model_loading import load_model_from_checkpoint, setup_random_seed
from pathlib import Path

# Set random seed for reproducibility
seed = setup_random_seed(42, output_dir=Path("models/"))

# Load model
model = load_model_from_checkpoint(
    checkpoint_path=Path("models/model_best.pt"),
    device="cuda"
)

# Model is now ready for inference
predictions = model(input_data)
```
