# Inference Module

Inference engine for applying trained models to new data.

## Overview

The inference module provides functions for running predictions on new POD5/BAM data.

::: leech.inference.run_inference
    options:
      show_root_heading: true
      show_source: true

::: leech.inference.load_predictions_from_bam
    options:
      show_root_heading: true
      show_source: true

## Example Usage

```python
from leech.inference import run_inference
from pathlib import Path

# Run inference on BAM file
run_inference(
    model_path=Path("models/model_best.pt"),
    pod5_path=Path("reads.pod5"),
    bam_path=Path("reads.bam"),
    output_bam=Path("predictions.bam"),
    batch_size=128,
    device="cuda"
)
```

## Output Format

The output BAM file contains the following additional tags:

| Tag | Type | Description |
|-----|------|-------------|
| `MP` | float | Modification probability (0-1) |
| `ML` | int | Model prediction label (0 or 1) |
| `MQ` | float | Prediction confidence score |
