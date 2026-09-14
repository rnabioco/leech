# Inference Module

Inference engine for applying trained models to new data.

## Overview

The inference module provides functions for running predictions on new POD5/BAM data.

::: leech.inference.run_inference
    options:
      show_root_heading: true
      show_source: true

## Example Usage

```python title="Python" linenums="1"
from leech.inference import run_inference
from pathlib import Path

# Run inference on BAM file
run_inference(
    model_path=Path("models/model_best.pt"),
    pod5_path=Path("reads.pod5"),
    bam_path=Path("reads.bam"),
    output_path=Path("predictions.bam"),
    batch_size=128,
    device="cuda",
    base_justify="center",  # Signal chunk centering strategy
)
```

## Output Format

The output BAM file contains the following additional tags. By default `ac`,
`am`, `pp`, and `pc` are written as compact `uint8` (0-255); pass `raw=True`
(CLI `--raw`) to write full-float values instead.

| Tag | Type | Description |
|-----|------|-------------|
| `aa` | str | Predicted class call, or `unc` if below threshold |
| `ac` | uint8 / float | Confidence (max class probability), regardless of whether the read passed the threshold |
| `am` | uint8 / float | Margin (top probability minus second-highest) |
| `pn` | str | Comma-separated class names |
| `pp` | uint8[] / float[] | Full probability distribution over classes |
| `pc` | uint8 / float | Predicted charging level (only when a CL regression head is present) |
| `ji` | int32 | CIGAR-measured indel length at the motif junction (`0` = exact). Omitted when no junction measurement was attempted (no motif, or a Remora model) |
| `jm` | uint8 | Whether the motif junction mapped to the reference (0/1). Omitted under the same conditions as `ji` |

`ji`/`jm` are written whenever a junction measurement exists, independent of
`--raw`, so the `--abstain-on-junction-indel` rule (see the
[CLI reference](../reference/cli.md#leech-predict)) can be re-applied or
audited offline without re-running inference.
