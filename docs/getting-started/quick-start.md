# Quick Start

This guide will walk you through using leech for the first time.

## Prerequisites

- [Install leech](installation.md)
- POD5 file with nanopore signal data
- BAM file with basecalls and move tables (from dorado/guppy with `--emit-moves`)

## Workflow Overview

The typical leech workflow consists of four steps:

```mermaid
graph LR
    A[Prepare Data] --> B[Train Model]
    B --> C[Test Model]
    C --> D[Run Inference]
```

## Step 1: Prepare Training Data

Extract features from your POD5 and BAM files:

```bash
uv run leech prepare \
  --pod5 reads.pod5 \
  --bam alignments.bam \
  --output-dir chunks/ \
  --feature-set signal+dwell+levels \
  --motif CCAGGC \
  --motif-offset 2 \
  --label 1
```

### Key Parameters

- `--pod5`: Path to POD5 file with raw signal
- `--bam`: Path to BAM file with alignments and move tables
- `--output-dir`: Directory to save training chunks
- `--feature-set`: Features to extract (`signal`, `dwell`, `levels`, or combinations)
- `--motif`: Sequence motif to center on (e.g., "CCAGGC" for tRNA 3' end)
- `--motif-offset`: Position within motif to focus on (0-indexed)
- `--label`: Class label (0 or 1)

### Parallel Processing

For large datasets, use parallel processing:

```bash
uv run leech prepare \
  --pod5 reads.pod5 \
  --bam alignments.bam \
  --output-dir chunks/ \
  --workers 8 \
  --chunk-size 100
```

This will process reads in parallel across 8 CPU cores.

## Step 2: Train a Model

Train a model on your prepared data:

```bash
uv run leech train \
  --train-data chunks/train.json \
  --val-data chunks/val.json \
  --model ConvLSTMDwell \
  --output-dir models/ \
  --epochs 50 \
  --batch-size 128 \
  --learning-rate 0.001
```

### Available Models

- `ConvLSTMDwell`: Multi-branch Conv-LSTM with dwell features (recommended)
- `ConvLSTMBase`: Baseline without dwell features
- `TransformerDwell`: Transformer with self-attention
- `ConvOnly`: Pure convolutional network
- `TCNDwell`: Temporal Convolutional Network
- `ResNetDwell`: Residual network

### Training Output

The training process will save:

- `model_best.pt`: Best model checkpoint (by validation loss)
- `model_last.pt`: Latest checkpoint
- `metrics.json`: Training metrics over time
- `summary.json`: Training summary statistics

## Step 3: Test the Model

Evaluate your trained model:

```bash
uv run leech test \
  --model models/model_best.pt \
  --test-data chunks/test.json \
  --output metrics.json
```

This will output:

- Accuracy, precision, recall, F1 score
- Confusion matrix
- ROC-AUC score
- Per-class metrics

### Example Output

```json
{
  "accuracy": 0.96,
  "precision": 0.95,
  "recall": 0.97,
  "f1": 0.96,
  "auc": 0.98,
  "confusion_matrix": [[1850, 50], [30, 1970]]
}
```

## Step 4: Run Inference

Apply your model to new data:

```bash
uv run leech infer \
  --model models/model_best.pt \
  --pod5 new_reads.pod5 \
  --bam new_alignments.bam \
  --output predictions.bam
```

The output BAM file will contain modification probability tags for each base.

## Complete Example

Here's a complete example workflow:

```bash
# 1. Prepare charged tRNA data
uv run leech prepare \
  --pod5 charged_ala.pod5 \
  --bam charged_ala.bam \
  --output-dir data/charged \
  --label 1 \
  --workers 8

# 2. Prepare uncharged tRNA data
uv run leech prepare \
  --pod5 uncharged_ala.pod5 \
  --bam uncharged_ala.bam \
  --output-dir data/uncharged \
  --label 0 \
  --workers 8

# 3. Train model
uv run leech train \
  --train-data data/*/train.json \
  --val-data data/*/val.json \
  --model ConvLSTMDwell \
  --output-dir models/ \
  --epochs 50

# 4. Test model
uv run leech test \
  --model models/model_best.pt \
  --test-data data/*/test.json \
  --output results/metrics.json

# 5. Run inference on new data
uv run leech infer \
  --model models/model_best.pt \
  --pod5 new_sample.pod5 \
  --bam new_sample.bam \
  --output predictions.bam
```

## Next Steps

- [CLI Usage](cli-usage.md): Detailed documentation of all commands
- [Implementation Guide](../guides/01-START_HERE_IMPLEMENTATION_GUIDE.md): In-depth guide for research use
- [Grid Search](../grid-search/grid-search-usage.md): Hyperparameter optimization
