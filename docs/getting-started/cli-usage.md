# CLI Usage

Complete reference for the `leech` command-line interface.

## Overview

The `leech` CLI provides four main commands:

- `prepare`: Extract features from POD5/BAM files
- `train`: Train a model on prepared data
- `test`: Evaluate a trained model
- `infer`: Run inference on new data

## Global Options

```bash
leech --help      # Show help message
leech --version   # Show version number
```

## prepare

Extract training chunks from POD5 and BAM files.

### Synopsis

```bash
leech prepare [OPTIONS] --pod5 FILE --bam FILE --output-dir DIR
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--pod5 FILE` | Path to POD5 file with raw signal |
| `--bam FILE` | Path to BAM file with alignments and move tables |
| `--output-dir DIR` | Directory to save training chunks |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--label INT` | 0 | Class label (0 or 1) |
| `--feature-set STR` | `signal+dwell+levels` | Features to extract |
| `--motif STR` | `CCAGGC` | Sequence motif to center on |
| `--motif-offset INT` | 2 | Position within motif (0-indexed) |
| `--motif-reference STR` | `fasta` | Where to search: `fasta` or `bam` |
| `--reference-fasta FILE` | - | Reference FASTA (if not in BAM header) |
| `--skip-motif-indels` | False | Skip motif sites with indels |
| `--signal-context INT` | 200 | Signal samples before/after |
| `--kmer-context INT` | 5 | K-mer bases before/after |
| `--min-mapping-quality INT` | 0 | Minimum MAPQ filter |
| `--workers INT` | 1 | Number of parallel workers |
| `--chunk-size INT` | 100 | Reads per batch (for parallel) |
| `--split-train-val-test` | `0.7,0.15,0.15` | Dataset split ratios |

### Feature Sets

Combine features with `+`:

- `signal`: Raw nanopore signal
- `dwell`: Per-base dwell times from move tables
- `levels`: Signal statistics (mean, median, std, range) per base

Examples:
- `signal+dwell+levels` (recommended)
- `signal+dwell`
- `signal` (baseline)

### Motif Search Strategies

#### Reference-based (default)

Search in reference sequence and map to query via CIGAR:

```bash
leech prepare \
  --motif CCAGGC \
  --motif-reference fasta \
  --reference-fasta genome.fa \
  --skip-motif-indels
```

**Advantages:**
- Avoids bias from basecalling errors at modification sites
- More accurate for trained models

#### Basecalled search

Search directly in basecalled sequence:

```bash
leech prepare \
  --motif CCAGGC \
  --motif-reference bam
```

**Use case:** Backward compatibility or when reference is unavailable

### Examples

#### Basic usage

```bash
leech prepare \
  --pod5 reads.pod5 \
  --bam alignments.bam \
  --output-dir chunks/ \
  --label 1
```

#### With parallel processing

```bash
leech prepare \
  --pod5 reads.pod5 \
  --bam alignments.bam \
  --output-dir chunks/ \
  --workers 8 \
  --chunk-size 100
```

#### Custom motif and features

```bash
leech prepare \
  --pod5 reads.pod5 \
  --bam alignments.bam \
  --output-dir chunks/ \
  --motif AGATCG \
  --motif-offset 3 \
  --feature-set signal+dwell \
  --signal-context 300 \
  --kmer-context 7
```

## train

Train a model on prepared training data.

### Synopsis

```bash
leech train [OPTIONS] --train-data FILES --val-data FILES --model MODEL --output-dir DIR
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--train-data FILES` | Training data JSON files (glob patterns supported) |
| `--val-data FILES` | Validation data JSON files |
| `--model MODEL` | Model architecture name |
| `--output-dir DIR` | Directory to save model checkpoints |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs INT` | 50 | Number of training epochs |
| `--batch-size INT` | 128 | Batch size |
| `--learning-rate FLOAT` | 0.001 | Learning rate |
| `--weight-decay FLOAT` | 0.0001 | L2 regularization |
| `--early-stopping-patience INT` | 5 | Stop if no improvement after N epochs |
| `--device STR` | `cuda` if available | Device: `cuda` or `cpu` |
| `--seed INT` | 42 | Random seed for reproducibility |

### Available Models

- `ConvLSTMDwell`: Conv-LSTM with dwell features (recommended)
- `ConvLSTMBase`: Baseline without dwell features
- `TransformerDwell`: Transformer with self-attention
- `ConvOnly`: Pure convolutional network
- `TCNDwell`: Temporal Convolutional Network
- `ResNetDwell`: Residual network

### Examples

#### Basic training

```bash
leech train \
  --train-data chunks/train.json \
  --val-data chunks/val.json \
  --model ConvLSTMDwell \
  --output-dir models/
```

#### With hyperparameters

```bash
leech train \
  --train-data data/*/train.json \
  --val-data data/*/val.json \
  --model ConvLSTMDwell \
  --output-dir models/ \
  --epochs 100 \
  --batch-size 256 \
  --learning-rate 0.0001 \
  --early-stopping-patience 10
```

## test

Evaluate a trained model.

### Synopsis

```bash
leech test [OPTIONS] --model FILE --test-data FILES --output FILE
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--model FILE` | Path to trained model (.pt file) |
| `--test-data FILES` | Test data JSON files |
| `--output FILE` | Output metrics JSON file |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size INT` | 128 | Batch size for evaluation |
| `--device STR` | `cuda` if available | Device: `cuda` or `cpu` |

### Examples

```bash
leech test \
  --model models/model_best.pt \
  --test-data chunks/test.json \
  --output metrics.json
```

## infer

Run inference on new data.

### Synopsis

```bash
leech infer [OPTIONS] --model FILE --pod5 FILE --bam FILE --output FILE
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--model FILE` | Path to trained model (.pt file) |
| `--pod5 FILE` | POD5 file with signal data |
| `--bam FILE` | BAM file with alignments |
| `--output FILE` | Output BAM file with predictions |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size INT` | 128 | Batch size for inference |
| `--device STR` | `cuda` if available | Device: `cuda` or `cpu` |
| `--motif STR` | Model default | Override motif |
| `--motif-offset INT` | Model default | Override motif offset |

### Examples

```bash
leech infer \
  --model models/model_best.pt \
  --pod5 new_reads.pod5 \
  --bam new_alignments.bam \
  --output predictions.bam
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LEECH_DEVICE` | Default device (cuda/cpu) | Auto-detect |
| `LEECH_WORKERS` | Default number of workers | 1 |
| `CUDA_VISIBLE_DEVICES` | Restrict GPU usage | All |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Data format error |

## Next Steps

- [API Reference](../api/index.md): Python API documentation
- [Grid Search](../grid-search/grid-search-usage.md): Hyperparameter optimization
