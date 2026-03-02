# CLI Usage

Complete reference for the `leech` command-line interface.

## Overview

The `leech` CLI is organized into workflow-based command groups that mirror the machine learning pipeline:

- **`leech data`** - Prepare and process training data
  - `prepare`: Extract features from POD5/BAM files
  - `merge`: Merge multi-sample data and split at read level
- **`leech model`** - Train and optimize models
  - `train`: Train a model on prepared data
  - `optimize`: Optimize hyperparameters via grid search
- **`leech eval`** - Evaluate and analyze models
  - `test`: Evaluate a trained model on test data
  - `compare`: Compare multiple models
  - `importance`: Analyze feature importance
  - `ablation`: Test sequence ablation
- **`leech predict`** - Run inference on new data

## Global Options

```bash title="Bash" linenums="1"
leech --help      # Show help message
leech --version   # Show version number
```

## Data Preparation Commands

### leech data prepare

Extract training chunks from POD5 and BAM files.

#### Synopsis

```bash title="Bash" linenums="1"
leech data prepare [OPTIONS] --pod5 FILE --bam FILE --output-dir DIR
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--pod5 FILE` | Path to POD5 file with raw signal |
| `--bam FILE` | Path to BAM file with alignments and move tables |
| `--output-dir DIR` | Directory to save training chunks |

#### Optional Arguments

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
| `--min-mapq INT` | 10 | Minimum MAPQ filter |
| `--workers INT` | 8 | Number of parallel workers |
| `--chunk-size INT` | 100 | Reads per batch (for parallel) |
| `--train-split FLOAT` | 0.7 | Fraction for training |
| `--val-split FLOAT` | 0.15 | Fraction for validation |
| `--base-justify STR` | `center` | Signal chunk centering: `start`, `center`, or `end` |
| `--dwell-margin INT` | 0 | Extra bases to include for runtime dwell_offset tuning |
| `--no-split` | False | Skip splitting (for later merge) |
| `--seed INT` | 42 | Random seed |

#### Feature Sets

Combine features with `+`:

- `signal`: Raw nanopore signal
- `dwell`: Per-base dwell times from move tables
- `levels`: Signal statistics (mean, median, std, range) per base

Examples:
- `signal+dwell+levels` (recommended)
- `signal+dwell`
- `signal` (baseline)

#### Motif Search Strategies

**Reference-based (default)** - Search in reference sequence and map to query via CIGAR:

```bash title="Bash" linenums="1"
leech data prepare \
  --motif CCAGGC \
  --motif-reference fasta \
  --reference-fasta genome.fa \
  --skip-motif-indels
```

**Advantages:**
- Avoids bias from basecalling errors at modification sites
- More accurate for trained models

**Basecalled search** - Search directly in basecalled sequence:

```bash title="Bash" linenums="1"
leech data prepare \
  --motif CCAGGC \
  --motif-reference bam
```

**Use case:** Backward compatibility or when reference is unavailable

#### Examples

Basic usage:
```bash title="Bash" linenums="1"
leech data prepare \
  --pod5 reads.pod5 \
  --bam alignments.bam \
  --output-dir chunks/ \
  --label 1
```

With parallel processing:
```bash title="Bash" linenums="1"
leech data prepare \
  --pod5 reads.pod5 \
  --bam alignments.bam \
  --output-dir chunks/ \
  --workers 8 \
  --chunk-size 100
```

Custom motif and features:
```bash title="Bash" linenums="1"
leech data prepare \
  --pod5 reads.pod5 \
  --bam alignments.bam \
  --output-dir chunks/ \
  --motif AGATCG \
  --motif-offset 3 \
  --feature-set signal+dwell \
  --signal-context 300 \
  --kmer-context 7
```

### leech data merge

Merge multiple chunk files from different samples and split at the read level to prevent data leakage. This is the correct workflow for multi-sample datasets.

#### Synopsis

```bash title="Bash" linenums="1"
leech data merge [OPTIONS] -i LABEL=FILE -i LABEL=FILE -o DIR
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `-i, --input-chunks` | Input chunks with labels (format: `label=file.npz`) |
| `-o, --output-dir` | Output directory for split chunks |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-split` | 0.7 | Fraction of reads for training |
| `--val-split` | 0.15 | Fraction of reads for validation |
| `--seed` | 42 | Random seed for reproducibility |
| `--comparison-spec` | - | TSV file with comparison specifications |

#### Examples

Pairwise amino acid comparison:
```bash title="Bash" linenums="1"
leech data merge \
  -i Ala=ala.npz \
  -i Gly=gly.npz \
  -o merged/
```

Multi-label comparison (chemical properties):
```bash title="Bash" linenums="1"
leech data merge \
  -i basic=lys.npz \
  -i basic=arg.npz \
  -i acidic=asp.npz \
  -i acidic=glu.npz \
  -o merged/
```

Batch processing with comparison spec:
```bash title="Bash" linenums="1"
leech data merge \
  -i chunks/dir1 \
  -i chunks/dir2 \
  --comparison-spec comparisons.tsv \
  -o merged/
```

## Model Training Commands

### leech model train

Train a model on prepared training data.

#### Synopsis

```bash title="Bash" linenums="1"
leech model train [OPTIONS] --train-data FILES --val-data FILES --model MODEL --output-dir DIR
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--train-data FILES` | Training data JSON files (glob patterns supported) |
| `--val-data FILES` | Validation data JSON files |
| `--model MODEL` | Model architecture name |
| `--output-dir DIR` | Directory to save model checkpoints |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs INT` | 50 | Number of training epochs |
| `--batch-size INT` | 128 | Batch size |
| `--learning-rate FLOAT` | 0.001 | Learning rate |
| `--device STR` | `cuda` if available | Device: `cuda` or `cpu` |
| `--seed INT` | 42 | Random seed for reproducibility |
| `--early-stopping INT` | 10 | Stop if no improvement after N epochs (0 to disable) |
| `--use-class-weights` | True | Auto-compute class weights for imbalance |
| `--pos-weight FLOAT` | - | Manual positive class weight (overrides auto) |

#### Available Models

- `ConvLSTMDwell`: Conv-LSTM with dwell features (recommended)
- `ConvLSTMBase`: Baseline without dwell features
- `TransformerDwell`: Transformer with self-attention
- `ConvOnly`: Pure convolutional network
- `TCNDwell`: Temporal Convolutional Network
- `ResNetDwell`: Residual network

#### Examples

Basic training:
```bash title="Bash" linenums="1"
leech model train \
  --train-data chunks/train.json \
  --val-data chunks/val.json \
  --model ConvLSTMDwell \
  --output-dir models/
```

With hyperparameters:
```bash title="Bash" linenums="1"
leech model train \
  --train-data data/*/train.json \
  --val-data data/*/val.json \
  --model ConvLSTMDwell \
  --output-dir models/ \
  --epochs 100 \
  --batch-size 256 \
  --learning-rate 0.0001 \
  --early-stopping 10
```

### leech model optimize

Run grid search over chunk context parameters to optimize model performance.

#### Synopsis

```bash title="Bash" linenums="1"
leech model optimize [OPTIONS] --train-data FILE --output-dir DIR --context-grid VALUES
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--train-data FILE` | Training dataset (.npz file) |
| `-o, --output-dir DIR` | Output directory for grid results |
| `--context-grid VALUES` | Comma-separated context values (e.g., `200,500,1000`) |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--val-data FILE` | - | Validation dataset (.npz) |
| `--model MODEL` | `ConvLSTMDwell` | Model architecture |
| `--left-contexts VALUES` | Uses `--context-grid` | Override left contexts |
| `--right-contexts VALUES` | Uses `--context-grid` | Override right contexts |
| `--kmer-context INT` | 5 | K-mer context for sequence |
| `--epochs INT` | 50 | Number of training epochs |
| `--batch-size INT` | 128 | Batch size |
| `--learning-rate FLOAT` | 0.001 | Learning rate |
| `--device STR` | `cuda` if available | Device: `cuda` or `cpu` |
| `--parallel INT` | 1 | Number of grid points to train concurrently |
| `--base-justify STR` | `center` | Signal chunk centering: `start`, `center`, or `end` |
| `--dwell-offsets VALUES` | - | Dwell offset values to search (comma-separated or `start:stop:step`) |
| `--seed INT` | 42 | Random seed |
| `--early-stopping INT` | 10 | Early stopping patience (0 to disable) |

#### Examples

Basic grid search:
```bash title="Bash" linenums="1"
leech model optimize \
  --train-data chunks/train.npz \
  --val-data chunks/val.npz \
  --output-dir grid_results/ \
  --context-grid 200,500,1000,2000
```

Asymmetric context search:
```bash title="Bash" linenums="1"
leech model optimize \
  --train-data chunks/train.npz \
  --output-dir grid_results/ \
  --left-contexts 200,500,1000 \
  --right-contexts 100,200,500
```

## Model Evaluation Commands

### leech eval test

Evaluate a trained model on a holdout test set.

#### Synopsis

```bash title="Bash" linenums="1"
leech eval test [OPTIONS] --model FILE --test-data FILES --output FILE
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--model FILE` | Path to trained model (.pt file) |
| `--test-data FILES` | Test data JSON files |
| `--output FILE` | Output metrics JSON file |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--device STR` | `cuda` if available | Device: `cuda` or `cpu` |

#### Examples

```bash title="Bash" linenums="1"
leech eval test \
  --model models/model_best.pt \
  --test-data chunks/test.json \
  --output metrics.json
```

### leech eval compare

Compare multiple trained models on the same test set.

#### Synopsis

```bash title="Bash" linenums="1"
leech eval compare [OPTIONS] -m DIR -m DIR -t FILE -o DIR
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `-m, --model-dirs` | Model directories to compare (specify multiple) |
| `-t, --test-data` | Test dataset for evaluation |
| `-o, --output-dir` | Output directory for comparison results |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--device STR` | `cuda` if available | Device: `cuda` or `cpu` |
| `--no-plot` | False | Skip generating plots |

#### Examples

```bash title="Bash" linenums="1"
leech eval compare \
  -m models/model1/ \
  -m models/model2/ \
  -m models/model3/ \
  -t chunks/test.npz \
  -o comparison/
```

### leech eval importance

Compute feature importance scores for a trained model.

#### Synopsis

```bash title="Bash" linenums="1"
leech eval importance [OPTIONS] -m FILE -t FILE -o DIR
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `-m, --model` | Path to trained model checkpoint |
| `-t, --test-data` | Test dataset for analysis |
| `-o, --output-dir` | Output directory for results |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--device STR` | `cuda` if available | Device: `cuda` or `cpu` |
| `--method STR` | `gradient` | Method: `gradient` or `integrated_gradients` |
| `--no-plot` | False | Skip generating plots |

#### Examples

```bash title="Bash" linenums="1"
leech eval importance \
  -m models/model_best.pt \
  -t chunks/test.npz \
  -o importance/ \
  --method gradient
```

### leech eval ablation

Test model performance with sequence ablation.

#### Synopsis

```bash title="Bash" linenums="1"
leech eval ablation [OPTIONS] -m FILE -t FILE -o DIR
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `-m, --model` | Path to trained model checkpoint |
| `-t, --test-data` | Test dataset for analysis |
| `-o, --output-dir` | Output directory for results |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--device STR` | `cuda` if available | Device: `cuda` or `cpu` |
| `--no-plot` | False | Skip generating plots |

#### Examples

```bash title="Bash" linenums="1"
leech eval ablation \
  -m models/model_best.pt \
  -t chunks/test.npz \
  -o ablation/
```

## Inference Command

### leech predict

Run inference on new data to generate predictions.

#### Synopsis

```bash title="Bash" linenums="1"
leech predict [OPTIONS] --model FILE --pod5 FILE --bam FILE --output FILE
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--model FILE` | Path to trained model (.pt file) |
| `--pod5 FILE` | POD5 file with signal data |
| `--bam FILE` | BAM file with alignments |
| `--output FILE` | Output BAM file with predictions |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--device STR` | `cuda` if available | Device: `cuda` or `cpu` |
| `--base-justify STR` | `center` | Signal chunk centering: `start`, `center`, or `end` |

#### Examples

```bash title="Bash" linenums="1"
leech predict \
  --model models/model_best.pt \
  --pod5 new_reads.pod5 \
  --bam new_alignments.bam \
  --output predictions.bam
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LEECH_DEVICE` | Default device (cuda/cpu) | Auto-detect |
| `LEECH_WORKERS` | Default number of workers | 8 |
| `CUDA_VISIBLE_DEVICES` | Restrict GPU usage | All |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Data format error |

## Typical Workflows

### Single-sample workflow

```bash title="Bash" linenums="1"
# 1. Prepare data
leech data prepare \
  --pod5 reads.pod5 \
  --bam alignments.bam \
  --output-dir chunks/

# 2. Train model
leech model train \
  --train-data chunks/train.json \
  --val-data chunks/val.json \
  --model ConvLSTMDwell \
  --output-dir models/

# 3. Evaluate
leech eval test \
  --model models/model_best.pt \
  --test-data chunks/test.json \
  --output metrics.json

# 4. Predict
leech predict \
  --model models/model_best.pt \
  --pod5 new_reads.pod5 \
  --bam new_alignments.bam \
  --output predictions.bam
```

### Multi-sample comparison workflow

```bash title="Bash" linenums="1"
# 1. Prepare each sample (no splitting)
leech data prepare --pod5 sample1.pod5 --bam sample1.bam --output-dir chunks/sample1/ --no-split --label 0
leech data prepare --pod5 sample2.pod5 --bam sample2.bam --output-dir chunks/sample2/ --no-split --label 1

# 2. Merge and split at read level
leech data merge \
  -i label0=chunks/sample1/all.npz \
  -i label1=chunks/sample2/all.npz \
  -o merged/

# 3. Train model
leech model train \
  --train-data merged/train.json \
  --val-data merged/val.json \
  --model ConvLSTMDwell \
  --output-dir models/

# 4. Evaluate
leech eval test \
  --model models/model_best.pt \
  --test-data merged/test.json \
  --output metrics.json
```

### Hyperparameter optimization workflow

```bash title="Bash" linenums="1"
# 1. Prepare data
leech data prepare --pod5 reads.pod5 --bam alignments.bam --output-dir chunks/

# 2. Optimize hyperparameters
leech model optimize \
  --train-data chunks/train.npz \
  --val-data chunks/val.npz \
  --context-grid 200,500,1000,2000 \
  --output-dir grid_results/

# 3. Train with best parameters (from grid_results/best_params.json)
leech model train \
  --train-data chunks/train.json \
  --val-data chunks/val.json \
  --model ConvLSTMDwell \
  --output-dir models/ \
  --model-config grid_results/best_params.json
```

## See Also

- [Quick Start](../getting-started/quick-start.md): Get started quickly
- [API Reference](../api/index.md): Python API documentation
- [Grid Search Guide](../grid-search/grid-search-usage.md): Hyperparameter optimization details
