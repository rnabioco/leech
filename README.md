# leech

**L**earning **E**nhanc**e**d **E**lectrical **C**lassifiers from **H**anopore signals

[![CI](https://github.com/rnabioco/leech/actions/workflows/ci.yml/badge.svg)](https://github.com/rnabioco/leech/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Leech classifies aminoacylation state and amino acid identity from Oxford
Nanopore tRNA sequencing data. It extracts **dwell time features** from move
tables (the BAM `mv` tag) and feeds them alongside raw signal and sequence
context into a multi-branch neural network, giving it information that
signal-only tools like [Remora](https://github.com/nanoporetech/remora)
discard.

## Installation

Requires Python 3.12+

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/rnabioco/leech.git
cd leech
uv sync
```

## Quick start

### 1. Prepare training data

```bash
uv run leech data prepare \
  --pod5 reads.pod5 \
  --bam alignments.bam \
  --output-dir chunks/ \
  --motif CCAGGC --motif-offset 2 \
  --label 1 --workers 8
```

### 2. Train a model

```bash
uv run leech model train \
  --train-data chunks/train.json \
  --val-data chunks/val.json \
  --model ConvLSTMDwell \
  --output-dir models/
```

### 3. Evaluate

```bash
uv run leech eval test \
  --model models/model_best.pt \
  --test-data chunks/test.json \
  --output metrics.json
```

### 4. Run inference

```bash
uv run leech predict \
  --model models/ \
  --pod5 new_reads.pod5 \
  --bam new_alignments.bam \
  --output predictions.bam
```

### 5. Bundle and deploy pairwise models

Package multiple pairwise models into a single file and run aggregated
inference:

```bash
# Bundle all pairwise models
uv run leech model bundle \
  --model-dir results/models/pairwise/ \
  --output bundle.pt --version 1.0.0

# Inspect bundle contents
uv run leech model bundle-info --bundle bundle.pt

# Run all models (aggregated amino acid prediction)
uv run leech predict \
  --bundle bundle.pt --all \
  --pod5 reads.pod5 --bam alignments.bam \
  --output predictions.bam
```

## CLI overview

| Group | Commands | Purpose |
|-------|----------|---------|
| `leech data` | `prepare`, `merge` | Extract features, merge and split datasets |
| `leech model` | `train`, `optimize`, `bundle`, `bundle-info`, `calibrate`, `export` | Train, tune, calibrate, and package models |
| `leech eval` | `test`, `compare`, `importance`, `ablation` | Evaluate and analyze models |
| `leech predict` | | Run inference (single model or bundle) |

## Model architectures

| Model | Description |
|-------|-------------|
| **ConvLSTMDwell** | Conv-LSTM with dwell features (recommended) |
| ConvLSTMBase | Baseline without dwell features (for comparison) |
| ConvLSTMRemora | Remora-compatible architecture with dwell features |
| ConvLSTMRemoraBase | Remora-compatible baseline (no dwell features) |
| TransformerDwell | Transformer with multi-head self-attention |
| ConvOnly | Pure CNN with multi-scale convolutions |
| TCNDwell | Temporal Convolutional Network |
| ResNetDwell | Residual network |

Batch normalization (BN) and attention pooling variants are available for ConvLSTMBase and ConvLSTMDwell models (e.g., ConvLSTMDwellBN, ConvLSTMDwellAttn, ConvLSTMDwellBNAttn).

## Training features

- **Loss functions**: BCE, focal loss (for class imbalance), and cross-entropy
- **Regularization**: weight decay, gradient clipping, dropout
- **LR scheduling**: reduce-on-plateau, cosine annealing with warmup
- **LR warmup**: linear warmup over configurable epochs
- **Data augmentation**: mixup (signal jitter + random scaling)
- **Mixed precision**: FP16 training on CUDA
- **Class balancing**: automatic class weight computation
- **Checkpoint resume**: continue training from any checkpoint
- **Sequence encoding**: base one-hot or signal-level kmer encoding
- **Balance-groups sampling**: equal contribution per source group per epoch
- **K-fold cross-validation**: stratified read-level k-fold splits
- **Platt calibration**: post-hoc Platt scaling for probability calibration
- **TorchScript export**: standalone model export for deployment without leech

## Snakemake pipeline

For production workloads, leech includes a Snakemake pipeline supporting:
- Charged vs. uncharged classification
- Pairwise amino acid discrimination
- Grid search optimization
- Multi-architecture comparison
- HPC clusters (SLURM/LSF)

See `pipeline/` for configuration and usage.

## Development

```bash
uv sync --all-extras      # Install with dev tools
uv run pytest              # Run tests
uv run ruff check .        # Lint
uv run ruff format .       # Format
uv run ty check src/leech/ # Type check
```

## Citation

If you use leech, please cite:

- This work (publication pending)
- [Remora](https://github.com/nanoporetech/remora) (underlying training framework)

## License

MIT License - see [LICENSE](LICENSE) for details.
