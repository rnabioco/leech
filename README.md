# leech

> [!WARNING]
> **leech is alpha quality and under active development.** APIs, CLI flags, and
> output formats may change without notice, and bugs are expected. Validate
> results before relying on it for anything important.

**L**earning **E**nhanc**e**d **E**lectrical **C**lassifiers from **H**anopore signals

[![PyPI](https://img.shields.io/pypi/v/leech.svg)](https://pypi.org/project/leech/)
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
uv add "leech[rust]"     # or: pip install "leech[rust]"
```

The `rust` extra pulls `leech-core`, the compiled accelerator for data
preparation and inference. `leech` runs without it — every accelerated path has
a pure-Python fallback — so plain `uv add leech` is fine if no wheel matches
your platform (wheels are built for manylinux x86_64 and aarch64).

To work on leech itself:

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

20 architectures across 5 families, all supporting multi-channel signal input (`signal_in_channels`):

| Family | Models | Description |
|--------|--------|-------------|
| **ConvLSTM** | ConvLSTMDwell (recommended), ConvLSTMBase | Conv-LSTM with 3 branches (signal, sequence, dwell/level features) |
| **ConvLSTM variants** | +BN, +Attn, +BNAttn, +GNAttn, +LNAttn | Batch/group/layer normalization and attention pooling |
| **Remora-compat** | ConvLSTMRemora, ConvLSTMRemoraBase | Remora-compatible architecture for direct comparison |
| **Transformer** | TransformerDwell, TransformerDwellResidual | Multi-head self-attention; Residual variant uses 2-channel signal (raw + kmer residual) |
| **TCN** | TCNDwell, +GN, +LN, +Residual | Temporal Convolutional Network with dilated convolutions |
| **Other** | ResNetDwell, ConvOnly | Residual network; pure CNN with multi-scale convolutions |

## Training features

- **Loss functions**: BCE, focal loss (for class imbalance), and cross-entropy
- **Regularization**: weight decay, gradient clipping, dropout
- **LR scheduling**: reduce-on-plateau, cosine annealing with warmup
- **Data augmentation**: mixup (signal jitter + random scaling)
- **Mixed precision**: FP16 training on CUDA; TF32 matmul on Ampere+
- **Performance**: `torch.compile` support, Rust-accelerated signal statistics (217x)
- **Class balancing**: automatic class weight computation
- **Balance-groups sampling**: equal contribution per source group per epoch
- **K-fold cross-validation**: stratified read-level k-fold splits
- **Platt calibration**: post-hoc Platt scaling for probability calibration
- **Signal map refinement**: Viterbi-based kmer level table refinement (matches Remora)
- **Kmer residual features**: expected level, signed/unsigned deviation from kmer table
- **Multi-channel signal**: 2-channel input (raw + kmer residual) for Residual model variants
- **Aggregation**: naive, confidence-weighted, and tournament pairwise aggregation
- **Composable config**: dataclass-based configuration shared between prep and inference
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
