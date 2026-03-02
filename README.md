# leech

**L**earning **E**nhanc**e**d **C**lassification from **H**anopore signals

A Python library for training machine learning models on nanopore signal data, with a focus on integrating dwell time features for modified base detection.

[![CI](https://github.com/rnabioco/leech/actions/workflows/ci.yml/badge.svg)](https://github.com/rnabioco/leech/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`leech` extends [Remora](https://github.com/nanoporetech/remora) with dwell time features extracted from move tables (`mv` tag in BAM files) to classify modified bases, specifically for distinguishing charged vs. uncharged tRNAs in aa-tRNA-seq experiments.

### Key Features

- **Dwell time integration**: Extract per-base dwell times from move tables
- **Signal level features**: Compute signal statistics (mean, median, std, range) per base
- **PyTorch models**: Conv-LSTM architectures with multi-branch inputs (signal + sequence + dwell features)
- **Snakemake pipeline**: Production-ready workflow for HPC clusters (SLURM/LSF)
- **Modern tooling**: Built with uv, ruff, and type hints

## Installation

Requires Python 3.12+

### Using uv (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/rnabioco/leech.git
cd leech
uv sync
```

### Using pip

```bash
git clone https://github.com/rnabioco/leech.git
cd leech
pip install -e .
```

## Quick Start

### CLI Usage

The CLI is organized into workflow-based command groups:

- `leech data` - Prepare and process training data
- `leech model` - Train and optimize models
- `leech eval` - Evaluate and analyze models
- `leech predict` - Run inference on new data

#### 1. Prepare training data

Extract features from POD5 and BAM files:

```bash
uv run leech data prepare \
  --pod5 reads.pod5 \
  --bam alignments.bam \
  --output-dir chunks/ \
  --feature-set signal+dwell+levels \
  --motif CCAGGC \
  --motif-offset 2 \
  --motif-reference fasta \
  --skip-motif-indels \
  --label 1
```

**Note**: By default, `--motif-reference fasta` searches for the motif in the reference sequence (from BAM @SQ header or `--reference-fasta`) and maps to query coordinates using CIGAR. This avoids training data bias from basecalling errors at modification sites. Use `--motif-reference bam` for the old behavior (search in basecalled sequence).

#### 2. Merge and split data (for multi-sample datasets)

For datasets with multiple samples, merge chunks and split at the read level to prevent data leakage:

```bash
uv run leech data merge \
  -i charged=charged_ala.npz \
  -i uncharged=uncharged_ala.npz \
  -o merged/
```

This prevents data leakage that can occur when splitting each sample independently.

#### 3. Train model

```bash
uv run leech model train \
  --train-data chunks/train.json \
  --val-data chunks/val.json \
  --model ConvLSTMDwell \
  --output-dir models/
```

#### 4. Evaluate model

```bash
uv run leech eval test \
  --model models/model_best.pt \
  --test-data chunks/test.json \
  --output metrics.json
```

#### 5. Run inference

```bash
uv run leech predict \
  --model models/model_best.pt \
  --pod5 new_reads.pod5 \
  --bam new_alignments.bam \
  --output predictions.bam
```

#### 6. Advanced: Model comparison and analysis

```bash
# Compare multiple models
uv run leech eval compare \
  -m models/model1/ -m models/model2/ \
  -t chunks/test.npz \
  -o comparison/

# Analyze feature importance
uv run leech eval importance \
  -m models/model_best.pt \
  -t chunks/test.npz \
  -o importance/

# Optimize hyperparameters (parallel on CPU)
uv run leech model optimize \
  --train-data chunks/train.npz \
  --val-data chunks/val.npz \
  --context-grid 200,500,1000,2000 \
  --parallel 4 \
  -o grid_results/
```

## Snakemake Pipeline

For production workloads, use the included Snakemake pipeline that handles:
- **Charged vs Uncharged Classification**: Distinguish between charged and uncharged tRNAs
- **Pairwise Amino Acid Classification**: Binary classifiers for all pairs of amino acids
- **Grid Search**: Hyperparameter optimization
- **Model Comparison**: Compare multiple architectures on the same data
- **HPC Integration**: SLURM and LSF cluster support

### Pipeline Structure

```
leech/
└── pipeline/
    ├── config/
    │   ├── config.yaml              # Main configuration file
    │   ├── alpine-config.yaml       # CU Boulder Alpine cluster (SLURM)
    │   └── bodhi-config.yaml        # Local Bodhi cluster (LSF)
    ├── workflow/
    │   ├── Snakefile                # Main workflow file
    │   ├── rules/                   # Modular rule files
    │   │   ├── common.smk
    │   │   ├── prepare.smk
    │   │   ├── grid_search.smk
    │   │   ├── train.smk
    │   │   ├── inference.smk
    │   │   ├── evaluate.smk
    │   │   └── compare_models.smk
    │   └── scripts/
    │       ├── summarize_metrics.py
    │       └── compare_architectures.py
    └── cluster/                     # Cluster execution profiles
        ├── slurm/
        │   ├── config.yaml
        │   ├── slurm_submit.sh
        │   └── slurm_status.py
        └── lsf/
            ├── config.yaml
            ├── lsf_submit.sh
            └── lsf_status.py
```

### Running the Pipeline

#### Configure Your Samples

Edit `pipeline/config/config.yaml`:

```yaml
samples:
  sample_charged_ala_rep1:
    pod5: "data/charged/ala/rep1.pod5"
    bam: "data/charged/ala/rep1.bam"
    label: "charged"
    amino_acid: "Ala"

  sample_uncharged_ala_rep1:
    pod5: "data/uncharged/ala/rep1.pod5"
    bam: "data/uncharged/ala/rep1.bam"
    label: "uncharged"
    amino_acid: "Ala"
```

#### Execute on SLURM

```bash
# Dry run
snakemake --profile pipeline/cluster/slurm -n

# Execute
snakemake --profile pipeline/cluster/slurm
```

#### Execute on LSF

```bash
# Dry run
snakemake --profile pipeline/cluster/lsf -n

# Execute
snakemake --profile pipeline/cluster/lsf
```

#### Local Execution (for testing)

```bash
snakemake --cores 8
```

### Pipeline Targets

**Single Model Mode** (when `compare_models: false`):
```bash
snakemake --profile pipeline/cluster/slurm all_prepare      # Prepare data only
snakemake --profile pipeline/cluster/slurm all_grid_search  # Grid search only
snakemake --profile pipeline/cluster/slurm all_train        # Train models only
snakemake --profile pipeline/cluster/slurm all_infer        # Run inference only
snakemake --profile pipeline/cluster/slurm all_single_model # Single model analysis
```

**Model Comparison Mode** (when `compare_models: true`):
```bash
snakemake --profile pipeline/cluster/slurm all_prepare                # Prepare data
snakemake --profile pipeline/cluster/slurm all_grid_search_comparison # Grid search for all
snakemake --profile pipeline/cluster/slurm all_train_comparison       # Train all architectures
snakemake --profile pipeline/cluster/slurm all_compare_models         # Full comparison
```

### Cluster Setup Guides

For detailed cluster-specific setup instructions, see:

- **[Alpine Setup (SLURM)](guides/ALPINE_SETUP.md)**: CU Boulder Alpine cluster with SLURM
- **[Bodhi Setup (LSF)](guides/BODHI_SETUP.md)**: Local Bodhi cluster with LSF

These guides cover:
- Storage architecture and quotas
- Module loading and dependencies
- Resource allocation
- Queue/partition configuration
- Monitoring and troubleshooting

### Pipeline Configuration

Key configuration options in `pipeline/config/config.yaml`:

```yaml
# Model architecture
model: "ConvLSTMDwell"  # or "ConvLSTMBase"

# Training hyperparameters
epochs: 50
batch_size: 128
learning_rate: 0.001
early_stopping_patience: 5

# Enable multi-architecture comparison
compare_models: true
models_to_compare:
  - "ConvLSTMBase"       # Baseline (no dwell features)
  - "ConvLSTMDwell"      # Original with dwell features
  - "TransformerDwell"   # Transformer with self-attention
  - "ConvOnly"           # Pure CNN baseline
  - "TCNDwell"           # Temporal Convolutional Network
  - "ResNetDwell"        # Residual Network

# Enable grid search
use_grid_search: true
grid_search:
  learning_rate: [0.0001, 0.001, 0.01]
  batch_size: [64, 128, 256]
  hidden_size: [128, 256, 512]
  num_layers: [1, 2, 3]
  dropout: [0.1, 0.2, 0.3]

# Amino acid pairs for pairwise classification
amino_acids:
  - "Ala"
  - "Gly"
  - "Val"
  # ... etc
```

### Pipeline Output

```
results/
├── chunks/                      # Prepared training data
│   └── {sample}/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── models/
│   ├── grid_search/             # Grid search results
│   │   ├── charged_vs_uncharged/
│   │   └── pairwise/{pair}/
│   ├── charged_vs_uncharged/    # Trained models
│   │   ├── model_best.pt
│   │   ├── model_checkpoint.pt
│   │   └── training_history.json
│   └── pairwise/{pair}/
│       └── ...
├── inference/                   # Prediction BAM files
│   ├── charged_vs_uncharged/
│   │   └── {sample}_predictions.bam
│   └── pairwise/{pair}/
│       └── {sample}_predictions.bam
└── metrics/                     # Evaluation metrics
    ├── charged_vs_uncharged/
    │   ├── {sample}_metrics.json
    │   └── ...
    ├── pairwise/{pair}/
    │   └── {sample}_metrics.json
    ├── charged_vs_uncharged_summary.tsv
    └── pairwise_summary.tsv
```

## Requirements

### Input Data

- **POD5 files**: Raw nanopore signal (from ONT sequencing)
- **BAM files**: Basecalls with move table tags (`mv`, `ns`, `ts`)
  - Generated by dorado/guppy basecaller with `--emit-moves` flag

### Move Table Tags

| Tag | Description |
|-----|-------------|
| `mv` | Move table (stride + binary array) |
| `ns` | Number of signal samples |
| `ts` | Trim offset (optional) |

### Software Dependencies

- **PyTorch**: Neural network training
- **pod5**: Reading ONT POD5 format
- **pysam**: BAM file parsing
- **polars**: Fast dataframe operations
- **numpy/scipy/scikit-learn**: Numerical operations and ML utilities
- **pydantic**: Config validation
- **ruff**: Linting and formatting
- **Snakemake**: Pipeline orchestration (≥ 7.0)

## Architecture

### Data Flow

```
POD5 + BAM → Feature Extraction → Training Chunks → Model Training → Predictions
     ↓              ↓                    ↓                ↓
  Signal    Dwell times         signal+dwell+      ConvLSTMDwell
            Signal levels        levels features
```

### Model Architectures

- **ConvLSTMDwell**: Multi-branch model with dwell time features (recommended)
- **ConvLSTMBase**: Baseline model without dwell features (for comparison)
- **TransformerDwell**: Transformer-based with multi-head self-attention
- **ConvOnly**: Pure CNN with multi-scale convolutions
- **TCNDwell**: Temporal Convolutional Network with dilated convolutions
- **ResNetDwell**: Deep residual network with skip connections

## Development

### Setup

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run ty check src/leech/
```

### Project Structure

```
leech/
├── src/leech/           # Main package
│   ├── cli.py          # Command-line interface (rich-click based)
│   ├── commands/       # CLI command handlers
│   ├── io/             # BAM/POD5 reading, reference handling
│   ├── preparation/    # Data preparation orchestration
│   ├── chunking/       # Training chunk extraction and serialization
│   ├── splitting/      # Train/val/test splitting
│   ├── features.py     # Dwell time & signal feature extraction
│   ├── dataset.py      # PyTorch Dataset classes
│   ├── training.py     # Training loop with Trainer class
│   ├── evaluation.py   # Model evaluation and testing
│   ├── inference.py    # Inference engine
│   ├── gridsearch.py   # Grid search for hyperparameters
│   ├── util.py         # Helper functions
│   ├── config.py       # Configuration management
│   ├── constants.py    # Project-wide constants
│   ├── logging_config.py  # Logging setup
│   └── models/         # PyTorch model architectures (6 models)
├── tests/              # Test suite
├── pipeline/           # Snakemake pipeline
│   ├── config/         # Pipeline configuration files
│   ├── workflow/       # Snakemake workflow
│   │   ├── Snakefile   # Main workflow
│   │   └── rules/      # Modular rule files
│   ├── cluster/        # Cluster execution profiles (SLURM/LSF)
│   └── resources/      # Pipeline resources
├── docs/               # Documentation site
└── pyproject.toml      # Project configuration
```

### Adding Dependencies

```bash
# Add a runtime dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Update all dependencies
uv sync --upgrade
```

## Citation

If you use `leech`, please cite:

- This work (publication pending)
- [Remora](https://github.com/nanoporetech/remora) (underlying training framework)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Related Projects

- [Remora](https://github.com/nanoporetech/remora) - Modified base detection for nanopore sequencing
- [POD5](https://github.com/nanoporetech/pod5-file-format) - Efficient file format for nanopore signal data
- [Snakemake](https://snakemake.readthedocs.io/) - Workflow management system
