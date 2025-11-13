# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0-alpha] - 2025-11-13

Initial alpha release of leech for aa-tRNA-seq nanopore signal classification.

### Features

- Complete CLI with 6 commands: `prepare`, `merge-and-split`, `train`, `test`, `infer`, `grid-search`
- Feature extraction from POD5 and BAM files with move table parsing for dwell time computation
- Six model architectures: ConvLSTMDwell, ConvLSTMBase, TransformerDwell, ConvOnly, TCNDwell, ResNetDwell
- Parallel data preparation with multiprocessing support (8 workers default)
- Reference-based motif search to prevent training bias from basecalling errors
- Read-level data splitting to prevent leakage in multi-sample datasets
- Class weighting for imbalanced datasets
- CPU/GPU training support with automatic device detection
- Rich CLI with progress bars and modern interface
- Grid search for chunk context optimization
- Automated GitHub release workflow

### Data Preparation

- Multi-sample merge-and-split with label=file syntax
- TSV-based comparison specifications for batch processing
- Parallel POD5/BAM processing with configurable workers and chunk size
- Reference-based and basecalled motif search strategies
- Optional indel filtering at motif sites

### Training & Evaluation

- Training with early stopping, checkpointing, and validation
- Comprehensive metrics: accuracy, precision, recall, F1, ROC AUC, confusion matrix
- Grid search over signal context parameters
- Model checkpoint management with best model tracking

### Documentation

- Complete MkDocs documentation site with API reference
- CLI usage guide with all commands documented
- Architecture documentation and ADRs (Architecture Decision Records)
- Guides for cluster setup (Alpine/SLURM, Bodhi/LSF)
- Troubleshooting and implementation guides

### Development

- Complete test suite with pytest
- Modern tooling: uv for dependencies, ruff for linting, mypy for type checking
- GitHub Actions CI/CD with linting, testing, and documentation deployment
- Snakemake pipeline for production workflows
