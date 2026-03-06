# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-03-06

### Added

- Tunable `dwell_offset` hyperparameter for motor-sensor offset correction
- `base_justify` parameter to control signal chunk centering ("start", "center", "end")
- Range syntax (`start:stop:step`) for grid search context parameters
- `best_params.json` output from grid search for Snakemake integration
- Self-documenting run summary for pipeline runs
- Motor-pore offset analysis notebook

### Changed

- Reorganize CLI into workflow-based command groups: `data`, `model`, `eval`, `predict`
- Speed up grid search with CPU optimizations and parallel execution
- Migrate docs from MkDocs + Material to Zensical
- Switch LSF profile to use snakemake-executor-plugin-lsf
- Consolidate guides into 3 professional documentation pages
- Replace mypy with ty for type checking
- Deduplicate `_TRAINING_PARAMS` set and extract `_instantiate_model()` helper in util.py
- Consolidate `FEATURE_MODELS` set (dataset.py now references ModelInferenceWrapper)
- Standardize project acronym across docs, pyproject.toml, and CLAUDE.md

### Fixed

- O(n²) BAM scan in inference.py: build alignment dict in one pass instead of rescanning per read
- Off-by-one error in `to_seq_to_sig_map` to match Remora convention
- Default `min_mapq` filtering that drops most tRNA reads
- TypeError from grid search context params passed to model constructor
- Python badge in docs/index.md now shows 3.12+ (matches requires-python)

### Removed

- Stale `data_prep.py.bak` backup file

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
- Modern tooling: uv for dependencies, ruff for linting, ty for type checking
- GitHub Actions CI/CD with linting, testing, and documentation deployment
- Snakemake pipeline for production workflows
