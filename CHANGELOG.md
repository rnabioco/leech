# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-03-14

### Added

- 12 new model architectures (20 total): BatchNorm (BN), Attention, GroupNorm (GN), LayerNorm (LN) variants for ConvLSTM; TCNDwellGN, TCNDwellLN, TCNDwellResidual; TransformerDwellResidual
- Multi-channel signal input (`signal_in_channels`) across all architectures for 2-channel (raw + kmer residual) models
- Kmer residual features: `kmer_expected`, `kmer_residual`, `kmer_residual_abs` from kmer level table lookup
- Signal map refinement rewritten to match Remora's banded Viterbi algorithm with iterative Theil-Sen rescaling
- Signal-level kmer encoding (`signal_kmer`) as default sequence encoding
- Reference-anchored mode (`--anchor reference`) and `pa_scaling` normalization
- Composable config dataclasses (`configs.py`) replacing 6-layer parameter threading
- `feature_start`/`feature_end` parameters replacing confusing `dwell_margin` params
- Dwell cross-attention in TransformerDwell, TCNDwell, ResNetDwell, ConvOnly
- Multi-class classification with confidence-weighted and tournament pairwise aggregation
- K-fold cross-validation with stratified read-level splits (`--k-fold`)
- Balance-groups sampling for equal source group contribution per epoch
- Platt scaling calibration (`leech model calibrate`) with guardrails and best-fold selection
- TorchScript export (`leech model export`) for standalone model deployment
- Rust-accelerated signal statistics via PyO3 (`leech-core` crate)
- `check-rust` CLI command to verify Rust extension availability
- Auto-load bundled kmer table when `--refine-signal-map` has no `--kmer-table`
- `--reference-fasta` support in `leech predict` for reference-anchored bundle inference
- Remora-compatible model variants (ConvLSTMRemora, ConvLSTMRemoraBase)
- Parallel inference with batch POD5 reads in workers
- GitHub Actions release workflow with platform wheel builds (linux x86_64/aarch64, macOS x86_64/arm64)

### Changed

- Default sequence encoding from `base_onehot` to `signal_kmer`
- Default `scale_iters` from 0 to 2 for signal map refinement
- Migrate model export from `torch.jit` to `torch.export` (PyTorch 2+)
- Unify `--anchor` flag across `prepare` and `predict` commands
- Lazy-load `MODEL_REGISTRY` to speed up CLI help (5.3s to 0.6s)
- Lazy imports in `__init__.py` to cut CLI startup from ~9s to ~0.5s
- Rust signal stats 217x faster than NumPy; pre-tensorize dataset; flatten serialization
- Enable `torch.compile` on CPU/GPU, TF32 matmul precision, `inference_mode` in eval
- Optimize test suite runtime from 287s to ~20s
- Speed up `leech eval test` with GPU optimizations and larger batch size
- Refactor CLI handlers into `commands/` subpackage
- Upgrade PyO3 and rust-numpy from 0.23 to 0.28

### Fixed

- Ensure `model_best.pt` always exists after training resume
- Load all checkpoints/bundles to CPU first to avoid device mismatch
- Batch bundle inference for GPU utilization
- Normalize `signal_in_channels` in architecture config comparison
- Return reference-relative coords from ReferenceMotifSearcher when `anchor=reference`
- Prevent `num_out` from leaking into model constructors that don't accept it
- Resolve constructor params for `**kwargs` subclasses in `_instantiate_model`
- Set `num_workers=0` in evaluation DataLoader to reduce memory usage
- Fix `torch.compile` `_orig_mod` prefix in checkpoint loading
- Fix missing `model_best.pt` when resuming completed training
- Fix bundle discovery for k-fold CV and batch size for small datasets
- Fix evaluation to use softmax for cross-entropy (multi-output) models
- Fix stale FASTA index by regenerating .fai before opening
- Fix DataLoader workers in parallel grid search
- Rename ResNetDwell `bn1/bn2` to `norm1/norm2` to match checkpoint migration
- Strip explicit kwargs from model config to prevent duplicates

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
