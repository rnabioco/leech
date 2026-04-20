# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2026-04-20

### Added

- `trna_id` adversarial confound for full isoacceptor identity debiasing
- Rust-accelerated training chunk extraction with rayon parallelism
- Dwell template features, motor/dwell-attn models, and rough-rescale option
- `--no-compress` flag and streaming BAM for data preparation
- `--split-by` for group-level train/test splits
- `signal_mode`, TCNDwellSplitResidual, and per-channel augmentation
- `--copy-tags` option to copy BAM tags into TSV predict output
- leech version and git commit captured in model config and bundles
- Rust monolithic extraction for bundle inference
- `leech_core` parallelization via `mp.Pool` workers for bundle inference

### Changed

- Bumped `escapepod-rs` to v0.1.3 (SSSE3 SIMD SVB16, audit-driven hot-path optimizations, dynamic versioning)
- Replaced `pod5` Python package with `escapepod` bindings for POD5 I/O
- `escapepod` moved to optional `pod5` extra for pixi compatibility
- Config.json is now the source of truth for model construction
- Replaced hardcoded model allowlists with signature introspection
- Consolidated `models/` from 27 to 13 files
- Split `util.py` into `model_loading`, `model_export`, `bundling`, `metrics`
- Split `inference.py` into `inference/` package
- Split Rust `inference_pipeline.rs` into 8 submodules
- Queue-based extraction pipeline for improved GPU inference throughput

### Fixed

- Module-level `np` shadowing from local numpy imports
- `_parse_and_validate_inputs` return type annotation
- Plumb `dwell_template_table` through calibrate/eval/predict
- Byte-identical Rust↔Python parity for CIGAR ref-to-signal mapping
- Pass `signal_in_channels` to model during calibration
- Resolve remaining clippy warnings in Rust code
- Excluded `vulture_whitelist.py` from ruff linting
- CI submodule checkout with `ESCAPEPOD_PAT`; ruff format compliance
- Benchmark script ruff lint errors

## [0.3.1] - 2026-03-20

### Added

- TSV prediction output (`TsvPredictionWriter`) as alternative to BAM tag output for multiclass models
- escapepod-rs integration for ~10x faster Rust-accelerated inference via POD5 batch reads
- Multiclass temperature scaling calibration with ECE improvement gating
- Adversarial training with gradient reversal layer and confound maps for discriminator base debiasing
- CL regression head for multi-task charging level prediction
- Cross-layer augmentation: time masking, cross-layer shift, per-channel feature noise
- Mega-batch streaming inference with double-buffered GPU pipeline
- `--signal-context` CLI option for `leech data prepare` to set asymmetric signal windows
- `--min-confidence` and `--min-margin` thresholds for `leech predict`
- `--oversample-minority` flag for class imbalance handling
- Auto-read `anchor` and `reference_fasta` from model config at predict time
- `am` (margin) BAM tag in predict output
- `enable_repr_capture()` for `ModelInferenceWrapper` internal activations
- Centralized Rich console with wide fallback for SLURM batch jobs
- TCNDwellResidualGN and TCNDwellResidualLN model variants (22 total architectures)
- Label smoothing and cosine annealing scheduler options
- `py.typed` PEP 561 marker for type checker support
- vulture dead code detection in dev tooling

### Changed

- K-fold merge now caches input files in RAM for faster processing
- Array-level merge without zlib compression for merge step
- Disable DataLoader workers for validation to reduce memory usage
- Checkpoint multiclass models on `val_f1` instead of `val_acc`
- Development status upgraded from Alpha to Beta
- Version string now uses `importlib.metadata` with git hash fallback
- Removed seaborn from notebook extras (plotnine-only policy)
- Bumped leech_core to v0.3.0 and Rust edition 2024

### Fixed

- Store `focus_signal_pos` in chunks for asymmetric `signal_context`
- Enable TF32 matmul precision in inference path
- Training summary reports actual best `val_acc`/F1 instead of last epoch
- Use `tolist()` for multiclass `pos_weight` serialization in config
- Handle k-fold directories in multiclass bundle discovery
- `label_map.json` lookup for k-fold adversarial training
- Use `reads_by_ids()` for O(1) indexed POD5 lookup instead of full scan
- Gate multiclass temperature scaling on ECE improvement
- Route multiclass through parallel inference path
- FASTA index race condition under parallel SLURM jobs
- Label smoothing no longer alters labels used for metrics
- Propagate `pa_mean`, `pa_stdev`, `skip_motif_indels`, and refiner params through config chain
- Align signal map refinement `scale_iters` between prepare and inference
- Auto-read `base_justify` from model config in single-model inference
- Use reference-based motif search in inference
- Signal kmer coordinate adjustment and `reference_fasta` plumbing
- Default `skip_motif_indels` to `False` everywhere

### Performance

- Prefetch pipeline with rayon contention fix in inference
- Sub-batch extraction with async BAM writes for GPU pipelining
- Multi-threaded extraction in sequential inference path
- Optimized inference pipeline for fast prediction

### Removed

- Dead code: `config.py` (replaced by `configs.py`), `calibrate_model_temperature()`, `_is_leech_export()`, `load_predictions_from_bam()`, `prepare_chunks_with_context()`, `extract_disc_bases_from_fasta()`, `handle_branch_contribution()`, `display_logo()`
- Unused constants: `DEFAULT_DWELL_MARGIN_LEFT/RIGHT`, `DEFAULT_NUM_WORKERS`, `DEFAULT_SEQ_ENCODING`, `DEFAULT_MIN_MAPQ`, `DEFAULT_MOTIF`, `DEFAULT_MOTIF_OFFSET`, `DEFAULT_REMORA_NUM_OUT`
- Unused Rust accel imports: `_rs_test_process_read`, `_rs_extract_levels`, `_rs_rough_rescale`
- Unused IO methods: `BAMReader.get_header()`, `BAMReader.count_alignments()`, `POD5Reader.get_signals_batch()`, `POD5Reader.iter_all_reads()`, `ReferenceManager.get_all_sequences()`

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
