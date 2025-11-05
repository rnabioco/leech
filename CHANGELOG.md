# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- Feature extraction from POD5 and BAM files
  - Move table parsing for dwell time computation
  - Per-base signal level statistics (mean, median, std, range)
  - Multiple normalization methods (median-MAD, z-score, quantile)
- PyTorch model architectures
  - `ConvLSTMDwell`: Multi-branch model with dwell features
  - `ConvLSTMBase`: Baseline model without dwell features
- Complete training pipeline
  - `Trainer` class with training loop, validation, and checkpointing
  - `train_model()` high-level training function
  - Early stopping and best model tracking
- Grid search functionality
  - Systematic optimization over signal window sizes (left/right context)
  - CSV output with performance metrics for each configuration
- Model evaluation and testing
  - `evaluate_model()` function with comprehensive metrics
  - Accuracy, precision, recall, F1, ROC AUC, confusion matrix
  - JSON output for test results
- Inference engine
  - `run_inference()` for predictions on new data
  - BAM output with modification probability tags (MP/ML)
  - Motif-based filtering for targeted predictions
- Utility functions (`util.py`)
  - `load_model_from_checkpoint()`: Load trained models with configs
  - `compute_metrics()`: Calculate classification metrics
  - `save_metrics()` and `print_metrics()`: Metrics I/O
- PyTorch Dataset classes
  - `LeechDataset`: Load and preprocess training chunks
  - Custom collate function for batching
- Chunk serialization (save/load .npz format)
- CLI interface with fully implemented subcommands:
  - `prepare`: Extract training chunks from POD5/BAM
  - `train`: Train models with hyperparameter support
  - `test`: Evaluate models on test data
  - `infer`: Run inference and write BAM predictions
  - `grid-search`: Optimize chunk context parameters
- Comprehensive test suite (8 tests for feature extraction)
- Development tooling:
  - `uv` for dependency management
  - `ruff` for linting and formatting
  - `mypy` for type checking
  - GitHub Actions CI/CD pipeline
  - Dependabot for automated dependency updates
- Documentation:
  - README.md with quickstart guide
  - CLAUDE.md for AI assistant guidance
  - Grid search documentation (docs/grid-search.md, docs/grid-search-usage.md)
  - Comprehensive docstrings throughout

## [0.1.0] - 2025-01-XX

Initial development release.
