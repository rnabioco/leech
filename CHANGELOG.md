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
- CLI interface with subcommands:
  - `prepare`: Extract training chunks from POD5/BAM
  - `train`: Train models (scaffold - to be implemented)
  - `test`: Test models (scaffold - to be implemented)
  - `infer`: Run inference (scaffold - to be implemented)
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
  - Comprehensive docstrings

### Todo
- Implement training loop (`cli.py:run_train`)
- Implement testing/evaluation (`cli.py:run_test`)
- Implement inference engine (`cli.py:run_infer`)
- Add chunk serialization format

## [0.1.0] - 2025-01-XX

Initial development release.
