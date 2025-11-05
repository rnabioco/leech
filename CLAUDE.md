# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`leech` (**L**earning **E**nhanced **A**minoacylation **C**lassification from **H**anopore signals) is a Python library for training machine learning models on nanopore signal data. It extends [Remora](https://github.com/nanoporetech/remora) with dwell time features extracted from move tables (`mv` tag in BAM files) to classify modified bases, specifically for distinguishing charged vs. uncharged tRNAs in aa-tRNA-seq experiments.

## Development Commands

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

### Installation
```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (creates .venv and installs everything)
uv sync

# Sync with dev dependencies
uv sync --all-extras
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_features.py

# Run with coverage
uv run pytest --cov=leech --cov-report=term-missing

# Run specific test function
uv run pytest tests/test_features.py::test_compute_dwell_times -v
```

### Linting and Formatting
```bash
# Check formatting
uv run ruff format --check .

# Format code
uv run ruff format .

# Lint with ruff
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .

# Type checking with mypy
uv run mypy src/leech/
```

### Running the CLI
```bash
# Prepare training data
uv run leech prepare --pod5 reads.pod5 --bam alignments.bam --output-dir chunks/

# Train model
uv run leech train --train-data chunks/train.json --val-data chunks/val.json \
  --model ConvLSTMDwell --output-dir models/

# Test model
uv run leech test --model models/model_best.pt --test-data chunks/test.json --output metrics.json

# Run inference
uv run leech infer --model models/model_best.pt --pod5 reads.pod5 --bam alignments.bam --output predictions.bam
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

## Architecture

### Core Data Flow

1. **Input**: POD5 files (raw nanopore signal) + BAM files (basecalls with `mv` move table tags)
2. **Feature Extraction** (`features.py`):
   - Parse move tables to compute per-base dwell times
   - Extract signal level statistics (mean, median, std, range) per base
   - Normalize raw signal using median-MAD
3. **Data Preparation** (`data_prep.py`):
   - Iterate BAM + POD5 together via `iter_bam_with_pod5()`
   - Extract training chunks centered on motifs (e.g., "CCA" for tRNA 3' end)
   - Create `LeechRead` objects with all features
4. **Model Training**: PyTorch models with three input branches (signal, sequence, dwell/level features)
5. **Output**: Trained models (.pt files) and predictions (BAM with modification probabilities)

### Key Classes and Functions

**`MoveTable` (features.py:15-56)**
- Parses move table from BAM `mv` tag
- `to_seq_to_sig_map()`: converts moves to base→signal index mapping
- Core data structure for dwell time computation

**`LeechRead` (data_prep.py:27-117)**
- Container for a single read's full feature set
- Includes: signal, sequence, dwell times, dwell features, signal features
- `get_chunk()`: extracts training chunks with signal/sequence/feature context

**`iter_bam_with_pod5()` (data_prep.py:147-224)**
- Main data loading function
- Iterates aligned BAM reads, fetches signal from POD5
- Yields `LeechRead` objects with all features computed
- Filters by mapping quality and required BAM tags

**`ConvLSTMDwell` (models/conv_lstm_dwell.py:13-141)**
- PyTorch model with three branches:
  - Signal branch: Conv1d on raw signal
  - Sequence branch: Conv1d on one-hot encoded k-mers
  - Feature branch: Conv1d on dwell+level features (NEW vs. Remora)
- Branches merge → BiLSTM → FC output
- Compare with `ConvLSTMBase` (no dwell features) to measure impact

### Module Organization

```
src/leech/           # Main package source
├── cli.py           # Command-line interface (argparse-based)
├── data_prep.py     # BAM/POD5 reading, LeechRead, chunk extraction
├── features.py      # MoveTable, dwell times, signal levels, normalization
├── dataset.py       # PyTorch Dataset classes for loading chunks
├── training.py      # Training loop with Trainer class
├── evaluation.py    # Model evaluation and testing
├── inference.py     # Inference engine for predictions
├── gridsearch.py    # Grid search for chunk context optimization
├── util.py          # Helper functions (model loading, metrics)
└── models/          # Model architectures
    ├── __init__.py  # Model registry and get_model()
    ├── conv_lstm_dwell.py  # ConvLSTMDwell architecture
    └── conv_lstm_base.py   # ConvLSTMBase architecture

tests/               # pytest tests
```

## Implementation Details

### Feature Engineering
- **Dwell times**: Number of signal samples per base, computed from move table using `np.diff(seq_to_sig_map)`
- **Signal normalization**: Median-MAD (default) is robust to outliers; z-score and quantile methods available
- **Feature concatenation**: Models expect 3 inputs: (signal, sequence, features) where features combines dwell and signal statistics

### Move Table Format
The BAM `mv` tag format:
- First element: stride (basecaller downsampling factor, typically 5)
- Remaining elements: binary array where 1 = new base, 0 = same base
- Convert to signal indices: `signal_idx = (move_position + 1) * stride + trim_offset`

### Training Data Structure
Training chunks are dictionaries:
```python
{
    'signal': np.ndarray,      # Raw signal chunk [signal_len]
    'sequence': str,           # K-mer context sequence
    'dwell': np.ndarray,       # Per-base dwell times [kmer_len]
    'features': np.ndarray,    # Stacked features [num_features, kmer_len]
    'label': int,              # 0=uncharged, 1=charged
    'read_id': str,
    'base_idx': int
}
```

### Snakemake Integration
The Snakemake workflow has been moved to a separate repository. The leech library is designed to integrate easily with Snakemake pipelines via its CLI commands.

## Important Constraints

1. **Move table requirement**: BAM files MUST have `mv` and `ns` tags (from dorado/guppy basecaller)
2. **Read ID matching**: POD5 read IDs must match BAM query names exactly
3. **Motif-based extraction**: Training focuses on specific motifs (e.g., "CCA" for tRNA); motif_offset specifies the focus base within the motif
4. **Edge handling**: Chunks require sufficient context (default: 200 samples left/right for signal, 5 bases for k-mer)
5. **Feature alignment**: All three model inputs (signal, sequence, features) must be temporally aligned after convolution layers

## Dependencies

- **PyTorch**: Neural network training
- **pod5**: Reading ONT POD5 format
- **pysam**: BAM file parsing
- **polars**: Fast dataframe operations (used for future batch processing)
- **numpy/scipy/scikit-learn**: Numerical operations and ML utilities
- **pydantic**: Config validation
- **ruff**: Linting and formatting (replaces black/flake8)

## Current Status

The codebase is feature-complete:
- ✓ Feature extraction fully implemented
- ✓ Model architectures defined (ConvLSTMDwell, ConvLSTMBase)
- ✓ CLI interface fully implemented
- ✓ Training loop with Trainer class
- ✓ Grid search for chunk context optimization
- ✓ Testing/evaluation with comprehensive metrics
- ✓ Inference engine with BAM output
- ✓ Chunk serialization (save_chunks/load_chunks in data_prep.py)
- ✓ Model loading utilities (load_model_from_checkpoint in util.py)
- ✓ Comprehensive tests for features.py

All core functionality is implemented and ready for use.
