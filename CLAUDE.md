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
# Prepare training data (sequential)
uv run leech prepare --pod5 reads.pod5 --bam alignments.bam --output-dir chunks/

# Prepare training data (parallel - recommended for large datasets)
# Use --workers to specify number of parallel processes
# Use --chunk-size to control batch size (default: 100 reads per batch)
uv run leech prepare --pod5 reads.pod5 --bam alignments.bam --output-dir chunks/ \
  --workers 8 --chunk-size 100

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
   - Extract training chunks centered on motifs (e.g., "CCAGGC" for tRNA 3' end)
   - Create `LeechRead` objects with all features
4. **Model Training**: PyTorch models with three input branches (signal, sequence, dwell/level features)
5. **Output**: Trained models (.pt files) and predictions (BAM with modification probabilities)

### Parallel Processing

The `prepare` command supports multiprocessing for large datasets:

**Implementation** (data_prep.py:525-812):
- `collect_read_infos_from_bam()`: First pass to collect lightweight read metadata from BAM
- `_process_read_chunk_worker()`: Worker function that processes batches of reads in parallel
- `prepare_training_data_parallel()`: Main parallel orchestration with configurable workers and chunk size

**Usage**:
```bash
# Use --workers N to enable parallel processing (N > 1)
# Use --chunk-size M to control batch size (default: 100 reads)
uv run leech prepare --pod5 data.pod5 --bam alignments.bam \
  --output-dir chunks/ --workers 8 --chunk-size 100
```

**Performance**:
- Expected speedup: 3-6x on typical multi-core machines
- CPU-bound tasks (feature extraction): near-linear speedup with cores
- I/O-bound tasks (POD5 reading): moderate speedup (2-4x)
- Memory-efficient: Chunks reads into batches to avoid loading entire dataset

**Implementation details**:
- Two-pass design: (1) collect read metadata from BAM, (2) parallel POD5 + feature extraction
- Each worker opens POD5 independently for thread-safe access
- Batching via `chunk_size` prevents memory issues with large datasets
- Reference sequences are passed to workers for reference-based motif search

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
3. **Motif-based extraction**: Training focuses on specific motifs (e.g., "CCAGGC" for tRNA); motif_offset specifies the focus base within the motif
   - **Reference-based search (default)**: Searches for motif in reference sequence, maps to query via CIGAR. Avoids bias from basecalling errors at modification sites.
   - **Basecalled search**: Searches in basecalled sequence (backward compatible). Use `--motif-reference bam` to enable.
4. **Reference sequences**: For reference-based motif search, BAM must contain @SQ sequences in header OR provide `--reference-fasta` path
5. **Edge handling**: Chunks require sufficient context (default: 200 samples left/right for signal, 5 bases for k-mer)
6. **Feature alignment**: All three model inputs (signal, sequence, features) must be temporally aligned after convolution layers

## Batch Effects and Data Leakage

**⚠️ CRITICAL**: If charged and uncharged samples are sequenced in **separate runs**, the model may learn run-specific artifacts instead of biological signal, leading to perfect AUC in validation but complete failure on new data.

### The Problem

The default per-read normalization (median-MAD) removes within-read signal drift but **preserves batch effects** between sequencing runs:
- Different pore types or conditions
- Different baseline signal characteristics
- Different temperature/voltage settings
- Different sequencing chemistry batches

Since train/test splits are done by read (not by run), the model sees examples from both runs in training and can learn to distinguish "Run A" (charged) vs "Run B" (uncharged) rather than the biological difference.

### Symptoms of Batch Effect Leakage

1. **Perfect or near-perfect AUC** (1.000) on validation data
2. Model trained quickly (few epochs to convergence)
3. High per-sample accuracy differences during validation
4. Model fails completely on new sequencing runs
5. Between-sample variance >> between-label variance

### Solutions

**Best practices (in order of preference)**:

1. **Multiplex samples in same run**: Barcode charged and uncharged samples together in the same flowcell
2. **Global normalization** (planned): Use `--global-normalization` flag to normalize across all reads from all samples
3. **Batch effect correction**: Apply explicit batch correction before training
4. **Cross-run validation**: Hold out entire runs for testing (not just reads)

### Diagnosing Batch Effects

Use the diagnostic script to detect batch effects in your data:

```bash
python scripts/diagnose_leakage.py \
  --chunks-dir path/to/chunks/ \
  --output-dir diagnostics/
```

This will:
- Compute between-sample vs between-label variance ratios
- Generate PCA/t-SNE plots colored by sample and label
- Flag if batch effects are larger than biological signal
- Provide recommendations for correction

**Interpretation**:
- Variance ratio > 2.0: ⚠️ Strong batch effects detected
- Variance ratio > 1.0: ⚠️ Moderate batch effects
- Variance ratio < 1.0: ✓ Label variance dominates (good)

### Global Normalization (Planned)

The `--global-normalization` flag will enable two-pass processing:

1. **Pass 1**: Extract all chunks with per-read normalization
2. **Pass 2**: Re-normalize all chunks using dataset-wide median/MAD
3. Save globally-normalized chunks for training

This ensures all samples are on the same scale, preventing the model from learning run-specific artifacts.

**Status**: CLI flag added, implementation in progress. See features.py for `compute_global_normalization_params()` and `apply_global_normalization()`.

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
