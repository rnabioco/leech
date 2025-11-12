# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`leech` (**L**earning **E**nhanc**e**d **C**lassification from **H**anopore signals) is a Python library for training machine learning models on nanopore signal data. It extends [Remora](https://github.com/nanoporetech/remora) with dwell time features extracted from move tables (`mv` tag in BAM files) to classify modified bases, specifically for distinguishing charged vs. uncharged tRNAs in aa-tRNA-seq experiments.

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
3. **Data Preparation** (`io/`, `preparation/`, `chunking/`):
   - Read BAM + POD5 files (`io/bam_reader.py`, `io/pod5_reader.py`)
   - Search for motifs in reference or basecalled sequence (`io/motif_search.py`)
   - Extract training chunks centered on motifs (e.g., "CCAGGC" for tRNA 3' end) (`chunking/extractor.py`)
   - Serialize chunks for training (`chunking/serialization.py`)
4. **Model Training**: PyTorch models with three input branches (signal, sequence, dwell/level features)
5. **Output**: Trained models (.pt files) and predictions (BAM with modification probabilities)

### Parallel Processing

The `prepare` command supports multiprocessing for large datasets:

**Implementation** (`preparation/parallel.py`, `preparation/orchestrator.py`):
- `collect_read_infos_from_bam()`: First pass to collect lightweight read metadata from BAM
- Worker functions: Process batches of reads in parallel
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

**`MoveTable` (features.py)**
- Parses move table from BAM `mv` tag
- `to_seq_to_sig_map()`: converts moves to base→signal index mapping
- Core data structure for dwell time computation

**`BamReader` (io/bam_reader.py)**
- Reads BAM files and extracts alignment information
- Handles move table tags and quality filtering
- Coordinates with POD5Reader for signal extraction

**`POD5Reader` (io/pod5_reader.py)**
- Reads raw signal from POD5 files
- Maps read IDs from BAM to POD5 signal data

**`ChunkExtractor` (chunking/extractor.py)**
- Extracts training chunks centered on motifs
- Handles signal/sequence/feature context windows
- Creates chunk dictionaries with all features aligned

**`MotifSearcher` (io/motif_search.py)**
- Searches for motifs in reference or basecalled sequences
- Maps motif positions from reference to query coordinates using CIGAR
- Filters reads with indels in motif regions (optional)

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
├── cli.py           # Command-line interface (rich-click based)
├── commands/        # CLI command handlers
│   ├── prepare.py   # Prepare command implementation
│   └── merge_split.py  # Merge-and-split command
├── io/              # Input/output operations
│   ├── bam_reader.py    # BAM file reading
│   ├── pod5_reader.py   # POD5 signal reading
│   ├── motif_search.py  # Motif searching in sequences
│   └── reference.py     # Reference sequence handling
├── preparation/     # Data preparation orchestration
│   ├── orchestrator.py  # Main preparation logic
│   ├── parallel.py      # Parallel processing
│   ├── reader.py        # Read iteration
│   └── encoding.py      # Sequence encoding
├── chunking/        # Training chunk extraction
│   ├── extractor.py     # Chunk extraction logic
│   └── serialization.py # Save/load chunks
├── splitting/       # Data splitting
│   └── splitter.py  # Train/val/test split
├── features.py      # MoveTable, dwell times, signal levels, normalization
├── dataset.py       # PyTorch Dataset classes for loading chunks
├── training.py      # Training loop with Trainer class
├── evaluation.py    # Model evaluation and testing
├── inference.py     # Inference engine for predictions
├── gridsearch.py    # Grid search for chunk context optimization
├── util.py          # Helper functions (model loading, metrics)
├── config.py        # Configuration management
├── constants.py     # Project-wide constants
├── logging_config.py  # Logging setup
└── models/          # Model architectures
    ├── __init__.py            # Model registry and get_model()
    ├── components.py          # Reusable model components
    ├── inference_wrapper.py   # Inference wrapper pattern
    ├── conv_lstm_dwell.py     # ConvLSTMDwell architecture
    ├── conv_lstm_base.py      # ConvLSTMBase architecture
    ├── transformer_dwell.py   # TransformerDwell architecture
    ├── tcn_dwell.py           # TCNDwell architecture
    ├── resnet_dwell.py        # ResNetDwell architecture
    └── conv_only.py           # ConvOnly architecture

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
The Snakemake workflow is included in this repository under `pipeline/`. It provides production-ready pipelines for:
- Charged vs uncharged tRNA classification
- Pairwise amino acid classification
- Grid search optimization
- Model comparison across architectures
- HPC cluster integration (SLURM/LSF)

The workflow is designed to integrate with the leech CLI commands and supports both local and cluster execution.

## Important Constraints

1. **Move table requirement**: BAM files MUST have `mv` and `ns` tags (from dorado/guppy basecaller)
2. **Read ID matching**: POD5 read IDs must match BAM query names exactly
3. **Motif-based extraction**: Training focuses on specific motifs (e.g., "CCAGGC" for tRNA); motif_offset specifies the focus base within the motif
   - **Reference-based search (default)**: Searches for motif in reference sequence, maps to query via CIGAR. Avoids bias from basecalling errors at modification sites.
   - **Basecalled search**: Searches in basecalled sequence (backward compatible). Use `--motif-reference bam` to enable.
4. **Reference sequences**: For reference-based motif search, BAM must contain @SQ sequences in header OR provide `--reference-fasta` path
5. **Edge handling**: Chunks require sufficient context (default: 200 samples left/right for signal, 5 bases for k-mer)
6. **Feature alignment**: All three model inputs (signal, sequence, features) must be temporally aligned after convolution layers

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
- ✓ Model architectures defined (ConvLSTMDwell, ConvLSTMBase, TransformerDwell, TCNDwell, ResNetDwell, ConvOnly)
- ✓ CLI interface fully implemented with 6 commands (prepare, merge-and-split, train, test, infer, grid-search)
- ✓ Training loop with Trainer class
- ✓ Grid search for chunk context optimization
- ✓ Testing/evaluation with comprehensive metrics
- ✓ Inference engine with BAM output
- ✓ Chunk serialization (chunking/serialization.py)
- ✓ Model loading utilities (load_model_from_checkpoint in util.py)
- ✓ Comprehensive tests for features.py
- ✓ Parallel data preparation with multiprocessing
- ✓ Reference-based motif search to avoid training bias
- ✓ Snakemake workflow for production pipelines

All core functionality is implemented and ready for use.
