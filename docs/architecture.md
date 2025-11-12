# Leech Architecture Overview

This document provides a high-level overview of the leech architecture after the Phase 1 & 2 refactoring.

## Table of Contents

- [Overview](#overview)
- [Module Structure](#module-structure)
- [Data Flow](#data-flow)
- [Key Design Patterns](#key-design-patterns)
- [Configuration Management](#configuration-management)

## Overview

Leech is a Python library for training machine learning models on nanopore signal data. It extends Remora with dwell time features extracted from move tables to classify modified bases, specifically for distinguishing charged vs. uncharged tRNAs in aa-tRNA-seq experiments.

### Core Capabilities

1. **Feature Extraction**: Extract dwell times and signal statistics from POD5/BAM files
2. **Data Preparation**: Create training chunks with motif-based filtering
3. **Model Training**: Train PyTorch models with signal + sequence + dwell features
4. **Inference**: Predict modification states on new data

## Module Structure

The codebase is organized into focused modules with clear responsibilities:

```
src/leech/
├── config.py              # Pydantic configuration models
├── constants.py           # Global constants
├── logging_config.py      # Logging setup
├── features.py            # Feature extraction (dwell times, signal levels)
├── data_prep.py           # High-level orchestration
├── dataset.py             # PyTorch Dataset classes
├── training.py            # Training loop and Trainer class
├── evaluation.py          # Model evaluation
├── inference.py           # Inference engine
├── gridsearch.py          # Hyperparameter grid search
├── util.py                # Utility functions
├── cli.py                 # Command-line interface
│
├── io/                    # I/O utilities (464 lines)
│   ├── __init__.py
│   ├── bam_reader.py      # BAM reading with filtering
│   ├── pod5_reader.py     # POD5 reading (batched access)
│   ├── reference.py       # Reference sequence management
│   └── motif_search.py    # Motif search strategies
│
├── chunking/              # Chunk management (380 lines)
│   ├── __init__.py
│   ├── extractor.py       # LeechRead and chunk extraction
│   └── serialization.py   # Save/load chunks
│
├── splitting/             # Data splitting (476 lines)
│   ├── __init__.py
│   └── splitter.py        # Read-level splitting
│
└── models/                # Model architectures
    ├── __init__.py
    ├── conv_lstm_dwell.py
    ├── conv_lstm_base.py
    ├── transformer_dwell.py
    ├── tcn_dwell.py
    ├── resnet_dwell.py
    ├── conv_only.py
    ├── components.py
    └── inference_wrapper.py
```

### Module Responsibilities

#### `io/` - Input/Output Operations

Handles all file I/O operations for nanopore data:

- **bam_reader.py**: BAM file reading with alignment filtering
  - `iter_bam_alignments()`: Filter and iterate BAM records
  - `collect_read_infos()`: Collect lightweight read metadata
  - `BAMReader`: Context manager for BAM access
  - `ReadInfo`: Lightweight read metadata container

- **pod5_reader.py**: POD5 signal reading with batched access
  - `read_pod5_signal()`: Read single signal
  - `read_pod5_signals_batch()`: Batch reading for performance
  - `POD5Reader`: Context manager with caching support

- **reference.py**: Reference sequence management
  - `get_reference_sequences()`: Load from BAM header or FASTA
  - `ReferenceManager`: Lazy-loading reference manager

- **motif_search.py**: Motif search strategies (Strategy pattern)
  - `MotifSearcher`: Abstract base class
  - `BasecalledMotifSearcher`: Search in basecalled sequence
  - `ReferenceMotifSearcher`: Search in reference (avoids basecalling errors)
  - `get_motif_searcher()`: Factory function

#### `chunking/` - Chunk Extraction and Serialization

Manages training chunk creation and storage:

- **extractor.py**: Chunk extraction logic
  - `LeechRead`: Container for read with all features
  - `extract_training_chunks()`: Extract chunks with motif filtering

- **serialization.py**: Chunk I/O
  - `save_chunks()`: Save to compressed .npz format
  - `load_chunks()`: Load from .npz
  - `get_chunk_statistics()`: Compute chunk statistics

#### `splitting/` - Read-Level Data Splitting

Prevents data leakage by splitting at read level:

- **splitter.py**: Splitting operations
  - `split_chunks_by_read()`: Split by read ID (not chunk)
  - `merge_and_split_chunks()`: Merge multiple files, then split
  - `parse_comparison_spec()`: Parse TSV comparison specs
  - `process_comparison_spec()`: Batch process comparisons

#### `data_prep.py` - High-Level Orchestration

Thin orchestration layer that composes functionality from other modules:
- `iter_bam_with_pod5()`: Iterate reads with feature extraction
- `prepare_training_data()`: Sequential data preparation
- `prepare_training_data_parallel()`: Parallel data preparation
- `prepare_training_data_with_split()`: Full pipeline with splitting

**Note**: This module re-exports public APIs from submodules for backward compatibility.

## Data Flow

### 1. Data Preparation Pipeline

```
POD5 + BAM Files
    ↓
[io.bam_reader] → Filter alignments, extract move tables
    ↓
[io.pod5_reader] → Read raw signals (batched for performance)
    ↓
[features] → Normalize signal, compute dwell times & features
    ↓
[chunking.extractor] → Create LeechRead objects
    ↓
[io.motif_search] → Find motif positions (bam or reference)
    ↓
[chunking.extractor] → Extract training chunks
    ↓
[chunking.serialization] → Save to .npz files
    ↓
[splitting.splitter] → Split at read level (train/val/test)
    ↓
Training Chunks (train.npz, val.npz, test.npz)
```

### 2. Training Pipeline

```
Chunk Files (.npz)
    ↓
[dataset.LeechDataset] → Load and preprocess chunks
    ↓
[DataLoader] → Batch chunks
    ↓
[models] → Forward pass through model
    ↓
[training.Trainer] → Training loop with validation
    ↓
Trained Model (.pt)
```

### 3. Inference Pipeline

```
POD5 + BAM + Model
    ↓
[data_prep] → Extract features from reads
    ↓
[inference] → Run model on chunks
    ↓
[inference] → Aggregate predictions
    ↓
Predictions (BAM with modification tags)
```

## Key Design Patterns

### 1. Strategy Pattern - Motif Search

Different strategies for finding motifs without code duplication:

```python
# Basecalled search (original)
searcher = BasecalledMotifSearcher()

# Reference search (avoids basecalling errors)
searcher = ReferenceMotifSearcher(references, skip_indels=True)

# Use the searcher (same interface)
positions = searcher.find_motif_positions(read_id, sequence, alignment, motif)
```

### 2. Context Managers - Resource Management

Automatic cleanup of file handles:

```python
# BAM reading
with BAMReader(bam_path) as reader:
    for aln in reader.iter_alignments():
        # ... process alignment

# POD5 reading with batching
with POD5Reader(pod5_path) as reader:
    signal, meta = reader.get_signal(read_id)
```

### 3. Factory Pattern - Configuration

Type-safe configuration with validation:

```python
# Create config with validation
config = DataPrepConfig(
    pod5_path=Path("reads.pod5"),
    bam_path=Path("alignments.bam"),
    output_dir=Path("chunks/"),
    motif="CCAGGC",
    workers=8
)

# Validation happens automatically
config.validate_splits()  # Raises if train_split + val_split > 1.0
```

### 4. Lazy Loading - Reference Manager

References loaded only when needed:

```python
manager = ReferenceManager(bam_path, fasta_path)
# No files opened yet

seq = manager.get_sequence("chr1")  # Loads on first access
# Subsequent calls use cached data
```

## Configuration Management

### Pydantic Configuration Models

All configuration is managed through Pydantic models in `config.py`:

```python
from leech.config import DataPrepConfig, TrainingConfig

# Data preparation config
data_config = DataPrepConfig(
    pod5_path=Path("reads.pod5"),
    bam_path=Path("alignments.bam"),
    output_dir=Path("chunks/"),
    motif="CCAGGC",
    motif_offset=0,
    motif_reference="fasta",  # Use reference-based search
    workers=8,
    chunk_size=100
)

# Training config
train_config = TrainingConfig(
    train_data_path=Path("train.npz"),
    val_data_path=Path("val.npz"),
    output_dir=Path("models/"),
    model=ModelConfig(
        model_name="ConvLSTMDwell",
        signal_len=400,
        kmer_len=11,
        num_features=9
    ),
    epochs=50,
    batch_size=128,
    learning_rate=0.001,
    device="cuda"
)
```

### Benefits of Pydantic Configs

1. **Type Safety**: Runtime validation of types and constraints
2. **Self-Documenting**: Field descriptions serve as documentation
3. **Serialization**: Easy JSON/YAML serialization
4. **Validation**: Custom validators for complex constraints
5. **IDE Support**: Autocomplete and type hints

## Performance Optimizations

### 1. Batched POD5 Reading

Instead of opening POD5 for each read:

```python
# Old: Open POD5 N times
for read_id in read_ids:
    with DatasetReader(pod5_path) as reader:
        signal = reader.reads([read_id])

# New: Open POD5 once
with POD5Reader(pod5_path) as reader:
    signals = reader.get_signals_batch(read_ids)
```

### 2. Parallel Processing

Two-pass parallel data preparation:

```python
# Pass 1: Collect lightweight read metadata (fast, sequential)
read_infos = collect_read_infos(bam_path)

# Pass 2: Process reads in parallel (CPU-intensive)
with multiprocessing.Pool(workers=8) as pool:
    chunks = pool.imap_unordered(process_chunk, read_batches)
```

### 3. Memory-Efficient Merging

Merge large datasets without loading all into memory:

```python
# First pass: Collect only read IDs for split assignment
all_read_ids = set()
for chunk_file in chunk_files:
    with np.load(chunk_file) as data:
        all_read_ids.update(data["read_ids"])

# Determine splits
train_ids, val_ids, test_ids = split_read_ids(all_read_ids)

# Second pass: Load and assign chunks one file at a time
for chunk_file in chunk_files:
    chunks = load_chunks(chunk_file)
    # Assign to appropriate split
    # Free memory before next file
```

## Error Handling

### Validation Errors

Pydantic models provide clear validation errors:

```python
try:
    config = DataPrepConfig(
        pod5_path="reads.pod5",
        train_split=0.8,
        val_split=0.3  # Invalid: sum > 1.0
    )
except ValidationError as e:
    print(e)
    # Clear error message about constraint violation
```

### Missing Data

Graceful handling of missing reads:

```python
# Missing POD5 reads logged, not fatal
signals = read_pod5_signals_batch(pod5_path, read_ids)
missing = set(read_ids) - set(signals.keys())
if missing:
    logger.warning(f"Could not find {len(missing)} reads in POD5")
```

## Testing Strategy

### Unit Tests

Each module has focused unit tests:

- `test_io_*.py`: Test I/O operations in isolation
- `test_chunking.py`: Test chunk extraction and serialization
- `test_splitting.py`: Test read-level splitting

### Integration Tests

End-to-end pipeline tests:

- `test_data_prep.py`: Full data preparation pipeline
- `test_training.py`: Training with real chunks

### Test Fixtures

Shared fixtures in `conftest.py`:

```python
@pytest.fixture
def mock_bam():
    """Create minimal BAM for testing."""
    # ... create test BAM with mv tags

@pytest.fixture
def mock_pod5():
    """Create minimal POD5 for testing."""
    # ... create test POD5
```

## Migration Guide

### For Existing Code

The refactoring maintains backward compatibility:

```python
# Old imports still work
from leech.data_prep import LeechRead, save_chunks, load_chunks

# New imports also work
from leech.chunking import LeechRead, save_chunks, load_chunks
from leech.io import POD5Reader, get_motif_searcher
```

### Recommended New Patterns

```python
# Use new I/O classes for better performance
from leech.io import POD5Reader, BAMReader, ReferenceManager

with POD5Reader(pod5_path) as pod5:
    with BAMReader(bam_path) as bam:
        for aln in bam.iter_alignments():
            signal, meta = pod5.get_signal(aln.query_name)

# Use Pydantic configs
from leech.config import DataPrepConfig

config = DataPrepConfig.model_validate(config_dict)
# or
config = DataPrepConfig(**kwargs)
```

## Future Enhancements

### Easy Additions (enabled by refactoring)

1. **New motif search strategies**: Just subclass `MotifSearcher`
2. **New feature types**: Add to `features.py` without touching I/O
3. **Alternative storage formats**: Implement new serializer in `chunking/`
4. **Different splitting strategies**: Add to `splitting/splitter.py`

### Performance Improvements

1. **Async I/O**: Replace synchronous POD5/BAM reading with asyncio
2. **Chunked processing**: Process data in streaming fashion
3. **Caching layer**: Add Redis/disk cache for frequently accessed reads

## See Also

- [Data Preparation Guide](data_preparation.md)
- [Model Training Guide](model_training.md)
- [API Reference](api_reference.md)
- [Performance Tuning](performance_tuning.md)
