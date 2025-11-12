# Data Preparation Modules

Data loading, feature extraction, and training chunk preparation.

## Overview

The data preparation functionality has been refactored into modular components for better maintainability and testing. The previous monolithic `data_prep.py` module has been split into:

- **`leech.io`** - Input/output operations (BAM/POD5 reading, motif search, reference handling)
- **`leech.preparation`** - Data preparation orchestration and parallel processing
- **`leech.chunking`** - Training chunk extraction and serialization
- **`leech.splitting`** - Train/val/test data splitting
- **`leech.commands`** - CLI command implementations

## I/O Module (`leech.io`)

### BAM Reading

Iterator for reading BAM alignments with filtering.

::: leech.io.bam_reader.iter_bam_alignments
    options:
      show_root_heading: true
      show_source: false

### POD5 Reading

Read raw signal from POD5 files.

::: leech.io.pod5_reader.read_pod5_signal
    options:
      show_root_heading: true
      show_source: false

### Motif Search

Base class for motif search strategies.

::: leech.io.motif_search.MotifSearcher
    options:
      show_root_heading: true
      show_source: false
      members:
        - find_motif_positions

## Chunking Module (`leech.chunking`)

### LeechRead

Container for a single read's data with all features.

::: leech.chunking.extractor.LeechRead
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - get_chunk

### Extract Training Chunks

Extract training chunks centered on motifs.

::: leech.chunking.extractor.extract_training_chunks
    options:
      show_root_heading: true
      show_source: false

### Chunk Serialization

Save and load training chunks.

::: leech.chunking.serialization.save_chunks
    options:
      show_root_heading: true
      show_source: false

::: leech.chunking.serialization.load_chunks
    options:
      show_root_heading: true
      show_source: false

## Preparation Module (`leech.preparation`)

### Sequential Preparation

Main data preparation function (sequential).

::: leech.preparation.orchestrator.prepare_training_data
    options:
      show_root_heading: true
      show_source: false

### Parallel Preparation

Parallel data preparation for large datasets.

::: leech.preparation.parallel.prepare_training_data_parallel
    options:
      show_root_heading: true
      show_source: false

## Splitting Module (`leech.splitting`)

### Split by Read

Split data into train/val/test sets at the read level to prevent data leakage.

::: leech.splitting.splitter.split_chunks_by_read
    options:
      show_root_heading: true
      show_source: false

## Usage

For most users, the CLI commands provide the easiest interface:

```bash
# Prepare data
uv run leech prepare --pod5 reads.pod5 --bam alignments.bam --output-dir chunks/

# Merge and split (multi-sample)
uv run leech merge-and-split -i charged=a.npz -i uncharged=b.npz -o merged/
```

For programmatic access, import the specific modules you need.
