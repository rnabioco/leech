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

### BamReader

Reads BAM files and extracts alignment information with move table tags.

::: leech.io.bam_reader.BamReader
    options:
      show_root_heading: true
      show_source: false

### POD5Reader

Reads raw signal from POD5 files.

::: leech.io.pod5_reader.POD5Reader
    options:
      show_root_heading: true
      show_source: false

### MotifSearcher

Searches for motifs in reference or basecalled sequences.

::: leech.io.motif_search.MotifSearcher
    options:
      show_root_heading: true
      show_source: false

## Chunking Module (`leech.chunking`)

### ChunkExtractor

Extracts training chunks centered on motifs.

::: leech.chunking.extractor.ChunkExtractor
    options:
      show_root_heading: true
      show_source: false

### Serialization

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

### Orchestrator

Main data preparation orchestration.

::: leech.preparation.orchestrator.prepare_chunks
    options:
      show_root_heading: true
      show_source: false

### Parallel Processing

Parallel data preparation for large datasets.

::: leech.preparation.parallel.prepare_chunks_parallel
    options:
      show_root_heading: true
      show_source: false

## Splitting Module (`leech.splitting`)

### DataSplitter

Split data into train/val/test sets at the read level.

::: leech.splitting.splitter.DataSplitter
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
