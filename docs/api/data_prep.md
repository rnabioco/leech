# Data Preparation Module

Data loading, feature extraction, and training chunk preparation.

## Overview

The data_prep module handles reading BAM and POD5 files together, extracting features, and preparing training chunks.

## Key Classes

::: leech.data_prep.LeechRead
    options:
      show_root_heading: true
      show_source: true

## Key Functions

::: leech.data_prep.iter_bam_with_pod5
    options:
      show_root_heading: true
      show_source: true

::: leech.data_prep.prepare_training_data
    options:
      show_root_heading: true
      show_source: true

::: leech.data_prep.prepare_training_data_parallel
    options:
      show_root_heading: true
      show_source: true

::: leech.data_prep.collect_read_infos
    options:
      show_root_heading: true
      show_source: true

## Chunk Serialization

::: leech.data_prep.save_chunks
    options:
      show_root_heading: true
      show_source: true

::: leech.data_prep.load_chunks
    options:
      show_root_heading: true
      show_source: true
