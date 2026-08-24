# Dataset Module

PyTorch Dataset classes for loading training data.

## Overview

The dataset module provides PyTorch Dataset implementations for efficient data loading during training.

::: leech.dataset.LeechDataset
    options:
      show_root_heading: true
      show_source: true

## Data Collation

::: leech.dataset.collate_fn
    options:
      show_root_heading: true
      show_source: true

## DataLoader Sizing

Every leech `DataLoader` -- training, validation and evaluation -- gets its
worker count from this one function, so the rules (auto on GPU, serial on CPU,
never workers inside a daemonic pool worker) cannot drift between call sites.

::: leech.dataset.resolve_dataloader_workers
    options:
      show_root_heading: true
      show_source: true
