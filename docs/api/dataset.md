# Dataset Module

PyTorch Dataset classes for loading training data.

## Overview

The dataset module provides PyTorch Dataset implementations for efficient data loading during training.

::: leech.dataset.LeechDataset
    options:
      show_root_heading: true
      show_source: true

## Data Collation

`LeechDataset` also implements `__getitems__`, so a `DataLoader` fetches a whole
batch in one call and gets back an already-collated dict rather than a list of
samples -- one gather per field instead of one slice per sample plus a
`torch.stack`. `collate_fn` passes such a dict through untouched, and still
stacks a list when it gets one (the per-sample path, used for the list-fallback
dataset and for the cross-layer shift/time-mask augmentations, which roll by a
per-sample offset).

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
