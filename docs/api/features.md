# Features Module

Dwell time extraction and signal feature computation.

## Overview

The features module provides functions for extracting dwell times from move tables and computing signal-level statistics.

## Move Table Parsing

::: leech.features.MoveTable
    options:
      show_root_heading: true
      show_source: true

## Dwell Time Computation

::: leech.features.compute_dwell_times
    options:
      show_root_heading: true
      show_source: true

## Signal Features

::: leech.features.compute_signal_levels
    options:
      show_root_heading: true
      show_source: true

## Signal Normalization

::: leech.features.normalize_read_signal
    options:
      show_root_heading: true
      show_source: true

## Feature Computation

::: leech.features.compute_dwell_features
    options:
      show_root_heading: true
      show_source: true

::: leech.features.compute_signal_features
    options:
      show_root_heading: true
      show_source: true

## Helper Functions

### Levels for Mapped Bases

Fits a per-sequence expected-level array to the per-mapped-base feature grid.
The two counts differ under `anchor="reference"` when an alignment ends in a
non-match CIGAR op; see `LeechRead.num_mapped_bases`.

::: leech.features.levels_for_mapped_bases
    options:
      show_root_heading: true
      show_source: false

::: leech.features.extract_move_table
    options:
      show_root_heading: true
      show_source: true
