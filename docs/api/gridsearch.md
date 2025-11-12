# Grid Search Module

Hyperparameter optimization for leech models.

## Overview

The gridsearch module provides utilities for optimizing chunk context parameters.

::: leech.gridsearch.GridSearchConfig
    options:
      show_root_heading: true
      show_source: true

::: leech.gridsearch.run_grid_search
    options:
      show_root_heading: true
      show_source: true

::: leech.gridsearch.parse_context_grid
    options:
      show_root_heading: true
      show_source: true

## Example Usage

```python
from leech.gridsearch import GridSearchConfig, run_grid_search
from pathlib import Path

# Create grid search config
config = GridSearchConfig(
    pod5=Path("reads.pod5"),
    bam=Path("alignments.bam"),
    model="ConvLSTMDwell",
    signal_context_grid="150,200,250",
    kmer_context_grid="3,5,7",
    output_dir=Path("grid_search_results/")
)

# Run grid search
results_path = run_grid_search(config)

print(f"Grid search complete. Results saved to: {results_path}")
```

For more details, see the [Grid Search Guide](../grid-search/grid-search-usage.md).
