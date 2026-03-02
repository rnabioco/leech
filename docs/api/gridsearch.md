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

```python title="Python" linenums="1"
from leech.gridsearch import GridSearchConfig, run_grid_search
from pathlib import Path

# Create grid search config
config = GridSearchConfig(
    train_data_path=Path("chunks/train.npz"),
    val_data_path=Path("chunks/val.npz"),
    model_name="ConvLSTMDwell",
    left_contexts=[200, 500, 1000, 2000],
    right_contexts=[200, 500, 1000, 2000],
    output_dir=Path("grid_search_results/"),
    n_parallel=4,  # Train 4 grid points concurrently
)

# Run grid search
results_path = run_grid_search(config)

print(f"Grid search complete. Results saved to: {results_path}")
```

For more details, see the [Grid Search Guide](../grid-search/grid-search-usage.md).
