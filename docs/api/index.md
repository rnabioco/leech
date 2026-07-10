# API Reference

Auto-generated reference for leech's Python API. For command-line usage,
see the [CLI Reference](../reference/cli.md).

## Modules

### Data handling

- **[Data Preparation](data_prep.md)** -- BAM/POD5 reading, chunk extraction, splitting
- **[Features](features.md)** -- Move table parsing, dwell time and signal feature computation
- **[Dataset](dataset.md)** -- PyTorch Dataset classes for training

### Models and training

- **[Models](models.md)** -- Neural network architectures (ConvLSTMDwell, TransformerDwell, etc.)
- **[Training](training.md)** -- Trainer class and training loop
- **[Evaluation](evaluation.md)** -- Model evaluation and metrics
- **[Inference](inference.md)** -- Inference engine and bundle inference
- **[Grid Search](gridsearch.md)** -- Hyperparameter optimization

### Utilities

- **[Utilities](util.md)** -- Model loading, checkpoint handling, bundle creation

## Quick examples

### Extract dwell times from a BAM record

```python
from leech.features import extract_move_table, compute_dwell_times

move_table = extract_move_table(alignment)
dwell_times = compute_dwell_times(move_table)
```

### Load and run a model

```python
from leech.model_loading import load_model_from_checkpoint

model, config = load_model_from_checkpoint("model_best.pt", device="cuda")
```

### Load a model from a bundle

```python
from leech.bundling import load_model_from_bundle

model, config = load_model_from_bundle("bundle.pt", pair="Ala_vs_Gly", device="cuda")
```
