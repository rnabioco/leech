# API Reference

Welcome to the leech API reference documentation. This section provides detailed documentation for all public modules, classes, and functions in the leech library.

## Module Overview

### Core Modules

- **[CLI](cli.md)**: Command-line interface
- **[Data Preparation](data_prep.md)**: BAM/POD5 reading and chunk extraction
- **[Features](features.md)**: Dwell time and signal feature extraction
- **[Dataset](dataset.md)**: PyTorch Dataset classes

### Model Modules

- **[Models](models.md)**: Neural network architectures
- **[Training](training.md)**: Training loop and Trainer class
- **[Evaluation](evaluation.md)**: Model evaluation and testing
- **[Inference](inference.md)**: Inference engine

### Utility Modules

- **[Grid Search](gridsearch.md)**: Hyperparameter optimization
- **[Utilities](util.md)**: Helper functions and model loading

## Quick Links

### Key Classes

- `LeechRead` - Container for read features ([data_prep.md](data_prep.md))
- `MoveTable` - Move table parser ([features.md](features.md))
- `ConvLSTMDwell` - Main model architecture ([models.md](models.md))
- `Trainer` - Training orchestration ([training.md](training.md))

### Key Functions

- `iter_bam_with_pod5()` - Main data loading iterator ([data_prep.md](data_prep.md))
- `compute_dwell_times()` - Extract dwell times ([features.md](features.md))
- `prepare_training_data()` - Prepare training chunks ([data_prep.md](data_prep.md))
- `load_model_from_checkpoint()` - Load trained models ([util.md](util.md))

## Usage Examples

### Loading and Processing Data

```python
from leech.data_prep import iter_bam_with_pod5
from leech.features import MoveTable

# Iterate over BAM reads with POD5 signal
for leech_read in iter_bam_with_pod5(
    bam_path="alignments.bam",
    pod5_path="reads.pod5"
):
    # Access features
    signal = leech_read.signal
    dwell_times = leech_read.dwell_times
    sequence = leech_read.sequence
```

### Training a Model

```python
from leech.training import Trainer
from leech.models import get_model

# Get model
model = get_model("ConvLSTMDwell")

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=0.001,
    device="cuda"
)

# Train
history = trainer.train(epochs=50)
```

### Running Inference

```python
from leech.inference import InferenceEngine
from leech.util import load_model_from_checkpoint

# Load model
model = load_model_from_checkpoint("model_best.pt")

# Create inference engine
engine = InferenceEngine(model=model, device="cuda")

# Run inference
predictions = engine.predict_bam(
    bam_path="reads.bam",
    pod5_path="reads.pod5",
    output_path="predictions.bam"
)
```

## Type Hints

All leech modules use Python type hints for better IDE support and type checking. We recommend using [mypy](http://mypy-lang.org/) for static type checking:

```bash
mypy src/leech/
```

## Contributing

When adding new functions or classes, please:

1. Include comprehensive docstrings (Google style)
2. Add type hints for all parameters and return values
3. Include usage examples in docstrings
4. Update this API reference if needed

See [CONTRIBUTING.md](https://github.com/rnabioco/leech/blob/main/CONTRIBUTING.md) for details.
