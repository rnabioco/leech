# API Reference

Welcome to the leech API reference documentation. This section provides detailed documentation for all public modules, classes, and functions in the leech library.

## Module Overview

### Core Modules

- **[CLI](cli.md)**: Command-line interface
- **[Features](features.md)**: Dwell time and signal feature extraction
- **[Dataset](dataset.md)**: PyTorch Dataset classes

### I/O and Data Preparation

The data preparation functionality has been refactored into modular components:

- **I/O Operations** (`leech.io`): BAM/POD5 reading, motif search, reference handling
- **Preparation** (`leech.preparation`): Data preparation orchestration and parallel processing
- **Chunking** (`leech.chunking`): Training chunk extraction and serialization
- **Splitting** (`leech.splitting`): Train/val/test data splitting
- **Commands** (`leech.commands`): CLI command implementations

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

- `MoveTable` - Move table parser ([features.md](features.md))
- `ConvLSTMDwell` - Main model architecture ([models.md](models.md))
- `Trainer` - Training orchestration ([training.md](training.md))
- `ChunkDataset` - PyTorch dataset for training ([dataset.md](dataset.md))

### Key Functions

- `compute_dwell_times()` - Extract dwell times ([features.md](features.md))
- `normalize_signal()` - Signal normalization ([features.md](features.md))
- `load_model_from_checkpoint()` - Load trained models ([util.md](util.md))
- `train_model()` - High-level training function ([training.md](training.md))

## Usage Examples

### Loading and Processing Data

Use the CLI commands for data preparation:

```bash
# Prepare training data
uv run leech prepare \
  --pod5 reads.pod5 \
  --bam alignments.bam \
  --output-dir chunks/
```

For programmatic access, use the modular components:

```python
from leech.io import BamReader, POD5Reader
from leech.features import MoveTable, compute_dwell_times

# Read BAM and POD5 data
bam_reader = BamReader("alignments.bam")
pod5_reader = POD5Reader("reads.pod5")

# Process reads
for alignment in bam_reader.fetch():
    signal = pod5_reader.get_signal(alignment.query_name)
    move_table = MoveTable.from_bam_tag(alignment.get_tag("mv"))
    dwell_times = compute_dwell_times(move_table, signal)
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
