# ADR 0004: Inference Wrapper Pattern

**Status:** Accepted

**Date:** 2025-01-08

## Context

Models in leech have different input signatures:

**Baseline model (ConvLSTMBase)**:
```python title="Python" linenums="1"
logits = model(signal, sequence)
```

**Models with dwell features** (ConvLSTMDwell, TransformerDwell, ConvOnly, TCNDwell, ResNetDwell):
```python title="Python" linenums="1"
logits = model(signal, sequence, features)
```

This led to conditional logic appearing in 4 locations:

1. **training.py** - `Trainer.train_epoch()`:
   ```python
   if "features" in batch:
       logits = model(signal, sequence, features)
   else:
       logits = model(signal, sequence)
   ```

2. **training.py** - `Trainer.validate()`: Same conditional

3. **evaluation.py** - `evaluate_model()`: Same conditional

4. **inference.py** - `run_inference()`: Similar conditional with model type check

This violated DRY and required each new feature to update 4 locations.

## Decision

Create `ModelInferenceWrapper` in `src/leech/models/inference_wrapper.py`:

```python title="Python" linenums="1"
class ModelInferenceWrapper:
    """Unified forward pass interface for all model types."""

    FEATURE_MODELS = {
        "ConvLSTMDwell", "TransformerDwell", "ConvOnly",
        "TCNDwell", "ResNetDwell"
    }

    def __init__(self, model: nn.Module, model_type: str):
        self.model = model
        self.requires_features = model_type in self.FEATURE_MODELS

    def forward_batch(self, batch: dict, device: str) -> torch.Tensor:
        """Forward pass from batch dict, handling tensor movement and model call."""
        signal = batch["signal"].to(device)
        sequence = batch["sequence"].to(device)

        if self.requires_features:
            features = batch["features"].to(device)
            return self.model(signal, sequence, features)
        else:
            return self.model(signal, sequence)
```

### Usage

**Before**:
```python title="Python" linenums="1"
# Duplicated in 4 places
if "features" in batch:
    features = batch["features"].to(device)
    logits = model(signal, sequence, features)
else:
    logits = model(signal, sequence)
```

**After**:
```python title="Python" linenums="1"
# Single line
wrapper = ModelInferenceWrapper(model, model_type)
logits = wrapper.forward_batch(batch, device)
```

## Consequences

### Positive

- **Single source of truth**: One place defines which models require features
- **Code reduction**: ~15 lines of duplicate conditional logic eliminated
- **Maintainability**: Adding a new feature model only requires updating `FEATURE_MODELS` set
- **Clarity**: Explicit declaration of model input requirements
- **Testability**: Can test wrapper independently

### Negative

- **Indirection**: One more layer between caller and model
- **Memory**: Wrapper instance created for each model (minimal overhead)

### Neutral

- Wrapper delegates `train()`, `eval()`, `to()`, `parameters()` to underlying model
- Wrapper can be used as drop-in replacement in most contexts via `__call__`

## Design Principles

1. **Single Responsibility**: Wrapper handles only input signature differences
2. **Dependency Inversion**: High-level code (training, inference) depends on wrapper abstraction, not concrete model signatures
3. **Open/Closed**: Adding new model types only requires updating `FEATURE_MODELS` set

## Implementation Details

### Integration Points

1. **training.py**: `Trainer.__init__()` wraps model
   - `train_epoch()` uses `wrapper.forward_batch()`
   - `validate()` uses `wrapper.forward_batch()`

2. **evaluation.py**: `evaluate_model()` wraps model
   - Evaluation loop uses `wrapper.forward_batch()`

3. **inference.py**: `run_inference()` wraps model
   - Uses `wrapper.requires_features` to conditionally add features to batch dict
   - Calls `wrapper.forward_batch()`

### Model Type Detection

Model type comes from:
- Training: CLI argument (`--model ConvLSTMDwell`)
- Evaluation/Inference: Loaded from checkpoint config (`config["model_name"]`)

## Alternatives Considered

1. **Unified model signature**: Make all models take optional `features=None`
   - Rejected: Requires changing all 6 model forward methods
   - Less explicit about requirements

2. **Protocol/ABC with multiple implementations**: Over-engineering for simple problem

3. **Keep conditionals**: Rejected due to duplication

4. **Batch dict with model dispatch**: Similar to wrapper but less explicit

## Testing Strategy

```python title="Python" linenums="1"
def test_wrapper_with_base_model():
    model = ConvLSTMBase()
    wrapper = ModelInferenceWrapper(model, "ConvLSTMBase")
    assert wrapper.requires_features == False

def test_wrapper_with_dwell_model():
    model = ConvLSTMDwell()
    wrapper = ModelInferenceWrapper(model, "ConvLSTMDwell")
    assert wrapper.requires_features == True
```

## Notes

This ADR implements Task 7 from the refactoring plan (Phase 2 high-priority refactoring). Complements ADR 0003 (Model Component Abstraction) by addressing runtime behavior rather than initialization.
