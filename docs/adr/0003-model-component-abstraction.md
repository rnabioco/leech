# ADR 0003: Model Component Abstraction

**Status:** Accepted

**Date:** 2025-01-08

## Context

The leech package contains 6 neural network architectures for RNA modification detection:
- `ConvLSTMBase`: Signal + sequence branches (baseline)
- `ConvLSTMDwell`: Signal + sequence + features branches (full model)
- `TransformerDwell`: Transformer-based architecture
- `ConvOnly`: Pure CNN baseline
- `TCNDwell`: Temporal Convolutional Network
- `ResNetDwell`: Residual Network

Each model had duplicated code:

1. **Duplicate branch architectures**: ConvLSTMBase and ConvLSTMDwell both implemented identical 3-layer Conv1d branches for signal and sequence processing (36 lines duplicated)

2. **Duplicate predict_proba()**: All 6 models had identical 15-line `predict_proba()` methods for converting logits to probabilities

3. **No code reuse**: Adding a new model required re-implementing standard branches

This violated DRY (Don't Repeat Yourself) and made maintenance difficult.

## Decision

Create a component library in `src/leech/models/components.py`:

### 1. Reusable Branch Components

```python
class SignalBranch(nn.Module):
    """Standardized 3-layer Conv1d for raw signal processing"""

class SequenceBranch(nn.Module):
    """Standardized 3-layer Conv1d for one-hot encoded sequences"""

class FeatureBranch(nn.Module):
    """Standardized 3-layer Conv1d for engineered features"""
```

All three follow the same pattern:
- Layer 1: in_channels → conv_channels[0]
- Layer 2: conv_channels[0] → conv_channels[1]
- Layer 3: conv_channels[1] → conv_channels[2]
- ReLU activations between layers

### 2. BaseModel Class

```python
class BaseModel(nn.Module):
    """Base class with shared predict_proba() implementation"""
```

All models inherit from `BaseModel` instead of `nn.Module`.

### 3. Refactored Models

- **ConvLSTMDwell**: Uses `SignalBranch`, `SequenceBranch`, `FeatureBranch` (reduced from 185→115 lines, 38% reduction)
- **ConvLSTMBase**: Uses `SignalBranch`, `SequenceBranch` (reduced from 163→95 lines, 42% reduction)
- **Other models**: Inherit from `BaseModel` for shared `predict_proba()` (14 lines saved each)

Specialized architectures (Transformer, TCN, ResNet, ConvOnly) keep their unique structures but eliminate `predict_proba()` duplication.

## Consequences

### Positive

- **Code reduction**: ~200 lines eliminated across 6 models
- **Maintainability**: Bug fixes to branch components benefit all models
- **Consistency**: All models using standard branches have identical implementations
- **Extensibility**: New models can reuse standard branches
- **Testing**: Can test branches independently of full models
- **Clear architecture**: Separation of concerns (branch vs. fusion logic)

### Negative

- **Indirection**: Reading model code requires checking components.py
- **Flexibility trade-off**: Models using standard branches lose some architectural freedom

### Neutral

- Standard branches are configurable (kernel sizes, channel counts via constants)
- Specialized architectures keep custom implementations

## Design Principles

1. **Composition over inheritance**: Models compose branch instances rather than inherit branch logic
2. **Single Responsibility**: Each component has one job (signal processing, sequence processing, feature processing)
3. **Open/Closed Principle**: Branches are open for extension (configurable) but closed for modification

## Alternatives Considered

1. **Full model inheritance hierarchy**: Rejected due to complexity and inflexibility
2. **Mixins**: Rejected in favor of simpler composition
3. **Keep duplication**: Rejected due to maintenance burden

## Migration Path

Existing checkpoints remain compatible:
- Models save `state_dict()` which is unchanged
- Component refactoring only affects initialization code
- Load/inference unchanged

## Notes

This ADR implements Task 6 from the refactoring plan (Phase 2 high-priority refactoring). The 38-42% code reduction in ConvLSTM models demonstrates significant impact.
