# Model Selection Guide

This guide helps you choose the appropriate model architecture for your nanopore signal classification task.

## Quick Decision Tree

```
Is your sequence context CONSTANT across all training examples?
│
├─ YES (e.g., tRNA with fixed CCAGGC motif)
│   └─ Use: TCNSignalFeatures or ConvLSTMSignalFeatures
│       • No sequence overfitting risk
│       • Faster training
│       • Optimal for constant-sequence applications
│
└─ NO (sequence varies across examples)
    └─ Use: TCNDwell or ConvLSTMDwell
        • Leverages sequence context information
        • Best for variable-sequence applications
        • Optional: Add sequence masking (--mask-sequence-prob 0.5) if sequences are partially constant
```

## Available Models

### Signal + Features Models (Constant Sequences)

#### TCNSignalFeatures ⭐ **RECOMMENDED FOR tRNA**
- **Architecture**: Temporal Convolutional Network with dilated convolutions
- **Branches**: Signal + Features only
- **Best for**: Constant-sequence applications (e.g., tRNA aminoacylation)
- **Pros**:
  - Best performance in initial testing
  - No sequence overfitting
  - Faster training (fewer parameters)
  - Large receptive field via dilated convolutions
- **Cons**: Cannot use sequence information (not suitable for variable sequences)

```bash
uv run leech train \
  --train-data chunks/train.npz \
  --val-data chunks/val.npz \
  --model TCNSignalFeatures \
  --output-dir models/tcn_signal_features/
```

#### ConvLSTMSignalFeatures
- **Architecture**: Conv + BiLSTM
- **Branches**: Signal + Features only
- **Best for**: Constant-sequence applications
- **Pros**:
  - No sequence overfitting
  - BiLSTM captures long-range dependencies
  - Simpler than TCN
- **Cons**:
  - Slower than TCN
  - Cannot use sequence information

```bash
uv run leech train \
  --train-data chunks/train.npz \
  --val-data chunks/val.npz \
  --model ConvLSTMSignalFeatures \
  --output-dir models/conv_lstm_signal_features/
```

### Full Models (Variable Sequences)

#### TCNDwell ⭐ **BEST OVERALL PERFORMANCE**
- **Architecture**: Temporal Convolutional Network
- **Branches**: Signal + Sequence + Features
- **Best for**: Variable-sequence applications
- **Pros**:
  - Best overall performance (when sequence is informative)
  - Fast training and inference
  - Large receptive field
- **Cons**: May overfit to constant sequences

```bash
uv run leech train \
  --train-data chunks/train.npz \
  --val-data chunks/val.npz \
  --model TCNDwell \
  --output-dir models/tcn_dwell/
```

#### ConvLSTMDwell
- **Architecture**: Conv + BiLSTM
- **Branches**: Signal + Sequence + Features
- **Best for**: General-purpose classification
- **Pros**:
  - Well-tested architecture
  - Captures long-range dependencies with BiLSTM
  - Good default choice
- **Cons**: Slower than TCN

```bash
uv run leech train \
  --train-data chunks/train.npz \
  --val-data chunks/val.npz \
  --model ConvLSTMDwell \
  --output-dir models/conv_lstm_dwell/
```

### Baseline Models

#### ConvLSTMBase
- **Architecture**: Conv + BiLSTM
- **Branches**: Signal + Sequence only (NO features)
- **Best for**: Control experiments to measure feature contribution
- **Use when**: You want to establish a performance baseline

```bash
uv run leech train \
  --train-data chunks/train.npz \
  --val-data chunks/val.npz \
  --model ConvLSTMBase \
  --output-dir models/base/
```

## Data Augmentation: Sequence Masking

If you must use a full model (Signal + Sequence + Features) on constant-sequence data, use **sequence masking** to prevent overfitting:

```bash
uv run leech train \
  --train-data chunks/train.npz \
  --val-data chunks/val.npz \
  --model ConvLSTMDwell \
  --mask-sequence-prob 0.5 \  # Randomize 50% of sequences
  --output-dir models/dwell_masked/
```

**How it works:**
- During training, randomly replaces sequences with random bases
- Forces model to ignore sequence branch
- Keeps architecture flexible for future variable-sequence use

## Experimental Comparison

To compare all approaches on your data:

```bash
# Compare 4 model variants
uv run leech analyze compare \
  -m models/base/ \
  -m models/tcn_signal_features/ \
  -m models/tcn_dwell/ \
  -m models/dwell_masked/ \
  -t chunks/test.npz \
  -o analysis/comparison/
```

See `configs/comparison/README.md` for pre-configured comparison experiments.

## Feature Importance Analysis

Understand which features contribute most to predictions:

```bash
# Compute gradient-based feature importance
uv run leech analyze feature-importance \
  -m models/tcn_signal_features/model_best.pt \
  -t chunks/test.npz \
  -o analysis/feature_importance/
```

## Sequence Ablation Test

Quantify sequence branch contribution:

```bash
# Test performance with/without sequence
uv run leech analyze sequence-ablation \
  -m models/tcn_dwell/model_best.pt \
  -t chunks/test.npz \
  -o analysis/sequence_ablation/
```

## Summary Table

| Model | Signal | Sequence | Features | Best For | Speed | Parameters |
|-------|--------|----------|----------|----------|-------|------------|
| **TCNSignalFeatures** | ✓ | ✗ | ✓ | Constant seq | Fast | Low |
| ConvLSTMSignalFeatures | ✓ | ✗ | ✓ | Constant seq | Medium | Low |
| **TCNDwell** | ✓ | ✓ | ✓ | Variable seq | Fast | High |
| ConvLSTMDwell | ✓ | ✓ | ✓ | General | Medium | High |
| ConvLSTMDwell + Masking | ✓ | ✓* | ✓ | Constant seq | Medium | High |
| ConvLSTMBase | ✓ | ✓ | ✗ | Baseline | Medium | Medium |

*With sequence masking, the sequence branch learns to be ignored

## Recommendations by Use Case

### tRNA Aminoacylation (CCAGGC motif)
**Best Choice**: `TCNSignalFeatures`
- Constant sequence across all examples
- No sequence information needed
- Optimal performance and speed

### Pairwise Amino Acid Classification (variable positions)
**Best Choice**: `TCNDwell`
- Sequence context varies by genomic position
- Sequence is informative
- Best overall performance

### General Base Modification Detection
**Best Choice**: `TCNDwell` or `ConvLSTMDwell`
- Flexible to different sequence contexts
- Well-tested architectures
- Good starting point for new problems

## Additional Resources

- **Model Architectures**: See `src/leech/models/` for implementation details
- **Training Examples**: See `CLAUDE.md` for CLI usage
- **Comparison Configs**: See `configs/comparison/` for experiment templates
- **Snakemake Workflow**: See `pipeline/` for automated comparisons
