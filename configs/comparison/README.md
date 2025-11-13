# Model Comparison Configs

This directory contains configuration files for comparing different model architectures and training strategies for tRNA aminoacylation classification.

## Comparison Variants

### 1. `base.yaml` - Baseline (Control)
**Model**: ConvLSTMBase
**Branches**: Signal + Sequence only
**Purpose**: Measure contribution of dwell/level features

- No dwell time or signal level features
- Establishes performance floor
- Expected to underperform feature-based models

### 2. `dwell.yaml` - Full Model (Standard)
**Model**: ConvLSTMDwell
**Branches**: Signal + Sequence + Features
**Purpose**: Standard three-branch architecture

- Uses all available information
- Potential sequence overfitting for constant motifs
- General-purpose architecture

### 3. `tcn_signal_features.yaml` - Optimized (Recommended for tRNA)
**Model**: TCNSignalFeatures
**Branches**: Signal + Features only
**Purpose**: Best architecture for constant-sequence applications

- No sequence branch (avoids overfitting to CCAGGC)
- Based on best-performing TCN architecture
- Fewer parameters, faster training
- **Recommended for tRNA aminoacylation**

### 4. `dwell_masked.yaml` - Augmented
**Model**: ConvLSTMDwell with sequence masking
**Branches**: Signal + Sequence + Features
**Purpose**: Test sequence masking effectiveness

- Randomly replaces sequences (50% probability)
- Forces model to ignore sequence
- Alternative approach to signal-features models

## Usage

### With leech CLI

```bash
# Train using a config
uv run leech train \
  --train-data chunks/train.npz \
  --val-data chunks/val.npz \
  --model TCNSignalFeatures \
  --model-config configs/comparison/tcn_signal_features.yaml \
  --output-dir models/tcn_signal_features/
```

### With Snakemake Workflow

The Snakemake workflow in `pipeline/` automates comparison across all variants:

```bash
# Run comparison workflow
snakemake --cores 4 --use-conda \
  --configfile configs/comparison_experiment.yaml \
  comparison_report
```

## Expected Performance Ranking

For **constant-sequence applications** (e.g., tRNA with CCAGGC motif):

1. **TCNSignalFeatures** - Optimal (no sequence overfitting)
2. **Dwell Masked** - Good (sequence masking reduces overfitting)
3. **Dwell** - Fair (may overfit to constant sequence)
4. **Base** - Baseline (lacks discriminative features)

For **variable-sequence applications**:

1. **Dwell** - Optimal (uses sequence context)
2. **Dwell Masked** - Good (some sequence information retained)
3. **Base** - Fair (lacks features)
4. **TCNSignalFeatures** - Not applicable (no sequence input)

## Metrics to Compare

- **Accuracy**: Overall correctness
- **Precision/Recall**: Class-specific performance
- **F1 Score**: Balanced metric
- **ROC-AUC**: Discrimination ability
- **Training time**: Computational efficiency
- **Parameter count**: Model complexity
