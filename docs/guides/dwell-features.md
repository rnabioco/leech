# Dwell Time Features

Leech extends [Remora](https://github.com/nanoporetech/remora) by adding a
third feature branch to the model architecture: per-base dwell time and signal
level statistics computed from move tables. This page describes the feature set,
how it integrates into the model, and how the convolutional architecture learns
to use these features.

## How leech extends Remora

Remora uses two input branches for base modification detection:

1. **Signal branch** -- raw nanopore signal processed by Conv1d layers
2. **Sequence branch** -- one-hot encoded k-mer context processed by Conv1d layers

These branches are concatenated and fed through a BiLSTM for classification.
Leech adds a **feature branch** that provides per-base dwell time and signal
level statistics as a third input. This is the novel contribution: by making
dwell information explicitly available as structured features (rather than
relying on the signal branch to implicitly learn timing patterns), the model
gains direct access to translocation kinetics.

## Feature components

Leech computes 9 feature channels for each base in the k-mer context window.
These are stacked into a feature matrix of shape `(9, kmer_len)`:

| Channel       | Description                                   | Source          |
|---------------|-----------------------------------------------|-----------------|
| `dwell`       | Raw dwell time (signal samples per base)      | Move table      |
| `dwell_log`   | Log-transformed dwell time                    | Move table      |
| `dwell_mean`  | Mean dwell in local window                    | Move table      |
| `dwell_std`   | Dwell standard deviation in local window      | Move table      |
| `dwell_ratio` | Ratio of base dwell to local mean             | Move table      |
| `level_mean`  | Mean signal amplitude for this base           | Signal + moves  |
| `level_median`| Median signal amplitude for this base         | Signal + moves  |
| `level_std`   | Signal amplitude standard deviation           | Signal + moves  |
| `level_range` | Signal amplitude range (max - min)            | Signal + moves  |

The dwell features capture translocation kinetics, while the level features
capture signal amplitude statistics. Together, they provide a compact
per-base summary that complements the raw signal.

!!! note
    The `dwell_ratio` feature (base dwell / local mean dwell) normalizes for
    read-level variation in translocation speed, making the feature more
    comparable across reads.

## Three-branch architecture

The `ConvLSTMDwell` model processes the three input types through parallel
convolutional branches before merging them:

```
Signal (1 ch)  →  Conv1d stack  →  256 features
                                        ↘
Sequence (4 ch) → Conv1d stack  →  256 features  →  Concat (768)  →  BiLSTM  →  FC  →  Output
                                        ↗
Features (9 ch) →  Conv1d stack  →  256 features
```

Each branch applies three Conv1d layers (increasing channels: 4 → 16 → 256)
with batch normalization and ReLU activation, followed by adaptive average
pooling. The concatenated 768-dimensional representation feeds a bidirectional
LSTM (768 → 96 × 2 directions), from which the center position is extracted
and passed through fully connected layers (192 → 64 → output).

### Architecture comparison

| Component         | ConvLSTMDwell          | ConvLSTMBase        | Remora            |
|-------------------|------------------------|---------------------|-------------------|
| Signal branch     | Conv1d (1→4→16→256)    | Conv1d (same)       | Conv1d            |
| Sequence branch   | Conv1d (4→4→16→256)    | Conv1d (same)       | Conv1d            |
| Feature branch    | Conv1d (9→4→16→256)    | None                | None              |
| BiLSTM input      | 768                    | 512                 | 512               |
| BiLSTM hidden     | 96 per direction       | 96 per direction    | varies            |
| Center extraction | Yes                    | Yes                 | Yes               |
| Output            | FC 192→64→num_classes  | FC 192→64→num_classes | FC→num_classes  |

The `ConvLSTMBase` model (without the feature branch) serves as a direct
comparison to measure the impact of dwell features.

## Feature learning

A natural question is whether explicit feature engineering (the 9 channels
above) is necessary, or whether the model could learn equivalent
representations from raw signal alone.

### Convolutional receptive fields

Each Conv1d layer with `kernel_size=3` contributes a receptive field of 3 bases.
Three stacked layers yield an effective receptive field of 7 bases. This means
the feature branch automatically learns to combine information across a 7-base
window without explicit windowed statistics.

The pre-computed features (like `dwell_mean` and `dwell_std` over a local
window) provide the model with summary statistics that would otherwise require
the network to learn averaging operations. This gives the model a head start,
particularly with limited training data.

### Grid search for context windows

Leech includes a grid search tool to find the optimal signal and k-mer context
window sizes for a given dataset:

```bash
uv run leech model optimize \
    --train-data chunks/train.npz \
    --val-data chunks/val.npz \
    --context-grid 200,500,1000 \
    --output-dir grid_results/
```

This evaluates model performance across different context window sizes. In
practice, the landscape is often flat (less than 2% difference between
reasonable window sizes), confirming that the convolutional layers effectively
adapt to the available context.

### When features matter most

The feature branch has the largest impact when:

- **Training data is limited** -- explicit features reduce what the model must
  learn from scratch
- **The classification task is subtle** -- amino acid discrimination relies on
  small dwell differences that benefit from direct feature access
- **The region of interest is the CCA tail** -- this is where aminoacylation
  produces the strongest dwell signal, and explicit features help the model
  focus there
