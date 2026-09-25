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

Leech computes a per-base feature channel for each base in the k-mer context
window; which channels a run extracts, and in what order, is a recorded
setting (`--feature-channels`, see below). The default list -- unchanged
since before that flag existed -- is 9 channels when no k-mer level table is
configured (12 with one, adding the three `kmer_*` residual rows below),
stacked into a feature matrix of shape `(F, kmer_len)`:

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
| `kmer_expected`| Expected signal level from the k-mer table   | Signal + moves + k-mer table |
| `kmer_residual`| `level_mean - kmer_expected` (signed)        | Signal + moves + k-mer table |
| `kmer_residual_abs`| absolute value of `kmer_residual`        | Signal + moves + k-mer table |

Two more channels are available but **not** in the default list -- opt in
with `--feature-channels` for a model that needs the tail of the per-base
signal distribution rather than just its first two moments:

| Channel          | Description                                                            | Source          |
|------------------|-------------------------------------------------------------------------|-----------------|
| `level_skew`     | Population skewness of signal amplitude (`scipy.stats.skew` defaults)   | Signal + moves  |
| `level_kurtosis` | Fisher excess kurtosis of signal amplitude (`scipy.stats.kurtosis` defaults) | Signal + moves |

Both read `0.0`, not `NaN`, for a degenerate (constant or single-sample) span
-- the same clamp `level_std` already has.

The dwell features capture translocation kinetics, while the level features
capture signal amplitude statistics. Together, they provide a compact
per-base summary that complements the raw signal.

### Choosing the channel list

`leech data prepare --feature-channels dwell,dwell_log,...` selects and
orders the feature rows a corpus is prepared with -- the row order *is* the
model's input channel order, so it is recorded in `prepare_config.json` and
carried into the trained checkpoint's `config.json`; `leech predict` reads it
back and extracts exactly that list rather than guessing from a flag at
predict time. Omitting `--feature-channels` uses the default list above, so
an existing corpus or checkpoint is unaffected by this option's existence.
See [`--feature-channels`](../reference/cli.md) in the CLI reference for the
full option description.

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

```bash title="Bash" linenums="1"
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
- **The classification task is subtle** -- fine-grained multi-class
  discrimination (e.g., amino acid identity in the tRNA charging example)
  relies on small dwell differences that benefit from direct feature access
- **The region of interest has a strong, localized dwell shift** -- for
  example, the CCA tail in tRNA charging, where aminoacylation produces the
  strongest dwell signal, and explicit features help the model focus there
