# Models Module

Neural network architectures for nanopore signal classification.

## Overview

The models module contains 29 PyTorch model architectures across 5 families. All models accept a `signal_in_channels` parameter for multi-channel signal input (default: 1).

Architectures come from two places, both reachable via `get_model()` and `MODEL_REGISTRY`:

- **TOML configs** in `leech/models/configs/` — the ConvLSTM and TCN families
  (22 of the 29 names). Each config is a fully declarative `kind = "graph"`
  architecture built from the layer registry in `leech.models.nn`, with one
  `[[variants]]` entry per registry name. Adding a normalization/pooling
  variant is a `[[variants]]` entry, not a new class.
- **Hand-written classes** for architectures not yet converted (Remora-compatible
  ConvLSTM, Transformer, ConvOnly, ResNet, SignalCNN).

Discovery is torch-free, so listing model names (e.g. for CLI `--model` choices)
does not pay the torch import cost.

## Model Registry

::: leech.models.get_model
    options:
      show_root_heading: true
      show_source: true

## Config-Driven Architectures (ConvLSTM and TCN)

The ConvLSTM and TCN families are declared in TOML configs, not Python
classes. `build_model_class()` turns each declaration into a real class on
first access, so `get_model("ConvLSTMDwell")`, `isinstance()` checks, and
checkpoint loading all behave as if the class were hand-written.

| Registry names | Config |
|---|---|
| `ConvLSTMBase`, `ConvLSTMBaseBN`, `ConvLSTMDwell`, `ConvLSTMDwellBN` | `configs/conv_lstm.toml` |
| `ConvLSTMBaseAttn`, `ConvLSTMBaseBNAttn`, `ConvLSTMDwellAttn`, `ConvLSTMDwellBNAttn`, `ConvLSTMDwellGNAttn`, `ConvLSTMDwellLNAttn` | `configs/conv_lstm_attn.toml` |
| `TCNDwell`, `TCNDwellGN`, `TCNDwellLN` | `configs/tcn_dwell.toml` |
| `TCNDwellResidual`, `TCNDwellResidualGN`, `TCNDwellResidualLN` | `configs/tcn_dwell_residual.toml` |
| `TCNDwellResidualMotor`, `TCNDwellResidualLNMotor` | `configs/tcn_dwell_residual_motor.toml` |
| `TCNDwellResidualDwellAttn`, `TCNDwellResidualLNDwellAttn` | `configs/tcn_dwell_residual_dwell_attn.toml` |
| `TCNDwellSplitResidual`, `TCNDwellSplitResidualLN` | `configs/tcn_dwell_split_residual.toml` |

**`ConvLSTMDwell`** (multi-branch Conv-LSTM with dwell time features) is the
recommended default. It has three branches — signal (Conv1d on raw signal),
sequence (Conv1d on one-hot k-mers), and features (Conv1d on dwell + level
statistics) — merged into a BiLSTM followed by a fully connected head.
**`ConvLSTMBase`** is the same architecture without the feature branch; compare
the two to measure the impact of dwell features.

The TCN family replaces the BiLSTM with stacks of dilated causal convolutions.
`Residual` variants take a 2-channel signal input (raw + k-mer model residual),
`SplitResidual` keeps separate branches for the raw signal and the residual,
`Motor` adds motor-region pooling, and `DwellAttn` adds dwell-only
cross-attention.

### Config Loader

::: leech.models.config_loader
    options:
      show_root_heading: true
      show_source: false
      members:
        - discover_configs
        - build_model_class

### Layer Registry

::: leech.models.nn
    options:
      show_root_heading: true
      show_source: false
      members:
        - register
        - to_dict
        - from_dict
        - Serial
        - Stack
        - Parallel
        - Graph

## Remora-compatible Architectures

::: leech.models.conv_lstm_remora.ConvLSTMRemora
    options:
      show_root_heading: true
      show_source: false

::: leech.models.conv_lstm_remora.ConvLSTMRemoraBase
    options:
      show_root_heading: true
      show_source: false

## Transformer Architectures

### TransformerDwell

Transformer with multi-head self-attention and dwell features.

::: leech.models.transformer_dwell.TransformerDwell
    options:
      show_root_heading: true
      show_source: false

### TransformerDwellResidual

Transformer with 2-channel signal input (raw + kmer residual).

::: leech.models.transformer_dwell.TransformerDwellResidual
    options:
      show_root_heading: true
      show_source: false

## Convolutional Architectures

### ConvOnly

Pure convolutional network with multi-scale convolutions.

::: leech.models.conv_only.ConvOnly
    options:
      show_root_heading: true
      show_source: false

### ResNetDwell

Deep residual network with skip connections.

::: leech.models.resnet_dwell.ResNetDwell
    options:
      show_root_heading: true
      show_source: false

### SignalCNN

Signal-only 1D-CNN classifier (ignores sequence and dwell inputs).

::: leech.models.signal_cnn.SignalCNN
    options:
      show_root_heading: true
      show_source: false

## Inference Wrappers

::: leech.models.inference_wrapper.ModelInferenceWrapper
    options:
      show_root_heading: true
      show_source: false

::: leech.models.remora_compat.RemoraModelWrapper
    options:
      show_root_heading: true
      show_source: false
