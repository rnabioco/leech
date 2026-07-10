# Models Module

Neural network architectures for nanopore signal classification.

## Overview

The models module contains 24 PyTorch model architectures across 5 families. All models accept a `signal_in_channels` parameter for multi-channel signal input (default: 1).

## Model Registry

::: leech.models.get_model
    options:
      show_root_heading: true
      show_source: true

## ConvLSTM Architectures

### ConvLSTMDwell

Multi-branch Conv-LSTM with dwell time features (recommended).

::: leech.models.ConvLSTMDwell
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - forward

### ConvLSTMBase

Baseline Conv-LSTM without dwell features (for comparison).

::: leech.models.ConvLSTMBase
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - forward

### Normalization and Attention Variants

Batch normalization (BN), group normalization (GN), layer normalization (LN), and attention pooling variants:

::: leech.models.ConvLSTMDwellBN
    options:
      show_root_heading: true
      show_source: false

::: leech.models.ConvLSTMDwellAttn
    options:
      show_root_heading: true
      show_source: false

::: leech.models.ConvLSTMDwellBNAttn
    options:
      show_root_heading: true
      show_source: false

::: leech.models.ConvLSTMDwellGNAttn
    options:
      show_root_heading: true
      show_source: false

::: leech.models.ConvLSTMDwellLNAttn
    options:
      show_root_heading: true
      show_source: false

::: leech.models.ConvLSTMBaseBN
    options:
      show_root_heading: true
      show_source: false

::: leech.models.ConvLSTMBaseAttn
    options:
      show_root_heading: true
      show_source: false

::: leech.models.ConvLSTMBaseBNAttn
    options:
      show_root_heading: true
      show_source: false

### Remora-compatible

::: leech.models.ConvLSTMRemora
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

::: leech.models.TransformerDwell
    options:
      show_root_heading: true
      show_source: false

### TransformerDwellResidual

Transformer with 2-channel signal input (raw + kmer residual).

::: leech.models.TransformerDwellResidual
    options:
      show_root_heading: true
      show_source: false

## Temporal Convolutional Networks

### TCNDwell

Temporal Convolutional Network with dilated convolutions.

::: leech.models.TCNDwell
    options:
      show_root_heading: true
      show_source: false

### TCN Normalization Variants

::: leech.models.TCNDwellGN
    options:
      show_root_heading: true
      show_source: false

::: leech.models.TCNDwellLN
    options:
      show_root_heading: true
      show_source: false

### TCNDwellResidual

TCN with 2-channel signal input (raw + kmer residual).

::: leech.models.TCNDwellResidual
    options:
      show_root_heading: true
      show_source: false

## Convolutional Architectures

### ConvOnly

Pure convolutional network with multi-scale convolutions.

::: leech.models.ConvOnly
    options:
      show_root_heading: true
      show_source: false

### ResNetDwell

Deep residual network with skip connections.

::: leech.models.ResNetDwell
    options:
      show_root_heading: true
      show_source: false

## Inference Wrappers

::: leech.models.ModelInferenceWrapper
    options:
      show_root_heading: true
      show_source: false

::: leech.models.RemoraModelWrapper
    options:
      show_root_heading: true
      show_source: false
