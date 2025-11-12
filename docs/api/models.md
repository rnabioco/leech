# Models Module

Neural network architectures for nanopore signal classification.

## Overview

The models module contains PyTorch model implementations for classifying nanopore signals.

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

## Transformer Architectures

### TransformerDwell

Transformer with multi-head self-attention and dwell features.

::: leech.models.TransformerDwell
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

### TCNDwell

Temporal Convolutional Network with dilated convolutions.

::: leech.models.TCNDwell
    options:
      show_root_heading: true
      show_source: false

### ResNetDwell

Deep residual network with skip connections.

::: leech.models.ResNetDwell
    options:
      show_root_heading: true
      show_source: false
