# GitHub Issue: Explore Alternative Model Architectures Beyond ConvLSTM

**Title:** Explore alternative model architectures beyond ConvLSTM

**Labels:** enhancement, model-architecture, research

---

## Overview

Currently, leech uses two model architectures:
- **ConvLSTMDwell**: Conv1D + BiLSTM with dwell time features (main model)
- **ConvLSTMBase**: Conv1D + BiLSTM without dwell features (ablation baseline)

Based on the grid-search findings (asymmetric contexts up to 9500 left / 500 right) and the multi-modal nature of the data (signal + sequence + dwell features), we should explore alternative architectures that may better capture long-range dependencies and multi-modal fusion.

## Proposed Architectures

### Priority 1: Must Evaluate

#### 1. Transformer-Based Model
**Rationale:**
- Self-attention can capture long-range dependencies better than LSTM (important given optimal context is 9500 samples)
- Multi-head attention naturally handles multi-modal fusion
- Attention weights provide interpretability (which bases/regions matter)
- State-of-the-art in sequence modeling tasks

**Architecture:**
```
Three branches → positional encoding → multi-head attention → classifier

Signal: 1D conv → transformer encoder blocks
Sequence: Embedding → transformer encoder blocks
Dwell: 1D conv → transformer encoder blocks
→ Cross-attention → MLP classifier
```

**Trade-offs:**
- ✅ Better long-range dependencies, parallelizable, interpretable
- ❌ More parameters, requires more data, slower inference

#### 2. Pure CNN Baseline (VGG/Inception-style)
**Rationale:**
- Test whether temporal dynamics (LSTM) are necessary
- Faster inference and training
- Easier to interpret via filter visualization
- May be sufficient if patterns are local

**Variants:**
- VGG-style: Deep stack of 3x1/5x1 conv + max pooling
- Inception-style: Multi-scale parallel convolutions (1, 3, 5, 7 kernel sizes)
- DenseNet-style: Dense connections between layers

**Trade-offs:**
- ✅ Simple, fast, interpretable
- ❌ Limited temporal modeling, may miss long-range patterns

### Priority 2: Worth Exploring

#### 3. Temporal Convolutional Networks (TCNs)
**Rationale:**
- Dilated convolutions provide large receptive fields with fewer parameters
- Faster than RNNs/LSTMs
- Causal convolutions preserve temporal ordering
- Good for variable-length sequences

**Architecture:**
```
Three parallel TCN branches → concatenate → dense layers
Each branch: Stack of dilated causal conv with residual connections
Dwell: 2^k dilation rates for multi-scale temporal patterns
```

**Trade-offs:**
- ✅ Fast training/inference, fewer parameters, handles long sequences
- ❌ Less expressive than transformers for very long-range dependencies

#### 4. Residual Networks (ResNets)
**Rationale:**
- Deep feature extraction with skip connections
- Proven in signal processing
- Can go deeper without vanishing gradients

**Architecture:**
```
Three ResNet branches (1D blocks) → global pooling → concatenate → FC

Signal: ResNet-18 style (8 residual blocks)
Sequence: Smaller ResNet (4 blocks)
Dwell: Lightweight ResNet (3 blocks)
```

**Trade-offs:**
- ✅ Deep feature learning, training stability, good generalization
- ❌ More computation than simple ConvLSTM

### Priority 3: Exploratory

#### 5. Attention-Enhanced ConvLSTM
**Rationale:**
- Add attention to existing architecture (minimal change)
- Attention over dwell features highlights important bases
- Attention over feature branches for smart fusion

**Modification:**
```python
# After LSTM:
lstm_out → attention over timesteps → weighted sum → FC

# Or cross-branch attention:
signal_feat, seq_feat, dwell_feat → attention → weighted fusion
```

**Trade-offs:**
- ✅ Incremental improvement, keeps LSTM benefits
- ❌ May not fully utilize attention's power

#### 6. WaveNet-Style Architecture
**Rationale:**
- Originally designed for audio (1D signal similar to nanopore)
- Gated dilated convolutions
- Proven for sequential dependencies

**Architecture:**
```
Signal → gated dilated conv stack → global pooling
Sequence → embedding → gated conv stack
Dwell → gated conv stack
→ Concatenate → output
```

**Trade-offs:**
- ✅ Excellent for raw signal modeling
- ❌ Complex training, designed for generation not classification

## Implementation Plan

### Phase 1: Core Alternatives
1. Implement **TransformerDwell** model
2. Implement **ConvOnly** (pure CNN) model
3. Update CLI to support new architectures:
```python
choices=[
    "ConvLSTMDwell",      # Current
    "ConvLSTMBase",       # Current
    "ConvOnly",           # NEW: Pure CNN
    "TransformerDwell",   # NEW: Transformer
    "TCNDwell",           # NEW: Temporal CNN
    "ResNetDwell",        # NEW: ResNet
]
```

### Phase 2: Grid Search Comparison
- Run grid search (200-10000 context) for each architecture
- Compare:
  - Validation accuracy
  - Training time
  - Inference speed
  - Model size
  - LoD/LoQ (from calibration)

### Phase 3: Biological Validation
- Test top 2-3 architectures on biological validation data
- Evaluate with distribution-based metrics (JS divergence, ΔECDF)
- Select production model

## Architecture-Specific Considerations

### For Long-Range Dependencies (9500 context)
- LSTM may struggle with very long sequences → gradient issues
- Transformer attention is designed for this
- TCN with high dilation rates is a good compromise

### For Multi-Modal Fusion
- Current: Simple concatenation before LSTM
- Better: Cross-attention between modalities (transformer)
- Alternative: Learned fusion weights per branch

### For Interpretability
- Attention weights show which bases/regions matter
- Helps understand: Is it dwell kinetics? Signal levels? Sequence context?
- Critical for biological insights

## Success Metrics

For each architecture, evaluate:
- [ ] Validation accuracy > 0.90
- [ ] Test AUC > 0.95
- [ ] LoD ≤ 2% (MLE estimator)
- [ ] LoQ ≤ 5%
- [ ] Training time (wall clock)
- [ ] Inference time per read
- [ ] Model size (parameters, disk)
- [ ] Biological validation: clear separation

## Related

- Grid search methodology: `docs/grid-search.md`
- Current models mentioned in: `CLAUDE.md`, `README.md`
- CLI architecture selection: `src/leech/cli.py:58-61`

## References

- Transformer: "Attention is All You Need" (Vaswani et al., 2017)
- TCN: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (Bai et al., 2018)
- WaveNet: "WaveNet: A Generative Model for Raw Audio" (van den Oord et al., 2016)
- Remora (baseline): https://github.com/nanoporetech/remora
