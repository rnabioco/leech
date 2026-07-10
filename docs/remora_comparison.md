# Remora vs Leech: Comprehensive Comparison

This document systematically documents the differences between [Remora](https://github.com/nanoporetech/remora) (ONT's official base modification detection tool) and Leech, to enable fair comparisons and informed architectural decisions.

## Architecture Differences

### ConvLSTM Architecture Comparison

| Aspect | Remora (`ConvLSTM`) | Leech (`ConvLSTMDwell`) |
|--------|---------------------|-------------------------|
| **Activation** | Swish (SiLU) | ReLU |
| **Normalization** | BatchNorm after every conv | None |
| **Signal conv** | Conv(1->4,k=5)+BN+SiLU -> Conv(4->16,k=5)+BN+SiLU -> Conv(16->size,k=9,stride=3)+BN+SiLU | Conv(1->4,k=5)+ReLU -> Conv(4->16,k=5)+ReLU -> Conv(16->256,k=5)+ReLU + AdaptiveAvgPool |
| **Seq conv** | Conv(36->16,k=5)+BN+SiLU -> Conv(16->size,k=13,stride=3)+BN+SiLU | Conv(36->4,k=3)+ReLU -> Conv(4->16,k=3)+ReLU -> Conv(16->256,k=3)+ReLU + AdaptiveAvgPool |
| **Merge** | Conv(size*2->size,k=5)+BN+SiLU | Concatenate (no conv) |
| **LSTM** | 2x unidirectional LSTM(size), manual bidir via flip, take **last** position | 1x nn.LSTM(768, bidirectional=True), take **center** position |
| **Width** | 64 channels (small, fast) | 256 channels (large) |
| **FC head** | Linear(size->num_out), Dropout(0.3) | Linear(192->64->1), Dropout(0.1) x2 |
| **Feature branch** | None | Yes (dwell + signal stats), Conv1d 3 layers |
| **Loss** | CrossEntropyLoss (2-class softmax) | BCEWithLogitsLoss (single logit) |
| **Output** | `(B, 2)` logits (softmax) | `(B, 1)` logit (sigmoid) |
| **Dropout** | 0.3, single location (before FC) | 0.1, two locations (before/after hidden FC) |

### Key Architectural Differences Explained

**BatchNorm + SiLU vs plain ReLU**: Remora's use of BatchNorm after every conv layer provides implicit regularization and faster convergence. SiLU (Swish) has smoother gradients than ReLU near zero.

**Strided convolutions vs AdaptiveAvgPool**: Remora uses stride=3 convolutions to downsample, which learns the downsampling. Leech uses AdaptiveAvgPool for length normalization.

**Merge convolution vs concatenation**: Remora fuses branches with a learned 1D convolution before the LSTM, reducing channel count. Leech concatenates all branches, leading to a much wider LSTM input (768 vs 128).

**LSTM design**: Remora manually implements bidirectionality by running two unidirectional LSTMs (one on flipped input) and taking the last position. Leech uses PyTorch's built-in bidirectional LSTM and takes the center position.

**CrossEntropy vs BCE**: Remora uses standard multi-class CrossEntropyLoss with 2 output classes. Leech uses BCEWithLogitsLoss with a single output logit.

## Data Processing Differences

| Aspect | Remora | Leech |
|--------|--------|-------|
| **Signal norm** | DAC -> pA -> median-MAD on pA (or pa_scaling from basecaller) | median-MAD on raw DACs |
| **Signal trimming** | Trim to reference-aligned region only | Full read signal |
| **Reference anchoring** | Default: uses ref sequence + ref->signal mapping via CIGAR | Default `--anchor reference`: uses ref sequence + ref->signal mapping via CIGAR (matches Remora); `--anchor basecall` available |
| **Signal map refinement** | Viterbi/dwell-penalty refinement of base boundaries with kmer level tables | None (uses raw move table boundaries) |
| **Seq encoding** | Signal-level kmer only (36 channels for 9-mer) | Signal-level kmer (default) + base_onehot fallback |
| **Chunk context** | Signal-level (raw samples around focus base) | Signal-level (configurable left/right context) |

## Signal Normalization Analysis

### Median-MAD Equivalence

**Key finding**: For median-MAD normalization, normalizing on raw DACs vs pA values is **mathematically identical**.

The POD5 calibration is an affine transform:
```
pA = (DAC - offset) * scale
```

Median-MAD normalization computes:
```
normalized = (x - median(x)) / (1.4826 * MAD(x))
```

Since median and MAD are both affine-equivariant statistics, applying the calibration before normalization cancels out:
```
norm(pA) = (pA - median(pA)) / (1.4826 * MAD(pA))
         = ((DAC - offset) * scale - median((DAC - offset) * scale)) / (1.4826 * MAD((DAC - offset) * scale))
         = (scale * (DAC - median(DAC))) / (scale * 1.4826 * MAD(DAC))
         = (DAC - median(DAC)) / (1.4826 * MAD(DAC))
         = norm(DAC)
```

So leech's current `median_mad` normalization on raw DACs produces **identical** values to Remora's when both use median-MAD.

### pa_scaling Mode (Remora-specific)

The **real** difference is Remora's `pa_scaling` normalization mode (available with Dorado 4.3+):
- Uses the basecaller model's global `(shift, scale)` parameters instead of per-read statistics
- Flow: `DAC -> pA -> (pA - global_shift) / global_scale`
- This is a **global** normalization, making it more consistent across reads/runs at the cost of per-read adaptivity
- The global parameters come from the basecaller model's training and are stored in POD5 metadata

## Signal Map Refinement

Remora includes an optional signal map refinement step that uses expected kmer signal levels to refine the move-table-based base boundaries via dynamic programming (Viterbi). This is implemented in `refine_signal_map.py` and `refine_signal_map_core.pyx`.

Key components:
- **kmer level tables**: Expected signal levels for each k-mer context (e.g., RNA004 9-mer levels)
- **Banded Viterbi DP**: Refines base boundaries within a band around the initial move-table estimate
- **Signal rescaling**: Rescales signal to match expected levels before refinement

This refinement can improve base boundary accuracy, particularly for bases with unusual dwell times.

## Leech Extensions (Work Streams)

### Work Stream A: Reference-Anchored Mode
When `--anchor reference` is used, leech uses the reference sequence for the model's sequence branch, trims signal to the aligned region, and computes ref->signal mapping via CIGAR.

### Work Stream B: pa_scaling Normalization
Adds Remora's global normalization as an option: `--signal-norm pa_scaling --pa-mean X --pa-stdev Y`.

### Work Stream C: Signal Map Refinement
Pure numpy port of Remora's Viterbi-based signal map refinement.

### Work Stream D: Remora-Style Model Variants
- `ConvLSTMRemora`: Remora architecture + leech's feature branch (3 branches)
- `ConvLSTMRemoraBase`: Pure Remora architecture reproduction (signal + sequence only)

### Work Stream E: CrossEntropy Loss
Adds `nn.CrossEntropyLoss` as an option (2-class output), matching Remora's training setup.
