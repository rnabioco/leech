# Remora Architecture Analysis & Leech Alignment

**Date**: 2025-11-06
**Purpose**: Deep dive into Remora's modeling approach to ensure Leech correctly captures core concepts while adding novel dwell time features.

---

## Executive Summary

✅ **Leech correctly implements Remora's core concepts:**
- Signal-sequence temporal alignment via move tables
- Fixed-length signal chunks with focus position
- K-mer context expansion around focus base
- ConvLSTM architecture with signal and sequence branches

🚀 **Leech's novel contribution:**
- **Explicit dwell time features** as a third model input branch
- Per-base temporal kinetics (duration information)
- Dwell time statistics (mean, std, ratio) for local context
- Spatiotemporal characterization (signal level + duration)

---

## 1. Core Concept: Temporal Signal-Sequence Alignment

### How Nanopore Sequencing Works (Temporal Process)

1. **DNA translocates through pore** → current changes over TIME
2. **ADC samples current** → raw signal array (temporal series)
3. **Basecaller processes signal** → outputs bases + move table
4. **Move table = temporal record** of basecaller decisions

### The Move Table: Critical Temporal Data Structure

```
Move Table Components (from BAM 'mv' tag):
├── stride: Downsampling factor (typically 5 or 6)
│   └── Basecaller looks at every Nth signal sample
├── moves: Binary array [1,1,0,1,0,0,0,1,...]
│   ├── 1 = "move to next base"
│   └── 0 = "stay on current base"
└── Temporal alignment: position i → signal index (i+1)*stride

Example:
    stride = 5
    moves = [1, 1, 0, 1, 0, 0, 0, 1]

    Temporal interpretation:
    Position 0: Move to base 0 (1) → base 0 starts at signal[0]
    Position 1: Move to base 1 (1) → base 1 starts at signal[5]
    Position 2: Stay on base 1 (0)
    Position 3: Move to base 2 (1) → base 2 starts at signal[10]
    Position 4-6: Stay on base 2 (0,0,0)
    Position 7: Move to base 3 (1) → base 3 starts at signal[20]

    Signal-to-Sequence Mapping:
    Base 0: signal[0:5]     (dwell = 5 samples)
    Base 1: signal[5:10]    (dwell = 5 samples)
    Base 2: signal[10:20]   (dwell = 10 samples)
    Base 3: signal[20:40]   (dwell = 20 samples)
```

### Remora's Use of Move Tables

**Remora's approach:**
1. Extract fixed-length signal chunks centered on focus base
2. Expand focus base to k-mer context (e.g., 5 bases: 2 left + focus + 2 right)
3. One-hot encode k-mers
4. Use move table to **implicitly** align signal with sequence
5. Feed aligned signal + sequence into ConvLSTM model

**Key insight:** Move table provides **deterministic alignment** (not learned attention).

### Leech's Enhancement

**Leech does everything Remora does PLUS:**
1. **Explicitly compute dwell times** from move table
2. **Extract dwell time features** (log, mean, std, ratio)
3. **Compute signal level features** per base (mean, median, std, range)
4. **Add third model branch** for temporal features

---

## 2. Remora's Data Preparation Pipeline

### Input Requirements

```
Required Files:
├── POD5: Raw signal data
│   └── signal: np.ndarray of float (picoamps over time)
└── BAM: Aligned basecalls with tags
    ├── mv: Move table [stride, moves...]
    ├── ns: Number of signal samples (int)
    ├── ts: Trim offset (int, optional)
    └── MD: Mismatch/deletion string (for alignment)
```

### Chunk Structure (Remora Standard)

```python
# Remora chunk = fundamental training unit
{
    'signal': np.ndarray,        # Fixed length (e.g., 400 samples)
    'sequence': str,             # K-mer context (e.g., 11 bases)
    'focus_base': int,           # Position being classified
    'label': int,                # Modified (1) vs canonical (0)
}

# Chunk extraction parameters:
├── signal_len: Fixed signal length (e.g., 400)
├── signal_context: Padding (left, right) around focus (e.g., 200, 200)
├── kmer_context: Bases on each side (e.g., 5 → 11 bases total)
└── focus_position: Typically center of chunk
```

### Leech's Chunk Structure (Extended)

```python
# Leech chunk = Remora chunk + dwell features
{
    'signal': np.ndarray,        # Same: fixed length signal
    'sequence': str,             # Same: k-mer context
    'dwell': np.ndarray,         # NEW: per-base dwell times
    'features': np.ndarray,      # NEW: stacked dwell + signal features
    'base_idx': int,             # Focus base index
    'label': int,                # Classification label
    'read_id': str,              # Read identifier
}

# Feature stack (num_features × kmer_len):
features = np.stack([
    dwell,           # Raw dwell time (temporal duration)
    dwell_log,       # Log-transformed dwell
    dwell_mean,      # Local mean (windowed)
    dwell_std,       # Local std (windowed)
    dwell_ratio,     # Ratio to local mean
    level_mean,      # Signal mean per base
    level_median,    # Signal median per base
    level_std,       # Signal variability per base
    level_range,     # Signal range per base
], axis=0)
```

---

## 3. Model Architecture Comparison

### Remora: ConvLSTM_w_ref

```
Architecture (inferred from documentation):
┌─────────────────────────────────────┐
│          Input Data                 │
├─────────────────────────────────────┤
│ Signal: (batch, signal_len)         │
│ Sequence: (batch, 4, kmer_len)      │ ← One-hot encoded
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│       Signal Branch                 │
├─────────────────────────────────────┤
│ Conv1d layers (extract features)    │
│ Adaptive pooling → kmer_len         │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│      Sequence Branch                │
├─────────────────────────────────────┤
│ Conv1d layers (k-mer patterns)      │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│      Concatenate Branches           │
│    (signal + sequence features)     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│         BiLSTM Layers               │
│  (capture sequential context)       │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   Extract Center Position           │
│     (focus base prediction)         │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│      Fully Connected → Output       │
│    (binary classification)          │
└─────────────────────────────────────┘
```

### Leech: ConvLSTMDwell

```
Architecture (THREE branches):
┌─────────────────────────────────────────────────────────┐
│                    Input Data                            │
├─────────────────────────────────────────────────────────┤
│ Signal: (batch, signal_len)                              │
│ Sequence: (batch, 4, kmer_len)                           │
│ Features: (batch, num_features, kmer_len)  ← NEW!        │
└─────────────────────────────────────────────────────────┘
              ↓                   ↓                  ↓
    ┌──────────────┐    ┌──────────────┐   ┌──────────────┐
    │Signal Branch │    │Sequence Branch│   │Feature Branch│ ← NEW!
    ├──────────────┤    ├──────────────┤   ├──────────────┤
    │Conv1d [1→4]  │    │Conv1d [4→4]  │   │Conv1d [N→4]  │
    │Conv1d [4→16] │    │Conv1d [4→16] │   │Conv1d [4→16] │
    │Conv1d [16→256]│   │Conv1d [16→256]│  │Conv1d [16→256]│
    │AdaptivePool  │    │              │   │              │
    └──────────────┘    └──────────────┘   └──────────────┘
              ↓                   ↓                  ↓
    ┌─────────────────────────────────────────────────────┐
    │      Concatenate All Three Branches                  │
    │         (768 features at kmer_len)                   │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │            BiLSTM (768 → 96*2)                       │
    │         (capture sequential context)                 │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │         Extract Center Position                      │
    │          (focus base output)                         │
    └─────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────┐
    │          FC → 64 → 1 (binary output)                 │
    └─────────────────────────────────────────────────────┘
```

### Architecture Comparison

| Component | Remora | Leech (ConvLSTMDwell) | Leech (ConvLSTMBase) |
|-----------|--------|------------------------|----------------------|
| Signal branch | ✅ Conv1d | ✅ Conv1d (same) | ✅ Conv1d (same) |
| Sequence branch | ✅ Conv1d | ✅ Conv1d (same) | ✅ Conv1d (same) |
| **Feature branch** | ❌ Not present | ✅ **NEW: Dwell+Level** | ❌ Not present |
| Adaptive pooling | ✅ Yes | ✅ Yes | ✅ Yes |
| BiLSTM | ✅ Yes | ✅ Yes (768→192) | ✅ Yes (512→192) |
| Output | Binary | Binary | Binary |

**Key difference:** Leech's ConvLSTMDwell adds a third branch for temporal features.

---

## 4. Novel Contribution: Dwell Time Features

### Why Dwell Times Matter for aa-tRNA-seq

**Biological hypothesis:**
- Charged tRNAs (with amino acid attached) have different physical properties
- Different molecular weight, charge, structure
- May affect translocation kinetics through nanopore
- **Dwell time captures these kinetic differences**

### Spatial vs. Temporal Information

| Feature Type | Information Content | Remora | Leech |
|--------------|---------------------|--------|-------|
| **Signal level** | Current amplitude (spatial) | ✅ Implicit | ✅ Explicit |
| **Dwell time** | Duration per base (temporal) | ❌ Not used | ✅ **NEW** |
| **Dwell variability** | Kinetic consistency | ❌ Not used | ✅ **NEW** |

**Example scenario:**
```
Two bases with SAME signal level but DIFFERENT dwell times:
    Base A: level=1.5, dwell=5  → Fast translocation
    Base B: level=1.5, dwell=15 → Slow translocation

Remora: Cannot distinguish (same signal amplitude)
Leech:  Can distinguish (different temporal kinetics)
```

### Dwell Feature Engineering

```python
# Raw dwell time (from move table)
dwell = np.diff(seq_to_sig_map)  # Temporal duration per base

# Log transform (handle skewed distribution)
dwell_log = np.log(dwell + 1e-6)

# Local context features (windowed statistics)
dwell_mean = rolling_mean(dwell, window=5)  # Local average
dwell_std = rolling_std(dwell, window=5)    # Local variability
dwell_ratio = dwell / (dwell_mean + eps)    # Normalized dwell

# Signal level features (per-base statistics)
level_mean = per_base_stat(signal, seq_to_sig_map, stat=mean)
level_median = per_base_stat(signal, seq_to_sig_map, stat=median)
level_std = per_base_stat(signal, seq_to_sig_map, stat=std)
level_range = per_base_stat(signal, seq_to_sig_map, stat=range)

# Stack all features → (num_features, kmer_len) tensor
features = np.stack([
    dwell, dwell_log, dwell_mean, dwell_std, dwell_ratio,
    level_mean, level_median, level_std, level_range
], axis=0)
```

---

## 5. Implementation Verification

### ✅ Correct Temporal Alignment

Our implementation correctly:
1. **Parses move tables** from BAM 'mv' tag (features.py:61-98)
2. **Computes seq_to_sig_map** with stride and trim_offset (features.py:40-58)
3. **Extracts dwell times** via np.diff(seq_to_sig_map) (features.py:101-121)
4. **Maintains temporal consistency** in chunk extraction (data_prep.py:61-116)

### ✅ Correct Feature Extraction

Our implementation correctly:
1. **Normalizes signal** using median-MAD (robust to outliers) (features.py:159-201)
2. **Computes per-base signal levels** using seq_to_sig_map slicing (features.py:124-156)
3. **Computes dwell features** with windowed statistics (features.py:204-245)
4. **Stacks features** into model-ready tensors (data_prep.py:104-112)

### ✅ Correct Model Architecture

Our models correctly:
1. **ConvLSTMBase**: Matches Remora's two-branch architecture (models/conv_lstm_base.py)
2. **ConvLSTMDwell**: Extends with third feature branch (models/conv_lstm_dwell.py)
3. **Adaptive pooling**: Aligns signal branch to kmer_len (both models)
4. **Center extraction**: Predicts at focus position (both models)

---

## 6. Comprehensive Test Coverage

### New Test Suite: `test_move_table_signal_temporal.py`

**Test Categories:**

1. **Temporal Relationship Tests**
   - Basic temporal mapping (stride × position → signal index)
   - Dwell time as temporal duration
   - Signal extraction with temporal alignment
   - Trim offset temporal shift
   - Stride consistency (5 vs 6)

2. **Remora Concept Alignment Tests**
   - Fixed-length signal chunks
   - K-mer expansion with move tables
   - Focus position extraction

3. **Novel Dwell Feature Tests**
   - Dwell features capture temporal dynamics
   - Dwell + signal level complementarity
   - Spatiotemporal characterization

4. **Edge Case Tests**
   - Single base reads
   - Consecutive moves (rapid transitions)
   - Long homopolymers (many zeros)

---

## 7. Recommendations

### ✅ Implementation is Sound

Our implementation correctly captures Remora's core concepts and adds meaningful extensions.

### 🎯 Key Strengths

1. **Explicit temporal modeling**: Dwell times as first-class features
2. **Spatiotemporal features**: Combines amplitude + duration information
3. **Robust feature engineering**: Log transforms, windowed statistics, normalization
4. **Clean abstraction**: `LeechRead` encapsulates all features consistently
5. **Comprehensive testing**: Tests verify temporal correctness

### 🚀 Future Enhancements (Optional)

1. **Dwell time normalization**: Consider per-read normalization for sequencing speed variation
2. **Advanced temporal features**: Dwell time derivatives, acceleration patterns
3. **Multi-scale features**: Different window sizes for local context
4. **Learned temporal encoding**: Let model learn from raw dwell sequence

### 📊 Validation Strategy

To validate our novel contribution:

1. **Ablation study**: Compare ConvLSTMDwell vs ConvLSTMBase
   - Same data, only difference is dwell feature branch
   - Measure performance improvement

2. **Feature importance**: Analyze which features contribute most
   - Use gradient-based attribution
   - Identify most informative dwell features

3. **Biological validation**: Check if dwell patterns match expectations
   - Do charged tRNAs have consistent dwell signatures?
   - Are dwell differences statistically significant?

---

## 8. Conclusion

**Leech correctly implements Remora's proven approach while adding a scientifically motivated extension.**

### Alignment with Remora ✅
- ✅ Move table-based temporal alignment
- ✅ Fixed-length signal chunks with focus position
- ✅ K-mer context expansion
- ✅ ConvLSTM architecture (signal + sequence branches)
- ✅ Adaptive pooling for dimension matching
- ✅ BiLSTM for sequential context
- ✅ Center position extraction for prediction

### Novel Contributions 🚀
- 🆕 Explicit dwell time computation from move tables
- 🆕 Dwell feature engineering (log, mean, std, ratio)
- 🆕 Per-base signal level statistics
- 🆕 Third model branch for temporal features
- 🆕 Spatiotemporal characterization (amplitude + duration)

### Scientific Rationale 🧬
- **Hypothesis**: Charged vs uncharged tRNAs have different translocation kinetics
- **Evidence**: Dwell time captures temporal kinetics that signal level alone misses
- **Validation**: Compare ConvLSTMDwell vs ConvLSTMBase performance

**The implementation is ready for training and evaluation.**
