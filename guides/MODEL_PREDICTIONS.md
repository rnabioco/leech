# Model Architecture Predictions for aa-tRNA-seq Classification

**Date**: 2025-11-06
**Question**: Which model architecture will be most useful for distinguishing charged vs uncharged tRNAs?

---

## TL;DR Predictions

🥇 **Winner: ConvLSTMDwell** (with dwell features)
📊 **Expected improvement**: 5-15% accuracy gain over ConvLSTMBase
🎯 **Key scenario**: When translocation kinetics differ between charged/uncharged states
⚠️ **Risk**: If dwell time differences are minimal, improvement may be small

---

## Detailed Predictions

### Prediction 1: ConvLSTMDwell Will Outperform ConvLSTMBase ✅

**Confidence**: HIGH (80%)

**Rationale:**

The biological hypothesis is that **charged tRNAs have different translocation kinetics** due to:
- **Increased molecular mass** (amino acid ~100-200 Da attached)
- **Altered charge distribution** (amino acid carries charge)
- **Structural changes** (aminoacyl group may affect tRNA conformation)

These physical differences should manifest as **temporal signatures** that raw signal alone might miss.

**Why ConvLSTMDwell should win:**

1. **Explicit temporal features**: Dwell time directly captures translocation speed
   - Charged tRNA: potentially slower → longer dwell times
   - Uncharged tRNA: potentially faster → shorter dwell times

2. **Temporal variability**: Dwell consistency may differ
   - Charged: more uniform (stable structure with amino acid)
   - Uncharged: more variable (flexible CCA tail)

3. **Complementary information**: Signal amplitude + duration > amplitude alone
   - Two bases with similar current but different dwells are distinguishable
   - Spatiotemporal characterization is richer

4. **Local context**: Windowed dwell statistics capture kinetic patterns
   - `dwell_mean`, `dwell_std`, `dwell_ratio` provide normalized temporal context
   - Context matters: dwell at CCA site relative to surrounding sequence

**Expected performance:**
```
Metric                  ConvLSTMBase    ConvLSTMDwell    Improvement
────────────────────────────────────────────────────────────────────
Accuracy                75-80%          85-90%           +5-15%
Precision (charged)     70-75%          80-88%           +10-13%
Recall (charged)        75-82%          85-92%           +8-12%
F1-score                72-78%          82-90%           +10-14%
AUC-ROC                 0.82-0.87       0.90-0.95        +0.08
```

---

### Prediction 2: Dwell Features Will Have High Importance Scores 📊

**Confidence**: MEDIUM-HIGH (70%)

**Rationale:**

If the biological hypothesis is correct, feature importance analysis should reveal:

**Top 5 most important features (predicted):**
1. **`dwell` or `dwell_log`** (raw temporal information)
2. **`level_mean`** (signal amplitude - baseline Remora feature)
3. **`dwell_ratio`** (normalized temporal context)
4. **`level_std`** (signal variability)
5. **`dwell_std`** (temporal variability)

**Why dwell features matter:**
- Direct measurement of translocation kinetics (the hypothesis!)
- Log transformation handles skewed distributions
- Ratio normalization accounts for read-to-read speed variation

**Validation approach:**
```python
# Use integrated gradients or SHAP values
from captum.attr import IntegratedGradients

# Compute feature attributions
attributions = integrated_gradients(model, test_data)

# Rank features by absolute attribution
feature_importance = attributions.abs().mean(dim=0)
```

---

### Prediction 3: Performance Will Vary by Sequence Context 🎯

**Confidence**: HIGH (85%)

**Predicted performance by region:**

| Region | ConvLSTMBase | ConvLSTMDwell | Why Dwell Helps More |
|--------|--------------|---------------|----------------------|
| **CCA tail (3' end)** | 72-78% | 85-92% | ⭐ **Largest effect** - amino acid directly attached here |
| **Acceptor stem** | 75-82% | 82-88% | Moderate effect - structural propagation |
| **T-loop / D-loop** | 78-85% | 80-86% | Small effect - distant from modification site |
| **Anticodon loop** | 80-87% | 82-88% | Small effect - furthest from CCA |

**Key insight:** Dwell features should help most at **motif site (CCA)** where the biological difference is strongest.

**Test this prediction:**
```python
# Stratify test set by position relative to CCA
cca_positions = [read for read in test_set if read.motif == "CCA"]
stem_positions = [read for read in test_set if read.region == "acceptor_stem"]

# Compare model performance by region
for region, subset in regions.items():
    base_acc = evaluate(ConvLSTMBase, subset)
    dwell_acc = evaluate(ConvLSTMDwell, subset)
    print(f"{region}: Improvement = {dwell_acc - base_acc:.2%}")
```

---

### Prediction 4: Optimal Architecture May Need Refinement 🔧

**Confidence**: MEDIUM (65%)

**Current ConvLSTMDwell may not be optimal.** Here are predicted improvements:

#### A. Attention Mechanism on Dwell Features ⭐ HIGH PRIORITY

**Prediction:** Adding attention to the feature branch will improve performance by 2-5%.

**Why:**
- Not all bases contribute equally to classification
- CCA bases are most relevant
- Model should learn to focus on informative positions

**Implementation:**
```python
class ConvLSTMDwellAttention(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... existing branches ...

        # Add attention to feature branch
        self.feature_attention = nn.Sequential(
            nn.Linear(conv_channels[2], 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, signal, sequence, features):
        # ... existing convolutions ...

        # Apply attention to feature branch
        feat_feat = self.feature_conv(features)  # (batch, 256, kmer_len)

        # Compute attention weights
        attn_logits = self.feature_attention(feat_feat.transpose(1, 2))  # (batch, kmer_len, 1)
        attn_weights = F.softmax(attn_logits, dim=1)

        # Apply attention
        feat_feat_weighted = feat_feat * attn_weights.transpose(1, 2)

        # Continue with merge...
```

#### B. Multi-Scale Dwell Features 📏 MEDIUM PRIORITY

**Prediction:** Multiple window sizes will capture both local and global temporal patterns.

**Current limitation:** Fixed window size (5 bases) may miss important patterns.

**Improvement:**
```python
# Extract dwell features at multiple scales
dwell_features = []
for window in [3, 5, 7, 9]:
    feats = compute_dwell_features(dwells, window=window)
    dwell_features.append(feats['dwell_mean'])
    dwell_features.append(feats['dwell_std'])

# Stack multi-scale features
features = np.stack(dwell_features, axis=0)  # More channels
```

**Expected gain:** 1-3% accuracy improvement

#### C. Per-Read Dwell Normalization 📊 MEDIUM PRIORITY

**Prediction:** Normalizing dwell times per read will reduce read-to-read variability.

**Issue:** Sequencing speed varies between reads (pore-to-pore variation, voltage drift)

**Solution:**
```python
def normalize_dwell_per_read(dwells: np.ndarray) -> np.ndarray:
    """
    Normalize dwell times relative to read-specific baseline.

    Accounts for global sequencing speed variation.
    """
    # Use median (robust to outliers)
    read_median = np.median(dwells)
    read_mad = np.median(np.abs(dwells - read_median))

    # Normalize
    normalized = (dwells - read_median) / (read_mad * 1.4826 + 1e-6)

    return normalized
```

**Expected gain:** 2-4% accuracy improvement, especially on multi-pore datasets

#### D. Temporal Convolutions (TCN) Instead of LSTM 🔄 LOW PRIORITY

**Prediction:** Temporal Convolutional Networks may be faster and equally effective.

**Rationale:**
- LSTMs are slow (sequential processing)
- TCNs can be parallelized
- Dilated convolutions capture long-range dependencies

**Trade-off:** May need more hyperparameter tuning.

---

### Prediction 5: Failure Modes and When Dwell Features Won't Help ⚠️

**Confidence**: HIGH (80%)

**Scenario A: Minimal Kinetic Differences**

If charged/uncharged tRNAs translocate at similar speeds:
- Dwell features will be uninformative
- ConvLSTMDwell ≈ ConvLSTMBase performance
- Should rely on signal amplitude instead

**Diagnostic:**
```python
# Compare dwell distributions
charged_dwells = [d for read in data if read.label == 1 for d in read.dwells]
uncharged_dwells = [d for read in data if read.label == 0 for d in read.dwells]

# Statistical test
from scipy.stats import mannwhitneyu
statistic, pvalue = mannwhitneyu(charged_dwells, uncharged_dwells)

if pvalue > 0.05:
    print("⚠️ WARNING: Dwell distributions not significantly different!")
    print("ConvLSTMDwell may not outperform ConvLSTMBase")
```

**Scenario B: High Dwell Time Noise**

If move table quality is poor (basecaller issues):
- Noisy dwell times may hurt more than help
- Signal features more reliable

**Diagnostic:**
```python
# Check move table quality
move_table_snr = compute_move_table_quality(reads)

if move_table_snr < threshold:
    print("⚠️ WARNING: Noisy move tables!")
    print("Consider using ConvLSTMBase or improving basecalling")
```

**Scenario C: Strong Sequence Context Effects**

If sequence alone predicts charging state (e.g., specific tRNA isoacceptors):
- Model may overfit to sequence
- Dwell features become less important

**Solution:** Use balanced sampling across tRNA families

---

### Prediction 6: Ensemble Will Outperform Individual Models 🎭

**Confidence**: HIGH (85%)

**Prediction:** Ensemble of ConvLSTMBase + ConvLSTMDwell will give best performance.

**Rationale:**
- Different models capture different aspects
- Base model: spatial signal patterns
- Dwell model: temporal kinetics
- Ensemble: combines both perspectives

**Implementation:**
```python
class EnsembleModel:
    def __init__(self, base_model, dwell_model):
        self.base_model = base_model
        self.dwell_model = dwell_model

    def predict_proba(self, signal, sequence, features):
        # Get predictions from both models
        p_base = self.base_model.predict_proba(signal, sequence)
        p_dwell = self.dwell_model.predict_proba(signal, sequence, features)

        # Weighted average (tune weights on validation set)
        p_ensemble = 0.4 * p_base + 0.6 * p_dwell

        return p_ensemble
```

**Expected performance:**
```
Single Models:
  ConvLSTMBase:  78% accuracy
  ConvLSTMDwell: 87% accuracy

Ensemble:
  Weighted:      89-91% accuracy  ⭐ BEST
```

---

## Recommended Experimental Plan 🧪

### Phase 1: Baseline Validation (Week 1-2)

1. **Train ConvLSTMBase**
   - Establish baseline performance
   - Verify data pipeline works
   - Tune hyperparameters

2. **Train ConvLSTMDwell**
   - Compare to baseline
   - Validate dwell feature utility
   - Check for overfitting

3. **Ablation studies**
   ```python
   # Test contribution of each feature type
   experiments = [
       ("full", all_features),
       ("no_dwell", signal_features_only),
       ("no_signal_feats", dwell_only),
       ("dwell_only", raw_dwell_only),
   ]
   ```

### Phase 2: Architecture Refinement (Week 3-4)

4. **Add attention mechanism**
   - Implement feature branch attention
   - Compare to base ConvLSTMDwell

5. **Try multi-scale dwell features**
   - Multiple window sizes
   - Compare to single-scale

6. **Per-read normalization**
   - Normalize dwell times per read
   - Check if variability reduces

### Phase 3: Ensemble & Optimization (Week 5-6)

7. **Build ensemble**
   - Combine best models
   - Tune ensemble weights

8. **Final optimization**
   - Hyperparameter search
   - Class balancing
   - Threshold tuning

### Phase 4: Validation & Analysis (Week 7-8)

9. **Feature importance**
   - Identify most useful features
   - Validate biological hypothesis

10. **Biological validation**
    - Check dwell patterns match expectations
    - Stratify by tRNA family, amino acid type
    - Correlation with biochemical properties

---

## Key Metrics to Track 📈

```python
metrics_to_track = {
    # Classification performance
    "accuracy": overall_accuracy,
    "precision_charged": precision_class_1,
    "recall_charged": recall_class_1,
    "f1_charged": f1_class_1,
    "auc_roc": area_under_roc,

    # Calibration
    "brier_score": calibration_quality,
    "ece": expected_calibration_error,

    # Feature importance
    "top_features": feature_importance_ranking,
    "dwell_importance": sum_dwell_feature_importance,

    # Biological validation
    "dwell_separation": cohen_d_charged_vs_uncharged,
    "per_family_accuracy": accuracy_by_trna_family,

    # Efficiency
    "training_time": wall_clock_time,
    "inference_speed": reads_per_second,
}
```

---

## Summary of Predictions

| Prediction | Confidence | Expected Outcome |
|------------|------------|------------------|
| **ConvLSTMDwell > ConvLSTMBase** | 80% | +5-15% accuracy gain |
| **Dwell features have high importance** | 70% | Top 5 features include dwell |
| **Performance varies by region** | 85% | Largest gain at CCA site |
| **Attention helps** | 75% | +2-5% accuracy gain |
| **Multi-scale dwell helps** | 60% | +1-3% accuracy gain |
| **Per-read normalization helps** | 65% | +2-4% accuracy gain |
| **Ensemble is best** | 85% | +2-4% over best single model |

---

## What Could Go Wrong 🚨

### Scenario 1: Dwell Features Don't Help Much
- **If dwell difference is small**: Use ConvLSTMBase (simpler, faster)
- **Mitigation**: Check dwell distributions early (Phase 1)

### Scenario 2: Overfitting on Dwell Features
- **If validation performance << training**: Too many dwell features
- **Mitigation**: Use dropout, regularization, feature selection

### Scenario 3: Computational Cost Too High
- **If ConvLSTMDwell is too slow**: 3 branches = more parameters
- **Mitigation**: Reduce feature branch channels, use TCN instead of LSTM

---

## Bottom Line

🥇 **Best bet: ConvLSTMDwell with attention**
- Captures both spatial (signal) and temporal (dwell) information
- Attention focuses on relevant positions (CCA site)
- Expected: 85-90% accuracy (vs 75-80% baseline)

📊 **Runner-up: Ensemble (Base + Dwell)**
- Combines different perspectives
- More robust to individual model failures
- Expected: 89-91% accuracy (best overall)

🎯 **Safe choice: Train both ConvLSTMBase and ConvLSTMDwell**
- Baseline establishes what Remora-style can do
- Dwell model tests biological hypothesis
- Comparison validates dwell feature utility

**My prediction: ConvLSTMDwell will win, and dwell features will prove biologically meaningful! 🚀**
