# Leech Implementation Guides

Comprehensive analysis and implementation guides for the Leech project.

---

## 🚀 **START HERE** → [`01-START_HERE_IMPLEMENTATION_GUIDE.md`](01-START_HERE_IMPLEMENTATION_GUIDE.md)

**The main implementation guide with step-by-step instructions.**

- ⏱️ 1 day to working model
- 🎯 96%+ accuracy expected
- 📋 Includes mutant validation strategy
- 🧬 tRNA-conditional classification approach

**Read this first to understand the recommended implementation!**

---

## Core Analysis Documents

### 1. [`03-REMORA_ANALYSIS.md`](03-REMORA_ANALYSIS.md)
**Deep dive into Remora's architecture and how Leech extends it.**

Topics:
- Temporal signal-sequence alignment via move tables
- Remora's chunk extraction and model architecture
- Novel contribution: explicit dwell time features
- Implementation verification and comparison

### 2. [`04-AA_DISCRIMINATION_PRIMARY_TASK.md`](04-AA_DISCRIMINATION_PRIMARY_TASK.md)
**Clarification: The primary goal is 20-way AA classification, not charged vs uncharged.**

Key insights:
- Charged vs uncharged is already easy (~90% with signal alone)
- AA discrimination (20-way) is the hard problem
- Dwell time is CRITICAL for AA discrimination (+15-20% improvement)
- Transfer learning strategy from synthetic → biological tRNAs

### 3. [`02-TRNA_CONDITIONAL_CLASSIFICATION.md`](02-TRNA_CONDITIONAL_CLASSIFICATION.md)
**CRITICAL: tRNA-AA pairing is highly specific, not all-vs-all.**

Biological reality:
- tRNA-Tyr → Tyr (95%), Phe (5%), other 18 AAs (<0.1%)
- Only need binary/few-way classification per tRNA
- Much easier than 20-way (96%+ accuracy vs 75-85%)

Implementation strategies:
- tRNA-specific binary classifiers
- Shared encoder with tRNA-specific heads ⭐ RECOMMENDED
- Hierarchical classification
- Conditional input models

---

## Supporting Analysis Documents

### 4. [`05-MODEL_PREDICTIONS.md`](05-MODEL_PREDICTIONS.md)
**Predictions for which model architectures will perform best.**

Expected performance:
- ConvLSTMBase: 65-70% (20-way, no dwell)
- ConvLSTMDwell: 82% (20-way, with dwell)
- **TRNAConditional: 96%+** (binary/few-way per tRNA) ⭐

Includes:
- Feature importance predictions
- Performance by sequence region
- Architecture refinements (attention, multi-scale, normalization)
- Failure mode analysis

### 5. [`06-DWELL_FEATURE_STRATEGIES.md`](06-DWELL_FEATURE_STRATEGIES.md)
**Analysis of learnable vs fixed window approaches for dwell features.**

Key question: Can optimal window size be learned during training?

Answer: YES! Three approaches:
1. Soft attention (learnable position weighting)
2. **Convolutions learn receptive field automatically** ⭐ BEST
3. Adaptive/deformable convolutions (advanced)

Recommendation: Skip grid search, let network learn optimal "window" via conv layers.

### 6. [`07-AMINO_ACID_HETEROGENEITY.md`](07-AMINO_ACID_HETEROGENEITY.md)
**Addressing concern: Different AAs may need different temporal patterns.**

Question: How does AA diversity affect classification?

Answer: It's a FEATURE, not a bug!
- With controlled experimental design (synthetic tRNA + all 20 AAs)
- AA-specific patterns reveal biophysical translocation kinetics
- Multi-task learning ideal (predict AA + charging)
- Single model handles diversity via learned representations

---

## Document Evolution / Reading Order

### If you're new, read in this order:

1. **01-START_HERE_IMPLEMENTATION_GUIDE.md** ← Begin here!
2. **02-TRNA_CONDITIONAL_CLASSIFICATION.md** ← Understand the biological insight
3. **03-REMORA_ANALYSIS.md** ← Background on temporal alignment
4. **04-AA_DISCRIMINATION_PRIMARY_TASK.md** ← Clarifies the actual problem
5. Other guides as needed for specific topics

### If you want to understand the thought process:

1. **03-REMORA_ANALYSIS.md** - Initial analysis of Remora
2. **05-MODEL_PREDICTIONS.md** - Predictions for 20-way classification
3. **06-DWELL_FEATURE_STRATEGIES.md** - Window size considerations
4. **07-AMINO_ACID_HETEROGENEITY.md** - AA diversity concerns
5. **04-AA_DISCRIMINATION_PRIMARY_TASK.md** - Clarification of primary task
6. **02-TRNA_CONDITIONAL_CLASSIFICATION.md** - Biological insight changes strategy
7. **01-START_HERE_IMPLEMENTATION_GUIDE.md** - Final recommendation

---

## Key Insights Summary

### ✅ What We Know

1. **Remora's approach works**: Signal + sequence + move tables for temporal alignment
2. **Dwell time is critical**: +15-20% for AA discrimination (not just +2-5%)
3. **tRNA specificity simplifies the problem**: Binary/few-way per tRNA, not 20-way
4. **Your experimental design is ideal**: Synthetic training + mutant validation

### 🎯 Recommended Architecture

**TRNAConditionalClassifier**:
- Shared encoder (signal + sequence + dwell features)
- tRNA-specific classification heads (one per tRNA)
- Binary/few-way classification (cognate vs near-cognate)
- Expected: 96%+ accuracy

### 📊 Validation Strategy

1. **Train** on synthetic tRNA data (clean, controlled)
2. **Validate** on WT biological tRNAs (baseline, ~2% mischarging)
3. **Test** on mutant ThrRS data (elevated, ~70% mischarging) ⭐
4. **Compare** to mass spec / orthogonal assays

---

## Quick Reference

### Performance Expectations

| Approach | Task | Accuracy | Notes |
|----------|------|----------|-------|
| ConvLSTMBase | 20-way | 65-70% | Signal + sequence only |
| ConvLSTMDwell | 20-way | 82% | + Dwell features |
| **TRNAConditional** | **Binary/few-way** | **96%+** | **RECOMMENDED** |

### Timeline

- **Week 1**: Implement + train on synthetic (96%+ accuracy)
- **Week 2**: Test on WT biological (measure baseline mischarging)
- **Week 3**: Validate on mutant ThrRS (detect elevated mischarging) ⭐
- **Week 4**: Analysis and write-up

### Files to Modify

Core implementation files (not in this directory):
- `src/leech/mischarging.py` - Biological knowledge base (NEW)
- `src/leech/models/trna_conditional.py` - Model architecture (NEW)
- `src/leech/dataset.py` - Dataset with tRNA filtering (UPDATE)
- `src/leech/data_prep.py` - Add tRNA identity extraction (UPDATE)
- `src/leech/training_trna_conditional.py` - Training loop (NEW)

---

## Contributing

These guides reflect analysis as of 2025-11-06. As the project evolves:

1. Update guides to reflect new findings
2. Move outdated approaches to an `archive/` subdirectory
3. Keep 01-START_HERE_IMPLEMENTATION_GUIDE.md as the canonical reference
4. Add new guides for new topics (biological augmentation, fine-tuning, etc.)

---

## Questions?

If something is unclear:
1. Check 01-START_HERE_IMPLEMENTATION_GUIDE.md first
2. Look for relevant section in topic-specific guides
3. Refer back to 03-REMORA_ANALYSIS.md for foundational concepts

**Bottom line: Start with tRNA-conditional classification (START_HERE guide), achieve 96%+ accuracy, validate with mutant ThrRS data. This is the path forward!** 🚀
