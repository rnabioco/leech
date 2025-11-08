# Amino Acid Heterogeneity in Dwell Time Patterns

**Date**: 2025-11-06
**Critical Question**: What if different amino acids have different optimal dwell time windows/patterns? How does this impact our ability to confidently determine charged vs uncharged status?

---

## The Problem Statement

### Biological Reality

Different amino acids have **drastically different physical properties**:

| Property | Range | Example |
|----------|-------|---------|
| **Molecular weight** | 75-204 Da | Gly (75) vs Trp (204) |
| **Size (volume)** | 60-228 Ų | Gly vs Trp |
| **Charge at pH 7** | -1, 0, +1 | Asp (-), Ala (0), Lys (+) |
| **Hydrophobicity** | -4.5 to +4.5 | Arg (-4.5) to Ile (+4.5) |

**These differences likely affect nanopore translocation kinetics!**

### The Concern

**Scenario 1: AA-specific dwell patterns**
```
tRNA-Gly (small, 75 Da):
  Charged:   dwell = 8 samples  (fast translocation)
  Uncharged: dwell = 5 samples

tRNA-Trp (large, 204 Da):
  Charged:   dwell = 20 samples (slow translocation)
  Uncharged: dwell = 5 samples
```

**Problem**: Different amino acids may require different "windows" or temporal patterns to distinguish charged vs uncharged.

**Question**: Does a single model architecture with fixed receptive field work? Or do we need AA-specific models?

---

## Is This a Problem or a Feature?

### Perspective 1: It's a PROBLEM ⚠️

**If dwell patterns are AA-specific:**

1. **Generalization is harder**
   - Model trained on tRNA-Ala may not generalize to tRNA-Trp
   - Need diverse training data across all 20 amino acids
   - Risk of overfitting to specific AA types in training set

2. **Inference is ambiguous**
   - At inference time, you don't know which AA the tRNA carries
   - How do you choose the right "window" or pattern to look for?
   - Multiple competing hypotheses per read

3. **Training data requirements increase**
   - Need balanced representation of all AA types
   - Some AAs are rare in biological samples
   - Sample preparation bias (some AAs more stable than others)

4. **Model complexity increases**
   - May need AA-conditional architecture
   - Or multi-task learning (predict AA type + charging state)
   - More parameters, more overfitting risk

---

### Perspective 2: It's a FEATURE ✅

**If handled correctly, AA diversity can help:**

1. **More discriminative signal**
   - Different AAs provide diverse training examples
   - Model learns robust features that generalize
   - "Charged" is a meta-pattern across diverse AA types

2. **Multi-task learning opportunity**
   - Predict both charging state AND amino acid type
   - AA prediction provides auxiliary supervision
   - Shared representations improve both tasks

3. **Biological validation**
   - If model learns AA-specific patterns, validates hypothesis
   - Can analyze which AAs show strongest dwell effects
   - Scientific insight into translocation kinetics

4. **Flexible architecture can handle diversity**
   - Attention mechanisms can focus on relevant scales per AA
   - Multi-scale features capture different pattern types
   - Learned representations abstract over AA diversity

---

## Quantifying the Problem: How Different Are AAs?

### Hypothesis Testing Strategy

Before worrying, **measure** if this is actually a problem:

```python
def analyze_aa_specific_dwell_patterns(data):
    """
    Quantify how much dwell patterns differ across amino acids.

    Returns:
        - Effect sizes for each AA
        - Separability metrics
        - Recommended modeling strategy
    """

    results = {}

    for aa in AMINO_ACIDS:
        # Get charged and uncharged reads for this AA
        charged = data[(data.label == 1) & (data.amino_acid == aa)]
        uncharged = data[(data.label == 0) & (data.amino_acid == aa)]

        # Compute dwell statistics
        charged_dwell = charged['dwell_at_CCA'].values
        uncharged_dwell = uncharged['dwell_at_CCA'].values

        # Effect size (Cohen's d)
        effect_size = cohen_d(charged_dwell, uncharged_dwell)

        # Separability (AUC using dwell alone)
        auc = roc_auc_score(
            labels=[1]*len(charged) + [0]*len(uncharged),
            scores=np.concatenate([charged_dwell, uncharged_dwell])
        )

        results[aa] = {
            'effect_size': effect_size,
            'auc_dwell_only': auc,
            'charged_mean': charged_dwell.mean(),
            'uncharged_mean': uncharged_dwell.mean(),
            'n_samples': len(charged) + len(uncharged)
        }

    return results


# Analyze results
results = analyze_aa_specific_dwell_patterns(training_data)

# Case 1: Consistent effect across AAs
if all(r['effect_size'] > 0.5 for r in results.values()):
    print("✅ All AAs show strong dwell effect - single model OK")

# Case 2: Variable effects
elif max(results.values(), key=lambda x: x['effect_size']) > 2 * min(...):
    print("⚠️ AA-specific effects - need AA-aware strategy")

# Case 3: No effect for some AAs
elif any(r['effect_size'] < 0.2 for r in results.values()):
    print("🚨 Some AAs show no dwell effect - may need exclusion")
```

### Expected Outcomes

**Scenario A: Consistent dwell effect across AAs** ✅
```
AA    Charged Mean    Uncharged Mean    Effect Size    AUC
Ala   12.3           8.1               1.2            0.85
Gly   10.8           7.9               0.9            0.82
Trp   15.7           8.3               1.5            0.89
...
All AAs show Cohen's d > 0.8
```
→ **Single model works!** Dwell time is universally informative.

**Scenario B: Variable but detectable effects** ⚠️
```
AA    Charged Mean    Uncharged Mean    Effect Size    AUC
Ala   12.3           8.1               1.2            0.85
Gly   9.2            8.9               0.3            0.58  ← Weak!
Trp   18.4           8.3               2.1            0.94  ← Strong!
...
Effect sizes range from 0.3 to 2.1
```
→ **Need AA-aware modeling** or multi-task learning.

**Scenario C: No consistent effect** 🚨
```
AA    Charged Mean    Uncharged Mean    Effect Size    AUC
Ala   9.1            9.3               0.05           0.51
Gly   8.8            9.1               0.08           0.49
Trp   9.5            9.2               0.07           0.52
...
All effect sizes < 0.2
```
→ **Dwell time not informative!** Fall back to signal-only model (ConvLSTMBase).

---

## Modeling Strategies for AA Heterogeneity

### Strategy 1: Single Model (Shared Representation) ✅ SIMPLEST

**When to use**: Scenario A (consistent effects)

**Approach**: Train one model on all AA types mixed together.

```python
# Train on all data
model = ConvLSTMDwell()
train_data = load_chunks('all_amino_acids_mixed.npz')
train(model, train_data)

# At inference: no AA-specific handling needed
pred = model.predict(test_read)  # Works for any AA
```

**Pros:**
- ✅ Simple - one model for everything
- ✅ Leverages all training data
- ✅ Learns AA-invariant features
- ✅ No need to know AA at inference time

**Cons:**
- ❌ May underfit AA-specific patterns
- ❌ Averages over AA diversity

**How it handles AA diversity:**
- Convolutional layers learn features robust to AA variation
- LSTM captures sequence context (may correlate with AA type)
- Model learns "charged" as abstract concept across AAs

**Expected performance:** 80-85% accuracy if effects are consistent

---

### Strategy 2: Multi-Task Learning (Predict AA Type + Charging) 🎯 RECOMMENDED

**When to use**: Scenario B (variable effects) OR when AA labels available

**Approach**: Simultaneously predict charging state AND amino acid type.

```python
class ConvLSTMDwellMultiTask(nn.Module):
    def __init__(self, num_amino_acids=20):
        super().__init__()

        # Shared feature extraction (signal + sequence + dwell)
        self.shared_encoder = SharedEncoder()  # Conv + LSTM

        # Task 1: Charging state (binary)
        self.charging_head = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Binary: charged vs uncharged
        )

        # Task 2: Amino acid type (20-way classification)
        self.amino_acid_head = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, num_amino_acids)  # 20 AAs
        )

    def forward(self, signal, sequence, dwell):
        # Shared representation
        shared_repr = self.shared_encoder(signal, sequence, dwell)

        # Task predictions
        charging_logits = self.charging_head(shared_repr)
        aa_logits = self.amino_acid_head(shared_repr)

        return {
            'charging': charging_logits,
            'amino_acid': aa_logits
        }


# Training with multi-task loss
def train_step(model, batch):
    outputs = model(batch['signal'], batch['sequence'], batch['dwell'])

    # Combined loss
    loss_charging = bce_loss(outputs['charging'], batch['charging_label'])
    loss_aa = cross_entropy(outputs['amino_acid'], batch['aa_label'])

    # Weighted combination
    total_loss = 0.7 * loss_charging + 0.3 * loss_aa

    return total_loss
```

**Pros:**
- ✅ **Best of both worlds** - learns AA-specific AND shared patterns
- ✅ AA prediction provides auxiliary supervision
- ✅ Shared encoder learns richer representations
- ✅ Can analyze AA-specific performance
- ✅ At inference: get both predictions

**Cons:**
- ❌ Need AA labels in training data
- ❌ More complex training (multi-task balancing)
- ❌ Harder to interpret

**Why this helps AA diversity:**
- Model learns AA-specific features in shared encoder
- Charging head learns to use AA context appropriately
- AA prediction enforces learning of AA-discriminative features
- Shared representations generalize better

**Expected performance:** 85-92% accuracy (best for variable effects)

---

### Strategy 3: AA-Conditional Model (Explicit Conditioning) 🔧 COMPLEX

**When to use**: Scenario B + AA identity known at inference

**Approach**: Explicitly condition model on amino acid type.

```python
class ConvLSTMDwellConditional(nn.Module):
    def __init__(self, num_amino_acids=20):
        super().__init__()

        # ... standard branches ...

        # Amino acid embedding
        self.aa_embedding = nn.Embedding(num_amino_acids, 64)

        # Conditioning: inject AA embedding into feature branch
        self.feature_conv_conditional = nn.Sequential(
            nn.Conv1d(num_features + 64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, signal, sequence, dwell, amino_acid_id):
        # Get AA embedding
        aa_emb = self.aa_embedding(amino_acid_id)  # (batch, 64)

        # Broadcast to kmer_len
        aa_emb = aa_emb.unsqueeze(-1).expand(-1, -1, kmer_len)  # (batch, 64, kmer_len)

        # Concatenate with dwell features
        dwell_with_aa = torch.cat([dwell_features, aa_emb], dim=1)  # (batch, num_feat+64, kmer_len)

        # Feature branch with AA conditioning
        feat_feat = self.feature_conv_conditional(dwell_with_aa)

        # ... rest of model ...
```

**Pros:**
- ✅ Explicitly models AA-specific patterns
- ✅ Can learn different "windows" per AA via conditioning
- ✅ Interpretable (AA has explicit effect)

**Cons:**
- ❌ **Requires knowing AA at inference time!**
- ❌ If AA unknown, need to evaluate all 20 possibilities
- ❌ More parameters
- ❌ Need balanced AA training data

**Inference with unknown AA:**
```python
# Option 1: Marginalize over all AAs (expensive)
probs = []
for aa in range(20):
    p = model.predict_proba(signal, sequence, dwell, aa_id=aa)
    probs.append(p)

# Average predictions
final_prob = np.mean(probs)  # or weighted by AA frequency

# Option 2: First predict AA, then use for charging
aa_pred = aa_classifier.predict(signal, sequence, dwell)
charging_prob = model.predict_proba(signal, sequence, dwell, aa_id=aa_pred)
```

**Expected performance:** 87-93% (IF AA known), 82-88% (IF AA must be inferred)

---

### Strategy 4: Ensemble of AA-Specific Models 🏗️ OVERKILL

**When to use**: Scenario B + abundant training data + AA known

**Approach**: Train separate model for each amino acid.

```python
# Train 20 models
models = {}
for aa in AMINO_ACIDS:
    aa_data = filter_by_amino_acid(train_data, aa)
    models[aa] = train_model(ConvLSTMDwell(), aa_data)

# Inference (requires knowing AA)
def predict(read, amino_acid):
    return models[amino_acid].predict(read)
```

**Pros:**
- ✅ Maximally flexible (each AA gets custom model)
- ✅ Can optimize architecture per AA
- ✅ Best performance IF AA known

**Cons:**
- ❌ **Requires 20x training data!**
- ❌ **Requires knowing AA at inference**
- ❌ No shared learning across AAs
- ❌ Maintenance nightmare (20 models!)
- ❌ Probably overkill

**Verdict:** Only if you have TONS of data and strong evidence of completely different mechanisms per AA.

---

## Recommended Approach: Progressive Strategy

### Phase 1: Characterize the Problem (Week 1) 🔬

**Before choosing architecture, measure AA diversity:**

```python
# Step 1: Analyze AA-specific effects
aa_analysis = analyze_aa_specific_dwell_patterns(train_data)

# Step 2: Visualize distributions
plot_dwell_distributions_by_aa(train_data)

# Step 3: Test separability
for aa in AMINO_ACIDS:
    auc = test_separability(train_data, amino_acid=aa)
    print(f"{aa}: AUC = {auc:.3f}")

# Step 4: Cluster analysis
# Do AAs cluster into groups with similar dwell patterns?
aa_similarity = compute_aa_similarity_matrix(train_data)
plot_dendrogram(aa_similarity)
```

**Decision tree based on results:**

```
IF all AAs show Cohen's d > 0.8:
    → Use Strategy 1 (Single Model) ✅

ELIF effect sizes vary but all > 0.3:
    → Use Strategy 2 (Multi-Task Learning) 🎯

ELIF some AAs show no effect (d < 0.2):
    → Exclude those AAs, use Strategy 1 on remainder

ELIF AAs cluster into 2-3 groups:
    → Train separate models per cluster
```

---

### Phase 2: Start Simple (Week 2) ✅

**Baseline: Strategy 1 (Single Model)**

```python
# Train one model on ALL data (mixed AAs)
model = ConvLSTMDwell()

# Ensure balanced AA representation
train_data_balanced = balance_amino_acids(train_data)

# Train
train(model, train_data_balanced)

# Evaluate overall
overall_acc = evaluate(model, test_data)

# Evaluate per AA
for aa in AMINO_ACIDS:
    aa_test = filter_by_amino_acid(test_data, aa)
    aa_acc = evaluate(model, aa_test)
    print(f"{aa}: {aa_acc:.3f}")
```

**What to look for:**
- Overall accuracy (target: >80%)
- Per-AA accuracy variance
- If some AAs perform poorly → need AA-aware strategy

---

### Phase 3: Add Complexity If Needed (Week 3-4) 🎯

**If per-AA performance varies significantly:**

```python
# Implement multi-task learning
model_multitask = ConvLSTMDwellMultiTask(num_amino_acids=20)

# Train with both labels
train_multitask(
    model,
    charging_labels=train_data['charging'],
    aa_labels=train_data['amino_acid']
)

# Compare to baseline
baseline_acc = 0.82
multitask_acc = 0.89  # Expected improvement

if multitask_acc > baseline_acc + 0.03:
    print("✅ Multi-task helps! Use this model.")
```

---

### Phase 4: Biological Interpretation (Week 5) 🧬

**Analyze what the model learned:**

```python
# Feature importance per AA
for aa in AMINO_ACIDS:
    aa_data = filter_by_amino_acid(test_data, aa)
    importance = compute_feature_importance(model, aa_data)

    print(f"\n{aa} (MW={aa_mw[aa]} Da):")
    print(f"  Dwell importance: {importance['dwell']:.3f}")
    print(f"  Signal importance: {importance['signal']:.3f}")

# Correlation with physical properties
correlate_performance_with_aa_properties(aa_analysis, aa_properties)

# Example hypothesis tests:
# H1: Larger AAs show stronger dwell effects
# H2: Charged AAs (Asp, Glu, Lys, Arg) show different patterns
# H3: Hydrophobic AAs cluster together
```

---

## Handling Unknown AA at Inference Time

### The Core Issue

**Problem**: At inference, you typically **don't know which amino acid** the tRNA carries!

**Options:**

### Option 1: AA-Agnostic Model (RECOMMENDED) ✅

```python
# Model never sees AA identity - learns AA-invariant features
model = ConvLSTMDwell()  # No AA input

# Inference: straightforward
pred = model.predict(read)  # Works for any AA
```

**When this works:**
- AA effects are consistent (Scenario A)
- Model learns to abstract over AA diversity
- Dwell features capture "charged" regardless of AA type

---

### Option 2: Multi-Task Model (Predict AA First) 🎯

```python
# Model learns to predict both
model = ConvLSTMDwellMultiTask()

# Inference: get both predictions
outputs = model.predict(read)
charging_prob = outputs['charging']  # Use this
aa_prediction = outputs['amino_acid']  # Bonus info!
```

**Benefits:**
- No need to know AA beforehand
- AA prediction is byproduct
- Can use AA confidence to weight charging prediction

**Advanced: Uncertainty-weighted prediction:**
```python
# If model is uncertain about AA, trust charging prediction less
aa_confidence = softmax(outputs['amino_acid']).max()

if aa_confidence < 0.5:
    # Model unsure about AA → less confident in charging
    charging_prob_adjusted = 0.5 + 0.5 * (charging_prob - 0.5)  # Shrink toward 0.5
```

---

### Option 3: Marginalize Over AAs (If AA-Conditional Model) 📊

```python
# AA-conditional model requires AA input
model = ConvLSTMDwellConditional()

# At inference, try all possibilities
charging_probs = []
aa_weights = []

for aa_id in range(20):
    prob = model.predict_proba(read, amino_acid_id=aa_id)
    charging_probs.append(prob)

    # Weight by AA frequency (from training data or biology)
    aa_weights.append(aa_frequency[aa_id])

# Weighted average
final_prob = np.average(charging_probs, weights=aa_weights)
```

**Pros:**
- Accounts for all AA possibilities
- Weights by biological plausibility

**Cons:**
- **20x slower** (need 20 forward passes!)
- Computationally expensive

---

### Option 4: Sequence-Based AA Prediction (Clever!) 🧠

**Insight**: tRNA sequence often correlates with AA identity!

- tRNA-Ala has specific sequence motifs
- Anticodon loop determines AA specificity
- Reference genome annotation may provide AA info

```python
# First: predict AA from tRNA sequence
aa_predictor = train_aa_from_sequence_model(reference_trnas)

# At inference:
def predict_charging_with_aa_inference(read):
    # Step 1: Predict AA from sequence/alignment
    aa_predicted = aa_predictor.predict(read.sequence, read.alignment_info)

    # Step 2: Use predicted AA for charging prediction
    if use_conditional_model:
        charging_prob = model.predict(read, amino_acid_id=aa_predicted)
    else:
        # Just use multi-task model
        charging_prob = model.predict(read)['charging']

    return charging_prob, aa_predicted
```

**When this works:**
- tRNA sequences are well-annotated
- Alignment to reference tRNAs available
- AA identity is deterministic from sequence

**This is probably the most practical approach!** You can often infer AA from the tRNA sequence itself.

---

## Practical Recommendations

### For Your Current Project 🎯

#### Recommendation 1: Start with Strategy 1 (Single Model)

**Rationale:**
- Simplest approach
- Likely to work if dwell effect is real
- Can always add complexity later

**Implementation:**
```python
# Use current ConvLSTMDwell as-is
model = ConvLSTMDwell()

# Key: ensure balanced AA representation in training
train_data = balance_amino_acids(load_chunks('train.npz'))

# Train
train(model, train_data)

# Evaluate overall and per-AA
evaluate_by_amino_acid(model, test_data)
```

**Success criterion:** Per-AA accuracy variance < 10%
- If Ala = 85% and Trp = 83% → Good! Model is AA-agnostic.
- If Ala = 90% and Gly = 60% → Problem. Need AA-aware strategy.

---

#### Recommendation 2: Track AA Labels (Metadata)

**Even if not used in model, track for analysis:**

```python
# In chunk extraction
chunk = {
    'signal': ...,
    'sequence': ...,
    'dwell': ...,
    'features': ...,
    'label': charging_label,
    'amino_acid': aa_label,  # ← ADD THIS
    'metadata': {
        'trna_gene': 'tRNA-Ala-AGC-1',
        'amino_acid': 'Ala',
        'anticodon': 'AGC',
    }
}
```

**Why:**
- Enables post-hoc AA-stratified analysis
- Can add multi-task learning later if needed
- Helps interpret failures

---

#### Recommendation 3: Analyze Per-AA Performance Early

**Week 1-2: After initial training:**

```python
# Stratified evaluation
results_by_aa = {}

for aa in AMINO_ACIDS:
    aa_test = [c for c in test_chunks if c['amino_acid'] == aa]

    if len(aa_test) > 10:  # Sufficient samples
        acc = evaluate(model, aa_test)
        results_by_aa[aa] = {
            'accuracy': acc,
            'n_samples': len(aa_test)
        }

# Check variance
accuracies = [r['accuracy'] for r in results_by_aa.values()]
variance = np.var(accuracies)

if variance > 0.01:  # >10% variance
    print("⚠️ High per-AA variance - investigate AA-specific patterns")
    print(sorted(results_by_aa.items(), key=lambda x: x[1]['accuracy']))
```

---

#### Recommendation 4: If AA Effects Vary → Multi-Task Learning

**If Strategy 1 shows high per-AA variance:**

1. **Implement Strategy 2** (Multi-Task Learning)
2. **Compare performance**:
   ```python
   baseline_acc = 0.82  # Strategy 1
   multitask_acc = 0.88  # Strategy 2

   if multitask_acc > baseline_acc + 0.03:
       use_multitask = True
   ```

3. **Analyze what improved**:
   ```python
   # Which AAs benefited most?
   for aa in AMINO_ACIDS:
       baseline = baseline_results[aa]
       multitask = multitask_results[aa]
       improvement = multitask - baseline

       if improvement > 0.05:
           print(f"{aa}: +{improvement:.2%} improvement")
   ```

---

## CRITICAL CONTEXT: Experimental Design

### Training Data Structure (UPDATED)

**User clarification:**
> "The training data is the same synthetic tRNA charged with all 20 amino acids, which is real ground truth. These were sequenced individually, so we have thousands of isolated samples."

**This is an IDEAL controlled experiment!** 🎉

```
Design:
  Synthetic tRNA backbone (e.g., tRNA-Ala scaffold)
    ├── Uncharged control (pure sample, n=1000s)
    ├── + Ala (pure sample, n=1000s)
    ├── + Arg (pure sample, n=1000s)
    ├── + Asn (pure sample, n=1000s)
    ├── ...
    └── + Val (pure sample, n=1000s)

Key properties:
  ✅ Same sequence backbone (removes sequence confounding)
  ✅ Pure samples (clean ground truth)
  ✅ Balanced representation (equal n per AA)
  ✅ Controlled conditions (same prep, sequencing)

Potential augmentation:
  Biological tRNAs (more modifications, more complexity)
```

### Why This Changes Everything

**Original concern:**
> "Different AAs may need different windows... how to handle at inference?"

**With this design:**

1. **AA-specific patterns are REAL signal, not noise!** ✅
   - Same tRNA + different AA = isolated AA effect
   - Dwell differences directly attributable to AA properties
   - This is biological gold - validate translocation kinetics!

2. **Multi-task learning is PERFECT here** 🎯
   - Predicting AA type is a valid auxiliary task
   - Shared representation benefits both tasks
   - Can analyze AA-specific kinetics scientifically

3. **Inference strategy depends on goal** 📊
   - **If goal = classify biological tRNAs**: AA unknown, use AA-agnostic model
   - **If goal = analyze synthetic tRNAs**: AA known, can use AA-conditional model
   - **If goal = both**: Multi-task model predicts both

4. **Controlled experiment enables causal inference** 🔬
   - Can measure: "Effect of Trp vs Ala on dwell time"
   - Can test: "Does AA mass correlate with dwell?"
   - Can validate: "Do charged AAs (Lys, Arg) differ from neutral?"

---

## REVISED Recommendations (Given Experimental Design)

### Strategy 1: Exploit the Controlled Design! 🎯

**Your data is PERFECT for multi-task learning:**

```python
class ConvLSTMDwellMultiTask(nn.Module):
    """
    Predict both charging state AND amino acid type.

    Training: Use AA labels from synthetic tRNA samples
    Inference: Depends on application (see below)
    """

    def __init__(self, num_amino_acids=20):
        super().__init__()

        # Shared encoder (signal + sequence + dwell branches)
        self.encoder = ConvLSTMEncoder(...)

        # Task 1: Charging state (primary task)
        self.charging_head = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)  # Binary: charged vs uncharged
        )

        # Task 2: Amino acid type (auxiliary task)
        self.amino_acid_head = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_amino_acids + 1)  # 20 AAs + uncharged
        )

    def forward(self, signal, sequence, features):
        # Shared representation
        encoding = self.encoder(signal, sequence, features)

        # Both predictions
        charging_logits = self.charging_head(encoding)
        aa_logits = self.amino_acid_head(encoding)

        return {
            'charging': charging_logits,
            'amino_acid': aa_logits
        }
```

**Training strategy:**

```python
# Your synthetic data has both labels!
train_chunks = [
    {
        'signal': ...,
        'sequence': ...,
        'dwell': ...,
        'features': ...,
        'charging_label': 1,       # Charged
        'amino_acid_label': 0,     # Ala (0-19 encoding)
    },
    {
        ...
        'charging_label': 0,       # Uncharged
        'amino_acid_label': 20,    # Special "uncharged" class
    },
    ...
]

# Multi-task loss
def compute_loss(outputs, labels):
    # Primary task: charging (weight more)
    loss_charging = F.binary_cross_entropy_with_logits(
        outputs['charging'],
        labels['charging']
    )

    # Auxiliary task: AA type (weight less)
    loss_aa = F.cross_entropy(
        outputs['amino_acid'],
        labels['amino_acid']
    )

    # Combined (tune weights on validation)
    total_loss = 0.7 * loss_charging + 0.3 * loss_aa

    return total_loss
```

**Why this is optimal for your data:**
- ✅ Leverages ALL available labels
- ✅ AA prediction provides rich auxiliary supervision
- ✅ Shared encoder learns better representations
- ✅ Can analyze AA-specific effects scientifically
- ✅ Flexible for different inference scenarios

---

### Strategy 2: Inference Depends on Application

**Scenario A: Inferring on synthetic tRNAs (AA known)**

```python
# You know which AA this is (from experimental design)
outputs = model.predict(read)

charging_prob = outputs['charging']
aa_prediction = outputs['amino_acid']

# Validate AA prediction matches expected
assert aa_prediction == expected_aa, "AA mismatch!"

# Use charging prediction
return charging_prob
```

**Scenario B: Inferring on biological tRNAs (AA unknown)**

```python
# Don't know AA, but model predicts it!
outputs = model.predict(biological_trna_read)

charging_prob = outputs['charging']
aa_prediction = outputs['amino_acid']  # Best guess from dwell pattern

# Can use AA prediction confidence to weight charging prediction
aa_confidence = softmax(outputs['amino_acid']).max()

if aa_confidence > 0.8:
    # Model confident in AA → trust charging more
    final_prob = charging_prob
else:
    # Model uncertain → shrink toward prior
    final_prob = 0.5 + 0.5 * (charging_prob - 0.5)
```

**Scenario C: Transfer to biological tRNAs (with adaptation)**

```python
# Train on synthetic, fine-tune on biological (if available)
model_pretrained = load_model('synthetic_trained.pt')

# Fine-tune on biological tRNAs (may have different modifications)
biological_train = load_chunks('biological_trnas.npz')
fine_tune(model_pretrained, biological_train, epochs=5, lr=1e-5)

# Now model handles biological complexity
```

---

### Strategy 3: Scientific Analysis of AA Effects

**Your controlled design enables causal inference!**

```python
def analyze_aa_specific_effects(model, test_data):
    """
    Measure how each AA affects dwell time and model predictions.

    This is possible because you have controlled experiments!
    """

    results = {}

    # Uncharged baseline
    uncharged = [c for c in test_data if c['charging_label'] == 0]
    uncharged_dwell = np.mean([c['dwell'][CCA_position] for c in uncharged])

    for aa in range(20):
        # Charged with this AA
        charged_aa = [c for c in test_data
                      if c['charging_label'] == 1 and c['amino_acid_label'] == aa]

        if len(charged_aa) < 10:
            continue

        # Measure dwell effect
        charged_dwell = np.mean([c['dwell'][CCA_position] for c in charged_aa])
        dwell_delta = charged_dwell - uncharged_dwell

        # Measure model performance
        accuracy = evaluate(model, charged_aa)

        # Correlate with physical properties
        results[aa] = {
            'dwell_delta': dwell_delta,
            'accuracy': accuracy,
            'molecular_weight': AA_PROPERTIES[aa]['mw'],
            'charge': AA_PROPERTIES[aa]['charge'],
            'hydrophobicity': AA_PROPERTIES[aa]['hydrophobicity'],
        }

    # Test hypotheses
    print("\nHypothesis 1: Does AA mass correlate with dwell time?")
    masses = [r['molecular_weight'] for r in results.values()]
    dwells = [r['dwell_delta'] for r in results.values()]
    correlation, pvalue = pearsonr(masses, dwells)
    print(f"Correlation = {correlation:.3f}, p = {pvalue:.4f}")

    print("\nHypothesis 2: Do charged AAs have different dwell patterns?")
    charged_aas = [aa for aa, r in results.items() if abs(r['charge']) > 0]
    neutral_aas = [aa for aa, r in results.items() if r['charge'] == 0]
    charged_dwells = [results[aa]['dwell_delta'] for aa in charged_aas]
    neutral_dwells = [results[aa]['dwell_delta'] for aa in neutral_aas]
    stat, pvalue = mannwhitneyu(charged_dwells, neutral_dwells)
    print(f"Mann-Whitney U test: p = {pvalue:.4f}")

    return results
```

**This analysis is ONLY possible with your controlled design!** 🔬

---

## Bottom Line Answers to Your Concern (REVISED)

### Your Question:
> "Different amino acids may need different training windows. How does this impact our ability to confidently determine AA identity if multiple windows need to be evaluated?"

### My Answers (Updated with Experimental Context):

#### 1. **You DON'T need to evaluate multiple windows at inference!** ✅

The convolutional architecture **automatically handles variable receptive fields**. The model learns which temporal scales matter during training.

#### 2. **Your controlled design is PERFECT for multi-task learning!** 🎯

With synthetic tRNA + all 20 AAs separately sequenced, you can:
- Train model to predict BOTH charging state AND AA type
- Use AA labels as auxiliary supervision (improves shared representations)
- Analyze AA-specific effects scientifically
- Get both predictions at inference (AA prediction is "free")

#### 3. **AA diversity is HIGHLY INFORMATIVE in your design!** 📈

Because you have controlled experiments (same tRNA + different AAs), AA-specific dwell patterns reveal true biophysical effects of amino acids on translocation kinetics. This is scientific gold!

#### 4. **Inference depends on your goal:**

**For synthetic tRNAs** (AA known from experiment):
- Use multi-task model
- Validate AA prediction matches expected
- High confidence in charging prediction

**For biological tRNAs** (AA unknown):
- Model predicts both charging AND AA type
- Use AA prediction confidence to weight charging prediction
- Can fine-tune on biological data if available

#### 5. **No need to evaluate multiple windows!** ✅

```python
# Multi-task model (RECOMMENDED for your data)
outputs = model.predict(read)
charging_pred = outputs['charging']   # Primary task
aa_pred = outputs['amino_acid']       # Auxiliary task (helps charging!)

# Single forward pass - no multiple window evaluation needed!
```

---

## Final Recommendations (For Your Specific Experimental Design)

### Immediate Actions ✅

#### 1. **Use Multi-Task Learning from Day 1**

Given your controlled design with AA labels, skip the "start simple" approach and go straight to multi-task:

```python
# Implement ConvLSTMDwellMultiTask
model = ConvLSTMDwellMultiTask(num_amino_acids=20)

# Your data already has both labels!
train_data = load_chunks_with_aa_labels('synthetic_trnas.npz')

# Train with multi-task loss
train_multitask(
    model,
    charging_labels=train_data['charging'],
    aa_labels=train_data['amino_acid'],
    loss_weights={'charging': 0.7, 'aa': 0.3}  # Tune on validation
)
```

**Why skip "simple" single-task?**
- You have AA labels (free supervision)
- Controlled design means AA effects are real signal
- Multi-task will outperform single-task
- Can analyze AA-specific effects

#### 2. **Track and Analyze Per-AA Performance**

```python
# Essential analysis for your controlled experiment
results_by_aa = evaluate_by_amino_acid(model, test_data)

# Check consistency
for aa, metrics in results_by_aa.items():
    print(f"{aa}:")
    print(f"  Charging accuracy: {metrics['charging_acc']:.3f}")
    print(f"  AA prediction acc: {metrics['aa_acc']:.3f}")
    print(f"  Mean dwell (charged): {metrics['dwell_charged']:.2f}")
    print(f"  Mean dwell (uncharged): {metrics['dwell_uncharged']:.2f}")
    print(f"  Effect size: {metrics['cohens_d']:.2f}")
```

#### 3. **Correlate Performance with AA Physical Properties**

Your controlled design enables hypothesis testing:

```python
# Test biological hypotheses
analyze_aa_effects(
    model=model,
    test_data=test_data,
    aa_properties={
        'molecular_weight': [75, 89, 105, ...],  # 20 values
        'charge': [0, 0, -1, +1, ...],
        'hydrophobicity': [-4.5, -3.5, ...],
        'volume': [60, 88, ...]
    }
)

# Expected insights:
# - Does dwell time correlate with AA mass?
# - Do charged AAs (Asp, Glu, Lys, Arg) cluster together?
# - Do hydrophobic AAs show different kinetics?
```

#### 4. **Plan for Transfer to Biological tRNAs**

Your synthetic data is for training, but biological validation is key:

**Strategy A: Direct inference** (if biological tRNAs similar)
```python
# Train on synthetic
model = train_on_synthetic_trnas()

# Test on biological (no fine-tuning)
bio_results = evaluate(model, biological_trna_data)

# Check if synthetic→biological transfer works
```

**Strategy B: Fine-tuning** (if biological tRNAs have other modifications)
```python
# Pre-train on synthetic
model_pretrained = train_on_synthetic_trnas()

# Fine-tune on biological (few examples needed)
model_finetuned = fine_tune(
    model_pretrained,
    biological_trna_data,
    epochs=5,
    lr=1e-5,
    freeze_encoder=False  # Allow adaptation
)
```

**Strategy C: Domain adaptation** (if significant distribution shift)
```python
# Train jointly on both domains
model = ConvLSTMDwellMultiTaskDomain(
    num_amino_acids=20,
    num_domains=2  # synthetic vs biological
)

# Adversarial training to learn domain-invariant features
train_domain_adversarial(
    model,
    synthetic_data=synthetic_trnas,
    biological_data=biological_trnas
)
```

---

### Long-Term Scientific Value 🔬

Your controlled experimental design is not just for classification - it's a **biophysical measurement platform**!

**Publications you can write:**

1. **"Amino acid-specific translocation kinetics in nanopores"**
   - Measure dwell time for each of 20 AAs
   - Correlate with physical properties (mass, charge, volume)
   - Validate computational models of translocation

2. **"Machine learning reveals translocation kinetics predict tRNA charging"**
   - Show that charging state affects kinetics
   - Quantify effect sizes per AA
   - Biological implications for aa-tRNA-seq

3. **"Multi-task learning improves charging state classification"**
   - Compare single-task vs multi-task architectures
   - Show that AA prediction helps charging prediction
   - Analyze learned representations

**Your concern about AA heterogeneity is actually your biggest STRENGTH!** 🎉

---

## Bottom Line: Your Specific Situation

**Your concern**: "Different AAs may need different windows... how to handle at inference?"

**My answer**:

1. ✅ **No multiple windows needed** - convolutions handle it automatically
2. ✅ **AA heterogeneity is SIGNAL, not noise** - your controlled design isolates AA effects
3. ✅ **Multi-task learning is PERFECT** - use AA labels as auxiliary supervision
4. ✅ **One model, one forward pass** - predicts both charging + AA simultaneously
5. ✅ **Scientific gold mine** - your data enables causal inference about translocation kinetics

**Recommendation**: Skip "simple" baseline, go straight to `ConvLSTMDwellMultiTask`. Your experimental design is too good to not exploit fully!

**Expected outcome**:
- Charging accuracy: 87-93% (better than single-task due to multi-task regularization)
- AA prediction accuracy: 70-85% (depends on how discriminative dwell patterns are)
- Rich scientific insights about AA-specific translocation kinetics
- Smooth transfer to biological tRNAs (with optional fine-tuning)

**Your concern is valid but solvable - and your data is ideal for the solution!** 🚀
