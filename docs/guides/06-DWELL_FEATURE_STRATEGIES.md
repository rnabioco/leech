# Dwell Feature Engineering Strategies

**Date**: 2025-11-06
**Context**: Reconsidering multi-scale dwell features vs grid search and better alternatives

---

## The Question

**Grid search vs Multi-scale**: If we're going to grid search for optimal window size anyway, why use multiple windows simultaneously? Isn't that redundant?

**Answer**: You're right! Let's reconsider the strategies.

---

## Strategy Comparison

### Strategy 1: Grid Search for Optimal Window ✅ RECOMMENDED

**Approach**: Use grid search to find the single best window size.

```python
# Grid search over window sizes
for window in [3, 5, 7, 9, 11]:
    features = compute_dwell_features(dwells, window=window)
    model = ConvLSTMDwell(num_features=5)  # Fixed number of features
    score = cross_validate(model, features)

# Pick best window (e.g., window=7 performs best)
best_window = 7
```

**Pros:**
- ✅ Simple and interpretable
- ✅ Fewer parameters (single scale)
- ✅ Less overfitting risk
- ✅ Faster training (fewer channels)
- ✅ **Empirically validated** choice

**Cons:**
- ❌ Assumes one scale is optimal for all patterns
- ❌ May miss complementary information across scales

**Verdict**: **This is probably the best starting point!**

---

### Strategy 2: Multi-Scale Features (All Windows) ⚠️ PROBABLY OVERKILL

**Approach**: Use ALL window sizes simultaneously.

```python
dwell_features = []
for window in [3, 5, 7, 9]:
    feats = compute_dwell_features(dwells, window=window)
    dwell_features.extend([feats['dwell_mean'], feats['dwell_std']])

features = np.stack(dwell_features, axis=0)  # 8 channels instead of 2
```

**Pros:**
- ✅ Captures patterns at multiple scales
- ✅ Model can learn which scales matter (via attention or weights)

**Cons:**
- ❌ 4x more channels → more parameters
- ❌ Likely redundant information (neighboring windows overlap heavily)
- ❌ Harder to interpret which scale matters
- ❌ More prone to overfitting
- ❌ **Adds complexity without clear benefit**

**When this helps:**
- Different scales capture fundamentally different patterns (rare in practice)
- You have TONS of training data (can afford the parameters)

**Verdict**: **Probably not worth it unless grid search shows multiple windows perform similarly well.**

---

### Strategy 3: Learnable Receptive Field 🚀 BETTER ALTERNATIVE

**Approach**: Let the model LEARN the optimal window size via architecture.

#### Option A: Dilated Convolutions

```python
class DwellFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # Different dilation rates = different receptive fields
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, dilation=1)  # window ~3
        self.conv2 = nn.Conv1d(1, 16, kernel_size=3, dilation=2)  # window ~5
        self.conv3 = nn.Conv1d(1, 16, kernel_size=3, dilation=4)  # window ~9

        # Concatenate multi-scale features
        self.merge = nn.Conv1d(48, 32, kernel_size=1)

    def forward(self, dwell):
        # Input: (batch, 1, kmer_len)
        f1 = F.relu(self.conv1(dwell))
        f2 = F.relu(self.conv2(dwell))
        f3 = F.relu(self.conv3(dwell))

        # Concatenate and merge
        merged = torch.cat([f1, f2, f3], dim=1)
        out = self.merge(merged)

        return out
```

**Pros:**
- ✅ Model learns optimal receptive field
- ✅ Efficient (uses convolutions, not explicit windowing)
- ✅ Multi-scale without feature explosion
- ✅ Standard deep learning practice

**Cons:**
- ❌ More complex architecture
- ❌ Harder to interpret learned patterns

#### Option B: Adaptive Pooling with Learnable Kernel

```python
class AdaptiveDwellPooling(nn.Module):
    def __init__(self, num_kernels=4):
        super().__init__()

        # Learnable pooling kernel sizes
        self.kernel_sizes = nn.Parameter(torch.tensor([3.0, 5.0, 7.0, 9.0]))

    def forward(self, dwell):
        # Interpolate to create custom kernel sizes
        # (simplified - real implementation would use differentiable pooling)
        pooled = []
        for k in self.kernel_sizes:
            pool_size = int(k.round())
            p = F.avg_pool1d(dwell, kernel_size=pool_size, stride=1, padding=pool_size//2)
            pooled.append(p)

        return torch.cat(pooled, dim=1)
```

**Verdict**: **Dilated convolutions are a proven approach for multi-scale feature learning.**

---

### Strategy 4: Hierarchical Feature Pyramid 🏗️ MODERATE COMPLEXITY

**Approach**: Build a feature pyramid like in image processing.

```python
class DwellFeaturePyramid(nn.Module):
    def __init__(self):
        super().__init__()

        # Hierarchical feature extraction
        self.level1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)  # Fine details
        self.level2 = nn.Sequential(
            nn.AvgPool1d(2),
            nn.Conv1d(1, 16, kernel_size=3, padding=1)  # Medium scale
        )
        self.level3 = nn.Sequential(
            nn.AvgPool1d(4),
            nn.Conv1d(1, 16, kernel_size=3, padding=1)  # Coarse scale
        )

        # Upsample and merge
        self.merge = nn.Conv1d(48, 32, kernel_size=1)

    def forward(self, dwell):
        # Extract features at multiple resolutions
        f1 = self.level1(dwell)
        f2 = F.interpolate(self.level2(dwell), size=dwell.shape[-1])
        f3 = F.interpolate(self.level3(dwell), size=dwell.shape[-1])

        # Concatenate and merge
        merged = torch.cat([f1, f2, f3], dim=1)
        out = self.merge(merged)

        return out
```

**Pros:**
- ✅ Captures multi-scale patterns efficiently
- ✅ Proven in computer vision (FPN, U-Net)

**Cons:**
- ❌ Complex architecture
- ❌ May be overkill for 1D sequence data

**Verdict**: **Interesting but probably over-engineered for dwell times.**

---

### Strategy 5: Context-Dependent Windowing 🎯 BIOLOGICALLY MOTIVATED

**Approach**: Use different windows for different sequence contexts.

**Rationale**:
- CCA tail (modification site): Use **small window** (local kinetics matter)
- Stem regions: Use **large window** (global structure matters)

```python
def compute_context_aware_dwell_features(
    dwells: np.ndarray,
    sequence: str,
    position: int,
) -> dict[str, np.ndarray]:
    """
    Compute dwell features with context-dependent windows.

    CCA motif → small window (focus on local kinetics)
    Other regions → larger window (capture context)
    """
    # Detect if we're at CCA site
    is_cca = sequence[position-1:position+2] == "CCA"

    # Adaptive window
    window = 3 if is_cca else 7

    # Compute features
    features = compute_dwell_features(dwells, window=window)

    return features
```

**Pros:**
- ✅ Biologically motivated
- ✅ No parameter explosion
- ✅ Interpretable (CCA is special!)

**Cons:**
- ❌ Requires domain knowledge
- ❌ Hard-coded heuristic (not learned)

**Verdict**: **Interesting hybrid approach - worth trying!**

---

## Recommended Strategy: Progressive Approach

### Phase 1: Grid Search (Week 1) ⭐ START HERE

**Do this first:**

```python
# Grid search for optimal window
window_sizes = [3, 5, 7, 9, 11]
results = {}

for window in window_sizes:
    # Compute features with this window
    train_features = compute_dwell_features(train_dwells, window=window)

    # Train model
    model = ConvLSTMDwell(num_features=5)
    score = cross_validate(model, train_features)

    results[window] = score
    print(f"Window {window}: {score:.4f}")

# Pick best
best_window = max(results, key=results.get)
print(f"Best window: {best_window}")
```

**Expected outcome:**
- Find single optimal window (e.g., 7)
- Use this in production model
- **Simple, validated, interpretable**

---

### Phase 2: Analyze Grid Search Results (Week 2)

**Check if multi-scale is needed:**

```python
# After grid search, analyze the landscape
scores = [results[w] for w in window_sizes]

# Case 1: Clear winner (one window much better)
if max(scores) - min(scores) > 0.05:  # 5% gap
    print("✅ Clear optimal window - use single scale")
    use_strategy = "single_optimal"

# Case 2: Flat landscape (all windows similar)
elif max(scores) - min(scores) < 0.02:  # <2% difference
    print("⚠️ All windows similar - multi-scale might help")
    use_strategy = "try_multiscale"

# Case 3: Bimodal (two windows good)
else:
    print("🤔 Multiple peaks - investigate further")
    use_strategy = "try_dilated_conv"
```

**Decision tree:**
- **Clear winner** → Use single optimal window ✅
- **Flat landscape** → All windows capture similar info, pick simplest
- **Multiple peaks** → Try dilated convolutions or pyramid

---

### Phase 3: If Multi-Scale Needed (Week 3) - Only If Phase 2 Suggests It

**Option A: Dilated Convolutions (Recommended)**

Instead of explicit multi-scale features, integrate dilated convolutions into the feature branch:

```python
class ConvLSTMDwellDilated(nn.Module):
    def __init__(self, ...):
        super().__init__()

        # ... signal and sequence branches unchanged ...

        # Feature branch: Dilated convolutions for multi-scale
        self.feature_conv = nn.Sequential(
            # Parallel dilated paths
            DilatedConvBlock(num_features, 64, dilations=[1, 2, 4]),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4]):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels//len(dilations),
                     kernel_size=3, dilation=d, padding=d)
            for d in dilations
        ])

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, dim=1)
```

**Why this is better than explicit multi-scale:**
- ✅ Efficient (no feature stacking)
- ✅ Learnable (model decides which scales matter)
- ✅ Standard practice (proven in WaveNet, TCN)
- ✅ Same number of output channels

**Option B: Attention Over Scales**

If you do use multi-scale features, add attention to learn which scales matter:

```python
class MultiScaleAttention(nn.Module):
    def __init__(self, num_scales=4):
        super().__init__()

        # Learn scale importance
        self.scale_attention = nn.Sequential(
            nn.Linear(256 * num_scales, 64),
            nn.ReLU(),
            nn.Linear(64, num_scales),
            nn.Softmax(dim=-1)
        )

    def forward(self, multi_scale_features):
        # multi_scale_features: list of (batch, 256, kmer_len) tensors

        # Stack scales
        stacked = torch.stack(multi_scale_features, dim=1)  # (batch, num_scales, 256, kmer_len)

        # Global pool per scale
        pooled = stacked.mean(dim=-1)  # (batch, num_scales, 256)

        # Compute attention weights
        attn_weights = self.scale_attention(pooled.flatten(1))  # (batch, num_scales)

        # Weighted combination
        weighted = (stacked * attn_weights.view(-1, num_scales, 1, 1)).sum(dim=1)

        return weighted  # (batch, 256, kmer_len)
```

---

## Updated Recommendations

### Start with Grid Search ✅

**File: src/leech/gridsearch.py** - Already exists!

```python
# Use existing grid search to optimize window size
uv run leech gridsearch \
    --pod5 reads.pod5 \
    --bam alignments.bam \
    --param window_size \
    --values 3,5,7,9,11 \
    --output gridsearch_results.json
```

Check if this is already implemented for window size parameter!

### Only Add Multi-Scale If Needed

**Don't add multi-scale features unless:**
1. Grid search shows flat landscape (all windows similar)
2. Different regions benefit from different windows
3. You have abundant training data (>50k examples)

### If Multi-Scale Needed, Use Dilated Convolutions

**Don't:** Stack explicit multi-scale features (feature explosion)

**Do:** Use dilated convolutions in feature branch (learnable, efficient)

---

## Practical Implementation

### Step 1: Check Current Grid Search Implementation

```python
# Check if gridsearch.py already handles window_size
```

Let me check your existing gridsearch implementation...

### Step 2: Add Window Size to Grid Search If Missing

If not already there, add:

```python
# In gridsearch.py
search_params = {
    'signal_context': [(100, 100), (150, 150), (200, 200)],
    'kmer_context': [3, 5, 7],
    'window_size': [3, 5, 7, 9, 11],  # ADD THIS
}
```

### Step 3: Run Grid Search

```bash
# Find optimal window
uv run leech gridsearch --config gridsearch_config.json
```

### Step 4: Use Optimal Window in Production

```python
# In features.py, use grid search result
DEFAULT_WINDOW_SIZE = 7  # From grid search

def compute_dwell_features(
    dwells: np.ndarray,
    window: int = DEFAULT_WINDOW_SIZE  # Use validated default
) -> dict[str, np.ndarray]:
    ...
```

---

## Better Idea: Make Window Size LEARNABLE 🚀

**User insight**: Can the optimal window be an OUTCOME of training, instead of testing multiple combinations?

**Answer**: YES! This is more elegant than grid search. Several approaches:

---

### Option 1: Soft Attention (Learnable Weighting) ⭐ BEST

**Idea**: Instead of hard window boundaries, learn soft weights over neighboring positions.

```python
class LearnableWindowDwellFeatures(nn.Module):
    """
    Learn optimal 'window' via soft attention over positions.

    No need to specify window size - model learns which neighbors matter!
    """

    def __init__(self, max_context=11):
        super().__init__()

        # Learn position-dependent weights
        self.position_embedding = nn.Embedding(max_context, 64)
        self.attention = nn.Sequential(
            nn.Linear(64 + 1, 32),  # +1 for dwell value
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, dwells):
        """
        Args:
            dwells: (batch, kmer_len) - raw dwell times

        Returns:
            features: (batch, num_features, kmer_len)
        """
        batch_size, kmer_len = dwells.shape
        max_ctx = self.position_embedding.num_embeddings

        # Compute windowed features with LEARNED weights
        dwell_mean = torch.zeros_like(dwells)
        dwell_std = torch.zeros_like(dwells)

        for i in range(kmer_len):
            # Extract local context
            left = max(0, i - max_ctx // 2)
            right = min(kmer_len, i + max_ctx // 2 + 1)
            context_dwells = dwells[:, left:right]  # (batch, context_len)

            # Position embeddings
            positions = torch.arange(left - i, right - i, device=dwells.device)
            pos_emb = self.position_embedding(positions + max_ctx // 2)  # (context_len, 64)

            # Combine dwell values with position embeddings
            combined = torch.cat([
                context_dwells.unsqueeze(-1),  # (batch, context_len, 1)
                pos_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, context_len, 64)
            ], dim=-1)  # (batch, context_len, 65)

            # Compute attention weights (learn which positions matter)
            attn_logits = self.attention(combined).squeeze(-1)  # (batch, context_len)
            attn_weights = F.softmax(attn_logits, dim=1)  # (batch, context_len)

            # Weighted mean and std
            dwell_mean[:, i] = (context_dwells * attn_weights).sum(dim=1)
            dwell_std[:, i] = torch.sqrt(
                ((context_dwells - dwell_mean[:, i:i+1])**2 * attn_weights).sum(dim=1)
            )

        # Stack features
        features = torch.stack([
            dwells,
            torch.log(dwells + 1e-6),
            dwell_mean,
            dwell_std,
            dwells / (dwell_mean + 1e-6)
        ], dim=1)

        return features
```

**Pros:**
- ✅ Window size learned during training (no grid search!)
- ✅ Can learn different "windows" for different positions
- ✅ Differentiable (end-to-end training)
- ✅ Interpretable (can visualize learned attention weights)

**Cons:**
- ❌ More complex than fixed window
- ❌ Slower computation (attention for each position)

---

### Option 2: Convolutions Learn Receptive Field Automatically 🎯 SIMPLEST

**Insight**: Convolutions ALREADY learn optimal receptive fields! Don't hand-engineer windows at all.

```python
class ConvolutionalDwellFeatures(nn.Module):
    """
    Let convolutions learn the optimal 'window' automatically.

    No windowing needed - conv kernels learn what context matters!
    """

    def __init__(self, num_output_features=32):
        super().__init__()

        # Stack of convolutions with increasing receptive fields
        self.conv_layers = nn.Sequential(
            # Layer 1: receptive field = 3
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            # Layer 2: receptive field = 3 + 2*1 = 5
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            # Layer 3: receptive field = 5 + 2*1 = 7
            nn.Conv1d(32, num_output_features, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, dwells):
        """
        Args:
            dwells: (batch, kmer_len) - raw dwell times

        Returns:
            features: (batch, num_output_features, kmer_len)
        """
        # Add channel dimension
        x = dwells.unsqueeze(1)  # (batch, 1, kmer_len)

        # Let convolutions learn optimal features
        # No need to manually compute dwell_mean, dwell_std, etc.
        # The network learns what matters!
        features = self.conv_layers(x)  # (batch, num_output_features, kmer_len)

        return features
```

**Pros:**
- ✅ **Simplest approach** - no manual feature engineering!
- ✅ Receptive field learned automatically through layer stacking
- ✅ Standard deep learning practice
- ✅ Fast (optimized conv operations)
- ✅ Model learns what features matter (no dwell_mean, dwell_std needed)

**Cons:**
- ❌ Less interpretable (what did it learn?)
- ❌ Fixed architecture (receptive field = function of depth)

**This is what modern deep learning does!** Let the network learn features instead of hand-crafting them.

---

### Option 3: Adaptive/Deformable Convolutions 🔬 ADVANCED

**Idea**: Learn offsets to sample points (variable receptive field per position).

```python
from torchvision.ops import DeformConv2d  # Or implement 1D version

class AdaptiveWindowConv(nn.Module):
    """
    Deformable convolutions - learn where to look for each position.

    Like attention but more efficient.
    """

    def __init__(self):
        super().__init__()

        # Predict offsets (where to sample)
        self.offset_conv = nn.Conv1d(1, 2 * 3, kernel_size=3, padding=1)  # 2 coords * kernel_size

        # Apply deformable convolution
        self.deform_conv = DeformConv1d(1, 32, kernel_size=3, padding=1)

    def forward(self, dwells):
        x = dwells.unsqueeze(1)

        # Learn where to sample
        offsets = self.offset_conv(x)

        # Apply deformable convolution
        out = self.deform_conv(x, offsets)

        return out
```

**Pros:**
- ✅ Each position learns custom receptive field
- ✅ More flexible than fixed convolutions
- ✅ Used in state-of-the-art vision models

**Cons:**
- ❌ Complex to implement
- ❌ Overkill for 1D sequence data
- ❌ Harder to debug

---

## REVISED Bottom Line

### 🥇 BEST: Just Use Convolutions! (Option 2)

**Replace manual windowing with convolutional feature extraction:**

```python
class ConvLSTMDwellLearnable(nn.Module):
    def __init__(self, signal_len=400, kmer_len=11):
        super().__init__()

        # ... signal and sequence branches unchanged ...

        # Feature branch: Let convolutions learn optimal features
        # NO manual dwell_mean, dwell_std computation needed!
        self.feature_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),   # RF = 3
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),  # RF = 5
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),  # RF = 7
            nn.ReLU(),
            nn.Conv1d(64, 256, kernel_size=3, padding=1), # RF = 9
            nn.ReLU(),
        )

    def forward(self, signal, sequence, dwell):
        # ... signal and sequence branches ...

        # Feature branch: Just pass raw dwell times!
        dwell_in = dwell.unsqueeze(1)  # (batch, 1, kmer_len)
        feat_feat = self.feature_conv(dwell_in)  # (batch, 256, kmer_len)

        # ... rest of model ...
```

**Why this is best:**
- ✅ **No grid search needed** - receptive field learned during training
- ✅ **No manual feature engineering** - network learns dwell_mean, dwell_std equivalents
- ✅ **Simpler code** - just pass raw dwell times to conv layers
- ✅ **Standard practice** - this is how modern deep learning works
- ✅ **Faster** - optimized operations
- ✅ **Scalable** - easily add more layers to increase receptive field

**Current implementation needs update:**
- ❌ Currently: Manually compute dwell_mean, dwell_std with fixed window
- ✅ Better: Pass raw dwell times, let convolutions learn features

---

### 🥈 SECOND BEST: Soft Attention (Option 1)

If you want interpretability (see which positions matter), use soft attention.

**Trade-off:**
- ✅ Can visualize learned attention weights
- ❌ Slower than convolutions
- ❌ More complex implementation

---

### 🥉 FALLBACK: Grid Search

Only if you really need interpretability AND simplicity:
- Grid search optimal window size
- Use fixed window with validated size
- But you're leaving performance on the table

---

## Recommended Implementation

### Immediate Update: ConvLSTMDwell Feature Branch

**Current implementation** (src/leech/models/conv_lstm_dwell.py):
```python
# Feature branch: Conv1d on dwell+level features
# Input: (batch, num_features, kmer_len)
self.feature_conv = nn.Sequential(
    nn.Conv1d(num_features, conv_channels[0], kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
    nn.ReLU(),
)
```

**This is ALREADY learning optimal receptive field!** ✅

The current implementation is actually correct! The conv layers are learning what context matters. The manual `compute_dwell_features()` with fixed windows is just **providing more input channels**, but the conv layers will learn to weight them appropriately.

**Two options going forward:**

#### Option A: Keep current approach (simpler)
- Compute dwell_mean, dwell_std, etc. with fixed window (say window=5)
- Feed to conv layers
- Convs learn which of these features matter

#### Option B: Simplify to raw dwell only (cleaner)
- Only pass raw dwell times to feature branch
- Let conv layers learn ALL features
- Fewer input channels, but more learnable

**My recommendation: Option B (simplify)**

---

## Action Items

### 1. Simplify Feature Extraction (Optional Refactor)

**Instead of:**
```python
features = compute_dwell_features(dwells, window=5)  # 5 channels
# → dwell, dwell_log, dwell_mean, dwell_std, dwell_ratio
```

**Try:**
```python
features = dwells.unsqueeze(0)  # Just raw dwell, 1 channel
# Let model learn log, mean, std, ratio via convolutions
```

**Pro**: Simpler, fewer hand-crafted features, model learns what matters
**Con**: Might need deeper feature branch to learn same features

### 2. Add More Conv Layers If Needed

If you remove manual features, add depth to feature branch:

```python
self.feature_conv = nn.Sequential(
    nn.Conv1d(1, 16, kernel_size=3, padding=1),    # 1 input channel now
    nn.ReLU(),
    nn.Conv1d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv1d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv1d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv1d(128, 256, kernel_size=3, padding=1),
    nn.ReLU(),
)
```

Receptive field = 1 + 2*(num_layers) = 1 + 2*5 = 11 bases

### 3. Experiment: Hand-crafted vs Learned Features

Compare both approaches:
- **ConvLSTMDwell-Manual**: Current implementation (manual dwell features)
- **ConvLSTMDwell-Auto**: Raw dwell only, deeper conv branch

See which performs better!

---

## Final Answer to Your Question

**Q: Can optimal window be an OUTCOME of training instead of testing multiple combinations?**

**A: YES! And your current model ALREADY does this!** 🎉

The convolutional layers in the feature branch are learning the optimal receptive field during training. The manual `compute_dwell_features(window=5)` is just giving the model pre-computed features, but the conv layers learn which are important.

**You can simplify even further by:**
1. Passing raw dwell times only (1 channel instead of 5)
2. Using deeper conv layers (5-6 layers)
3. Letting the model learn dwell_mean, dwell_std, etc. automatically

**This is more elegant, requires no grid search, and is standard deep learning practice!** 🚀
