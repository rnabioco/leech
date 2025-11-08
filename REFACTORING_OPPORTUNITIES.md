# Refactoring Opportunities for Leech

**Generated:** 2025-11-05
**Codebase Version:** commit 1280fc2

## Executive Summary

The leech codebase is well-structured and functional, but has several opportunities for refactoring that would improve:
- Code maintainability and DRY (Don't Repeat Yourself) compliance
- Type safety and error handling
- Testing coverage
- Performance
- Configuration management

This document identifies **11 major categories** of refactoring opportunities with **40+ specific improvements**.

---

## 1. Model Architecture Code Duplication ⭐⭐⭐

**Priority: HIGH**
**Impact: Reduces ~300 lines of duplicated code**

### Issues

#### 1.1 Shared Components Across Models
All models share nearly identical code for:
- `predict_proba()` methods (identical in all 6 models)
- Signal branch convolutions (similar patterns in ConvLSTMBase, ConvLSTMDwell, TransformerDwell)
- Sequence branch convolutions (duplicated across models)
- Feature branch convolutions (in ConvLSTMDwell, TransformerDwell, ConvOnly, TCNDwell, ResNetDwell)

**Example:** Compare `conv_lstm_base.py:133-145` with `conv_lstm_dwell.py:155-171` - identical `predict_proba()` logic.

#### 1.2 ConvLSTMBase vs ConvLSTMDwell Overlap
These two models share 70% of their code:
- Signal branch: `conv_lstm_base.py:45-52` vs `conv_lstm_dwell.py:49-56` (identical)
- Sequence branch: `conv_lstm_base.py:56-63` vs `conv_lstm_dwell.py:60-67` (identical)
- LSTM layers: `conv_lstm_base.py:73-80` vs `conv_lstm_dwell.py:89-96` (nearly identical, only input_size differs)
- FC layers: `conv_lstm_base.py:84-90` vs `conv_lstm_dwell.py:100-106` (identical)

### Recommendations

#### 1.1 Create Shared Model Components Module
Create `src/leech/models/components.py`:

```python
class SignalBranch(nn.Module):
    """Reusable signal processing branch."""
    def __init__(self, conv_channels=[4, 16, 256], kernel_size=5):
        # Shared signal conv implementation

class SequenceBranch(nn.Module):
    """Reusable sequence processing branch."""
    def __init__(self, conv_channels=[4, 16, 256], kernel_size=3):
        # Shared sequence conv implementation

class FeatureBranch(nn.Module):
    """Reusable feature processing branch."""
    def __init__(self, num_features, conv_channels=[4, 16, 256]):
        # Shared feature conv implementation

class BaseModel(nn.Module):
    """Base class for all models with shared predict_proba()."""
    def predict_proba(self, *args):
        logits = self.forward(*args)
        return torch.sigmoid(logits)
```

#### 1.2 Refactor Models to Use Components
Refactor each model to inherit from `BaseModel` and use shared branches. This would reduce:
- `ConvLSTMBase`: from 167 → ~100 lines
- `ConvLSTMDwell`: from 170 → ~110 lines
- Other models similarly

**Estimated Impact:**
- Reduces code by ~300 lines
- Makes adding new models easier (just compose branches differently)
- Centralizes component improvements

---

## 2. Forward Pass Conditional Logic Duplication ⭐⭐⭐

**Priority: HIGH**
**Impact: Improves maintainability across 4 files**

### Issues

The pattern `if "features" in batch: ... else: ...` appears in 4 locations:

1. `training.py:94-100` (train_epoch)
2. `training.py:145-149` (validate)
3. `evaluation.py:84-88` (evaluate_model)
4. `inference.py:131-136` (run_inference)

This creates maintenance burden - any change to model interfaces requires updating 4 places.

### Recommendations

#### 2.1 Unified Model Wrapper
Create a unified inference wrapper:

```python
# src/leech/models/inference_wrapper.py
class ModelInferenceWrapper:
    """Wraps models to provide unified forward pass interface."""

    def __init__(self, model, model_type: str):
        self.model = model
        self.model_type = model_type

    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        signal = batch["signal"]
        sequence = batch["sequence"]

        if self.model_type in ["ConvLSTMDwell", "TransformerDwell", ...]:
            return self.model(signal, sequence, batch["features"])
        else:
            return self.model(signal, sequence)
```

Then replace all 4 locations with:
```python
logits = model_wrapper(batch)
```

**Estimated Impact:**
- Reduces duplication from 4 places to 1
- Makes adding new model types easier
- Centralizes model calling logic

---

## 3. Sequence Encoding Duplication ⭐⭐

**Priority: MEDIUM**
**Impact: Eliminates confusion, improves consistency**

### Issues

Two nearly identical functions exist:

1. `data_prep.py:300-325` - `one_hot_encode_sequence()` - unused in current code
2. `models/conv_lstm_base.py:148-167` - `encode_kmer()` - actively used

Both do one-hot encoding but with slightly different approaches. The `data_prep.py` version is more complex (handles k-mer context) but appears unused.

### Recommendations

#### 3.1 Consolidate Encoding Functions
- Move `encode_kmer()` from `models/conv_lstm_base.py` to `data_prep.py`
- Make it the canonical implementation
- Import it in models and dataset modules
- Remove unused `one_hot_encode_sequence()` or clearly document why both exist

```python
# src/leech/data_prep.py
def encode_kmer(sequence: str) -> torch.Tensor:
    """Canonical DNA sequence one-hot encoding."""
    # Move from conv_lstm_base.py
```

Update imports in:
- `src/leech/dataset.py:12` - already imports from conv_lstm_base
- `src/leech/inference.py:16` - already imports from conv_lstm_base
- `src/leech/models/conv_lstm_base.py` - keep or remove based on dependencies

---

## 4. CLI Argument Parser Duplication ⭐⭐

**Priority: MEDIUM**
**Impact: Reduces ~50 lines, improves consistency**

### Issues

#### 4.1 Model Choices Repeated
Model choices list appears in 2 places:
- `cli.py:60-67` (train parser)
- `cli.py:124-134` (grid-search parser)

#### 4.2 Training Arguments Repeated
These arguments appear in both train and grid-search parsers:
- `--epochs`, `--batch-size`, `--learning-rate`, `--device`, `--seed`

### Recommendations

#### 4.1 Create Shared Argument Groups

```python
# src/leech/cli.py

MODEL_CHOICES = [
    "ConvLSTMDwell", "ConvLSTMBase", "TransformerDwell",
    "ConvOnly", "TCNDwell", "ResNetDwell",
]

def add_training_args(parser):
    """Add common training arguments."""
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    return parser

def add_model_args(parser):
    """Add common model arguments."""
    parser.add_argument("--model", type=str, default="ConvLSTMDwell", choices=MODEL_CHOICES)
    return parser
```

Then use in `add_train_parser()` and `add_grid_search_parser()`:
```python
parser = add_model_args(parser)
parser = add_training_args(parser)
```

---

## 5. Feature Computation Redundancy ⭐

**Priority: LOW-MEDIUM**
**Impact: Cleaner API, potential performance gain**

### Issues

#### 5.1 Overlapping Functions
`features.py` has two functions with overlapping responsibilities:

1. `compute_signal_levels()` (124-156) - Computes single stat per base
2. `compute_signal_features()` (248-285) - Computes multiple stats per base

Both iterate over bases and compute signal statistics. `compute_signal_features()` essentially calls the logic of `compute_signal_levels()` multiple times.

### Recommendations

#### 5.1 Vectorize Feature Computation
Refactor to compute all statistics in a single pass:

```python
def compute_signal_features_vectorized(
    signal: np.ndarray,
    seq_to_sig_map: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute all per-base signal features in one pass."""
    num_bases = len(seq_to_sig_map) - 1

    # Pre-allocate arrays
    features = {
        'level_mean': np.zeros(num_bases, dtype=np.float32),
        'level_median': np.zeros(num_bases, dtype=np.float32),
        'level_std': np.zeros(num_bases, dtype=np.float32),
        'level_range': np.zeros(num_bases, dtype=np.float32),
    }

    # Single loop to compute all stats
    for i in range(num_bases):
        base_sig = signal[seq_to_sig_map[i]:seq_to_sig_map[i + 1]]
        if len(base_sig) > 0:
            features['level_mean'][i] = np.mean(base_sig)
            features['level_median'][i] = np.median(base_sig)
            features['level_std'][i] = np.std(base_sig)
            features['level_range'][i] = np.ptp(base_sig)  # peak-to-peak

    return features
```

Keep `compute_signal_levels()` for single-stat use cases, but have it call a shared helper.

---

## 6. Configuration Management ⭐⭐

**Priority: MEDIUM**
**Impact: Better organization, validation, extensibility**

### Issues

#### 6.1 Config Dict Scattered Across Files
Model configuration is manually constructed in multiple places:
- `training.py:346-357` - Training config dict
- `gridsearch.py:232-242` - Grid search config dict
- Loaded/saved with manual JSON operations in `training.py` and `util.py`

#### 6.2 No Validation
No validation of config values (e.g., signal_len > 0, valid model names, etc.)

### Recommendations

#### 6.1 Create Config Classes with Pydantic

```python
# src/leech/config.py
from pydantic import BaseModel, Field, validator

class TrainingConfig(BaseModel):
    """Training configuration with validation."""
    model_name: str
    signal_len: int = Field(gt=0)
    kmer_len: int = Field(gt=0)
    num_features: int = Field(ge=0)
    epochs: int = Field(gt=0, le=1000)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0, lt=1)
    device: str = Field(pattern="^(cuda|cpu)$")
    seed: int = 42

    @validator('model_name')
    def validate_model_name(cls, v):
        from leech.models import MODEL_REGISTRY
        if v not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {v}")
        return v

    def save(self, path: Path):
        """Save config to JSON."""
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path):
        """Load config from JSON."""
        return cls.model_validate_json(path.read_text())

class GridSearchConfig(BaseModel):
    """Grid search config inheriting common training params."""
    # Similar structure
```

Benefits:
- Automatic validation
- Clear documentation of expected values
- Type safety
- Easy serialization/deserialization

---

## 7. Error Handling and Logging ⭐⭐

**Priority: MEDIUM**
**Impact: Better debugging, production readiness**

### Issues

#### 7.1 Inconsistent Error Handling
- `data_prep.py:227-229` - Catches all exceptions and prints warning
- No structured logging - uses `print()` everywhere
- Some file operations lack proper cleanup (try/finally)

#### 7.2 Silent Failures
Grid search (gridsearch.py:196-204) catches exceptions but only stores error string, losing stack trace.

### Recommendations

#### 7.1 Add Structured Logging

```python
# src/leech/logging_config.py
import logging

def setup_logging(level=logging.INFO, log_file=None):
    """Configure structured logging for leech."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

# In each module:
logger = logging.getLogger(__name__)
```

Replace all `print()` statements with appropriate logging levels:
- `print(f"Warning: ...")` → `logger.warning(...)`
- `print(f"Extracted {n} chunks")` → `logger.info(...)`
- Add `logger.debug()` for detailed diagnostics

#### 7.2 Improve Context Managers

```python
# data_prep.py - Better BAM handling
def iter_bam_with_pod5(...):
    bam = pysam.AlignmentFile(str(bam_path), "rb")
    try:
        for aln in bam:
            # ... processing ...
            yield leech_read
    finally:
        bam.close()

# Or use context manager:
with pysam.AlignmentFile(str(bam_path), "rb") as bam:
    for aln in bam:
        # ... processing ...
```

#### 7.3 Better Exception Info in Grid Search

```python
# gridsearch.py
except Exception as e:
    logger.exception(f"Training failed for grid point: {e}")  # Logs stack trace
    result = {
        ...
        "error": str(e),
        "error_type": type(e).__name__,
    }
```

---

## 8. Testing Gaps ⭐⭐⭐

**Priority: HIGH**
**Impact: Critical for code reliability**

### Issues

Only `tests/test_features.py` exists. Missing tests for:
- Models (6 architectures untested)
- Training pipeline
- Dataset loading
- Data preparation
- CLI commands
- Utility functions

### Recommendations

#### 8.1 Create Comprehensive Test Suite

```
tests/
├── test_features.py           # ✓ Exists
├── test_models.py             # Test all 6 model architectures
├── test_dataset.py            # Test LeechDataset, collate_fn
├── test_data_prep.py          # Test chunk extraction, serialization
├── test_training.py           # Test Trainer class
├── test_evaluation.py         # Test evaluation metrics
├── test_inference.py          # Test inference pipeline
├── test_cli.py                # Test CLI commands
├── test_util.py               # Test utility functions
├── fixtures/                  # Test data
│   ├── sample.pod5
│   ├── sample.bam
│   └── sample_chunks.npz
└── conftest.py                # Pytest fixtures
```

#### 8.2 Priority Test Cases

**High Priority:**
1. Model forward pass with different input shapes
2. Dataset loading with corrupted/missing data
3. Training loop convergence (simple synthetic data)
4. Chunk extraction edge cases (boundaries, short reads)

**Medium Priority:**
5. Feature computation correctness
6. Config validation
7. CLI argument parsing

**Example Model Test:**
```python
# tests/test_models.py
import pytest
import torch
from leech.models import get_model

@pytest.mark.parametrize("model_name", [
    "ConvLSTMBase", "ConvLSTMDwell", "TransformerDwell",
    "ConvOnly", "TCNDwell", "ResNetDwell"
])
def test_model_forward_pass(model_name):
    """Test that all models can do forward pass."""
    model = get_model(
        model_name,
        signal_len=400,
        kmer_len=11,
        num_features=5
    )

    batch_size = 4
    signal = torch.randn(batch_size, 400)
    sequence = torch.randn(batch_size, 4, 11)
    features = torch.randn(batch_size, 5, 11)

    if model_name == "ConvLSTMBase":
        output = model(signal, sequence)
    else:
        output = model(signal, sequence, features)

    assert output.shape == (batch_size, 1)
    assert not torch.isnan(output).any()
```

---

## 9. Performance Optimizations ⭐

**Priority: LOW-MEDIUM**
**Impact: Faster inference and training**

### Issues

#### 9.1 Double BAM Read in Inference
`inference.py:88-96` reads BAM twice:
1. Once via `iter_bam_with_pod5()` to get LeechRead
2. Again by resetting and fetching to get alignment for output

This is inefficient for large BAM files.

#### 9.2 Method Assignment Hack in Grid Search
`gridsearch.py:102` uses dynamic method assignment:
```python
read.get_chunk = custom_get_chunk  # type: ignore[method-assign]
```

This is non-idiomatic and could have performance implications.

#### 9.3 Sequential Feature Computation
`data_prep.py:201-203` computes features sequentially:
```python
dwells = compute_dwell_times(move_table)
dwell_feats = compute_dwell_features(dwells)
signal_feats = compute_signal_features(norm_signal, seq_to_sig_map)
```

Could potentially be parallelized for large batches.

### Recommendations

#### 9.1 Refactor Inference to Single BAM Pass

```python
# inference.py
def run_inference(...):
    # Open input BAM once
    with pysam.AlignmentFile(str(bam_path), "rb") as bam_in, \
         pysam.AlignmentFile(str(output_path), "wb", template=bam_in) as bam_out:

        # Process reads directly from BAM
        for aln in bam_in:
            # Check filters
            if aln.is_unmapped or ...: continue

            # Extract LeechRead from alignment + POD5
            leech_read = read_from_alignment_and_pod5(aln, pod5_path)

            # Make predictions
            predictions = predict_on_read(model, leech_read, ...)

            # Add tags to same alignment object
            aln.set_tag("MP", ...)
            aln.set_tag("ML", ...)

            # Write immediately
            bam_out.write(aln)
```

#### 9.2 Fix Grid Search Context Handling

Instead of method assignment, pass context as parameters:

```python
# gridsearch.py
def extract_training_chunks_with_context(
    leech_read: LeechRead,
    left_context: int,
    right_context: int,
    kmer_context: int,
    motif: str | None = None,
    motif_offset: int = 0,
    label: int = 0,
) -> list[dict]:
    """Extract chunks with custom signal context."""
    # ... find focus bases ...

    for base_idx in focus_bases:
        chunk = leech_read.get_chunk(
            base_idx,
            signal_context=(left_context, right_context),
            kmer_context=kmer_context
        )
        # ...
```

---

## 10. Magic Numbers and Constants ⭐

**Priority: LOW**
**Impact: Better maintainability**

### Issues

Hardcoded values scattered throughout:
- Kernel sizes: `kernel_size=5` in multiple models
- Conv channels: `[4, 16, 256]` in multiple models
- Default contexts: `(200, 200)` in data_prep.py:64
- Feature names: `"dwell"`, `"level_mean"`, etc. as strings

### Recommendations

#### 10.1 Create Constants Module

```python
# src/leech/constants.py

# Signal processing
DEFAULT_SIGNAL_CONTEXT = (200, 200)
DEFAULT_KMER_CONTEXT = 5

# Model architecture
DEFAULT_CONV_CHANNELS = [4, 16, 256]
DEFAULT_SIGNAL_KERNEL = 5
DEFAULT_SEQ_KERNEL = 3
DEFAULT_LSTM_HIDDEN = 96
DEFAULT_DROPOUT = 0.1

# Feature names
DWELL_FEATURES = ["dwell", "dwell_log", "dwell_mean", "dwell_std", "dwell_ratio"]
SIGNAL_FEATURES = ["level_mean", "level_median", "level_std", "level_range"]

# Training
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
```

Import and use:
```python
from leech.constants import DEFAULT_CONV_CHANNELS, DEFAULT_SIGNAL_KERNEL

self.signal_conv = nn.Sequential(
    nn.Conv1d(1, conv_channels[0], kernel_size=DEFAULT_SIGNAL_KERNEL, padding=2),
    # ...
)
```

---

## 11. Documentation and Code Organization ⭐

**Priority: LOW-MEDIUM**
**Impact: Easier onboarding, better maintainability**

### Issues

#### 11.1 Inconsistent Docstrings
- Some functions have detailed Args/Returns, others don't
- No Examples in docstrings for complex functions
- Missing module-level docstrings in some files

#### 11.2 Complex Logic Lacks Comments
- `MoveTable.to_seq_to_sig_map()` (features.py:40-58) - complex logic, minimal comments
- `one_hot_encode_sequence()` (data_prep.py:300-325) - k-mer encoding logic unclear

#### 11.3 No Architecture Decision Records (ADRs)
No documentation of why certain design decisions were made (e.g., why median-MAD normalization, why specific model architectures)

### Recommendations

#### 11.1 Standardize Docstring Format

Use Google or NumPy docstring style consistently:

```python
def complex_function(arg1: type1, arg2: type2) -> return_type:
    """
    Brief one-line description.

    More detailed explanation if needed. Can span multiple lines
    and explain the rationale, algorithm, or important notes.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When this happens

    Example:
        >>> result = complex_function(x, y)
        >>> print(result)
        42

    Note:
        Important implementation details or caveats
    """
```

#### 11.2 Add Inline Comments for Complex Logic

```python
def to_seq_to_sig_map(self) -> np.ndarray:
    """Convert move table to sequence-to-signal mapping."""
    # Find indices where moves occur (value == 1)
    # These represent boundaries between bases
    move_positions = np.where(self.moves == 1)[0]

    # Convert move indices to signal sample indices
    # Formula: (move_idx + 1) * stride + offset
    # The +1 accounts for 0-indexing vs 1-indexed moves
    seq_to_sig = (move_positions + 1) * self.stride + self.trim_offset

    # Prepend 0 for the start of the first base
    # This gives us a mapping: base_i starts at seq_to_sig[i]
    seq_to_sig = np.concatenate([[self.trim_offset], seq_to_sig])

    return seq_to_sig
```

#### 11.3 Create docs/ Directory

```
docs/
├── architecture/
│   ├── 001-model-architecture-choices.md
│   ├── 002-feature-engineering-rationale.md
│   └── 003-training-pipeline-design.md
├── tutorials/
│   ├── getting-started.md
│   ├── adding-new-models.md
│   └── extending-features.md
└── api/
    └── (auto-generated from docstrings)
```

---

## Summary and Prioritization

### High Priority Refactorings (Do First)

1. **Model Architecture Code Duplication** - Creates shared components module, reduces ~300 lines
2. **Forward Pass Conditional Logic** - Creates unified model wrapper, fixes 4 duplication sites
3. **Testing Gaps** - Essential for code confidence and preventing regressions

**Estimated Time:** 2-3 days
**Lines Saved:** ~350-400
**Risk:** Low (mostly additive, with tests)

### Medium Priority Refactorings (Do Next)

4. **Configuration Management** - Adds Pydantic configs for validation
5. **CLI Argument Parser Duplication** - Creates shared argument groups
6. **Error Handling and Logging** - Adds structured logging throughout
7. **Sequence Encoding Duplication** - Consolidates encoding functions

**Estimated Time:** 1-2 days
**Lines Saved:** ~100
**Risk:** Low-Medium (requires careful migration)

### Low Priority Refactorings (Nice to Have)

8. **Feature Computation Redundancy** - Vectorizes computation
9. **Performance Optimizations** - Fixes double BAM read, method assignment
10. **Magic Numbers and Constants** - Creates constants module
11. **Documentation and Code Organization** - Improves docstrings, adds ADRs

**Estimated Time:** 1-2 days
**Lines Saved:** ~50
**Risk:** Low

---

## Refactoring Strategy

### Phase 1: Foundation (Week 1)
1. Create comprehensive test suite (block other work on this)
2. Add structured logging
3. Create shared model components module

### Phase 2: Consolidation (Week 2)
4. Refactor models to use shared components
5. Create unified model wrapper
6. Add configuration management with Pydantic

### Phase 3: Polish (Week 3)
7. Fix CLI duplication
8. Consolidate encoding functions
9. Performance optimizations
10. Documentation improvements

### Phase 4: Continuous
- Add new tests as features are added
- Update documentation
- Monitor for new duplication patterns

---

## Metrics to Track

- **Code Coverage:** Currently unknown → Target 80%+
- **Duplication:** Currently ~300-400 lines → Target <100 lines
- **Cyclomatic Complexity:** Monitor functions >10
- **Type Coverage:** Use mypy strict mode, target 95%+

---

## Conclusion

The leech codebase is well-structured but would benefit significantly from these refactorings. The high-priority items (model duplication, forward pass logic, testing) would provide the most immediate value. The medium and low-priority items would improve long-term maintainability and developer experience.

**Total Estimated Impact:**
- Reduce codebase by ~500-600 lines (15-18%)
- Improve test coverage from 0% to 80%+
- Centralize configuration and validation
- Make adding new models 50% faster

These refactorings align with software engineering best practices (DRY, SOLID, proper testing) and would make the codebase more maintainable as the project grows.
