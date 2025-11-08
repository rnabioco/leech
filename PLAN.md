# Leech Refactoring Implementation Plan

**Branch:** `refactor/quick-wins`
**Strategy:** Quick wins first (high impact, low risk, fast implementation)
**Scope:** All 11 refactoring categories from REFACTORING_OPPORTUNITIES.md
**Target:** Reduce codebase by ~500-600 lines (15-18%), achieve 80%+ test coverage

---

## Overview

This plan addresses all refactoring opportunities identified in REFACTORING_OPPORTUNITIES.md, organized by **quick wins** (easiest/highest-impact first) rather than following the suggested phased approach. Each refactoring is marked with:
- **Priority:** HIGH/MEDIUM/LOW from original analysis
- **Effort:** FAST (<4 hours) / MEDIUM (4-8 hours) / COMPLEX (1-2 days)
- **Risk:** Minimal/Low/Medium/High
- **Impact:** Expected benefit

---

## Phase 1: Quick Wins (1-2 days)

### 1. Magic Numbers and Constants ⭐

**Priority:** LOW | **Effort:** FAST | **Risk:** Minimal

**Problem:** Hardcoded values scattered throughout codebase
- Kernel sizes: `kernel_size=5` in multiple models
- Conv channels: `[4, 16, 256]` in multiple models
- Default contexts: `(200, 200)` in data_prep.py
- Feature names as strings: `"dwell"`, `"level_mean"`, etc.

**Solution:**
1. Create `src/leech/constants.py`:
```python
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

# Training defaults
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
```

2. Update imports in:
   - All 6 model files (conv_lstm_base.py, conv_lstm_dwell.py, transformer_dwell.py, conv_only.py, tcn_dwell.py, resnet_dwell.py)
   - data_prep.py
   - training.py
   - cli.py

**Files to modify:**
- NEW: `src/leech/constants.py`
- EDIT: `src/leech/models/*.py` (6 files)
- EDIT: `src/leech/data_prep.py`
- EDIT: `src/leech/training.py`
- EDIT: `src/leech/cli.py`

**Testing:**
- Run existing tests to ensure no behavior change
- Visual inspection of constant usage

**Impact:** Better maintainability, ~50 lines cleaner, single source of truth

---

### 2. CLI Argument Parser Duplication ⭐⭐

**Priority:** MEDIUM | **Effort:** FAST | **Risk:** Low

**Problem:**
- Model choices list repeated in train and grid-search parsers (cli.py:60-67, 124-134)
- Training arguments repeated in both parsers: `--epochs`, `--batch-size`, `--learning-rate`, `--device`, `--seed`

**Solution:**
1. Add to cli.py:
```python
MODEL_CHOICES = [
    "ConvLSTMDwell", "ConvLSTMBase", "TransformerDwell",
    "ConvOnly", "TCNDwell", "ResNetDwell",
]

def add_training_args(parser):
    """Add common training arguments to parser."""
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser

def add_model_args(parser):
    """Add common model arguments to parser."""
    parser.add_argument("--model", type=str, default="ConvLSTMDwell",
                       choices=MODEL_CHOICES, help="Model architecture to use")
    return parser
```

2. Refactor `add_train_parser()` and `add_grid_search_parser()` to use helpers

**Files to modify:**
- EDIT: `src/leech/cli.py`

**Testing:**
- Test `uv run leech train --help`
- Test `uv run leech grid-search --help`
- Verify all arguments still work

**Impact:** Saves ~50 lines, consistent CLI interface, easier to add new arguments

---

### 3. Sequence Encoding Duplication ⭐⭐

**Priority:** MEDIUM | **Effort:** FAST | **Risk:** Low

**Problem:**
- Two nearly identical one-hot encoding functions:
  1. `data_prep.py:300-325` - `one_hot_encode_sequence()` - UNUSED
  2. `models/conv_lstm_base.py:148-167` - `encode_kmer()` - actively used
- Creates confusion and maintenance burden

**Solution:**
1. Move `encode_kmer()` from `models/conv_lstm_base.py` to `data_prep.py`
2. Make it the canonical implementation with comprehensive docstring
3. Remove or clearly document why `one_hot_encode_sequence()` exists (if kept)
4. Update imports in:
   - `src/leech/dataset.py:12`
   - `src/leech/inference.py:16`
   - Remove from `models/conv_lstm_base.py` or keep wrapper

**Files to modify:**
- EDIT: `src/leech/data_prep.py` (move encode_kmer here)
- EDIT: `src/leech/models/conv_lstm_base.py` (remove or keep wrapper)
- EDIT: `src/leech/dataset.py` (update import)
- EDIT: `src/leech/inference.py` (update import)

**Testing:**
- Run all tests (especially dataset and inference)
- Verify encoding output matches previous

**Impact:** Single source of truth, eliminates confusion, cleaner API

---

### 4. Error Handling and Logging ⭐⭐

**Priority:** MEDIUM | **Effort:** MEDIUM | **Risk:** Low

**Problem:**
- Inconsistent error handling (data_prep.py:227-229 catches all exceptions)
- No structured logging - uses `print()` everywhere (~40 locations)
- Some file operations lack proper cleanup
- Grid search loses stack traces on errors

**Solution:**
1. Create `src/leech/logging_config.py`:
```python
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

    return logging.getLogger('leech')
```

2. Add to each module:
```python
import logging
logger = logging.getLogger(__name__)
```

3. Replace all `print()` statements:
   - `print(f"Warning: ...")` → `logger.warning(...)`
   - `print(f"Extracted {n} chunks")` → `logger.info(...)`
   - Add `logger.debug()` for detailed diagnostics

4. Add context managers for BAM handling in data_prep.py

5. Improve grid search exception handling:
```python
except Exception as e:
    logger.exception(f"Training failed for grid point: {e}")
    result = {
        ...
        "error": str(e),
        "error_type": type(e).__name__,
    }
```

**Files to modify:**
- NEW: `src/leech/logging_config.py`
- EDIT: All modules with print statements:
  - `src/leech/cli.py`
  - `src/leech/data_prep.py`
  - `src/leech/training.py`
  - `src/leech/evaluation.py`
  - `src/leech/inference.py`
  - `src/leech/gridsearch.py`

**Testing:**
- Run CLI commands and verify log output
- Test with `--verbose` flag (if added)
- Verify log file creation works

**Impact:** Professional logging, better debugging, production-ready error handling

---

### 5. Configuration Management ⭐⭐

**Priority:** MEDIUM | **Effort:** MEDIUM | **Risk:** Medium

**Problem:**
- Model config manually constructed in multiple places (training.py:346-357, gridsearch.py:232-242)
- No validation of config values
- Manual JSON operations for load/save

**Solution:**
1. Create `src/leech/config.py`:
```python
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from typing import Literal

class TrainingConfig(BaseModel):
    """Training configuration with automatic validation."""
    model_name: str
    signal_len: int = Field(gt=0, description="Signal chunk length")
    kmer_len: int = Field(gt=0, description="K-mer sequence length")
    num_features: int = Field(ge=0, description="Number of dwell/level features")
    epochs: int = Field(gt=0, le=1000, description="Training epochs")
    batch_size: int = Field(gt=0, description="Batch size")
    learning_rate: float = Field(gt=0, lt=1, description="Learning rate")
    device: Literal["cuda", "cpu"] = "cuda"
    seed: int = 42

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v):
        from leech.models import MODEL_REGISTRY
        if v not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {v}. Choose from {list(MODEL_REGISTRY.keys())}")
        return v

    def save(self, path: Path):
        """Save config to JSON."""
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path):
        """Load config from JSON."""
        return cls.model_validate_json(path.read_text())

class GridSearchConfig(BaseModel):
    """Grid search configuration."""
    signal_len_range: list[int]
    kmer_context_range: list[int]
    model_name: str
    epochs: int = 10
    batch_size: int = 128
    # ... other fields
```

2. Update training.py to use TrainingConfig
3. Update gridsearch.py to use GridSearchConfig
4. Update util.py model loading to use config classes

**Files to modify:**
- NEW: `src/leech/config.py`
- EDIT: `src/leech/training.py`
- EDIT: `src/leech/gridsearch.py`
- EDIT: `src/leech/util.py`

**Testing:**
- Test config save/load roundtrip
- Test validation (invalid model name, negative values)
- Run training with config file
- Verify backward compatibility with old config JSONs

**Impact:** Type safety, automatic validation, cleaner config handling, extensibility

---

## Phase 2: High-Impact Refactorings (2-3 days)

### 6. Model Architecture Code Duplication ⭐⭐⭐

**Priority:** HIGH | **Effort:** COMPLEX | **Risk:** Medium

**Problem:**
- All 6 models share nearly identical code for:
  - `predict_proba()` methods (identical in all models)
  - Signal branch convolutions (duplicated patterns)
  - Sequence branch convolutions (duplicated)
  - Feature branch convolutions (duplicated in 5 models)
- ConvLSTMBase vs ConvLSTMDwell share 70% of code
- Estimated ~300 lines of duplication

**Solution:**
1. Create `src/leech/models/components.py`:
```python
import torch
import torch.nn as nn

class SignalBranch(nn.Module):
    """Reusable 1D convolutional branch for raw signal processing."""

    def __init__(self, conv_channels=[4, 16, 256], kernel_size=5):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, conv_channels[0], kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
        )

    def forward(self, signal):
        # signal: [batch_size, signal_len]
        signal = signal.unsqueeze(1)  # [batch_size, 1, signal_len]
        return self.conv_layers(signal)  # [batch_size, conv_channels[-1], signal_len]

class SequenceBranch(nn.Module):
    """Reusable 1D convolutional branch for one-hot sequence processing."""

    def __init__(self, conv_channels=[4, 16, 256], kernel_size=3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, conv_channels[0], kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
        )

    def forward(self, sequence):
        # sequence: [batch_size, 4, kmer_len]
        return self.conv_layers(sequence)  # [batch_size, conv_channels[-1], kmer_len]

class FeatureBranch(nn.Module):
    """Reusable 1D convolutional branch for engineered features (dwell + level)."""

    def __init__(self, num_features, conv_channels=[4, 16, 256], kernel_size=3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(num_features, conv_channels[0], kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
        )

    def forward(self, features):
        # features: [batch_size, num_features, kmer_len]
        return self.conv_layers(features)  # [batch_size, conv_channels[-1], kmer_len]

class BaseModel(nn.Module):
    """Base class for all leech models with shared predict_proba() method."""

    def predict_proba(self, *args):
        """Predict probability of positive class (charged tRNA)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(*args)
            probs = torch.sigmoid(logits)
        return probs
```

2. Refactor each model to inherit from BaseModel and use components:
   - ConvLSTMBase: Use SignalBranch + SequenceBranch
   - ConvLSTMDwell: Use all three branches
   - TransformerDwell: Use all three branches
   - ConvOnly: Use all three branches
   - TCNDwell: Use all three branches
   - ResNetDwell: Use all three branches

3. Expected line reduction:
   - ConvLSTMBase: 167 → ~100 lines
   - ConvLSTMDwell: 170 → ~110 lines
   - Other models: Similar reductions

**Files to modify:**
- NEW: `src/leech/models/components.py`
- EDIT: `src/leech/models/conv_lstm_base.py`
- EDIT: `src/leech/models/conv_lstm_dwell.py`
- EDIT: `src/leech/models/transformer_dwell.py`
- EDIT: `src/leech/models/conv_only.py`
- EDIT: `src/leech/models/tcn_dwell.py`
- EDIT: `src/leech/models/resnet_dwell.py`

**Testing:**
- **CRITICAL:** Test each model's forward pass with known inputs
- Compare old vs new model outputs (should be identical)
- Test predict_proba() for all models
- Run training for 1 epoch to ensure gradients flow
- Load old checkpoints and verify they still work

**Impact:**
- Saves ~300 lines of code
- Makes adding new models 50% faster (just compose branches)
- Centralizes improvements (fix once, all models benefit)
- Cleaner, more maintainable architecture

---

### 7. Forward Pass Conditional Logic Duplication ⭐⭐⭐

**Priority:** HIGH | **Effort:** MEDIUM | **Risk:** Low-Medium

**Problem:**
- Pattern `if "features" in batch: model(signal, sequence, features) else: model(signal, sequence)` appears in 4 locations:
  1. training.py:94-100 (train_epoch)
  2. training.py:145-149 (validate)
  3. evaluation.py:84-88 (evaluate_model)
  4. inference.py:131-136 (run_inference)
- Maintenance burden: changes to model interfaces require updating 4 places

**Solution:**
1. Create `src/leech/models/inference_wrapper.py`:
```python
import torch
from torch import nn

class ModelInferenceWrapper:
    """
    Wraps models to provide unified forward pass interface.

    Handles differences between models that take (signal, sequence) vs
    models that take (signal, sequence, features).
    """

    # Models that require dwell/level features
    FEATURE_MODELS = {
        "ConvLSTMDwell", "TransformerDwell", "ConvOnly",
        "TCNDwell", "ResNetDwell"
    }

    def __init__(self, model: nn.Module, model_type: str):
        self.model = model
        self.model_type = model_type
        self.requires_features = model_type in self.FEATURE_MODELS

    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Unified forward pass.

        Args:
            batch: Dictionary with keys "signal", "sequence", and optionally "features"

        Returns:
            Model logits [batch_size, 1]
        """
        signal = batch["signal"]
        sequence = batch["sequence"]

        if self.requires_features:
            if "features" not in batch:
                raise ValueError(f"Model {self.model_type} requires 'features' in batch")
            return self.model(signal, sequence, batch["features"])
        else:
            return self.model(signal, sequence)
```

2. Update all 4 locations to use wrapper:
```python
# In training.py, evaluation.py, inference.py
from leech.models.inference_wrapper import ModelInferenceWrapper

model_wrapper = ModelInferenceWrapper(model, model_name)
logits = model_wrapper(batch)  # Single line instead of if/else
```

**Files to modify:**
- NEW: `src/leech/models/inference_wrapper.py`
- EDIT: `src/leech/training.py` (2 locations)
- EDIT: `src/leech/evaluation.py`
- EDIT: `src/leech/inference.py`

**Testing:**
- Test training with both ConvLSTMBase and ConvLSTMDwell
- Test evaluation with both model types
- Test inference with both model types
- Verify wrapper raises error when features missing

**Impact:**
- Reduces duplication from 4 places to 1
- Makes adding new models easier (update wrapper list)
- Centralizes model calling logic
- Cleaner code in training/eval/inference

---

### 8. Documentation and Code Organization ⭐

**Priority:** LOW-MEDIUM | **Effort:** ONGOING | **Risk:** Low

**Problem:**
- Inconsistent docstrings (some functions have detailed Args/Returns, others don't)
- No Examples in docstrings for complex functions
- Missing module-level docstrings
- Complex logic lacks inline comments (e.g., MoveTable.to_seq_to_sig_map)
- No Architecture Decision Records (ADRs)

**Solution:**

**8.1 Standardize Docstring Format**

Use Google-style docstrings consistently:
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

**8.2 Add Inline Comments for Complex Logic**

Priority areas:
- `MoveTable.to_seq_to_sig_map()` (features.py:40-58)
- `one_hot_encode_sequence()` (data_prep.py:300-325)
- Feature computation loops
- Model forward passes

Example:
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

**8.3 Create Documentation Structure**

Create `docs/` directory:
```
docs/
├── architecture/
│   ├── 001-model-architecture-choices.md
│   ├── 002-feature-engineering-rationale.md
│   ├── 003-training-pipeline-design.md
│   └── 004-move-table-format.md
├── tutorials/
│   ├── getting-started.md
│   ├── adding-new-models.md
│   ├── extending-features.md
│   └── understanding-dwell-times.md
└── api/
    └── (auto-generated from docstrings via pdoc or sphinx)
```

**Architecture Decision Records (ADRs):**
- Why median-MAD normalization over z-score?
- Why ConvLSTM + feature branches?
- Why motif-based chunk extraction?
- Why separate signal/sequence/feature branches?

**Files to modify:**
- EDIT: All Python files (improve docstrings)
- EDIT: Complex functions (add inline comments)
- NEW: `docs/architecture/*.md`
- NEW: `docs/tutorials/*.md`

**Testing:**
- Run `uv run pdoc leech` to generate API docs
- Review rendered documentation
- Have someone unfamiliar with code review tutorials

**Impact:**
- Easier onboarding for new contributors
- Better understanding of design decisions
- Reduced time to answer "why" questions
- Professional-grade documentation

---

## Phase 3: Testing and Optimization (2-3 days)

### 9. Testing Gaps ⭐⭐⭐

**Priority:** HIGH | **Effort:** COMPLEX (TIME-CONSUMING) | **Risk:** Low

**Problem:**
- Only `tests/test_features.py` exists
- Missing tests for: models, training, dataset, data prep, CLI, utilities
- No integration tests
- Current coverage: ~0% (only features.py tested)

**Solution:**

**9.1 Test Structure**
```
tests/
├── test_features.py           # ✓ Already exists
├── test_models.py             # NEW: Test all 6 architectures
├── test_components.py         # NEW: Test shared components
├── test_dataset.py            # NEW: Test LeechDataset, collate_fn
├── test_data_prep.py          # NEW: Test chunk extraction, LeechRead
├── test_training.py           # NEW: Test Trainer class
├── test_evaluation.py         # NEW: Test evaluation metrics
├── test_inference.py          # NEW: Test inference pipeline
├── test_cli.py                # NEW: Test CLI parsing
├── test_util.py               # NEW: Test utility functions
├── test_config.py             # NEW: Test config validation
├── fixtures/                  # NEW: Test data
│   ├── sample.pod5            # Small POD5 for testing
│   ├── sample.bam             # Small BAM with mv tags
│   ├── sample_chunks.npz      # Pre-computed chunks
│   └── sample_model.pt        # Tiny trained model
└── conftest.py                # Pytest fixtures
```

**9.2 Priority Test Cases**

**HIGH PRIORITY:**

1. **test_models.py** - Model forward passes
```python
@pytest.mark.parametrize("model_name", [
    "ConvLSTMBase", "ConvLSTMDwell", "TransformerDwell",
    "ConvOnly", "TCNDwell", "ResNetDwell"
])
def test_model_forward_pass(model_name):
    """Test that all models can do forward pass with correct shapes."""
    model = get_model(model_name, signal_len=400, kmer_len=11, num_features=5)

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

def test_model_predict_proba():
    """Test predict_proba() returns probabilities in [0, 1]."""
    model = get_model("ConvLSTMDwell", signal_len=400, kmer_len=11, num_features=5)
    # ... test that output is in [0, 1]
```

2. **test_components.py** - Shared components
```python
def test_signal_branch():
    """Test SignalBranch forward pass."""
    branch = SignalBranch(conv_channels=[4, 16, 32], kernel_size=5)
    signal = torch.randn(2, 400)
    output = branch(signal)
    assert output.shape == (2, 32, 400)

# Similar for SequenceBranch, FeatureBranch
```

3. **test_data_prep.py** - Chunk extraction edge cases
```python
def test_leech_read_get_chunk():
    """Test chunk extraction with various contexts."""
    # Create mock LeechRead
    # Test edge cases: start of read, end of read, invalid indices

def test_chunk_extraction_boundaries():
    """Test that chunks respect read boundaries."""
    # Test left/right context at boundaries

def test_serialization_roundtrip():
    """Test save_chunks() and load_chunks() roundtrip."""
    # Save chunks, load them, verify they match
```

4. **test_dataset.py** - Dataset loading
```python
def test_leech_dataset_loading(sample_chunks):
    """Test LeechDataset can load chunks."""
    dataset = LeechDataset(sample_chunks)
    assert len(dataset) > 0

    batch = dataset[0]
    assert "signal" in batch
    assert "sequence" in batch

def test_collate_fn():
    """Test batch collation."""
    # Test that batches are properly stacked
```

**MEDIUM PRIORITY:**

5. **test_training.py** - Training loop
```python
def test_trainer_convergence(tiny_synthetic_dataset):
    """Test that Trainer can fit synthetic data."""
    # Simple dataset: all 1s → label 1, all 0s → label 0
    # Train for few epochs, verify loss decreases

def test_trainer_checkpoint_saving(tmp_path):
    """Test that checkpoints are saved correctly."""
    # Train for 2 epochs, verify checkpoint files exist
```

6. **test_evaluation.py** - Metrics
```python
def test_evaluate_model_metrics():
    """Test that evaluation returns all expected metrics."""
    # Mock model and dataset
    # Verify metrics dict has accuracy, precision, recall, etc.
```

7. **test_inference.py** - Inference pipeline
```python
def test_inference_output_format(sample_bam, sample_pod5):
    """Test inference produces valid BAM with MP/ML tags."""
    # Run inference, verify output BAM has expected tags
```

8. **test_cli.py** - CLI parsing
```python
def test_train_parser():
    """Test train command parser."""
    args = parser.parse_args(["train", "--model", "ConvLSTMDwell", ...])
    assert args.model == "ConvLSTMDwell"
```

9. **test_config.py** - Config validation
```python
def test_config_validation():
    """Test that invalid configs raise errors."""
    with pytest.raises(ValidationError):
        TrainingConfig(model_name="InvalidModel", ...)
```

**9.3 Test Fixtures**

Create `conftest.py`:
```python
import pytest
import torch
import numpy as np

@pytest.fixture
def sample_signal():
    """Generate synthetic signal data."""
    return np.random.randn(1000).astype(np.float32)

@pytest.fixture
def sample_move_table():
    """Generate synthetic move table."""
    moves = np.zeros(180, dtype=np.int8)
    moves[::10] = 1  # Move every 10 samples
    return moves

@pytest.fixture
def tiny_model():
    """Create small model for fast testing."""
    return get_model("ConvLSTMDwell", signal_len=100, kmer_len=5, num_features=3)

# ... more fixtures
```

**Files to modify:**
- NEW: 11 test files + conftest.py
- NEW: `tests/fixtures/` directory with sample data

**Testing the tests:**
- Run `uv run pytest -v`
- Run `uv run pytest --cov=leech --cov-report=term-missing`
- Target: 80%+ coverage

**Impact:**
- **CRITICAL** for code reliability
- Prevents regressions when refactoring
- Enables confident changes
- Documents expected behavior
- Professional software quality

---

### 10. Performance Optimizations ⭐

**Priority:** LOW-MEDIUM | **Effort:** MEDIUM | **Risk:** Medium

**Problem:**

**10.1 Double BAM Read in Inference**
- inference.py:88-96 reads BAM twice:
  1. Once via `iter_bam_with_pod5()` to get LeechRead
  2. Again by resetting and fetching to get alignment for output
- Inefficient for large BAM files

**10.2 Method Assignment Hack in Grid Search**
- gridsearch.py:102 uses dynamic method assignment:
  ```python
  read.get_chunk = custom_get_chunk  # type: ignore[method-assign]
  ```
- Non-idiomatic, confusing, potential performance issues

**10.3 Sequential Feature Computation**
- data_prep.py:201-203 computes features sequentially
- Could potentially be parallelized for batches

**Solution:**

**10.1 Refactor Inference to Single BAM Pass**

```python
# inference.py
def run_inference(
    model,
    pod5_path: Path,
    bam_path: Path,
    output_path: Path,
    ...
):
    """Run inference with single BAM pass."""
    pod5_reader = pod5.Reader(str(pod5_path))

    with pysam.AlignmentFile(str(bam_path), "rb") as bam_in, \
         pysam.AlignmentFile(str(output_path), "wb", template=bam_in) as bam_out:

        for aln in bam_in:
            # Apply filters
            if aln.is_unmapped or aln.mapping_quality < min_mapq:
                bam_out.write(aln)  # Write unmodified
                continue

            # Check required tags
            if not aln.has_tag("mv") or not aln.has_tag("ns"):
                bam_out.write(aln)
                continue

            try:
                # Construct LeechRead directly from alignment
                leech_read = leech_read_from_alignment(aln, pod5_reader, ...)

                # Make predictions
                predictions = predict_on_read(model, leech_read, motif, motif_offset)

                # Add MM/ML tags to same alignment object
                if predictions:
                    aln.set_tag("MP", predictions["probs"])
                    aln.set_tag("ML", predictions["labels"])

                # Write immediately
                bam_out.write(aln)

            except Exception as e:
                logger.warning(f"Failed to process read {aln.query_name}: {e}")
                bam_out.write(aln)  # Write unmodified on error

    pod5_reader.close()
```

Benefits:
- Only iterate BAM once
- Lower memory usage
- Faster inference

**10.2 Fix Grid Search Context Handling**

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
    """
    Extract training chunks with custom signal context.

    This is used by grid search to test different context sizes.
    """
    # Find focus bases (based on motif)
    if motif:
        focus_bases = find_motif_positions(leech_read.sequence, motif, motif_offset)
    else:
        focus_bases = list(range(len(leech_read.sequence)))

    chunks = []
    for base_idx in focus_bases:
        try:
            chunk = leech_read.get_chunk(
                base_idx,
                signal_context=(left_context, right_context),
                kmer_context=kmer_context
            )
            chunk["label"] = label
            chunks.append(chunk)
        except ValueError:
            continue  # Skip bases at boundaries

    return chunks

# In grid search loop:
chunks = extract_training_chunks_with_context(
    read,
    left_context,
    right_context,
    kmer_context,
    motif="CCA",
    motif_offset=2,
    label=1
)
```

**10.3 Vectorize Feature Computation (Optional)**

Profile first to see if this is actually a bottleneck. If yes:
- Batch process multiple reads
- Use numpy vectorization more aggressively
- Consider numba JIT compilation for hot loops

**Files to modify:**
- EDIT: `src/leech/inference.py` (major refactor)
- EDIT: `src/leech/gridsearch.py`
- EDIT: `src/leech/data_prep.py` (if vectorizing)

**Testing:**
- **CRITICAL:** Compare old vs new inference output (must be identical)
- Benchmark inference speed (should be faster)
- Test grid search still finds optimal contexts
- Profile to verify performance improvement

**Impact:**
- Faster inference (especially for large BAM files)
- Cleaner code (no method assignment hack)
- More idiomatic Python
- Potential memory savings

---

### 11. Feature Computation Redundancy ⭐

**Priority:** LOW-MEDIUM | **Effort:** MEDIUM | **Risk:** Low-Medium

**Problem:**
- `compute_signal_levels()` (features.py:124-156) computes single stat per base
- `compute_signal_features()` (features.py:248-285) computes multiple stats per base
- Both iterate over bases and compute signal statistics
- `compute_signal_features()` essentially calls logic of `compute_signal_levels()` multiple times

**Solution:**

Create vectorized version that computes all statistics in single pass:

```python
# features.py
def compute_signal_features_vectorized(
    signal: np.ndarray,
    seq_to_sig_map: np.ndarray,
    stats: list[str] = ["mean", "median", "std", "range"]
) -> dict[str, np.ndarray]:
    """
    Compute all per-base signal features in one pass.

    This is more efficient than calling compute_signal_levels() multiple times.

    Args:
        signal: Raw signal array
        seq_to_sig_map: Mapping from base indices to signal indices
        stats: List of statistics to compute. Options: "mean", "median", "std", "range"

    Returns:
        Dictionary mapping feature names to arrays of shape [num_bases]

    Example:
        >>> features = compute_signal_features_vectorized(signal, seq_to_sig, ["mean", "std"])
        >>> features["level_mean"].shape
        (100,)
    """
    num_bases = len(seq_to_sig_map) - 1

    # Pre-allocate output arrays
    result = {}
    if "mean" in stats:
        result["level_mean"] = np.zeros(num_bases, dtype=np.float32)
    if "median" in stats:
        result["level_median"] = np.zeros(num_bases, dtype=np.float32)
    if "std" in stats:
        result["level_std"] = np.zeros(num_bases, dtype=np.float32)
    if "range" in stats:
        result["level_range"] = np.zeros(num_bases, dtype=np.float32)

    # Single loop to compute all requested stats
    for i in range(num_bases):
        base_signal = signal[seq_to_sig_map[i]:seq_to_sig_map[i + 1]]

        if len(base_signal) == 0:
            # Handle zero-length segments (shouldn't happen but be safe)
            for key in result:
                result[key][i] = 0.0
            continue

        if "mean" in stats:
            result["level_mean"][i] = np.mean(base_signal)
        if "median" in stats:
            result["level_median"][i] = np.median(base_signal)
        if "std" in stats:
            result["level_std"][i] = np.std(base_signal)
        if "range" in stats:
            result["level_range"][i] = np.ptp(base_signal)  # peak-to-peak

    return result
```

Keep `compute_signal_levels()` for backward compatibility or single-stat use cases:
```python
def compute_signal_levels(
    signal: np.ndarray,
    seq_to_sig_map: np.ndarray,
    stat: str = "mean"
) -> np.ndarray:
    """
    Compute a single per-base signal statistic.

    For computing multiple statistics, use compute_signal_features_vectorized()
    which is more efficient.
    """
    features = compute_signal_features_vectorized(signal, seq_to_sig_map, [stat])
    return features[f"level_{stat}"]
```

Update `compute_signal_features()` to use vectorized version:
```python
def compute_signal_features(
    signal: np.ndarray,
    seq_to_sig_map: np.ndarray
) -> np.ndarray:
    """Compute all signal features and stack into array."""
    features_dict = compute_signal_features_vectorized(
        signal, seq_to_sig_map,
        stats=["mean", "median", "std", "range"]
    )

    # Stack in consistent order
    features = np.stack([
        features_dict["level_mean"],
        features_dict["level_median"],
        features_dict["level_std"],
        features_dict["level_range"]
    ], axis=0)  # Shape: [4, num_bases]

    return features
```

**Files to modify:**
- EDIT: `src/leech/features.py`
- EDIT: `src/leech/data_prep.py` (if API changes)

**Testing:**
- Test that new vectorized version produces identical results to old version
- Add parametrized tests for different stat combinations
- Benchmark to verify it's actually faster (profile first!)

**Impact:**
- Cleaner API
- More flexible (can request specific stats)
- Potentially faster (single loop, pre-allocated arrays)
- Better documented

---

## Success Metrics

### Code Quality Metrics

- **Code Reduction:** ~500-600 lines (15-18% of codebase)
  - Model duplication: -300 lines
  - Other refactorings: -200-300 lines

- **Test Coverage:** 0% → 80%+
  - All models tested
  - Critical paths covered
  - Edge cases handled

- **Duplication:** ~300-400 duplicate lines → <100 lines
  - Model architectures: unified
  - Forward pass logic: centralized
  - CLI arguments: shared functions
  - Encoding functions: single source

- **Type Coverage:**
  - Enable mypy strict mode
  - Target 95%+ type coverage
  - All config classes use Pydantic

### Performance Metrics

- **Inference Speed:**
  - Measure before/after single-pass BAM refactor
  - Target: 20-30% faster for large BAM files

- **Memory Usage:**
  - Monitor during inference refactor
  - Should be lower with single-pass approach

### Maintainability Metrics

- **Time to Add New Model:**
  - Before: ~2-3 hours (copy-paste-modify)
  - After: ~1 hour (compose shared components)
  - Target: 50% reduction

- **Documentation Coverage:**
  - All public functions have docstrings
  - All complex logic has inline comments
  - Architecture decisions documented in ADRs

---

## Implementation Checklist

### Phase 1: Quick Wins ✅

- [ ] 1. Magic Numbers and Constants
  - [ ] Create constants.py
  - [ ] Update all model files
  - [ ] Update data_prep.py, training.py, cli.py
  - [ ] Test: run existing tests

- [ ] 2. CLI Argument Parser Deduplication
  - [ ] Add MODEL_CHOICES constant
  - [ ] Create add_training_args() helper
  - [ ] Create add_model_args() helper
  - [ ] Refactor train and grid-search parsers
  - [ ] Test: `leech train --help`, `leech grid-search --help`

- [ ] 3. Sequence Encoding Deduplication
  - [ ] Move encode_kmer() to data_prep.py
  - [ ] Update imports in dataset.py, inference.py
  - [ ] Remove/document one_hot_encode_sequence()
  - [ ] Test: run dataset and inference tests

- [ ] 4. Error Handling and Logging
  - [ ] Create logging_config.py
  - [ ] Replace all print() statements (~40 locations)
  - [ ] Add context managers for BAM files
  - [ ] Improve grid search exception handling
  - [ ] Test: run CLI commands, verify log output

- [ ] 5. Configuration Management
  - [ ] Create config.py with Pydantic models
  - [ ] Update training.py to use TrainingConfig
  - [ ] Update gridsearch.py to use GridSearchConfig
  - [ ] Update util.py model loading
  - [ ] Test: config save/load, validation

### Phase 2: High-Impact Refactorings ✅

- [ ] 6. Model Architecture Code Duplication
  - [ ] Create components.py
  - [ ] Implement SignalBranch, SequenceBranch, FeatureBranch, BaseModel
  - [ ] Refactor ConvLSTMBase
  - [ ] Refactor ConvLSTMDwell
  - [ ] Refactor TransformerDwell
  - [ ] Refactor ConvOnly
  - [ ] Refactor TCNDwell
  - [ ] Refactor ResNetDwell
  - [ ] Test: forward pass for all models, compare outputs
  - [ ] Test: load old checkpoints

- [ ] 7. Forward Pass Conditional Logic Deduplication
  - [ ] Create inference_wrapper.py
  - [ ] Implement ModelInferenceWrapper
  - [ ] Update training.py (2 locations)
  - [ ] Update evaluation.py
  - [ ] Update inference.py
  - [ ] Test: train with both model types, evaluate, infer

- [ ] 8. Documentation and Code Organization
  - [ ] Standardize all docstrings
  - [ ] Add inline comments to complex functions
  - [ ] Create docs/architecture/ with ADRs
  - [ ] Create docs/tutorials/
  - [ ] Test: generate API docs with pdoc

### Phase 3: Testing and Optimization ✅

- [ ] 9. Testing Gaps
  - [ ] Create test structure (11 files + fixtures)
  - [ ] Implement test_models.py (HIGH PRIORITY)
  - [ ] Implement test_components.py
  - [ ] Implement test_data_prep.py (HIGH PRIORITY)
  - [ ] Implement test_dataset.py
  - [ ] Implement test_training.py
  - [ ] Implement test_evaluation.py
  - [ ] Implement test_inference.py
  - [ ] Implement test_cli.py
  - [ ] Implement test_util.py
  - [ ] Implement test_config.py
  - [ ] Create test fixtures (sample POD5, BAM, chunks, model)
  - [ ] Create conftest.py
  - [ ] Test: run pytest, verify 80%+ coverage

- [ ] 10. Performance Optimizations
  - [ ] Refactor inference.py to single BAM pass
  - [ ] Fix grid search method assignment
  - [ ] Profile and optimize if needed
  - [ ] Test: compare inference output, benchmark speed

- [ ] 11. Feature Computation Redundancy
  - [ ] Create compute_signal_features_vectorized()
  - [ ] Update compute_signal_levels() to use vectorized version
  - [ ] Update compute_signal_features() to use vectorized version
  - [ ] Test: verify identical output, benchmark

### Final Steps ✅

- [ ] Run full test suite: `uv run pytest -v`
- [ ] Check test coverage: `uv run pytest --cov=leech --cov-report=html`
- [ ] Run linting: `uv run ruff check .`
- [ ] Run formatting: `uv run ruff format .`
- [ ] Run type checking: `uv run mypy src/leech/`
- [ ] Update CLAUDE.md with any architecture changes
- [ ] Update README.md if needed
- [ ] Create PR with detailed description
- [ ] Request code review

---

## Estimated Timeline

| Phase | Duration | Effort |
|-------|----------|--------|
| Phase 1: Quick Wins | 1-2 days | Items 1-5 |
| Phase 2: High-Impact | 2-3 days | Items 6-8 |
| Phase 3: Testing & Optimization | 2-3 days | Items 9-11 |
| **Total** | **5-8 days** | **All 11 items** |

**Note:** Timeline assumes full-time work. Adjust based on availability.

---

## Risk Mitigation

### High-Risk Items

1. **Model Architecture Refactoring (Item 6)**
   - **Risk:** Breaking existing model behavior
   - **Mitigation:**
     - Test each model individually
     - Compare old vs new outputs with same inputs
     - Keep old model files until verified
     - Test with old checkpoints

2. **Inference Single-Pass Refactoring (Item 10)**
   - **Risk:** Changing inference output format
   - **Mitigation:**
     - Compare output BAM files byte-by-byte if possible
     - Test on small known dataset first
     - Keep old implementation until verified

3. **Configuration Management (Item 5)**
   - **Risk:** Breaking backward compatibility with old configs
   - **Mitigation:**
     - Add migration script for old configs
     - Support both old and new formats temporarily
     - Warn users about config format changes

### Medium-Risk Items

- Forward pass wrapper (Item 7): Test thoroughly with both model types
- Feature computation vectorization (Item 11): Verify numeric equivalence
- Performance optimizations (Item 10): Benchmark before/after

### Low-Risk Items

- Constants extraction (Item 1): Pure refactoring
- CLI deduplication (Item 2): Test with --help
- Documentation (Item 8): No code changes
- Testing (Item 9): Additive only

---

## Notes

- Commit frequently with descriptive messages
- Run tests after each major change
- Use feature flags if needed to toggle new behavior
- Keep REFACTORING_OPPORTUNITIES.md as reference
- Update this PLAN.md as items are completed
- Track any blockers or issues encountered

---

## Questions for Review

1. Should we maintain backward compatibility with old model checkpoints, or require retraining?
2. Should we add CLI --version flag to track when refactoring was applied?
3. Should we create a migration guide for users?
4. Should we add deprecation warnings for any old APIs?
5. Should we create a separate branch for each phase, or do all in one branch?

---

**Last Updated:** 2025-11-08
**Status:** Ready for implementation
**Assignee:** TBD
