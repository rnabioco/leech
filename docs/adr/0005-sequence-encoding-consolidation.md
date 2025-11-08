# ADR 0005: Sequence Encoding Consolidation

**Status:** Accepted

**Date:** 2025-01-08

## Context

DNA sequence encoding (converting "ACGT" strings to one-hot tensors) appeared in two places:

1. **`models/conv_lstm_base.py`**:
   ```python
   def encode_kmer(sequence: str) -> torch.Tensor:
       """One-hot encode a DNA sequence."""
       # Implementation...
   ```

2. **`data_prep.py`**:
   ```python
   def one_hot_encode_sequence(sequence: str) -> np.ndarray:
       """One-hot encode DNA sequence as NumPy array."""
       # Similar implementation...
   ```

Both functions performed the same task with minor differences:
- `encode_kmer()` returned PyTorch tensors (used in models)
- `one_hot_encode_sequence()` returned NumPy arrays (used in data loading)

This led to:
- Code duplication (~15 lines duplicated)
- Confusion about which function to use
- Two slightly different implementations for the same concept
- Imports from different modules depending on context

## Decision

Consolidate to a single canonical function in `data_prep.py`:

```python
def encode_kmer(sequence: str) -> torch.Tensor:
    """
    One-hot encode a DNA sequence for model input.

    This is the canonical sequence encoding function used throughout leech.
    Returns PyTorch tensor suitable for direct model input.

    Args:
        sequence: DNA sequence string (e.g., "ACGT")

    Returns:
        One-hot encoded tensor of shape (4, len(sequence))
        where channels are ordered [A, C, G, T]

    Example:
        >>> encode_kmer("ACG")
        tensor([[1., 0., 0.],  # A
                [0., 1., 0.],  # C
                [0., 0., 1.],  # G
                [0., 0., 0.]]) # T
    """
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq_len = len(sequence)
    encoded = torch.zeros(4, seq_len, dtype=torch.float32)

    for i, base in enumerate(sequence.upper()):
        if base in base_to_idx:
            encoded[base_to_idx[base], i] = 1.0

    return encoded
```

### Rationale for Location

Placed in `data_prep.py` because:
1. **Central data processing module**: Natural home for data transformation functions
2. **Used in multiple contexts**: Training data prep, inference, dataset loading
3. **Not model-specific**: Encoding is independent of model architecture
4. **Imported alongside other data functions**: `LeechRead`, `iter_bam_with_pod5()`

### Migration

- Removed `encode_kmer()` from `models/conv_lstm_base.py`
- Removed `one_hot_encode_sequence()` (unused after refactoring)
- Updated imports in:
  - `dataset.py`: `from leech.data_prep import encode_kmer`
  - `inference.py`: `from leech.data_prep import encode_kmer`

## Consequences

### Positive

- **Single source of truth**: One canonical implementation
- **Discoverability**: Clear where to import sequence encoding
- **Consistency**: All code uses identical encoding logic
- **Documentation**: Enhanced docstring with example
- **Maintainability**: Bug fixes benefit all callers

### Negative

- **Import path change**: Code importing from old location needs updating (one-time migration)

### Neutral

- Returns PyTorch tensors (models expect tensors, not NumPy arrays)
- Uppercase normalization handles case-insensitive input
- Unknown bases (not ACGT) result in all-zeros encoding (graceful degradation)

## Design Decisions

### 1. PyTorch vs. NumPy

**Decision**: Return PyTorch tensors

**Rationale**:
- Models consume tensors directly
- Dataset `__getitem__()` should return tensors
- PyTorch is already a dependency
- Conversion overhead minimal if needed

### 2. Base Ordering

**Decision**: [A, C, G, T] alphabetical ordering

**Rationale**:
- Intuitive and easy to remember
- Matches common conventions
- Consistent across codebase

### 3. Error Handling

**Decision**: Unknown bases → all-zeros (silent handling)

**Rationale**:
- Graceful degradation for ambiguous bases (N, R, Y, etc.)
- Avoids crashes on real-world data
- Models learn that all-zeros = unknown

### 4. Output Shape

**Decision**: `(4, seq_len)` - channels first

**Rationale**:
- Matches PyTorch Conv1d expectation: `(batch, channels, length)`
- Consistent with signal processing convention

## Alternatives Considered

1. **Keep both functions**: Rejected due to duplication
2. **Place in `util.py`**: Rejected - util.py is for misc helpers, not core data processing
3. **Place in models module**: Rejected - encoding is not model-specific
4. **Use sklearn or BioPython**: Rejected - too heavyweight for simple one-hot encoding

## Usage Examples

### Training Data Preparation
```python
from leech.data_prep import encode_kmer

sequence = read.get_sequence(base_idx, context)
seq_encoded = encode_kmer(sequence)  # (4, kmer_len)
```

### Dataset Loading
```python
from leech.data_prep import encode_kmer

sequence = chunk["sequence"]
seq_tensor = encode_kmer(sequence)
```

### Inference
```python
from leech.data_prep import encode_kmer

sequence = chunk["sequence"]
seq_input = encode_kmer(sequence).to(device)
```

## Notes

This ADR implements Task 3 from the refactoring plan. The consolidation eliminates confusion and ensures all sequence encoding is identical across training, evaluation, and inference.
