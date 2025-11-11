# Data Leakage Validation Report

## Summary

Comprehensive investigation and validation of data splitting in the leech pipeline to prevent data leakage, comparing our approach with Remora (the framework we extend).

**Related**: [Issue #42](https://github.com/rnabioco/leech/issues/42)

## Key Findings

### Leech's Approach (More Rigorous)

✅ **Automatic read-level splitting** - Our implementation prevents data leakage by design:

1. **`split_chunks_by_read()`** (data_prep.py:1032-1114)
   - Groups chunks by `read_id` BEFORE splitting
   - Shuffles READ IDs, not individual chunks
   - Ensures no read appears in multiple splits

2. **`merge_and_split_chunks()`** (data_prep.py:1117-1283)
   - Handles multi-sample datasets correctly
   - Merges all samples first
   - Then splits at read level
   - Memory-efficient two-pass design

3. **CLI integration**
   - `leech prepare` uses read-level splitting by default
   - `leech merge-and-split` for multi-sample workflows

### Remora's Approach (Less Rigorous)

⚠️ **Manual, prone to user error**:

1. **No built-in read-level splitting**
   - `shuffle()` function (remora/src/remora/data_chunks.py:1997) shuffles chunks randomly
   - No consideration of read IDs

2. **External validation strategy**
   - Relies on users to prepare completely separate datasets
   - Uses `--ext-val` for validation
   - Documentation doesn't mention read-level splitting

3. **Potential for leakage**
   - If users extract all chunks, shuffle, and manually split
   - Chunks from same read can end up in train and validation sets

## Validation Tests

Created comprehensive test suite (12 tests, all passing):

### 1. Read ID Preservation Tests
- ✅ `test_leech_read_preserves_id` - LeechRead stores read_id correctly
- ✅ `test_chunk_contains_read_id` - Extracted chunks contain correct read_id
- ✅ `test_extract_training_chunks_preserves_read_id` - Motif extraction preserves read_id

### 2. Read-Level Splitting Tests
- ✅ `test_split_chunks_by_read_no_leakage` - No read ID overlap between splits
- ✅ `test_split_chunks_by_read_all_chunks_assigned` - All chunks assigned to exactly one split
- ✅ `test_split_chunks_by_read_respects_proportions` - Split proportions are correct
- ✅ `test_split_chunks_by_read_reproducible` - Same seed produces identical splits

### 3. Multi-Sample Merging Tests
- ✅ `test_merge_and_split_no_leakage` - No leakage across merged samples
- ✅ `test_merge_and_split_preserves_labels` - Labels from different samples preserved

### 4. Persistence Tests
- ✅ `test_save_load_preserves_read_ids` - Read IDs preserved through save/load

### 5. Leakage Detection Tests
- ✅ `test_detect_leakage_in_splits` - Utility correctly detects leakage
- ✅ `test_no_leakage_in_clean_splits` - Utility confirms clean splits

## Read ID Flow Through Pipeline

Traced read ID preservation through entire pipeline:

```
POD5 → BAM → LeechRead → Chunk → Split
━━━━   ━━━   ━━━━━━━━━   ━━━━━   ━━━━━
 │      │        │          │       │
 │      │        │          │       └─→ split_chunks_by_read()
 │      │        │          │           groups by read_id
 │      │        │          │
 │      │        │          └─→ chunk["read_id"] = read_id
 │      │        │              (extract_training_chunks)
 │      │        │
 │      │        └─→ LeechRead(read_id=...)
 │      │            (iter_bam_with_pod5)
 │      │
 │      └─→ read_id = aln.query_name
 │          (BAM query name)
 │
 └─→ str(read.read_id)
     (POD5 read ID)
```

**Confirmed**: Read IDs flow correctly through all stages with no loss or corruption.

## Parallel Processing Validation

Verified that parallel processing maintains read-level grouping:

1. **Two-pass design** (data_prep.py:747-869):
   - Pass 1: `collect_read_infos_from_bam()` - collect lightweight read metadata
   - Pass 2: Parallel POD5 reading + feature extraction

2. **Worker function** (`_process_read_chunk_worker`):
   - Each worker processes a batch of complete reads
   - Read IDs preserved through workers
   - All chunks from a read stay together

3. **Post-processing**:
   - CLI calls `split_chunks_by_read()` after parallel extraction
   - Ensures read-level splitting regardless of parallelization

**Confirmed**: Parallel processing doesn't break read-level grouping.

## Validation Utilities

### CLI Command
```bash
leech validate-splits --data-dir /path/to/splits --verbose
```

Features:
- Checks for read ID leakage between train/val/test
- Provides severity levels (none/minor/moderate/severe)
- Detailed reports with leaked read IDs (--verbose)
- Exit code 1 if leakage detected (CI/CD friendly)

### Python API
```python
from leech.validate_splits import detect_leakage, print_leakage_report

report = detect_leakage(
    train_path=Path("train.npz"),
    val_path=Path("val.npz"),
    test_path=Path("test.npz")
)

print_leakage_report(report, verbose=True)
```

### Standalone Script
```bash
python -m leech.validate_splits /path/to/splits --verbose
```

## Example Output

### Clean Splits (No Leakage)
```
======================================================================
DATA LEAKAGE VALIDATION REPORT
======================================================================

✅ NO LEAKAGE DETECTED - All splits are clean!

----------------------------------------------------------------------
STATISTICS
----------------------------------------------------------------------
Total unique reads: 1000
  Train reads: 700 (3500 chunks)
  Val reads: 150 (750 chunks)
  Test reads: 150 (750 chunks)

----------------------------------------------------------------------
RECOMMENDATIONS
----------------------------------------------------------------------

✅ Your data splits are clean!
   • No reads appear in multiple splits
   • Safe to proceed with training
```

### Detected Leakage
```
======================================================================
DATA LEAKAGE VALIDATION REPORT
======================================================================

❌ LEAKAGE DETECTED - Severity: SEVERE

----------------------------------------------------------------------
STATISTICS
----------------------------------------------------------------------
Total unique reads: 1000
  Train reads: 700 (3500 chunks)
  Val reads: 150 (750 chunks)
  Test reads: 150 (750 chunks)

Leaked reads: 50 (5.00%)
  Train ↔ Val: 30 reads
  Train ↔ Test: 20 reads
  Val ↔ Test: 0 reads

----------------------------------------------------------------------
RECOMMENDATIONS
----------------------------------------------------------------------

⚠️  Data leakage detected! This can lead to:
   • Inflated validation metrics
   • Overfitting to specific reads
   • Poor generalization to new data

💡 To fix this issue:
   1. Use `leech merge-and-split` to merge samples and split at read level
   2. Or use `--no-split` during prepare, then merge-and-split afterward
   3. Ensure you're using split_chunks_by_read() in your code
```

## Comparison Table

| Aspect | Remora | Leech |
|--------|--------|-------|
| **Splitting level** | Not explicitly handled; relies on separate datasets | **Read-level splitting** |
| **Data leakage prevention** | Manual - users must use different samples | **Automatic** - enforced by code |
| **Multi-sample handling** | Merge then manually separate | Merge then split at read level |
| **Shuffle behavior** | Shuffles chunks randomly | Groups by read first, then shuffles reads |
| **Robustness** | Prone to leakage if users don't understand workflow | Protected against leakage |
| **Validation tools** | None built-in | `validate-splits` command + utilities |
| **Documentation** | No mention of read-level splitting | Comprehensive validation tests |

## Recommendations

### For Users

1. **Always use the built-in splitting**:
   ```bash
   leech prepare --pod5 data.pod5 --bam alignments.bam --output-dir chunks/
   # Automatically splits at read level
   ```

2. **For multi-sample datasets**:
   ```bash
   # Prepare each sample without splitting
   leech prepare --pod5 uncharged.pod5 --bam uncharged.bam --output-dir uncharged/ --no-split --label 0
   leech prepare --pod5 charged.pod5 --bam charged.bam --output-dir charged/ --no-split --label 1

   # Merge and split at read level
   leech merge-and-split -i uncharged/all.npz -i charged/all.npz -o merged/
   ```

3. **Validate your splits**:
   ```bash
   leech validate-splits --data-dir merged/ --verbose
   ```

### For Developers

1. **Always group by read_id before splitting**
2. **Use `split_chunks_by_read()` or `merge_and_split_chunks()`**
3. **Add validation tests for any new splitting code**
4. **Run `test_data_leakage.py` before committing**

## Conclusion

**Leech's approach is more rigorous than Remora's** and automatically prevents data leakage:

✅ Read-level splitting enforced by design
✅ Comprehensive validation tests (12/12 passing)
✅ Built-in validation utilities
✅ Clear documentation and recommendations
✅ Safe for production use

The validation work in this branch confirms that the leech implementation correctly prevents data leakage at all stages of the pipeline.
