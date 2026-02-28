# Data Preparation Guide

This guide explains how to prepare training data for leech models.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Sequential Processing](#sequential-processing)
- [Parallel Processing](#parallel-processing)
- [Motif Search Strategies](#motif-search-strategies)
- [Read-Level Splitting](#read-level-splitting)
- [Pairwise Comparisons](#pairwise-comparisons)
- [Troubleshooting](#troubleshooting)

## Overview

Data preparation in leech involves:

1. Reading POD5 (raw signal) and BAM (alignments with move tables) files
2. Extracting dwell times and signal features
3. Creating training chunks centered on modification sites (motifs)
4. Splitting data at the read level to prevent data leakage
5. Saving chunks to compressed .npz format

## Quick Start

### Basic Usage

```bash title="Bash" linenums="1"
# Prepare data with default settings
uv run leech data prepare \
    --pod5 reads.pod5 \
    --bam alignments.bam \
    --output-dir chunks/ \
    --motif CCAGGC \
    --motif-offset 0
```

This will:
- Extract chunks centered on "CCAGGC" motif
- Split into train/val/test at read level
- Save to `chunks/train.npz`, `chunks/val.npz`, `chunks/test.npz`

### Parallel Processing

```bash title="Bash" linenums="1"
# Use 8 workers for faster processing
uv run leech data prepare \
    --pod5 reads.pod5 \
    --bam alignments.bam \
    --output-dir chunks/ \
    --motif CCAGGC \
    --workers 8 \
    --chunk-size 100
```

Expected speedup: 3-6x on typical multi-core machines.

## Sequential Processing

### Python API

```python title="Python" linenums="1"
from pathlib import Path
from leech.data_prep import prepare_training_data_with_split

# Prepare data with splitting
result = prepare_training_data_with_split(
    pod5_path=Path("reads.pod5"),
    bam_path=Path("alignments.bam"),
    output_dir=Path("chunks/"),
    motif="CCAGGC",
    motif_offset=0,
    motif_reference="fasta",  # Search in reference sequence
    min_mapq=10,
    train_split=0.7,
    val_split=0.15,
    seed=42
)

print(f"Extracted {result['n_chunks']} chunks")
print(f"Train: {result['n_train']}, Val: {result['n_val']}, Test: {result['n_test']}")
```

### Without Splitting

```python title="Python" linenums="1"
# Extract chunks without splitting (for later merge-and-split)
result = prepare_training_data_with_split(
    pod5_path=Path("reads.pod5"),
    bam_path=Path("alignments.bam"),
    output_dir=Path("chunks/"),
    motif="CCAGGC",
    label="charged",  # Add label for identification
    no_split=True  # Save to all.npz without splitting
)
```

## Parallel Processing

Parallel processing is recommended for large datasets (>10,000 reads).

### Python API

```python title="Python" linenums="1"
from leech.data_prep import prepare_training_data_parallel
from leech.io import get_reference_sequences
from leech.splitting import split_chunks_by_read
from leech.chunking import save_chunks

# Load reference sequences (for reference-based motif search)
references = get_reference_sequences(
    bam_path=Path("alignments.bam"),
    fasta_path=Path("reference.fasta")
)

# Parallel extraction
chunks, stats = prepare_training_data_parallel(
    bam_path=Path("alignments.bam"),
    pod5_path=Path("reads.pod5"),
    motif="CCAGGC",
    motif_offset=0,
    label="charged",
    motif_reference="fasta",
    reference_sequences=references,
    num_workers=8,
    chunk_size=100
)

print(f"Extracted {stats['total_chunks']} chunks from {stats['total_reads']} reads")

# Split at read level
train, val, test = split_chunks_by_read(chunks, train_frac=0.7, val_frac=0.15, seed=42)

# Save splits
save_chunks(train, Path("chunks/train.npz"))
save_chunks(val, Path("chunks/val.npz"))
save_chunks(test, Path("chunks/test.npz"))
```

### Performance Tuning

```python title="Python" linenums="1"
# Optimize for your hardware
num_workers = 8  # Set to number of CPU cores
chunk_size = 100  # Reads per batch

# CPU-bound tasks (feature extraction): near-linear speedup
# I/O-bound tasks (POD5 reading): moderate speedup (2-4x)

# For large datasets with many small reads:
chunk_size = 200  # Larger batches reduce overhead

# For datasets with long reads:
chunk_size = 50  # Smaller batches prevent memory issues
```

## Motif Search Strategies

Leech supports two strategies for finding modification sites:

### 1. Basecalled Search (Legacy)

Search directly in the basecalled sequence:

```python title="Python" linenums="1"
from leech.io import get_motif_searcher

searcher = get_motif_searcher(mode="bam")
positions = searcher.find_motif_positions(
    read_id="read_001",
    sequence="ACGTCCAGGCTT",
    alignment=None,  # Not needed for basecalled search
    motif="CCAGGC"
)
```

**Pros:**
- Simple and fast
- No reference needed

**Cons:**
- Affected by basecalling errors at modification sites
- May miss true modification sites

### 2. Reference-Based Search (Recommended)

Search in reference sequence, then map to query:

```python title="Python" linenums="1"
from leech.io import get_motif_searcher, get_reference_sequences

# Load reference sequences
references = get_reference_sequences(
    bam_path=Path("alignments.bam"),
    fasta_path=Path("reference.fasta")
)

# Create reference-based searcher
searcher = get_motif_searcher(
    mode="fasta",
    reference_sequences=references,
    skip_indels=True  # Skip positions with indels
)

positions = searcher.find_motif_positions(
    read_id="read_001",
    sequence="...",  # Basecalled sequence (unused in reference mode)
    alignment=bam_alignment,  # Required for CIGAR-based mapping
    motif="CCAGGC"
)
```

**Pros:**
- Avoids basecalling errors at modification sites
- More accurate for modified bases
- Recommended for aa-tRNA-seq

**Cons:**
- Requires reference sequence
- Slightly slower due to CIGAR parsing

### CLI Usage

```bash title="Bash" linenums="1"
# Basecalled search
uv run leech data prepare \
    --motif CCAGGC \
    --motif-reference bam

# Reference-based search (default)
uv run leech data prepare \
    --motif CCAGGC \
    --motif-reference fasta \
    --reference-fasta reference.fasta
```

## Read-Level Splitting

**Why read-level splitting?**

Splitting at the chunk level can cause data leakage - the same molecule (read) appearing in both training and validation sets. This leads to overly optimistic performance estimates.

### Correct Workflow

```python title="Python" linenums="1"
from leech.splitting import split_chunks_by_read

# Load all chunks
chunks = load_chunks(Path("all_chunks.npz"))

# Split by read ID (not by chunk)
train, val, test = split_chunks_by_read(
    chunks,
    train_frac=0.7,
    val_frac=0.15,
    seed=42
)

# Now no read appears in multiple splits
```

### Incorrect Workflow (DO NOT DO THIS)

```python title="Python" linenums="1"
# ❌ Wrong: Split chunks directly
from sklearn.model_selection import train_test_split

train, test = train_test_split(chunks, test_size=0.3)
# Same read can appear in both train and test!
```

### Merge-Then-Split for Multi-Sample Datasets

When you have multiple samples (e.g., charged and uncharged), merge first, then split:

```python title="Python" linenums="1"
from leech.splitting import merge_and_split_chunks

# Merge multiple samples, then split at read level
result = merge_and_split_chunks(
    input_paths=[
        Path("charged_all.npz"),
        Path("uncharged_all.npz")
    ],
    output_dir=Path("merged/"),
    train_frac=0.7,
    val_frac=0.15,
    seed=42
)
```

**Why merge-then-split?**

If you split each sample independently and then merge the splits, you can still have the same read in different splits (if it appears in both samples due to technical replicates or batch effects).

## Pairwise Comparisons

For binary classification tasks, use pairwise relabeling:

### Single Labels

```python title="Python" linenums="1"
from leech.splitting import merge_and_split_chunks

# Ala vs Gly comparison
result = merge_and_split_chunks(
    input_paths=[
        Path("ala_all.npz"),
        Path("gly_all.npz")
    ],
    output_dir=Path("comparisons/Ala_vs_Gly/"),
    relabel_pairwise=("Ala", "Gly"),  # Ala=0, Gly=1
    seed=42
)
```

### Multiple Labels (Groups)

```python title="Python" linenums="1"
# Basic vs Acidic amino acids
result = merge_and_split_chunks(
    input_paths=[
        Path("lys_all.npz"),
        Path("arg_all.npz"),
        Path("glu_all.npz"),
        Path("asp_all.npz")
    ],
    output_dir=Path("comparisons/basic_vs_acidic/"),
    relabel_pairwise=(
        ["Lys", "Arg"],  # Group 0: basic
        ["Glu", "Asp"]   # Group 1: acidic
    ),
    seed=42
)
```

### Batch Processing with TSV Spec

For many comparisons, use a TSV specification file:

```tsv title="TSV" linenums="1"
# comparisons.tsv (4 columns, no header)
basic	Lys,Arg	acidic	Glu,Asp
polar	Ser,Thr	nonpolar	Ala,Val
small	Gly,Ala	large	Trp,Tyr
```

```python title="Python" linenums="1"
from leech.splitting import process_comparison_spec

result = process_comparison_spec(
    chunk_dirs=[Path("chunks/")],  # Directories with .npz files
    comparison_spec=Path("comparisons.tsv"),
    output_dir=Path("comparisons/"),
    seed=42
)

# Creates:
# comparisons/basic_vs_acidic/train.npz
# comparisons/basic_vs_acidic/val.npz
# comparisons/basic_vs_acidic/test.npz
# comparisons/basic_vs_acidic/metadata.json
# ... and same for polar_vs_nonpolar, small_vs_large
```

### CLI Usage

```bash title="Bash" linenums="1"
# Pairwise comparison
uv run leech data merge \
    -i Ala=ala.npz \
    -i Gly=gly.npz \
    -o comparisons/Ala_vs_Gly/ \
    --seed 42

# Batch processing
uv run leech data merge \
    -i chunks/dir1 \
    -i chunks/dir2 \
    --comparison-spec comparisons.tsv \
    -o comparisons/
```

## Data Validation

### Check Chunk Statistics

```python title="Python" linenums="1"
from leech.chunking import load_chunks, get_chunk_statistics

chunks = load_chunks(Path("chunks/train.npz"))
stats = get_chunk_statistics(chunks)

print(f"Total chunks: {stats['n_chunks']}")
print(f"Unique reads: {stats['n_reads']}")
print(f"Labels: {stats['labels']}")
print(f"Signal length: {stats['signal_lengths']['mean']:.1f} ± {stats['signal_lengths']['std']:.1f}")
```

### Verify No Data Leakage

```python title="Python" linenums="1"
from leech.chunking import load_chunks

# Load all splits
train = load_chunks(Path("train.npz"))
val = load_chunks(Path("val.npz"))
test = load_chunks(Path("test.npz"))

# Extract read IDs
train_reads = {c["read_id"] for c in train}
val_reads = {c["read_id"] for c in val}
test_reads = {c["read_id"] for c in test}

# Check for overlap (should be empty sets)
assert len(train_reads & val_reads) == 0, "Train-Val overlap!"
assert len(train_reads & test_reads) == 0, "Train-Test overlap!"
assert len(val_reads & test_reads) == 0, "Val-Test overlap!"

print("✓ No data leakage detected")
```

## Troubleshooting

### Issue: No chunks extracted

**Symptoms:**
```
Extracted 0 training chunks
```

**Possible causes:**

1. **Motif not found in reference:**
   ```python
   # Check if motif exists in reference
   from leech.io import get_reference_sequences
   refs = get_reference_sequences(bam_path, fasta_path)
   for name, seq in refs.items():
       count = seq.count("CCAGGC")
       print(f"{name}: {count} occurrences")
   ```

2. **Mapping quality too high:**
   ```bash
   # Lower the min_mapq threshold
   --min-mapq 5  # instead of default 10
   ```

3. **Missing required tags:**
   ```python
   # Check BAM has mv and ns tags
   import pysam
   with pysam.AlignmentFile("alignments.bam") as bam:
       aln = next(bam)
       print(f"Has mv tag: {aln.has_tag('mv')}")
       print(f"Has ns tag: {aln.has_tag('ns')}")
   ```

### Issue: Slow processing

**Solution:** Use parallel processing

```bash title="Bash" linenums="1"
# Use all CPU cores
uv run leech data prepare \
    --workers $(nproc) \
    --chunk-size 100 \
    ...
```

### Issue: Memory errors with parallel processing

**Solution:** Reduce chunk size or workers

```python title="Python" linenums="1"
# Reduce memory usage
prepare_training_data_parallel(
    ...,
    num_workers=4,  # Fewer workers
    chunk_size=50   # Smaller batches
)
```

### Issue: Read IDs don't match between POD5 and BAM

**Symptoms:**
```
ValueError: Read read_001 not found in reads.pod5
```

**Solution:** Check read ID format

```python title="Python" linenums="1"
# Check POD5 read IDs
from pod5 import DatasetReader
with DatasetReader(Path("reads.pod5")) as reader:
    for i, read in enumerate(reader.reads()):
        print(f"POD5 ID: {read.read_id}")
        if i >= 5:
            break

# Check BAM query names
import pysam
with pysam.AlignmentFile("alignments.bam") as bam:
    for i, aln in enumerate(bam):
        print(f"BAM ID: {aln.query_name}")
        if i >= 5:
            break
```

### Issue: All chunks have same label after pairwise relabeling

**Symptoms:**
```
⚠️  WARNING: All chunks have the same label (Ala)!
```

**Cause:** Label mismatch in relabel_pairwise

**Solution:** Check label values in chunk files

```python title="Python" linenums="1"
import numpy as np

# Check what labels are in the file
with np.load("ala_all.npz", allow_pickle=True) as data:
    labels = set(data["labels"])
    print(f"Labels in file: {labels}")

# Make sure relabel_pairwise matches these labels
merge_and_split_chunks(
    ...,
    relabel_pairwise=("Ala", "Gly")  # Must match exactly
)
```

## Best Practices

1. **Always use reference-based motif search** for modified bases:
   ```python
   motif_reference="fasta"
   ```

2. **Set a random seed** for reproducibility:
   ```python
   seed=42
   ```

3. **Split at read level**, never at chunk level:
   ```python
   split_chunks_by_read(chunks, ...)  # ✓ Correct
   ```

4. **Use parallel processing** for large datasets:
   ```python
   num_workers=8, chunk_size=100
   ```

5. **Validate data** after preparation:
   ```python
   stats = get_chunk_statistics(chunks)
   assert stats['n_chunks'] > 0
   assert len(stats['labels']) > 0
   ```

6. **Save intermediate results** when processing multiple samples:
   ```python
   # Save individual samples with --no-split
   # Then merge-and-split together
   ```

## See Also

- [Architecture Overview](architecture.md)
- [API Reference](api/index.md)
- [Getting Started](getting-started/quick-start.md)
- [CLI Reference](reference/cli.md)
