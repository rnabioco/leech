# Data Preparation

This guide covers preparing training data for leech models: extracting
features from nanopore reads, handling multi-sample experiments, and validating
your data.

## Input requirements

You need two files per sample:

- **POD5 file** -- raw nanopore signal from ONT sequencing
- **BAM file** -- basecalls with move table tags (`mv`, `ns`), produced by
  Dorado with `--emit-moves`

Read IDs must match exactly between the POD5 and BAM files.

## Basic preparation

Extract training chunks centered on a sequence motif:

```bash
leech data prepare \
  --pod5 reads.pod5 \
  --bam alignments.bam \
  --output-dir chunks/ \
  --motif CCAGGC \
  --motif-offset 2 \
  --label 1
```

This will:

1. Read each alignment from the BAM, extract its move table and raw signal
   from the POD5
2. Search for the motif in the reference sequence and map to query coordinates
3. Extract a signal chunk and per-base features centered on the motif site
4. Split into `train.npz`, `val.npz`, `test.npz` at the read level

### Parallel processing

For large datasets, use multiple workers:

```bash
leech data prepare \
  --pod5 reads.pod5 \
  --bam alignments.bam \
  --output-dir chunks/ \
  --workers 8 \
  --chunk-size 100
```

Expected speedup: 3--6x on typical multi-core machines. CPU-bound tasks
(feature extraction) scale near-linearly; I/O-bound tasks (POD5 reading)
see 2--4x improvement.

If you run into memory issues, reduce `--chunk-size` or `--workers`.

## Motif search strategies

Leech supports two strategies for locating modification sites.

### Reference-based search (default)

Searches for the motif in the reference sequence, then maps the position to
query coordinates via the CIGAR string. This avoids bias from basecalling
errors at modification sites, which is important because modifications can
cause systematic miscalls.

```bash
leech data prepare \
  --motif CCAGGC \
  --motif-reference fasta \
  --reference-fasta genome.fa \
  --skip-motif-indels
```

Use `--skip-motif-indels` to discard reads with insertions or deletions in the
motif region, where coordinate mapping is unreliable.

### Basecalled search

Searches directly in the basecalled sequence. Use when a reference is
unavailable or for backward compatibility:

```bash
leech data prepare \
  --motif CCAGGC \
  --motif-reference bam
```

## Signal orientation

By default, leech reverses the raw signal for direct RNA sequencing. POD5
files store RNA signal in 3'->5' order, but the basecaller (and therefore the
move table) operates 5'->3'. Leech reverses to match.

For DNA data, disable reversal:

```bash
leech data prepare --no-reverse-signal ...
```

## Signal justification

The `--base-justify` option controls where the signal chunk is centered within
the focus base:

- `center` (default) -- midpoint of the focus base's signal region
- `start` -- first signal sample of the focus base
- `end` -- last signal sample (useful for 3' modifications like aminoacylation)

## Multi-sample datasets

When you have multiple samples (e.g., charged and uncharged tRNAs from
different experiments), the correct workflow is:

1. Prepare each sample separately with `--no-split`
2. Merge and split together with `leech data merge`

This ensures that read-level splitting is done across *all* samples
simultaneously, preventing data leakage.

### Step 1: Prepare each sample

```bash
leech data prepare --pod5 charged.pod5 --bam charged.bam \
  --output-dir chunks/charged/ --no-split --label 1

leech data prepare --pod5 uncharged.pod5 --bam uncharged.bam \
  --output-dir chunks/uncharged/ --no-split --label 0
```

### Step 2: Merge and split

```bash
leech data merge \
  -i charged=chunks/charged/all.npz \
  -i uncharged=chunks/uncharged/all.npz \
  -o merged/
```

This creates `merged/train.npz`, `merged/val.npz`, `merged/test.npz` with no
read appearing in more than one split.

### Pairwise amino acid comparisons

For binary amino acid classification:

```bash
leech data merge \
  -i Ala=ala.npz \
  -i Gly=gly.npz \
  -o comparisons/Ala_vs_Gly/
```

For group comparisons (e.g., chemical properties):

```bash
leech data merge \
  -i basic=lys.npz -i basic=arg.npz \
  -i acidic=asp.npz -i acidic=glu.npz \
  -o comparisons/basic_vs_acidic/
```

### Batch comparisons with a TSV spec

For many comparisons at once, use a TSV file:

```
basic	Lys,Arg	acidic	Glu,Asp
polar	Ser,Thr	nonpolar	Ala,Val
```

```bash
leech data merge \
  -i chunks/dir1 -i chunks/dir2 \
  --comparison-spec comparisons.tsv \
  -o comparisons/
```

## Why read-level splitting matters

Splitting at the *chunk* level (e.g., using sklearn's `train_test_split` on
individual chunks) allows the same molecule to appear in both training and
test sets. This inflates performance metrics because the model has already seen
signal from that molecule.

Leech always splits at the *read* level: all chunks from a given read go into
the same split.

## Widening the feature window for offset tuning

If you plan to search over dwell offsets during grid search, prepare data with a
wider feature window using `--feature-start` and `--feature-end`. These set the
feature window bounds as signed offsets from the focus base (negative = toward
the tRNA body, positive = toward the adaptor). The defaults are `-5` and `5`
(i.e., `±kmer_context`).

```bash
leech data prepare --feature-start -20 --feature-end 20 ...
```

Storing extra bases on each side of the dwell/feature arrays allows
`leech model optimize --dwell-offsets` to shift the window at runtime without
re-preparing data. For a right-only window, use e.g. `--feature-start 0
--feature-end 20`.

## Troubleshooting

### No chunks extracted

Check that:

1. Your BAM file has `mv` and `ns` tags (run with `--emit-moves` during basecalling)
2. The motif exists in your reference sequences
3. `--min-mapq` isn't filtering out all reads (try `--min-mapq 0`)

### Read IDs don't match

POD5 and BAM read IDs must be identical. Check a few IDs from each file to
confirm they use the same format.

### Memory errors with parallel processing

Reduce `--chunk-size` (fewer reads per batch) or `--workers` (fewer parallel
processes).
