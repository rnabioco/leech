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

`--workers` sets how many read batches are processed at once, and it applies to
both backends (see below): worker processes without `leech_core`, threads with
it. It is not advisory on either path.

If you run into memory issues, reduce `--chunk-size` or `--workers`.

### Backends, and checking which one you got

`prepare` runs on one of two interchangeable backends. If the optional
`leech_core` Rust extension is installed and your options are all supported, it
handles each batch in a single call; otherwise the work goes to a pool of
Python worker processes. Both produce identical chunks.

The backend is named in the log and in the progress bar, along with the
achieved rate:

```text
Starting parallel data preparation with 32 workers [Rust (rayon)]
Progress [Rust (rayon)]: 40 batches, 40000 reads | 38112 chunks extracted | 1064 reads/s
```

Watch the reads/s figure on a new dataset or after changing your install. It is
the quickest way to notice that a run is heading somewhere much worse than the
last one -- a 12-hour cluster allocation was once lost to a backend regression
that showed no other symptom (issue #176).

Some options force the Python backend because the Rust pipeline cannot honor
them (`--focus-map`, non-median-MAD normalization, softclip signal recovery).
When that happens the log says so explicitly:

```text
Using Python workers instead of the Rust pipeline: focus_map is set (...)
```

### Read yield

Every run ends with the fraction of reads that produced at least one chunk:

```text
Read yield [Rust (rayon)]: 992576/1052751 reads produced chunks (94.28%); 60175 produced none
```

Reads produce no chunk when the motif is absent, or when the focus base has no
signal boundaries. That is normal and the figure is usually stable for a given
sample and motif -- which is what makes it useful: **the two backends must
agree on it.** They once did not. The Rust path silently discarded ~1% of reads
that the Python path kept, biased toward supplementary-aligned and indel-heavy
alignments, and nothing in the output said so; it took a performance comparison
to notice (issue #185).

If you switch backends and the yield moves, that is a bug, not a tuning knob.
The same number is in the returned statistics as `reads_with_motif`, which
counts reads -- not chunks.

The yield is a necessary check, not a sufficient one: the backends can agree on
which reads produced chunks and still disagree on what is *in* them. Since
v0.6.3 CI compares every field of a serialized corpus across both backends --
signals, dwells, features, the stored feature window, the `signal_kmer` fields
-- over a matrix of anchors, refinement settings, justifications and feature
windows, and fails on any chunk field it has not been told how to compare.

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

The assignment depends only on `--seed`. Read IDs are sorted before shuffling,
so the same reads at the same seed land in the same split regardless of how the
corpus was prepared -- worker count, backend, or chunk arrival order. Note that
seeds do not reproduce splits generated before v0.6.2, which were sensitive to
that ordering; re-splitting an older corpus moves reads across the train/test
boundary.

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

`prepare` echoes the window it resolved -- `Feature window: [+0, +20] relative
to the focus base, 21 bases wide` -- and records `feature_start_resolved`,
`feature_end_resolved`, and `feature_width` in `prepare_config.json`. Check
those rather than the requested values, which are null when unset. `data merge`
warns if the corpora being merged disagree on the resolved window.

!!! warning "Re-prepare `--feature-start 0` corpora built before v0.6.2"

    On the Rust backend, a falsy `--feature-start 0` was stored as the default
    `-5` even though the arrays held the requested window. Training and
    inference then sliced the k-mer window five bases off, silently. Chunk
    files written before v0.6.2 with an explicit `--feature-start 0` need
    re-preparing (or `feature_starts` rewritten in the `.npz`).

!!! warning "Re-prepare refined corpora built on the Python backend before v0.6.3"

    Signal-map refinement is on by default, and until v0.6.3 the two backends
    refined differently. The Python refiner took escapepod's fixed
    `dwell_target=4.0` -- roughly 8x too fast for RNA004, which sits near 31
    samples/base -- and then rewrote the signal with the per-read affine fit
    that the Rust path deliberately discards. Every dwell and every
    level-derived feature differed between backends.

    You are affected if you prepared with refinement on **and** the run used
    the Python backend: no `leech_core` installed, or `--workers 1`, a
    `--signal-norm` other than `median_mad`, `--recover-softclip-signal`, or a
    focus TSV. The startup line names the backend that ran.

    `--scale-iters -1` was also split: Python skipped refinement, Rust ran one
    DP pass. It now means "no refinement" on both.

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
