# Troubleshooting

## Data preparation

### No chunks extracted

```
Extracted 0 training chunks
```

**Possible causes:**

1. **BAM missing move tables.** Basecall with `dorado basecaller --emit-moves`.
   Verify tags exist:
   ```bash
   uv run python -c "
   import pysam
   with pysam.AlignmentFile('alignments.bam') as bam:
       aln = next(bam)
       print(f'mv: {aln.has_tag(\"mv\")}, ns: {aln.has_tag(\"ns\")}')
   "
   ```

2. **Motif not found.** Check that your motif exists in the reference:
   ```bash
   grep -c CCAGGC reference.fa
   ```

3. **All reads filtered.** Lower `--min-mapq` or check alignment rates.

4. **Insufficient context.** Reads near chromosome ends may lack enough
   flanking signal. Reduce `--signal-context` if needed.

### Read ID mismatch between POD5 and BAM

```
ValueError: Read read_001 not found in reads.pod5
```

Read IDs must match exactly. Print a few from each file to compare:

```bash
# BAM read IDs
uv run python -c "
import pysam
with pysam.AlignmentFile('alignments.bam') as bam:
    for i, aln in enumerate(bam):
        print(aln.query_name)
        if i >= 3: break
"

# POD5 read IDs
uv run python -c "
from escapepod import DatasetReader
reader = DatasetReader('reads.pod5')
for i, read in enumerate(reader.reads()):
    print(read.read_id)
    if i >= 3: break
"
```

### Slow data preparation

Use parallel processing: `--workers 8 --chunk-size 100`. Set workers to
your core count and adjust chunk size based on read length (smaller for
long reads, larger for short reads).

Check the reported rate before tuning anything else. Every progress line
carries the backend and the achieved reads/s:

```text
Progress [Rust (rayon)]: 40 batches, 40000 reads | 38112 chunks extracted | 1064 reads/s
```

If that number is far below what the same data managed before, the problem is
not your `--workers` setting. Compare the two backends directly with
`--backend python`, which forces the worker pool for one run.

Preparation on a large POD5 over a network filesystem (BeeGFS, Lustre, NFS) is
usually latency-bound on POD5 reads rather than CPU-bound, so a run can crawl
while the cores sit idle. `--workers` is the lever that helps: it sets how many
read batches are in flight, and therefore how many POD5 reads are outstanding
at once. Watching CPU alone will mislead you here -- sample it as a delta over
an interval, since `sstat`'s `AveCPU` is cumulative and hides an idle pool.

### Fewer chunks than a previous run, or than the other backend

Compare the read yield line at the end of each run:

```text
Read yield [Rust (rayon)]: 992576/1052751 reads produced chunks (94.28%); 60175 produced none
```

The two backends must report the same number on the same input. Force the
Python worker pool for one run with `--backend python` and compare.

A yield that differs by backend is a bug -- please report it. Up to v0.6.0 the
Rust path dropped ~1% of reads the Python path kept, non-randomly, and said
nothing (issue #185); a corpus prepared with the Rust backend before v0.6.1
should be re-prepared.

A yield that is simply lower than you expected, equally on both backends, is
about your data rather than your install: reads whose motif is absent, or whose
focus base falls outside the aligned region, legitimately produce no chunk. See
[No chunks extracted](#no-chunks-extracted).

### The feature window is not the one I asked for

`prepare` logs the window it resolved:

```text
Feature window: [+0, +20] relative to the focus base, 21 bases wide (kmer_context=5)
```

`--feature-start` / `--feature-end` are optional and fall back to
+/-`kmer_context` when unset, so a window that looks ignored is usually one
that was never passed through. `prepare_config.json` records the same values as
`feature_start_resolved` / `feature_end_resolved` / `feature_width`; the plain
`feature_start` / `feature_end` keys are null when unset, and are the request
rather than the result.

Up to v0.6.1 the Rust backend stored `-5` for a requested `--feature-start 0`
even though the arrays held the requested window (issue #189). The corpus is
not visibly wrong -- the shapes are right -- but training and inference then
slice the k-mer window `kmer_context` bases off. Chunk files written before
v0.6.2 with an explicit `--feature-start 0` should be re-prepared.

### Memory errors during preparation

Reduce `--chunk-size` (fewer reads per batch) or `--workers` (fewer
parallel processes).

## Training

### CUDA out of memory

```
RuntimeError: CUDA out of memory
```

- Reduce `--batch-size` (try 64 or 32)
- Reduce `--signal-context` during data preparation
- Use `--device cpu` (slower but no VRAM limit)

### Training doesn't converge

- Check that your data has both classes (`label 0` and `label 1`)
- Try a lower learning rate: `--learning-rate 0.0001`
- Enable focal loss for imbalanced data: `--loss focal`
- Run grid search to find optimal signal context

### Validation accuracy plateaus early

- Add regularization: `--weight-decay 0.01`
- Use gradient clipping: `--max-grad-norm 1.0`
- Enable LR scheduling: `--scheduler reduce_on_plateau`
- Add data augmentation: `--augment-jitter 0.01`

## Inference

### Model loading errors

```
RuntimeError: Error loading model checkpoint
```

- Ensure the model architecture matches what was used during training
  (the `config.json` alongside `model_best.pt` records this)
- Check PyTorch version compatibility
- Verify the `.pt` file is not corrupted

### Bundle inference errors

```
KeyError: Pair 'Ala_vs_Gly' not in bundle
```

List available pairs with `leech model bundle-info --bundle FILE`.
Pair names must match exactly (case-sensitive).

### Wrong predictions on RNA data

If predictions seem random, check signal orientation. Direct RNA data
needs signal reversal (the default). If you're running on DNA data, use
`--no-reverse-signal`.

## Snakemake pipeline

### Lock files

```
LockException: Error: Directory cannot be locked
```

A previous Snakemake run was interrupted. Unlock with:

```bash
snakemake --unlock
```

### QoS error (SLURM)

```
sbatch: error: A Quality of Service (QoS) has not been provided
```

Set `slurm-qos: "normal"` as a top-level key in your SLURM profile's
`config.yaml` (not under `default-resources`).

### Jobs timing out

Increase `runtime` in rule resources, or use the `long` QoS for grid
search jobs.

### POD5 files not found

Verify paths in your pipeline config. On Alpine, ensure raw data is in
`/projects/` (not `/scratch/`, which is auto-purged after 90 days).

## General

### `uv run leech` not found

```
error: No module named leech
```

Run `uv sync` from the project root to install the package.

### Checking your installation

```bash
uv run leech --version
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```
