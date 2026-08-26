# CLI Reference

Complete reference for the `leech` command-line interface.

## Command overview

The CLI is organized into workflow-based command groups:

| Group | Commands | Purpose |
|-------|----------|---------|
| `leech data` | `prepare`, `merge` | Extract features, merge and split datasets |
| `leech model` | `train`, `optimize`, `benchmark`, `bundle`, `bundle-info`, `calibrate`, `export`, `release`, `list`, `fetch` | Train, tune, calibrate, package, and publish models |
| `leech eval` | `test`, `compare`, `importance`, `ablation` | Evaluate and analyze models |
| `leech predict` | *(top-level)* | Run inference on new data |

```bash
leech --help       # Show all commands
leech --version    # Show version
```

---

## Data commands

### leech data prepare

Extract training chunks from POD5 and BAM files centered on a sequence motif.

```bash
leech data prepare --pod5 FILE --bam FILE --output-dir DIR [OPTIONS]
```

**Required:**

| Option | Description |
|--------|-------------|
| `--pod5 FILE` | POD5 file with raw nanopore signal |
| `--bam FILE` | BAM file with alignments and move tables (`mv`, `ns` tags) |
| `--output-dir DIR` | Directory to save training chunks |

**Feature extraction:**

| Option | Default | Description |
|--------|---------|-------------|
| `--feature-set STR` | `signal+dwell+levels` | Features to extract (combine with `+`) |
| `--signal-context LEFT RIGHT` | *(symmetric 200/200)* | Asymmetric signal window as two ints (e.g. `--signal-context 90 450`) |
| `--feature-start INT` | `-5` | Feature window start offset from focus base (negative = toward tRNA body) |
| `--feature-end INT` | `5` | Feature window end offset from focus base (positive = toward adaptor) |

**Motif search:**

| Option | Default | Description |
|--------|---------|-------------|
| `--motif STR` | -- | Sequence motif to center on |
| `--motif-offset INT` | `0` | Focus base within motif (0-indexed) |
| `--motif-reference STR` | `fasta` | Search in `fasta` (reference) or `bam` (basecalled sequence) |
| `--reference-fasta FILE` | -- | Reference FASTA if not in BAM header |
| `--skip-motif-indels` | `False` | Skip motif sites with indels in alignment |
| `--require-query-mapping / --no-require-query-mapping` | enabled | Require a reference motif to also map cleanly to query coordinates (`--motif-reference fasta`, `--anchor reference`). See note below. |

!!! note "When to use `--no-require-query-mapping`"

    By default a motif found in the reference must also map cleanly to query
    coordinates. That mapping is only a quality gate -- its result is discarded,
    because the chunk is positioned from the reference through CIGAR
    interpolation -- so turning it off keeps reads whose motif basecalled badly
    without moving any chunk.

    Use it when the modification under study mis-calls the motif itself.
    Otherwise those reads are dropped for carrying the very signal being
    measured.

    The setting is recorded in `prepare_config.json`, carried into the model
    config by `leech model train`, and applied again by `leech predict`, so a
    model is scored on the same read population it was trained on.

**Signal handling:**

| Option | Default | Description |
|--------|---------|-------------|
| `--base-justify STR` | `center` | Where to center signal chunk within the focus base: `start`, `center`, or `end` |
| `--no-reverse-signal` | *(off)* | Do NOT reverse raw signal. By default signal is reversed for direct RNA (POD5 stores 3'->5'). Use this flag for DNA data. |

**Signal map refinement:**

| Option | Default | Description |
|--------|---------|-------------|
| `--refine-signal-map / --no-refine-signal-map` | enabled | Refine move-table base boundaries against expected k-mer levels. Use `--no-refine-signal-map` for DNA. |
| `--kmer-table FILE` | *(bundled)* | K-mer level table; defaults to the table shipped with leech |
| `--scale-iters INT` | `2` | `-1` = no refinement; `0` = one banded-DP pass without rescaling; `N` = N passes with Theil-Sen rescaling between them |
| `--rough-rescale / --no-rough-rescale` | enabled | **Not honored.** Refinement is delegated to escapepod, whose `refine_signal_map` always applies its own least-squares rough rescale and exposes no switch. Passing `--no-rough-rescale` logs a warning. |

!!! note "Refinement changes chunk contents"

    Refinement is **on by default** and moves base boundaries, so it changes
    every dwell and every level-derived feature. A corpus prepared with it on
    is not interchangeable with one prepared with it off.

    The refined boundaries are taken; the affine `(scale, shift, drift)` fit
    that escapepod returns alongside them is deliberately discarded. Applying
    it would replace one median-MAD transform shared by every read with a
    per-read fit estimated on a near-constant 3' adapter, where it is weakly
    identified -- and cross-read comparability is what k-mer residual features
    depend on.

**Splitting and parallelism:**

| Option | Default | Description |
|--------|---------|-------------|
| `--label STR` | -- | Label identifier for this sample (e.g. `Ala`, `charged`); numeric labels are assigned during merge |
| `--train-split FLOAT` | `0.7` | Fraction for training |
| `--val-split FLOAT` | `0.15` | Fraction for validation |
| `--no-split` | `False` | Save all chunks to `all.npz` (for later merge) |
| `--workers INT` | `8` | Parallel workers |
| `--chunk-size INT` | `100` | Reads per batch |
| `--min-mapq INT` | `0` | Minimum mapping quality |
| `--seed INT` | *(none)* | Random seed |

**Examples:**

```bash
# Basic: prepare charged tRNA data
leech data prepare \
  --pod5 reads.pod5 --bam alignments.bam \
  --output-dir chunks/ --label 1

# Parallel with reference-based motif search
leech data prepare \
  --pod5 reads.pod5 --bam alignments.bam \
  --output-dir chunks/ --workers 8 \
  --motif CCAGGC --motif-reference fasta --skip-motif-indels

# Prepare without splitting (for multi-sample merge later)
leech data prepare \
  --pod5 reads.pod5 --bam alignments.bam \
  --output-dir chunks/ --no-split --label 1
```

### leech data merge

Merge chunk files from multiple samples and split at the read level to prevent data leakage.

```bash
leech data merge -i LABEL=FILE -i LABEL=FILE -o DIR [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input-chunks` | *(required)* | Input chunks with labels (`label=file.npz`) |
| `-o, --output-dir` | *(required)* | Output directory for split chunks |
| `--train-split` | `0.7` | Fraction for training |
| `--val-split` | `0.15` | Fraction for validation |
| `--seed` | `42` | Random seed |
| `--k-fold` | `1` | Number of cross-validation folds. When > 1, creates k-fold splits (must be >= 3) |
| `--comparison-spec` | -- | TSV file with batch comparison specs |
| `--split-by` | -- | NPZ field to split by group instead of by read (e.g. `reference_names`); reads sharing a value stay in the same split |

**Examples:**

```bash
# Pairwise amino acid comparison
leech data merge \
  -i Ala=ala.npz -i Gly=gly.npz \
  -o merged/

# Group comparison (chemical properties)
leech data merge \
  -i basic=lys.npz -i basic=arg.npz \
  -i acidic=asp.npz -i acidic=glu.npz \
  -o merged/

# 5-fold cross-validation
leech data merge \
  -i Ala=ala.npz -i Gly=gly.npz \
  -o kfold/ --k-fold 5
```

---

## Model commands

### leech model train

Train a model on prepared training data.

```bash
leech model train --train-data FILES --val-data FILES --model MODEL --output-dir DIR [OPTIONS]
```

**Required:**

| Option | Description |
|--------|-------------|
| `--train-data FILES` | Training data JSON files (glob patterns supported) |
| `--val-data FILES` | Validation data JSON files |
| `--model MODEL` | Architecture name (see below) |
| `--output-dir DIR` | Directory for model checkpoints |

**Available architectures (24 total):**

| Family | Models |
|--------|--------|
| ConvLSTM | `ConvLSTMDwell` (recommended), `ConvLSTMBase`, +BN, +Attn, +BNAttn, +GNAttn, +LNAttn variants |
| Remora-compat | `ConvLSTMRemora`, `ConvLSTMRemoraBase` |
| Transformer | `TransformerDwell`, `TransformerDwellResidual` (2-channel signal) |
| TCN | `TCNDwell`, `TCNDwellGN`, `TCNDwellLN`, `TCNDwellResidual`, `TCNDwellResidualGN`, `TCNDwellResidualLN`, `TCNDwellSplitResidual`, `TCNDwellSplitResidualLN` |
| Other | `ResNetDwell`, `ConvOnly` |

**Core training options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs INT` | `50` | Training epochs |
| `--batch-size INT` | `128` | Batch size |
| `--learning-rate FLOAT` | `0.001` | Learning rate |
| `--device STR` | `cuda` | `cuda` or `cpu` |
| `--seed INT` | *(none)* | Random seed |
| `--early-stopping INT` | `10` | Stop after N epochs without improvement (0 = disable) |
| `--resume FILE` | -- | Resume from a checkpoint file |

**Class balancing:**

| Option | Default | Description |
|--------|---------|-------------|
| `--use-class-weights / --no-class-weights` | enabled | Auto-compute class weights for imbalanced data |
| `--pos-weight FLOAT` | -- | Manual positive class weight (overrides auto) |

**Regularization and optimization:**

| Option | Default | Description |
|--------|---------|-------------|
| `--weight-decay FLOAT` | `0` | L2 weight decay |
| `--max-grad-norm FLOAT` | `0` | Gradient clipping (0 = disabled) |
| `--scheduler STR` | `none` | LR scheduler: `none`, `reduce_on_plateau`, or `cosine` |
| `--scheduler-patience INT` | `5` | Epochs before reducing LR |
| `--scheduler-factor FLOAT` | `0.5` | Factor to reduce LR by |
| `--warmup-epochs INT` | `0` | Linear warmup epochs |

With `--scheduler cosine`, warmup is part of the schedule itself: the LR ramps
linearly for `--warmup-epochs` and then follows a cosine decay down to a floor of
`1e-6`. Past the epoch budget the LR holds at the floor rather than cycling back
up. `reduce_on_plateau` keeps its own warmup handling and is driven by
`--checkpoint-metric`, not validation loss.

**Optimizer loop:**

| Option | Default | Description |
|--------|---------|-------------|
| `--grad-accum-split INT` | `1` | Split each batch into N sub-batches, accumulating gradients before stepping once (1 = disabled) |
| `--quantile-grad-clip / --no-quantile-grad-clip` | disabled | Clip at 2x the median of the last 100 gradient norms instead of a fixed threshold |
| `--save-optim-every INT` | `1` | Write optimizer state only every N epochs (weights are still written on every save) |

Use `--grad-accum-split` to raise the *effective* batch size when a real batch
will not fit in GPU memory: `--batch-size 64 --grad-accum-split 4` costs the
memory of 16 samples but takes the optimizer step of 64. Keep `--batch-size`
divisible by N -- on a ragged final sub-batch, per-sample weighting is slightly
off.

`--quantile-grad-clip` adapts the clipping threshold to the gradient scale the
run is actually seeing, which is useful when a fixed `--max-grad-norm` is hard to
pick. It takes precedence over `--max-grad-norm` if both are given.

`--save-optim-every N` trades resumability for checkpoint size. Note that it
applies by epoch number, so `model_last.pt` carries optimizer state only if the
final epoch is a multiple of N; resuming from a checkpoint without it starts
from a fresh optimizer and logs a warning.

**Loss function:**

| Option | Default | Description |
|--------|---------|-------------|
| `--loss STR` | `bce` | Loss function: `bce`, `focal`, or `cross_entropy` |
| `--focal-gamma FLOAT` | `2.0` | Focal loss gamma (only with `--loss focal`) |

**Data augmentation:**

| Option | Default | Description |
|--------|---------|-------------|
| `--augment-jitter FLOAT` | `0` | Signal jitter noise std dev (0 = disabled) |
| `--augment-scale-min FLOAT` | `1.0` | Min random scale factor |
| `--augment-scale-max FLOAT` | `1.0` | Max random scale factor |
| `--mixed-precision / --no-mixed-precision` | disabled | Mixed precision training (CUDA only) |

**Provenance (recorded in config for inference):**

| Option | Default | Description |
|--------|---------|-------------|
| `--motif STR` | -- | Motif used for chunk extraction |
| `--motif-offset INT` | `0` | Focus base within motif (0-indexed) |
| `--base-justify STR` | `center` | Signal justification |
| `--model-config FILE` | -- | JSON file with model architecture overrides |
| `--seq-encoding STR` | `signal_kmer` | Sequence encoding: `base_onehot` or `signal_kmer` |
| `--encoding-fallback / --no-encoding-fallback` | auto | Allow `signal_kmer` to fall back to `base_onehot` when the corpus has no base-to-signal maps. Auto = allowed only when `--seq-encoding` was left at its default |
| `--num-workers INT` | `0` | DataLoader workers (0=auto) |
| `--balance-groups / --no-balance-groups` | disabled | Balance sampling across source groups (e.g., per-AA) so each group contributes equally per epoch |

**Output files:**

- `model_best.pt` -- best checkpoint by `--checkpoint-metric` (default `auto`: val_auc for binary, val_f1 for multiclass)
- `model_last.pt` -- final epoch checkpoint
- `config.json` -- full training configuration (needed for inference)
- `metrics.json` -- per-epoch metrics

**Examples:**

```bash
# Basic training
leech model train \
  --train-data chunks/train.json --val-data chunks/val.json \
  --model ConvLSTMDwell --output-dir models/

# With focal loss, gradient clipping, and LR scheduling
leech model train \
  --train-data chunks/train.json --val-data chunks/val.json \
  --model ConvLSTMDwell --output-dir models/ \
  --loss focal --focal-gamma 2.0 \
  --max-grad-norm 1.0 \
  --scheduler reduce_on_plateau --scheduler-patience 5

# Resume from checkpoint
leech model train \
  --train-data chunks/train.json --val-data chunks/val.json \
  --model ConvLSTMDwell --output-dir models/ \
  --resume models/model_last.pt

# Large effective batch on a small GPU, with adaptive clipping
leech model train \
  --train-data chunks/train.json --val-data chunks/val.json \
  --model ConvLSTMDwell --output-dir models/ \
  --batch-size 128 --grad-accum-split 4 \
  --quantile-grad-clip \
  --scheduler cosine --warmup-epochs 2
```

### leech model optimize

Grid search over signal context windows and dwell offsets to find the best configuration.

```bash
leech model optimize --train-data FILE --output-dir DIR --context-grid VALUES [OPTIONS]
```

**Required:**

| Option | Description |
|--------|-------------|
| `--train-data FILE` | Training dataset (.npz) |
| `-o, --output-dir DIR` | Output directory |
| `--context-grid VALUES` | Comma-separated context values or `start:stop:step` range |

**Grid options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--val-data FILE` | -- | Validation dataset |
| `--left-contexts VALUES` | uses `--context-grid` | Override left context grid |
| `--right-contexts VALUES` | uses `--context-grid` | Override right context grid |
| `--dwell-offsets VALUES` | -- | Dwell offset values to search (comma-separated or `start:stop:step`) |
| `--kmer-context INT` | `5` | K-mer context |
| `--base-justify STR` | `center` | Signal chunk centering |
| `--parallel INT` | `1` | Grid points to train concurrently |

Training options (`--model`, `--epochs`, `--batch-size`, `--learning-rate`, `--device`, `--seed`, `--early-stopping`) work the same as in `model train`.

**Output:**

- `grid_summary.csv` -- results for all grid points
- `best_params.json` -- best configuration (for use with `model train --model-config`)
- Per-grid-point subdirectories with model checkpoints

**Examples:**

```bash
# Coarse symmetric grid
leech model optimize \
  --train-data chunks/train.npz --val-data chunks/val.npz \
  --context-grid 200,500,1000,2000,5000 \
  --output-dir grid_results/

# Asymmetric fine grid with range syntax
leech model optimize \
  --train-data chunks/train.npz --val-data chunks/val.npz \
  --left-contexts 8000:10000:500 \
  --right-contexts 0:2000:500 \
  --output-dir grid_results/

# With dwell offset search, parallel on CPU
leech model optimize \
  --train-data chunks/train.npz --val-data chunks/val.npz \
  --context-grid 500,1000 \
  --dwell-offsets -5:5:1 \
  --device cpu --parallel 8 \
  --output-dir grid_results/
```

### leech model benchmark

Time one training step end to end -- per-phase timing plus GPU utilization -- to find out which
phase is the bottleneck before changing anything.

```bash
leech model benchmark --train-data FILE --output-dir DIR [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--train-data PATH` | *(required)* | Training dataset (`.npz` or JSON chunks config) to benchmark against |
| `-o, --output-dir DIR` | *(required)* | Directory for `benchmark.json` (and `trace.json` with `--trace`) |
| `--model STR` | `ConvLSTMDwell` | Architecture to benchmark |
| `--batch-size INT` | `128` | Batch size |
| `--device STR` | `cuda` | `cuda` or `cpu` |
| `--num-steps INT` | `100` | Timed training steps |
| `--warmup-steps INT` | `10` | Warmup steps (cuDNN benchmark, `torch.compile`, prefetch queue) |
| `--num-workers INT` | `8` | DataLoader workers (0 disables multiprocessing) |
| `--prefetch-factor INT` | `4` | DataLoader `prefetch_factor` |
| `--mixed-precision / --no-mixed-precision` | enabled | `torch.amp` autocast + `GradScaler` |
| `--non-blocking / --blocking` | blocking | `.to(device, non_blocking=True)` for host-to-device copies |
| `--signal-len INT` | `400` | Signal length |
| `--kmer-len INT` | `11` | K-mer length |
| `--seq-encoding STR` | `signal_kmer` | `base_onehot` or `signal_kmer` |
| `--signal-mode STR` | `both` | `both`, `residual`, or `signal` |
| `--trace / --no-trace` | off | Also collect a `torch.profiler` Chrome trace |
| `--trace-active-steps INT` | `10` | Active steps captured in the trace |

**Example:**

```bash
leech model benchmark \
  --train-data chunks/train.npz \
  --model TCNDwellResidual \
  --num-workers 8 \
  -o bench/
```

### leech model bundle

Package multiple trained pairwise models into a single versioned bundle file for deployment.

```bash
leech model bundle --model-dir DIR --output FILE --version VERSION [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model-dir DIR` | *(required)* | Root directory containing pair subdirectories (each with `model_best.pt` + `config.json`) |
| `-o, --output FILE` | *(required)* | Output `.pt` bundle file |
| `-v, --version STR` | *(required)* | Semantic version (e.g., `"0.1.0-alpha.1"`) |
| `--comparison-type STR` | `pairwise` | `pairwise`, `one_vs_all`, `group`, or `multiclass` |

The command auto-discovers all subdirectories containing `model_best.pt` and `config.json`.

**Example:**

```bash
# Bundle all pairwise models from a training run
leech model bundle \
  --model-dir results/models/pairwise/ \
  --output bundles/aa_classifier_v0.1.0.pt \
  --version 0.1.0

# One-vs-all bundle
leech model bundle \
  --model-dir results/models/one_vs_all/ \
  --output bundles/ova_v0.1.0.pt \
  --version 0.1.0 \
  --comparison-type one_vs_all
```

### leech model bundle-info

Display metadata from a model bundle.

```bash
leech model bundle-info --bundle FILE
```

Shows architecture, version, comparison type, included pairs, and file size.

### leech model calibrate

Learn post-hoc calibration on the validation set. Binary models use Platt scaling (fits `a, b` so `sigmoid(a*logit + b)` is better calibrated). Multiclass models use temperature (default), matrix, or Dirichlet scaling.

```bash
leech model calibrate --model-dir DIR --val-data FILE [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model-dir DIR` | *(required)* | Model directory or parent with pair subdirs |
| `--val-data FILE` | *(required)* | Validation data (`.npz`) |
| `--device STR` | `cpu` | Device for inference |
| `--batch-size INT` | `1024` | Batch size for validation pass |
| `--num-workers INT` | `0` | DataLoader workers |
| `--method STR` | `temperature` | Multiclass method: `temperature`, `matrix`, or `dirichlet` (binary always uses Platt) |
| `--reg-lambda FLOAT` | `0.01` | L2 regularization toward identity for `matrix`/`dirichlet` |
| `-o, --output FILE` | -- | Output calibration JSON path (single model only) |

Binary models write `platt.json`; multiclass models write `calibration.json` to the model directory. For a parent directory with pair subdirs, calibrates each pair independently.

**Examples:**

```bash
# Calibrate a single model
leech model calibrate --model-dir models/Ala_vs_Gly/ --val-data val.npz

# Calibrate all pairs in a directory
leech model calibrate --model-dir models/one_vs_all/ --val-data val.npz
```

### leech model export

Export a trained model as a standalone `.pt` file, loadable with `torch.export.load()` without the leech codebase.

```bash
leech model export --model-dir DIR --output FILE
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model-dir DIR` | *(required)* | Model checkpoint directory |
| `-o, --output FILE` | *(required)* | Output `.pt` file |

**Example:**

```bash
leech model export --model-dir models/best/ -o exported_model.pt
```

### leech model release

Publish a model bundle to GitHub Releases from a YAML spec. Requires the GitHub CLI (`gh`),
authenticated (`gh auth login`) unless `--dry-run`.

```bash
leech model release --spec FILE [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--spec FILE` | *(required)* | YAML release spec |
| `--dry-run` | off | Render the release notes and preview without publishing |
| `--overwrite` | off | Delete an existing release with the same tag first |
| `--repo STR` | `origin` | GitHub repo as `owner/repo` |

The spec requires `name`, `version`, `description` and `bundle` (path to the `.pt`), and accepts
`organism`, `experiment`, `data_version`, `training`, `metrics`, `prerelease` and `extra_assets`.
The release tag is `model-{name}-v{version}`.

**Example:**

```bash
leech model release --spec release/ivc-pairwise.yaml --dry-run
```

### leech model list

Browse published model releases on GitHub.

```bash
leech model list [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--repo STR` | `origin` | GitHub repo as `owner/repo` |
| `--limit INT` | `30` | Maximum releases to show |

### leech model fetch

Download a published bundle from GitHub Releases.

```bash
leech model fetch --name NAME --version VERSION [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--name STR` | -- | Model release name (e.g. `ivc-pairwise-v1`) |
| `--version STR` | -- | Model version |
| `--tag STR` | -- | Full release tag, as an alternative to `--name` + `--version` |
| `-o, --output-dir DIR` | `.` | Directory to download into |
| `--repo STR` | `origin` | GitHub repo as `owner/repo` |

**Example:**

```bash
leech model fetch --name ivc-pairwise --version 1.0.0 -o models/
```

---

## Evaluation commands

### leech eval test

Evaluate a trained model on a held-out test set.

```bash
leech eval test --model FILE --test-data FILES --output FILE [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model FILE` | *(required)* | Trained model checkpoint (`.pt`) |
| `--test-data FILES` | *(required)* | Test data JSON files |
| `--output FILE` | *(required)* | Output metrics JSON |
| `--device STR` | auto | `cuda` or `cpu` |
| `--num-workers INT` | `0` (auto) | DataLoader workers; auto is 8 on GPU (capped by the job's CPU allocation) and 0 on CPU |
| `--emit-scores PATH` | *(off)* | Also write per-chunk `read_ids`, `labels`, `probs` to an `.npz` |

!!! note "Why `--num-workers` matters on a GPU"

    Collate, the host-to-device copy and the forward pass all run in whichever
    process owns the loader. With no workers that is one core feeding an
    accelerator that then waits: a 7.8M-chunk test set evaluated at **8% GPU**
    on an A5000, while training the same corpus on the same card ran at 98%.
    The default now resolves to workers on CUDA, capped by the CPUs the job is
    actually allowed to use, so a 2-core allocation gets 1 worker rather than 8
    thrashing ones. Raise it when the eval job has cores to spare.

!!! note "Why `--emit-scores` exists"

    The metrics JSON is a summary at **one** threshold. The per-chunk scores
    behind it are computed either way; this flag only decides whether they
    survive the run. Without them, AUPRC for the minority class, per-group
    error breakdowns (per barcode, per isotype), paired model comparison
    (McNemar), calibration, and any operating point other than the reported one
    cannot be answered without re-running inference.

    The join to read ids is positional and is verified: if the test `.npz` holds
    a different number of `read_ids` than there are scored chunks, it raises
    rather than writing a well-formed file with every score attached to the
    wrong read. Multiclass emits the full `(N, C)` softmax.

    The metrics JSON also carries a `threshold_sweep` with the best operating
    points (`at_youden`, `at_mcc`, `at_f1`) alongside `prevalence`. Prefer
    `at_youden` when the deployment class ratio is unknown or varies per
    sample -- it is the only one of the three that is prevalence-invariant.

### leech eval compare

Compare multiple trained models on the same test set.

```bash
leech eval compare -m DIR -m DIR -t FILE -o DIR [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model-dirs` | *(required)* | Model directories to compare (specify multiple) |
| `-t, --test-data` | *(required)* | Test dataset |
| `-o, --output-dir` | *(required)* | Output directory |
| `--device STR` | auto | `cuda` or `cpu` |
| `--no-plot` | `False` | Skip generating plots |

### leech eval importance

Compute feature importance scores.

```bash
leech eval importance -m FILE -t FILE -o DIR [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model` | *(required)* | Trained model checkpoint |
| `-t, --test-data` | *(required)* | Test dataset |
| `-o, --output-dir` | *(required)* | Output directory |
| `--method STR` | `gradient` | `gradient` or `integrated_gradients` |
| `--no-plot` | `False` | Skip generating plots |

### leech eval ablation

Test model performance with sequence ablation.

```bash
leech eval ablation -m FILE -t FILE -o DIR [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model` | *(required)* | Trained model checkpoint |
| `-t, --test-data` | *(required)* | Test dataset |
| `-o, --output-dir` | *(required)* | Output directory |
| `--no-plot` | `False` | Skip generating plots |

---

## Inference command

### leech predict

Run inference on new data. Supports single models, single pairs from a bundle, or all-model aggregation from a bundle.

```bash
leech predict --pod5 FILE --bam FILE --output FILE (--model DIR | --bundle FILE (--pair NAME | --all)) [OPTIONS]
```

**Required:**

| Option | Description |
|--------|-------------|
| `--pod5 FILE` | POD5 file with raw signal |
| `--bam FILE` | BAM file with alignments |
| `-o, --output FILE` | Output BAM with predictions |

**Model selection (mutually exclusive):**

| Option | Description |
|--------|-------------|
| `--model DIR` | Single model checkpoint directory |
| `--bundle FILE` | Model bundle `.pt` file (requires `--pair` or `--all`) |

**Bundle options (require `--bundle`):**

| Option | Description |
|--------|-------------|
| `--pair NAME` | Run a single pair's model from the bundle |
| `--all` | Run every model in the bundle, aggregate to a single amino acid prediction |
| `--raw` | Write full-float probabilities instead of the compact uint8 encoding (`ac`/`pp` tags) |

**Signal handling:**

| Option | Default | Description |
|--------|---------|-------------|
| `--device STR` | auto | `cuda` or `cpu` |
| `--base-justify STR` | `center` | Signal chunk centering |
| `--no-reverse-signal` | *(off)* | Disable signal reversal (use for DNA data) |

**Backend:**

| Option | Default | Description |
|--------|---------|-------------|
| `--backend STR` | `auto` | Chunk-extraction backend: `auto` (Rust when available), `rust` (force Rust, error if unavailable), `python` (force Python). Useful for comparing the two paths. |

!!! note "When `auto` falls back"

    Some options cannot be honored by the Rust pipeline -- non-median-MAD
    `--signal-norm` and softclip signal recovery among them. Under `auto` those
    fall back to Python with a warning; under `--backend rust` they raise rather
    than silently producing different chunks.

**Examples:**

```bash
# Single model inference
leech predict \
  --model models/model_best.pt \
  --pod5 reads.pod5 --bam alignments.bam \
  --output predictions.bam

# Run one pair from a bundle
leech predict \
  --bundle bundles/aa_classifier.pt --pair Ala_vs_Gly \
  --pod5 reads.pod5 --bam alignments.bam \
  --output predictions.bam

# Run all models in bundle (aggregated amino acid prediction)
leech predict \
  --bundle bundles/aa_classifier.pt --all \
  --pod5 reads.pod5 --bam alignments.bam \
  --output predictions.bam

# Same but also write per-pair raw probabilities
leech predict \
  --bundle bundles/aa_classifier.pt --all --raw \
  --pod5 reads.pod5 --bam alignments.bam \
  --output predictions.bam
```

---

## Typical workflows

### Single-sample classification

```bash
# 1. Prepare data
leech data prepare \
  --pod5 reads.pod5 --bam alignments.bam \
  --output-dir chunks/ --label 1

# 2. Train
leech model train \
  --train-data chunks/train.json --val-data chunks/val.json \
  --model ConvLSTMDwell --output-dir models/

# 3. Evaluate
leech eval test \
  --model models/model_best.pt \
  --test-data chunks/test.json --output metrics.json

# 4. Predict
leech predict \
  --model models/ \
  --pod5 new_reads.pod5 --bam new_alignments.bam \
  --output predictions.bam
```

### Multi-sample pairwise comparison

```bash
# 1. Prepare each sample without splitting
leech data prepare --pod5 ala.pod5 --bam ala.bam --output-dir chunks/ala/ --no-split --label 0
leech data prepare --pod5 gly.pod5 --bam gly.bam --output-dir chunks/gly/ --no-split --label 1

# 2. Merge and split at read level
leech data merge -i Ala=chunks/ala/all.npz -i Gly=chunks/gly/all.npz -o merged/

# 3. Train, test, predict as above
```

### Bundle workflow (multi-pair deployment)

```bash
# 1. Train pairwise models (one per amino acid pair)
leech model train --train-data ala_gly/train.json ... --output-dir models/Ala_vs_Gly/
leech model train --train-data ala_val/train.json ... --output-dir models/Ala_vs_Val/
# ... repeat for each pair

# 2. Bundle all models
leech model bundle \
  --model-dir models/ --output bundle.pt --version 1.0.0

# 3. Inspect the bundle
leech model bundle-info --bundle bundle.pt

# 4. Run aggregated inference
leech predict \
  --bundle bundle.pt --all \
  --pod5 reads.pod5 --bam alignments.bam \
  --output predictions.bam
```

### Hyperparameter optimization

```bash
# 1. Grid search for optimal context
leech model optimize \
  --train-data chunks/train.npz --val-data chunks/val.npz \
  --context-grid 200,500,1000,2000,5000 \
  --output-dir grid_results/

# 2. Train final model with best params
leech model train \
  --train-data chunks/train.json --val-data chunks/val.json \
  --model ConvLSTMDwell --output-dir models/ \
  --model-config grid_results/best_params.json
```
