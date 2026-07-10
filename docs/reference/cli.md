# CLI Reference

Complete reference for the `leech` command-line interface.

## Command overview

The CLI is organized into workflow-based command groups:

| Group | Commands | Purpose |
|-------|----------|---------|
| `leech data` | `prepare`, `merge` | Extract features, merge and split datasets |
| `leech model` | `train`, `optimize`, `benchmark`, `bundle`, `bundle-info`, `calibrate`, `export` | Train, tune, calibrate, and package models |
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

**Signal handling:**

| Option | Default | Description |
|--------|---------|-------------|
| `--base-justify STR` | `center` | Where to center signal chunk within the focus base: `start`, `center`, or `end` |
| `--no-reverse-signal` | *(off)* | Do NOT reverse raw signal. By default signal is reversed for direct RNA (POD5 stores 3'->5'). Use this flag for DNA data. |

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
