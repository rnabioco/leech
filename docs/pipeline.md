# Snakemake Pipeline

For production workloads, leech includes a Snakemake pipeline that handles
data preparation, grid search, training, inference, and model comparison on
HPC clusters.

## Pipeline structure

```
pipeline/
├── config/
│   ├── config.yaml              # Main configuration
│   ├── alpine-config.yaml       # CU Boulder Alpine (SLURM)
│   └── bodhi-config.yaml        # Bodhi cluster (LSF)
├── workflow/
│   ├── Snakefile                # Main workflow
│   └── rules/                   # Modular rule files
│       ├── common.smk
│       ├── prepare.smk
│       ├── grid_search.smk
│       ├── train.smk
│       ├── inference.smk
│       ├── evaluate.smk
│       └── compare_models.smk
└── cluster/                     # Cluster execution profiles
    ├── slurm/
    └── lsf/
```

## Configuration

Edit `pipeline/config/config.yaml` to define your samples and parameters.

### Sample definitions

```yaml
samples:
  sample_charged_ala_rep1:
    pod5: "data/charged/ala/rep1.pod5"
    bam: "data/charged/ala/rep1.bam"
    label: "charged"
    amino_acid: "Ala"

  sample_uncharged_ala_rep1:
    pod5: "data/uncharged/ala/rep1.pod5"
    bam: "data/uncharged/ala/rep1.bam"
    label: "uncharged"
    amino_acid: "Ala"
```

### Model and training settings

```yaml
model: "ConvLSTMDwell"
epochs: 50
batch_size: 128
learning_rate: 0.001
early_stopping_patience: 5
```

### Model comparison mode

Train and compare multiple architectures on the same data:

```yaml
compare_models: true
models_to_compare:
  - "ConvLSTMBase"
  - "ConvLSTMDwell"
  - "TransformerDwell"
  - "ConvOnly"
  - "TCNDwell"
  - "ResNetDwell"
```

### Grid search

```yaml
use_grid_search: true
grid_search:
  learning_rate: [0.0001, 0.001, 0.01]
  batch_size: [64, 128, 256]
  hidden_size: [128, 256, 512]
```

### Pairwise amino acid pairs

```yaml
amino_acids:
  - "Ala"
  - "Gly"
  - "Val"
  # ... etc
```

## Running the pipeline

### SLURM clusters

```bash
cd pipeline/workflow

# Dry run
snakemake --profile ../cluster/slurm -n

# Execute
snakemake --profile ../cluster/slurm
```

### LSF clusters

```bash
# Using the modern executor plugin (recommended)
snakemake --executor lsf --configfile ../config/bodhi-config.yaml \
  --default-resources lsf_queue=gpuqueue --jobs 100

# Or using the profile
snakemake --profile ../cluster/lsf
```

### Local execution

```bash
snakemake --cores 8
```

## Pipeline targets

**Single model mode** (when `compare_models: false`):

```bash
snakemake --profile ../cluster/slurm all_prepare        # Prepare data only
snakemake --profile ../cluster/slurm all_grid_search    # Grid search only
snakemake --profile ../cluster/slurm all_train          # Train models
snakemake --profile ../cluster/slurm all_infer          # Run inference
snakemake --profile ../cluster/slurm all_single_model   # Full single-model analysis
```

**Model comparison mode** (when `compare_models: true`):

```bash
snakemake --profile ../cluster/slurm all_prepare                # Prepare data
snakemake --profile ../cluster/slurm all_grid_search_comparison # Grid search all architectures
snakemake --profile ../cluster/slurm all_train_comparison       # Train all architectures
snakemake --profile ../cluster/slurm all_compare_models         # Full comparison
```

## Output structure

```
results/
├── chunks/{sample}/              # Prepared training data
│   ├── train.json
│   ├── val.json
│   └── test.json
├── models/
│   ├── grid_search/              # Grid search results
│   ├── charged_vs_uncharged/     # Trained models
│   │   ├── model_best.pt
│   │   ├── config.json
│   │   └── training_history.json
│   └── pairwise/{pair}/
├── inference/                    # Prediction BAMs
│   └── {sample}_predictions.bam
└── metrics/                      # Evaluation results
    ├── charged_vs_uncharged_summary.tsv
    └── pairwise_summary.tsv
```

## Cluster setup guides

For detailed cluster-specific setup (storage layout, modules, quotas):

- [Alpine Setup (SLURM)](setup/ALPINE_SETUP.md) -- CU Boulder Alpine
- [Bodhi Setup (LSF)](setup/BODHI_SETUP.md) -- Local Bodhi cluster
