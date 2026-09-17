# Snakemake Pipeline

For production workloads, leech includes a Snakemake pipeline that handles
data preparation, grid search, training, inference, and model comparison on
HPC clusters. The pipeline is task-agnostic -- samples, labels, and
comparisons all come from your own config -- but the examples on this page
use the tRNA charging/amino acid assay that drove its development, since
that is the config it has been run against in production.

## Pipeline structure

```
pipeline/
├── config/
│   ├── config.yaml              # Main configuration
│   ├── samples-alpine.yaml      # Sample definitions (passed via --configfile)
│   └── comparisons_*.tsv        # Comparison specs (pairwise, chemical, etc.)
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

`pod5`, `bam`, and `label` are the only keys the pipeline rules read; `label`
is an arbitrary string passed straight to `leech data prepare --label`. Any
other key (like `amino_acid` below) is just metadata for your own
bookkeeping -- shown here because it's how the tRNA charging example names
its samples.

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

`ConvOnly` is deprecated (`leech.models.model_tier`): it has zero references
in escapepod-models and a known ONNX-export defect with no fix planned, so
`get_model("ConvOnly", ...)` raises a `DeprecationWarning`. Drop it from
`models_to_compare` rather than silencing the warning.

### Grid search

`leech model optimize` searches signal context windows and dwell offset. It does
not search learning rate, batch size, or layer sizes — set those directly under
the training hyperparameters above.

```yaml
use_grid_search: true
grid_search:
  context_grid: "200:1000:200"   # symmetric fallback when left/right unset
  left_contexts: [200, 500, 1000]
  right_contexts: [200, 500]
  dwell_offsets: "0"
grid_search_epochs: 20
grid_search_parallel: 1
```

Each value may be a list or a `start:stop:step` range string.

### Pairwise comparisons

Comparisons are driven by a TSV spec (`comparison_spec_file` in
`config.yaml`, pointing at one of `pipeline/config/comparisons_*.tsv`; see
[Batch comparisons with a TSV spec](data_preparation.md#batch-comparisons-with-a-tsv-spec)).
The bundled specs compare amino acid identity, since that's the tRNA charging
example the pipeline ships configured for -- `comparisons_charged_uncharged.tsv`,
`comparisons_all_pairwise.tsv`, and `comparisons_chemical_properties.tsv`. A
different task supplies its own TSV of `meta_label	label_set` pairs.

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
snakemake --executor lsf --configfile ../config/samples-alpine.yaml \
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
│   ├── {comparison}/              # Trained models, one dir per top-level
│   │   ├── model_best.pt         # comparison (e.g. charged_vs_uncharged)
│   │   ├── config.json
│   │   └── metrics.json
│   └── pairwise/{pair}/
├── inference/                    # Prediction BAMs
│   └── {sample}_predictions.bam
└── metrics/                      # Evaluation results
    ├── {comparison}_summary.tsv
    └── pairwise_summary.tsv
```
