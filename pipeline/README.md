# Leech Snakemake Pipeline

A Snakemake pipeline for training and applying leech models on nanopore tRNA data. This pipeline handles:

1. **Charged vs Uncharged Classification**: Distinguish between charged and uncharged tRNAs
2. **Pairwise Amino Acid Classification**: Binary classifiers for all pairs of amino acids

## Pipeline Structure

Following the [Snakemake deployment best practices](https://snakemake.readthedocs.io/en/stable/snakefiles/deployment.html):

```
pipeline/
├── config/
│   └── config.yaml              # Main configuration file
├── workflow/
│   ├── Snakefile                # Main workflow file
│   ├── rules/                   # Modular rule files
│   │   ├── common.smk           # Common variables and functions
│   │   ├── prepare.smk          # Data preparation
│   │   ├── grid_search.smk      # Hyperparameter tuning
│   │   ├── train.smk            # Model training
│   │   ├── inference.smk        # Model inference
│   │   └── evaluate.smk         # Evaluation and summarization
│   └── scripts/
│       └── summarize_metrics.py # Aggregate metrics across samples
├── profiles/                    # Cluster execution profiles
│   ├── slurm/
│   │   ├── config.yaml          # SLURM configuration
│   │   ├── slurm_submit.sh      # Job submission script
│   │   └── slurm_status.py      # Status checking script
│   └── lsf/
│       ├── config.yaml          # LSF configuration
│       ├── lsf_submit.sh        # Job submission script
│       └── lsf_status.py        # Status checking script
└── README.md                    # This file
```

## Quick Start

### 1. Configure Your Samples

Edit `config/config.yaml` to define your samples:

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

### 2. Choose Your Cluster

#### SLURM
```bash
cd pipeline
snakemake --profile profiles/slurm
```

#### LSF
```bash
cd pipeline
snakemake --profile profiles/lsf
```

#### Local (for testing)
```bash
cd pipeline
snakemake --cores 8
```

### 3. Run Specific Targets

```bash
# Prepare data only
snakemake --profile profiles/slurm all_prepare

# Grid search only
snakemake --profile profiles/slurm all_grid_search

# Train models only
snakemake --profile profiles/slurm all_train

# Run inference only
snakemake --profile profiles/slurm all_infer

# Everything (default)
snakemake --profile profiles/slurm all
```

## Configuration

### Sample Configuration

Each sample requires:
- `pod5`: Path to POD5 file containing raw signal
- `bam`: Path to aligned BAM file (must have `mv` and `ns` tags)
- `label`: Either "charged" or "uncharged" (for charged vs uncharged classification)
- `amino_acid`: Amino acid identity (e.g., "Ala", "Gly") for pairwise classification

### Amino Acid Pairs

List all amino acids you want to compare:

```yaml
amino_acids:
  - "Ala"
  - "Gly"
  - "Val"
  # ... etc
```

The pipeline automatically generates all pairwise combinations: `C(n, 2) = n!/(2!(n-2)!)`

### Model Training Parameters

```yaml
# Model architecture
model: "ConvLSTMDwell"  # or "ConvLSTMBase"

# Training hyperparameters
epochs: 50
batch_size: 128
learning_rate: 0.001
early_stopping_patience: 5
```

### Grid Search

Enable hyperparameter tuning:

```yaml
use_grid_search: true

grid_search:
  learning_rate: [0.0001, 0.001, 0.01]
  batch_size: [64, 128, 256]
  hidden_size: [128, 256, 512]
  num_layers: [1, 2, 3]
  dropout: [0.1, 0.2, 0.3]

grid_search_epochs: 20  # Shorter training for grid search
```

When `use_grid_search: true`, the pipeline will:
1. Run grid search to find optimal hyperparameters
2. Use those hyperparameters for final training

## Cluster Configuration

### SLURM

Edit `profiles/slurm/config.yaml` to customize:

```yaml
jobs: 100  # Max concurrent jobs
default-resources:
  - partition="gpu"
  - mem_mb=8000
  - runtime=120
```

GPU jobs automatically request:
```bash
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"  # or v100, rtx8000, etc.
```

### LSF

Edit `profiles/lsf/config.yaml` similarly:

```yaml
jobs: 100
default-resources:
  - queue="gpuqueue"
  - mem_mb=8000
  - runtime=120
```

GPU jobs automatically request:
```bash
#BSUB -gpu "num=1:mode=shared:mps=no:j_exclusive=yes"
#BSUB -R "select[gpu_model==a100]"
```

### Module Loading

If your cluster requires environment modules, edit the submission scripts:

**SLURM** (`profiles/slurm/slurm_submit.sh`):
```bash
module purge
module load cuda/11.8
module load gcc/11.2.0
```

**LSF** (`profiles/lsf/lsf_submit.sh`):
```bash
module purge
module load cuda/11.8
module load gcc/11.2.0
```

## Pipeline Workflow

The pipeline follows this workflow:

```
1. prepare_chunks
   ├─ Extract training data from POD5/BAM
   └─ Split into train/val/test sets

2. grid_search (optional)
   ├─ Search hyperparameter space
   └─ Save best parameters

3. train
   ├─ Train charged vs uncharged classifier
   └─ Train pairwise amino acid classifiers

4. infer
   ├─ Apply models to generate predictions
   └─ Output: BAM files with modification probabilities

5. evaluate
   ├─ Test models on held-out data
   ├─ Generate per-sample metrics
   └─ Aggregate into summary tables
```

## Resource Requirements

Default resource allocations:

| Rule                | CPUs | Memory | Time | GPU |
|---------------------|------|--------|------|-----|
| prepare_chunks      | 4    | 8 GB   | 2h   | 0   |
| grid_search         | 4    | 16 GB  | 24h  | 1   |
| train               | 4    | 16 GB  | 8h   | 1   |
| infer               | 4    | 8 GB   | 4h   | 1   |
| test                | 2    | 4 GB   | 1h   | 1   |

Adjust in `config/config.yaml` or profile configs as needed.

## Output Structure

```
results/
├── chunks/                      # Prepared training data
│   └── {sample}/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── models/
│   ├── grid_search/             # Grid search results
│   │   ├── charged_vs_uncharged/
│   │   └── pairwise/{pair}/
│   ├── charged_vs_uncharged/    # Trained models
│   │   ├── model_best.pt
│   │   ├── model_checkpoint.pt
│   │   └── training_history.json
│   └── pairwise/{pair}/
│       └── ...
├── inference/                   # Prediction BAM files
│   ├── charged_vs_uncharged/
│   │   └── {sample}_predictions.bam
│   └── pairwise/{pair}/
│       └── {sample}_predictions.bam
└── metrics/                     # Evaluation metrics
    ├── charged_vs_uncharged/
    │   ├── {sample}_metrics.json
    │   └── ...
    ├── pairwise/{pair}/
    │   └── {sample}_metrics.json
    ├── charged_vs_uncharged_summary.tsv
    └── pairwise_summary.tsv
```

## Troubleshooting

### Check Job Status

**SLURM:**
```bash
squeue -u $USER
sacct -j <job_id>
```

**LSF:**
```bash
bjobs
bjobs -l <job_id>
```

### View Logs

All rules generate log files in the output directories:
```bash
# Example: view training log
cat results/models/charged_vs_uncharged/train.log

# Example: view data preparation log
cat results/chunks/sample_name/prepare.log
```

### Dry Run

Always test with a dry run first:
```bash
snakemake --profile profiles/slurm -n
```

### Rerun Failed Jobs

```bash
# Rerun all failed jobs
snakemake --profile profiles/slurm --rerun-incomplete

# Rerun specific rule
snakemake --profile profiles/slurm --forcerun train_charged_vs_uncharged
```

### GPU Issues

If GPU allocation fails:
1. Check available GPU types: `sinfo -o "%N %G"` (SLURM) or `bhosts -gpu` (LSF)
2. Update `gpu_type` or `gpu_model` in config
3. Verify GPU resources in submission scripts

## Advanced Usage

### Running Subsets

Process only specific samples by modifying `SAMPLES` in `workflow/rules/common.smk`:

```python
# Process only samples matching a pattern
SAMPLES = [s for s in config["samples"].keys() if "rep1" in s]
```

### Custom Rules

Add custom rules in new files under `workflow/rules/` and include them in `workflow/Snakefile`:

```python
include: "rules/my_custom_rules.smk"
```

### DAG Visualization

Generate a workflow diagram:
```bash
snakemake --dag | dot -Tpdf > dag.pdf
```

## Requirements

- **Snakemake** ≥ 7.0
- **Python** ≥ 3.10
- **uv** (for running leech commands)
- **samtools** (for BAM indexing)
- **GPU** with CUDA support (for training/inference)

## Citation

If you use this pipeline, please cite:
```
[Citation information for leech]
```

## Support

For issues related to:
- **Pipeline**: Open an issue in the leech repository
- **leech library**: See main README.md
- **Cluster configuration**: Consult your HPC documentation
