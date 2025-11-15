# Biological tRNA Discriminative Models Workflow

This workflow trains discriminative models from biological tRNA samples by:
1. Aligning reads to a tRNA reference database to identify isodecoder/amino acid
2. Classifying reads by charging status using a pre-trained model
3. Splitting reads by amino acid and charging status
4. Training two types of discriminative models:
   - **Pairwise amino acid discrimination** (Ala vs Arg, Ala vs Asn, etc.)
   - **Per-amino acid charging discrimination** (Ala-charged vs Ala-uncharged, etc.)

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Usage](#usage)
- [Workflow Steps](#workflow-steps)
- [Output Structure](#output-structure)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The biological tRNA workflow is designed to handle real biological samples where tRNAs from multiple amino acids are mixed together. Unlike the synthetic workflow where samples are pre-labeled by amino acid, the biological workflow must:

1. **Identify amino acid** from alignment to a tRNA reference database
2. **Classify charging status** using a pre-trained charged/uncharged model
3. **Demultiplex reads** into groups: `{AA}_{charged|uncharged}`
4. **Train discriminative models** for amino acid and charging classification

### Workflow Architecture

```
Biological POD5/BAM
    ↓
1. Align to tRNA reference (minimap2)
    ↓
2. Classify charging status (leech infer with pre-trained model)
    ↓
3. Split BAM by AA + charging status
    ↓
    ├─→ Ala_charged.bam
    ├─→ Ala_uncharged.bam
    ├─→ Arg_charged.bam
    └─→ ...
    ↓
4. Prepare chunks for each group
    ↓
    ├─→ Path A: Pairwise AA models
    │   - Merge all Ala (charged+uncharged) vs all Arg (charged+uncharged)
    │   - Train 190 models (20 choose 2)
    │
    └─→ Path B: Per-AA charging models
        - Merge Ala_charged vs Ala_uncharged
        - Train 20 models (one per amino acid)
```

## Prerequisites

### 1. tRNA Reference Database

You need a FASTA file containing tRNA sequences with names formatted as:
```
>tRNA-Ala-AGC-1-1
>tRNA-Arg-ACG-1-1
>tRNA-Asn-GTT-1-1
...
```

The workflow parses amino acid identity from the reference name using the pattern `tRNA-{AA}-`.

**Recommended sources:**
- [GtRNAdb](http://gtrnadb.ucsc.edu/) - Genomic tRNA database
- [tRNAscan-SE](http://lowelab.ucsc.edu/tRNAscan-SE/) - tRNA gene prediction
- Custom synthetic tRNA sequences

Place your reference at:
```bash
pipeline/resources/references/trna-reference.fa
```

### 2. Pre-trained Charging Classification Model

You need a trained model to classify charged vs uncharged status. This should be generated from the synthetic workflow:

```bash
# First, train a charged/uncharged model using synthetic samples
snakemake --configfile config/samples-alpine.yaml \
  --config comparison_spec_file=config/comparisons_charged_uncharged.tsv \
  --profile profiles/slurm \
  all_train

# The model will be saved at:
# results/models/pairwise/charged_uncharged/model_best.pt
```

Update the path in your config:
```yaml
charging_model: "results/models/pairwise/charged_uncharged/model_best.pt"
```

### 3. Biological Samples

Each biological sample requires:
- **POD5 file(s)**: Raw nanopore signal data
- **BAM file**: Basecalled reads with move tables (`mv` tag from dorado/guppy)

## Configuration

### 1. Configure Sample Paths

Edit `pipeline/config/samples-bio-trna.yaml`:

```yaml
bio_samples:
  ecoli_bio_rep1:
    pod5: "/path/to/ecoli_bio_rep1/pod5_pass"
    bam: "/path/to/ecoli_bio_rep1/basecalls.bam"
    label: "biological"
    description: "E. coli biological sample, replicate 1"
```

### 2. Set Reference and Model Paths

```yaml
# tRNA reference database
trna_reference: "pipeline/resources/references/trna-reference.fa"

# Pre-trained charging model
charging_model: "results/models/pairwise/charged_uncharged/model_best.pt"

# Charging probability threshold
charging_threshold: 0.5  # P(charged) >= 0.5 → charged
```

### 3. Configure Output Directories

```yaml
bio_bam_dir: "/scratch/alpine/user/leech/bam/bio"
bio_chunks_dir: "/scratch/alpine/user/leech/chunks/bio"
bio_models_dir: "/scratch/alpine/user/leech/models/bio"
bio_metrics_dir: "/scratch/alpine/user/leech/metrics/bio"
```

### 4. Configure Comparisons

The workflow uses TSV spec files to define comparisons:

**Pairwise AA** (`config/comparisons_bio_pairwise_aa.tsv`):
```
Ala	Ala	Arg	Arg
Ala	Ala	Asn	Asn
...
```

**Per-AA Charging** (`config/comparisons_bio_per_aa_charging.tsv`):
```
charged	Ala	uncharged	Ala
charged	Arg	uncharged	Arg
...
```

## Usage

### Run Complete Workflow

```bash
# Run full biological workflow (alignment → splitting → training)
snakemake --configfile config/samples-bio-trna.yaml \
  --profile profiles/slurm \
  all_bio
```

### Run Individual Steps

```bash
# 1. Align to tRNA reference
snakemake --configfile config/samples-bio-trna.yaml \
  --profile profiles/slurm \
  all_bio_align

# 2. Classify charging status
snakemake --configfile config/samples-bio-trna.yaml \
  --profile profiles/slurm \
  all_bio_classify

# 3. Split by AA + charging
snakemake --configfile config/samples-bio-trna.yaml \
  --profile profiles/slurm \
  all_bio_split

# 4. Prepare training chunks
snakemake --configfile config/samples-bio-trna.yaml \
  --profile profiles/slurm \
  all_bio_prepare

# 5. Train pairwise AA models
snakemake --configfile config/samples-bio-trna.yaml \
  --profile profiles/slurm \
  all_bio_train_pairwise_aa

# 6. Train per-AA charging models
snakemake --configfile config/samples-bio-trna.yaml \
  --profile profiles/slurm \
  all_bio_train_charging
```

### Dry Run

```bash
# Preview what will be executed
snakemake --configfile config/samples-bio-trna.yaml \
  --profile profiles/slurm \
  all_bio -n
```

## Workflow Steps

### Step 1: Align to tRNA Reference

**Rule:** `align_bio_trna_to_reference`

Aligns biological reads to the tRNA reference database using minimap2:

```bash
minimap2 -ax map-ont --secondary=no -L \
  trna-reference.fa \
  sample.bam
```

**Outputs:**
- `{sample}/{sample}.aligned.bam` - Aligned reads
- `{sample}/{sample}.aligned.bam.bai` - BAM index

The alignment identifies which tRNA isodecoder each read came from.

### Step 2: Classify Charging Status

**Rule:** `infer_charging_status`

Uses the pre-trained charging model to predict charging status:

```bash
leech infer \
  --model charging_model.pt \
  --pod5 sample.pod5 \
  --bam sample.aligned.bam \
  --output sample.charging_predictions.bam
```

**Outputs:**
- `{sample}/{sample}.charging_predictions.bam` - BAM with ML tags containing charging probabilities
- `{sample}/{sample}.charging_predictions.bam.bai` - BAM index

Each read gets an `ML` tag with the probability of being charged.

### Step 3: Split by AA + Charging

**Rule:** `split_bam_by_aa_and_charging`

Demultiplexes reads into groups based on amino acid and charging status:

1. Parse amino acid from reference name (`tRNA-Ala-AGC-1-1` → `Ala`)
2. Classify charging based on ML tag probability threshold
3. Write reads to separate BAM files

**Outputs:**
- `{sample}/Ala_charged.bam`
- `{sample}/Ala_uncharged.bam`
- `{sample}/Arg_charged.bam`
- `{sample}/Arg_uncharged.bam`
- ... (one file per AA+charging combination found)
- `{sample}/split.done` - Completion stamp with summary

### Step 4: Prepare Training Chunks

**Rule:** `prepare_bio_chunks`

Extracts training chunks for each AA+charging group:

```bash
leech prepare \
  --pod5 sample.pod5 \
  --bam Ala_charged.bam \
  --output-dir chunks/sample/Ala_charged/ \
  --motif CCATGGC \
  --motif-offset 3 \
  --workers 8 \
  --no-split
```

**Outputs:**
- `{sample}/{group}/all.npz` - Training chunks for each group

### Step 5a: Train Pairwise AA Models

**Rules:** `merge_bio_chunks_pairwise_aa`, `train_bio_pairwise_aa`

Trains models to discriminate between amino acids:

1. Merge all reads for two amino acids (both charged and uncharged)
2. Relabel: AA1 → label 0, AA2 → label 1
3. Split at read level into train/val/test
4. Train binary classifier

**Outputs:**
- `pairwise_aa/{AA1}_{AA2}/train.npz`
- `pairwise_aa/{AA1}_{AA2}/val.npz`
- `pairwise_aa/{AA1}_{AA2}/test.npz`
- `pairwise_aa/{AA1}_{AA2}/model_best.pt`

**Example:** `Ala_Arg` model discriminates all Ala tRNAs from all Arg tRNAs, regardless of charging status.

### Step 5b: Train Per-AA Charging Models

**Rules:** `merge_bio_chunks_per_aa_charging`, `train_bio_per_aa_charging`

Trains models to discriminate charging status within each amino acid:

1. Merge charged and uncharged reads for one amino acid
2. Relabel: charged → label 1, uncharged → label 0
3. Split at read level into train/val/test
4. Train binary classifier

**Outputs:**
- `per_aa_charging/{AA}/train.npz`
- `per_aa_charging/{AA}/val.npz`
- `per_aa_charging/{AA}/test.npz`
- `per_aa_charging/{AA}/model_best.pt`

**Example:** `Ala` model discriminates Ala-charged from Ala-uncharged.

## Output Structure

```
results/
└── bam/
    └── bio/
        └── {sample}/
            ├── {sample}.aligned.bam              # Aligned to tRNA reference
            ├── {sample}.charging_predictions.bam # With ML tags
            ├── Ala_charged.bam                   # Split by AA+charging
            ├── Ala_uncharged.bam
            ├── Arg_charged.bam
            ├── ...
            └── split.done                        # Split summary

└── chunks/
    └── bio/
        ├── {sample}/
        │   ├── Ala_charged/all.npz
        │   ├── Ala_uncharged/all.npz
        │   └── ...
        └── merged/
            ├── pairwise_aa/
            │   └── {AA1}_{AA2}/
            │       ├── train.npz
            │       ├── val.npz
            │       └── test.npz
            └── per_aa_charging/
                └── {AA}/
                    ├── train.npz
                    ├── val.npz
                    └── test.npz

└── models/
    └── bio/
        ├── pairwise_aa/
        │   └── {AA1}_{AA2}/
        │       ├── model_best.pt
        │       ├── model_last.pt
        │       └── metrics.json
        └── per_aa_charging/
            └── {AA}/
                ├── model_best.pt
                ├── model_last.pt
                └── metrics.json
```

## Examples

### Example 1: E. coli Biological Sample

```yaml
# config/samples-bio-trna.yaml
bio_samples:
  ecoli_bio:
    pod5: "/data/ecoli_bio/pod5_pass"
    bam: "/data/ecoli_bio/basecalls.bam"
    label: "biological"
```

Run workflow:
```bash
snakemake --configfile config/samples-bio-trna.yaml \
  --profile profiles/slurm \
  all_bio
```

Expected outputs:
- 40 split BAM files (20 AA × 2 charging states)
- 190 pairwise AA models
- 20 per-AA charging models

### Example 2: Multiple Replicates

```yaml
bio_samples:
  human_rep1:
    pod5: "/data/human_rep1/pod5_pass"
    bam: "/data/human_rep1/basecalls.bam"
    label: "biological"

  human_rep2:
    pod5: "/data/human_rep2/pod5_pass"
    bam: "/data/human_rep2/basecalls.bam"
    label: "biological"
```

The workflow will:
1. Process each replicate independently (align, classify, split)
2. Merge chunks from both replicates when training models
3. Perform read-level splitting to prevent data leakage

### Example 3: Subset of Amino Acids

If you only expect certain amino acids, specify them:

```yaml
bio_samples:
  yeast_partial:
    pod5: "/data/yeast/pod5_pass"
    bam: "/data/yeast/basecalls.bam"
    label: "biological"
    amino_acids: ["Ala", "Gly", "Leu", "Val"]
```

Only the specified amino acids will be processed.

## Troubleshooting

### Issue: No reads in split BAM files

**Possible causes:**
1. Reference names don't match expected format (`tRNA-{AA}-...`)
2. No reads aligned to tRNA reference
3. All reads filtered out during splitting

**Solutions:**
- Check alignment rate: `samtools flagstat {sample}.aligned.bam`
- Verify reference names: `samtools view -H {sample}.aligned.bam | grep '@SQ'`
- Check ML tags: `samtools view {sample}.charging_predictions.bam | grep 'ML:'`

### Issue: Charging classification fails

**Possible causes:**
1. Pre-trained model not found
2. Model incompatible with data
3. POD5 file doesn't match BAM read IDs

**Solutions:**
- Verify model path exists: `ls -lh {charging_model}`
- Check POD5/BAM compatibility: read IDs must match exactly
- Re-train charging model with same parameters as biological data

### Issue: Training fails with "No chunks found"

**Possible causes:**
1. No motif instances found in reads
2. Reads too short for chunk context
3. All reads filtered by quality/indels

**Solutions:**
- Check motif: verify it exists in your tRNA sequences
- Reduce chunk context: try `chunk_context: [100, 100]`
- Disable indel filtering: `skip_motif_indels: false`

### Issue: Out of memory during splitting

**Possible causes:**
1. Too many reads being processed at once
2. Too many output BAM files open simultaneously

**Solutions:**
- Process samples individually
- Increase memory allocation in cluster profile
- Split large POD5/BAM files into smaller chunks

## Advanced Configuration

### Custom Amino Acid Groupings

You can define custom comparisons beyond pairwise:

```
# config/comparisons_bio_custom.tsv
hydrophobic	Ala,Ile,Leu,Val	hydrophilic	Ser,Thr
basic	Arg,Lys,His	acidic	Asp,Glu
```

### Adjusting Charging Threshold

The threshold determines how reads are classified:

```yaml
charging_threshold: 0.5  # Default: 50% probability
charging_threshold: 0.7  # Conservative: higher confidence
charging_threshold: 0.3  # Permissive: more reads classified as charged
```

### Using Different Models per Sample

```yaml
bio_samples:
  sample1:
    pod5: "/data/s1.pod5"
    bam: "/data/s1.bam"
    charging_model: "models/charging_v1.pt"  # Sample-specific model

  sample2:
    pod5: "/data/s2.pod5"
    bam: "/data/s2.bam"
    charging_model: "models/charging_v2.pt"
```

## Performance Tips

1. **Parallel processing:** Set `workers: 16` for faster chunk preparation
2. **GPU acceleration:** Set `use_cpu_training: false` for faster training
3. **Batch size:** Increase `batch_size: 256` if you have enough memory
4. **Subset testing:** Test with one sample first before running all replicates

## Related Documentation

- [Main Pipeline README](../../../README.md)
- [Synthetic Workflow](README_dorado.md)
- [Model Comparison](compare_models.smk)
- [Configuration Guide](../../config/config.yaml)
