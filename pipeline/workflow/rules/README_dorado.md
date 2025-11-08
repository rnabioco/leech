# Dorado Rebasecalling Rules

This file documents the POD5 rebasecalling workflow using dorado.

## Overview

The dorado rules provide a complete workflow for rebasecalling raw POD5 files:

1. **merge_pods**: Consolidates multiple raw POD5 files into a single merged file
2. **rebasecall**: Runs dorado basecaller with move table generation
3. **align_rebasecalled**: (Optional) Aligns rebasecalled reads to a reference

## Configuration

Add the following to your sample configuration in `config.yaml`:

```yaml
samples:
  my_sample:
    # Required for standard leech workflow
    pod5: "results/pod5/my_sample/my_sample.pod5"  # Will be created by merge_pods
    bam: "results/bam/rebasecall/my_sample/my_sample.rbc.bam"  # Will be created by rebasecall
    label: "charged"
    amino_acid: "Ala"

    # Required for rebasecalling workflow
    raw_pod5: "data/raw/my_sample"  # Directory with raw POD5 files
    # OR
    raw_pod5_list:  # List of specific POD5 files
      - "data/raw/my_sample/file1.pod5"
      - "data/raw/my_sample/file2.pod5"

    # Optional for alignment
    reference: "references/genome.fasta"
```

### Dorado Configuration Parameters

```yaml
# Scratch directory for intermediate files (improves I/O performance)
scratch_dir: "/scratch/alpine/jhesselberth@xsede.org"

# Dorado binary path
dorado_bin: "dorado"  # or "/path/to/dorado"

# Basecalling model (default: RNA SUP for best accuracy)
base_calling_model: "rna004_130bps_sup@v5.2.0"

# Modified base detection (comma-separated list)
# Available RNA mods for rna004_130bps_sup@v5.2.0:
#   m5C_2OmeC, m6A_DRACH, inosine_m6A_2OmeA, pseU_2OmeU, 2OmeG
modifications: ""  # or "m6A_DRACH,pseU_2OmeU" for example

# Dorado options (--emit-moves is REQUIRED for leech)
dorado_opts: "--emit-moves"

# Resource allocation
merge_pods_threads: 12
rebasecall_gpu: 1
```

## Usage

### Merge POD5 files

```bash
# Merge POD5 files for all samples
snakemake --profile profiles/slurm all_merge_pods

# Merge POD5 for specific sample
snakemake --profile profiles/slurm results/pod5/my_sample/my_sample.pod5
```

### Rebasecall with dorado

```bash
# Rebasecall all samples
snakemake --profile profiles/slurm all_rebasecall

# Rebasecall specific sample
snakemake --profile profiles/slurm results/bam/rebasecall/my_sample/my_sample.rbc.bam
```

### Complete workflow

```bash
# 1. Rebasecall
snakemake --profile profiles/slurm all_rebasecall

# 2. Prepare training data
snakemake --profile profiles/slurm all_prepare

# 3. Train models
snakemake --profile profiles/slurm all_train
```

## Output Files

- `results/pod5/{sample}/{sample}.pod5`: Merged POD5 file
- `results/bam/rebasecall/{sample}/{sample}.rbc.bam`: Rebasecalled BAM with move tables
- `results/bam/rebasecall/{sample}/{sample}.aligned.bam`: Aligned BAM (if using align_rebasecalled)

## Requirements

- **dorado**: Install from https://github.com/nanoporetech/dorado
- **pod5**: `pip install pod5` or `uv add pod5`
- **samtools**: For alignment workflow
- **minimap2**: For alignment workflow

## RNA Modifications

The pipeline supports detection of RNA modifications during basecalling. Available modifications for `rna004_130bps_sup@v5.2.0`:

| Modification Code | Description |
|------------------|-------------|
| `m5C_2OmeC` | 5-methylcytosine and 2'-O-methylcytosine |
| `m6A_DRACH` | N6-methyladenosine in DRACH motif context |
| `inosine_m6A_2OmeA` | Inosine, N6-methyladenosine, and 2'-O-methyladenosine |
| `pseU_2OmeU` | Pseudouridine and 2'-O-methyluridine |
| `2OmeG` | 2'-O-methylguanosine |

To enable modification calling, set the `modifications` parameter in config.yaml:

```yaml
# Single modification
modifications: "m6A_DRACH"

# Multiple modifications (comma-separated, no spaces)
modifications: "m6A_DRACH,pseU_2OmeU"

# All modifications
modifications: "m5C_2OmeC,m6A_DRACH,inosine_m6A_2OmeA,pseU_2OmeU,2OmeG"
```

**Note**: Modification calling increases basecalling time and computational requirements.

## Important Notes

1. **Move tables required**: The `--emit-moves` flag is REQUIRED for leech to extract dwell time features
2. **GPU recommended**: Dorado basecalling is much faster with GPU acceleration
3. **Storage**: POD5 and BAM files can be large; ensure adequate storage space
4. **Model selection**: Choose appropriate model based on flowcell and kit (see dorado documentation)
5. **Scratch directory**: Using scratch storage significantly improves I/O performance for large files
