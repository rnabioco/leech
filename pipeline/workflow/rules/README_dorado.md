# Dorado Rebasecalling Rules

This file documents the POD5 rebasecalling workflow using dorado.

## Overview

The dorado rules provide a complete workflow for rebasecalling raw POD5 files:

1. **merge_pods**: Consolidates multiple raw POD5 files into a single merged file
2. **rebasecall**: Runs dorado basecaller with move table and modification detection
3. **align_rebasecalled**: Aligns rebasecalled reads to reference (REQUIRED - preserves mv, ns, MM, ML tags)

## Configuration

Add the following to your sample configuration in `config.yaml`:

```yaml
# Global reference for alignment (required)
reference: "references/transcriptome.fasta"

samples:
  my_sample:
    # Required for leech workflow (use aligned BAM after rebasecalling)
    pod5: "results/pod5/my_sample/my_sample.pod5"  # Created by merge_pods
    bam: "results/bam/rebasecall/my_sample/my_sample.aligned.bam"  # Created by align_rebasecalled
    label: "charged"
    amino_acid: "Ala"

    # Required for rebasecalling workflow
    raw_pod5: "data/raw/my_sample"  # Directory with raw POD5 files
    # OR
    raw_pod5_list:  # List of specific POD5 files
      - "data/raw/my_sample/file1.pod5"
      - "data/raw/my_sample/file2.pod5"
```

### Dorado Configuration Parameters

```yaml
# Reference genome/transcriptome for alignment (REQUIRED)
reference: "references/transcriptome.fasta"

# Scratch directory for intermediate files (improves I/O performance)
scratch_dir: "/scratch/alpine/jhesselberth@xsede.org"

# Dorado binary path
dorado_bin: "dorado"  # or "/path/to/dorado"

# Basecalling model (default: RNA SUP for best accuracy)
base_calling_model: "rna004_130bps_sup@v5.2.0"

# Modified base detection (comma-separated list, all enabled by default)
# Available RNA mods for rna004_130bps_sup@v5.2.0:
#   m5C_2OmeC, m6A_DRACH, inosine_m6A_2OmeA, pseU_2OmeU, 2OmeG
modifications: "m5C_2OmeC,m6A_DRACH,inosine_m6A_2OmeA,pseU_2OmeU,2OmeG"

# Dorado options (--emit-moves is REQUIRED for leech)
dorado_opts: "--emit-moves"

# Resource allocation
merge_pods_threads: 12
rebasecall_gpu: 1
align_threads: 8
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

### Align rebasecalled reads

```bash
# Align all samples (REQUIRED - preserves mv, ns, MM, ML tags)
snakemake --profile profiles/slurm all_align

# Align specific sample
snakemake --profile profiles/slurm results/bam/rebasecall/my_sample/my_sample.aligned.bam
```

### Complete workflow

```bash
# 1. Merge POD5 files
snakemake --profile profiles/slurm all_merge_pods

# 2. Rebasecall with dorado
snakemake --profile profiles/slurm all_rebasecall

# 3. Align to reference (preserves all tags)
snakemake --profile profiles/slurm all_align

# 4. Prepare training data
snakemake --profile profiles/slurm all_prepare

# 5. Train models
snakemake --profile profiles/slurm all_train
```

## Output Files

- `results/pod5/{sample}/{sample}.pod5`: Merged POD5 file
- `results/bam/rebasecall/{sample}/{sample}.rbc.bam`: Rebasecalled BAM with move tables (intermediate)
- `results/bam/rebasecall/{sample}/{sample}.aligned.bam`: Aligned BAM with preserved tags (use this for leech)
- `results/bam/rebasecall/{sample}/{sample}.aligned.bam.bai`: BAM index file

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

1. **Tag preservation**: The alignment step preserves critical tags:
   - `mv`: Move table (REQUIRED for leech dwell time features)
   - `ns`: Number of samples per base
   - `MM`: Modified base positions/types
   - `ML`: Modified base probabilities
2. **Move tables required**: The `--emit-moves` flag is REQUIRED for leech to extract dwell time features
3. **Use aligned BAM**: Always use the `.aligned.bam` file for leech prepare/inference, not the `.rbc.bam`
4. **Reference required**: A reference transcriptome/genome is required for alignment
5. **GPU recommended**: Dorado basecalling is much faster with GPU acceleration
6. **Storage**: POD5 and BAM files can be large; ensure adequate storage space
7. **Scratch directory**: Using scratch storage significantly improves I/O performance for large files
