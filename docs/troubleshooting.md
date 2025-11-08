# Troubleshooting Guide

## SLURM Executor Issues

### QoS Error

**Symptom:**
```
sbatch: error: Error: A Quality of Service (QoS) has not been provided,
specifying a QoS is now required.
```

**Solution:**
Ensure `cluster/slurm/config.yaml` has `slurm-qos` (with hyphen) as a top-level parameter:

```yaml
# Correct configuration
slurm-qos: "normal"
```

NOT as a resource with underscore:
```yaml
# INCORRECT
default-resources:
  slurm_qos: "normal"  # Wrong - this doesn't work
```

See `TESTING.md` for detailed explanation and verification steps.

### Lock Files

**Symptom:**
```
LockException: Error: Directory cannot be locked
```

**Solution:**
```bash
snakemake --unlock
```

This can happen if a previous Snakemake run was killed or terminated unexpectedly.

### Running in SLURM Job Context

**Warning:**
```
You are running snakemake in a SLURM job context. This is not recommended
```

**Explanation:**
This warning appears when you run Snakemake from within an sbatch job (like the orchestrator pattern in `scripts/test_merge_pods.sh`). This is expected behavior - the orchestrator job submits worker jobs via sbatch.

You can safely ignore this warning when using the orchestrator pattern, but avoid running Snakemake directly on compute nodes.

## Data Pipeline Issues

### POD5 Files Not Found

**Symptom:**
```
FileNotFoundError: POD5 file not found
```

**Check:**
1. Verify paths in `config/samples.yml`
2. Ensure POD5 files exist at specified locations
3. Check file permissions

### BAM Missing Move Tables

**Symptom:**
```
KeyError: 'mv' tag not found in BAM file
```

**Solution:**
BAM files must be basecalled with dorado or guppy to include move table tags (`mv` and `ns`). Re-basecall if necessary:

```bash
snakemake --profile cluster/slurm all_rebasecall
```

### Memory Errors During Training

**Symptom:**
```
CUDA out of memory
```

**Solutions:**
1. Reduce batch size in training config
2. Use CPU instead of GPU (slower)
3. Request more GPU memory in SLURM config

Edit `cluster/slurm/config.yaml`:
```yaml
set-resources:
  train_model:
    mem_mb: 64000  # Increase from 32000
```

## Model Issues

### Poor Model Performance

**Check:**
1. Verify training/validation split is balanced
2. Check chunk context sizes (use grid search)
3. Ensure sufficient training data
4. Verify feature normalization

**Debug:**
```bash
# Run grid search to optimize chunk context
snakemake --profile cluster/slurm all_grid_search

# Check training metrics
cat results/models/charged_vs_uncharged/training_history.json
```

### Model Loading Errors

**Symptom:**
```
RuntimeError: Error loading model checkpoint
```

**Solutions:**
1. Verify model architecture matches checkpoint
2. Check PyTorch version compatibility
3. Ensure model file is not corrupted

```bash
# Verify checkpoint integrity
uv run python -c "import torch; print(torch.load('path/to/model.pt').keys())"
```

## Common Command Issues

### Profile Not Found

**Symptom:**
```
Could not find profile: cluster/slurm
```

**Solution:**
Run from project root directory where `cluster/` exists, or specify full path:
```bash
snakemake --profile /full/path/to/cluster/slurm target
```

### Conda/Apptainer Errors

**Symptom:**
```
CondaError: environment not found
```

**Solution:**
Snakemake uses apptainer (singularity) for containerized execution. Ensure apptainer cache directory is accessible:

```yaml
# In cluster/slurm/config.yaml
apptainer-prefix: '/scratch/alpine/username/apptainer_cache'
```

Create the directory if it doesn't exist:
```bash
mkdir -p /scratch/alpine/$USER/apptainer_cache
```

## Getting Help

1. Check this troubleshooting guide
2. Review `TESTING.md` for diagnostic scripts
3. Check Snakemake logs in `logs/slurm/`
4. Enable verbose mode: `snakemake --verbose ...`
5. Review rule-specific logs in `results/*/logs/`

For SLURM-specific issues, consult:
- Alpine docs: https://curc.readthedocs.io/en/latest/clusters/alpine/
- Snakemake SLURM plugin: https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html
