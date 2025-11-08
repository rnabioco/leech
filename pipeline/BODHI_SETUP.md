# Running Leech Pipeline on Bodhi Cluster

This guide explains how to run the Leech Snakemake pipeline on your local Bodhi cluster with LSF scheduler.

## Bodhi Cluster Overview

Bodhi is your local HPC cluster with:
- **Scheduler**: LSF (Load Sharing Facility)
- **Storage**: Essentially unlimited home directory space
- **Structure**: Everything runs from `/home/$USER`
- **Flexibility**: No strict quotas or I/O restrictions
- **GPU Queues**: Accessible via LSF queue system

## Prerequisites

### 1. Access to Bodhi

Ensure you have:
- Active Bodhi cluster account
- SSH access to Bodhi login nodes
- Access to appropriate LSF queues (especially GPU queues if needed)

Check your queue access:
```bash
bqueues  # List all queues
bjobs    # List your running/pending jobs
```

### 2. Install uv (Python Package Manager)

```bash
# Install uv in your home directory
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (add to ~/.bashrc)
export PATH="$HOME/.cargo/bin:$PATH"

# Verify installation
uv --version
```

### 3. Install Snakemake LSF Executor Plugin (Recommended)

The modern approach uses the LSF executor plugin:

```bash
# Install the LSF executor plugin
uv pip install snakemake-executor-plugin-lsf

# Or if using pip directly:
pip install snakemake-executor-plugin-lsf
```

**Why use the executor plugin?**
- More efficient job submission
- Better resource management
- Native LSF integration
- Actively maintained
- Better error handling

**Repository**: https://github.com/BEFH/snakemake-executor-plugin-lsf

### 4. Clone and Setup

```bash
# Clone repository
cd /home/$USER
git clone <repository_url> leech
cd leech

# Install dependencies
uv sync --all-extras
```

## Configuration

The pipeline includes `config/bodhi-config.yaml` optimized for Bodhi:

```yaml
# Simple relative paths (everything in home directory)
chunks_dir: "results/chunks"
models_dir: "results/models"
inference_dir: "results/inference"
metrics_dir: "results/metrics"

# Bodhi-specific cluster settings
cluster:
  default_queue: "general"     # Adjust to your Bodhi setup
  gpu_queue: "gpuqueue"        # GPU queue name
  gpu_model: "a100"            # GPU model for resource requests
```

### Update for Your Bodhi Environment

Edit `config/bodhi-config.yaml` to match your Bodhi setup:

```yaml
cluster:
  default_queue: "your_queue_name"     # Check with: bqueues
  gpu_queue: "your_gpu_queue"          # GPU queue name
  gpu_model: "your_gpu_model"          # e.g., "a100", "v100", "h100"
```

## Running the Pipeline

### Option 1: Using LSF Executor Plugin (Recommended)

This is the modern, recommended approach:

```bash
cd /home/$USER/leech/pipeline/workflow

# Dry run
snakemake \
  --executor lsf \
  --configfile ../config/bodhi-config.yaml \
  --default-resources lsf_queue=gpuqueue \
  --jobs 100 \
  -n

# Execute pipeline
snakemake \
  --executor lsf \
  --configfile ../config/bodhi-config.yaml \
  --default-resources lsf_queue=gpuqueue \
  --jobs 100
```

**Create an alias** for convenience (add to `~/.bashrc`):

```bash
alias snakemake-bodhi='snakemake --executor lsf --configfile config/bodhi-config.yaml --default-resources lsf_queue=gpuqueue --jobs 100'
```

Then use:
```bash
cd /home/$USER/leech/pipeline/workflow
snakemake-bodhi -n  # Dry run
snakemake-bodhi     # Execute
```

### Option 2: Using Traditional Profile (Legacy)

The traditional approach still works:

```bash
cd /home/$USER/leech/pipeline/workflow

# Dry run
snakemake --profile ../profiles/lsf --configfile ../config/bodhi-config.yaml -n

# Execute
snakemake --profile ../profiles/lsf --configfile ../config/bodhi-config.yaml
```

### Common Workflows

#### 1. Prepare Training Data

```bash
snakemake-bodhi all_prepare
```

#### 2. Train Models with GPU

```bash
# Training jobs automatically request GPU resources
snakemake-bodhi all_train
```

#### 3. Full Pipeline

```bash
snakemake-bodhi all
```

#### 4. Run Specific Rule

```bash
snakemake-bodhi results/models/charged_vs_uncharged/model_best.pt
```

## LSF-Specific Features

### GPU Resource Requests

The pipeline automatically requests GPUs for training/inference jobs:

```bash
# LSF directives added automatically:
# #BSUB -gpu "num=1:mode=shared:mps=no:j_exclusive=yes"
# #BSUB -R "select[gpu_model==a100]"
```

GPU requirements are set in the Snakemake rules:
- Training: 1 GPU, 16GB RAM, 8h runtime
- Inference: 1 GPU, 8GB RAM, 4h runtime
- Grid search: 1 GPU, 16GB RAM, 24h runtime

### Monitoring Jobs

```bash
# View your jobs
bjobs

# Detailed job info
bjobs -l <job_id>

# View job output
bpeek <job_id>

# Check queue status
bqueues -l gpuqueue

# Historical job info
bhist -l <job_id>
```

### Managing Jobs

```bash
# Kill a job
bkill <job_id>

# Kill all your jobs
bkill 0

# Suspend a job
bstop <job_id>

# Resume a job
bresume <job_id>
```

## Directory Structure

On Bodhi, everything can run from your home directory:

```bash
/home/$USER/leech/
├── leech/                          # Code repository (if cloned here)
├── pipeline/
│   ├── config/
│   │   └── bodhi-config.yaml      # Bodhi-specific config
│   ├── profiles/
│   │   └── lsf/                   # LSF profile
│   └── workflow/                  # Snakemake workflow
├── data/                          # Raw data (POD5, BAM)
├── results/
│   ├── chunks/                    # Training chunks
│   ├── models/                    # Trained models
│   ├── inference/                 # Inference results
│   └── metrics/                   # Evaluation metrics
```

## Troubleshooting

### Common Issues

#### 1. "Command not found: bsub"

**Solution**: LSF commands not in PATH
```bash
# Load LSF module (if available)
module load lsf

# Or ask your Bodhi admin for LSF setup
```

#### 2. GPU Jobs Not Starting

**Symptoms**: Jobs pending with `PEND` status, reason shows resource requirements

**Solutions**:
```bash
# Check GPU queue availability
bqueues -l gpuqueue

# Check GPU hosts
bhosts -l | grep gpu

# Verify your queue access
bqueues | grep gpu

# Check specific GPU availability
bhosts -gpu
```

#### 3. "uv: command not found"

**Solution**: Install uv or add to PATH
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

#### 4. Executor Plugin Not Found

**Symptoms**: Error about LSF executor not available

**Solution**:
```bash
# Install the plugin
uv pip install snakemake-executor-plugin-lsf

# Verify installation
python -c "import snakemake_executor_plugin_lsf; print('OK')"
```

#### 5. Jobs Failing with Memory Errors

**Symptoms**: Jobs exit with `TERM_MEMLIMIT` or similar

**Solutions**:
- Increase memory in rule resources
- Check memory limits with: `bqueues -l <queue_name>`
- Adjust batch sizes in config.yaml

#### 6. Module Loading Issues

If your jobs need specific modules:

Edit `profiles/lsf/lsf_submit.sh` and uncomment/add:
```bash
module purge
module load cuda/12.1
module load gcc/11.2
# Add any other modules needed
```

## Performance Optimization

### 1. Queue Selection

Use appropriate queues for different job types:
```bash
# CPU jobs → general queue
# GPU jobs → GPU queue
# Short test jobs → test/debug queue (if available)
```

### 2. Resource Requests

Request only what you need:
```yaml
# In Snakemake rules:
resources:
  mem_mb=8000,      # Don't over-request memory
  runtime=120,      # Set realistic time limits
  gpu=1,            # 1 GPU usually sufficient
```

### 3. Parallel Job Submission

The pipeline submits up to 100 jobs in parallel by default:
```bash
# Adjust with --jobs flag
snakemake-bodhi --jobs 50  # More conservative
snakemake-bodhi --jobs 200 # More aggressive
```

### 4. Local Execution for Small Tasks

Run data preparation locally (faster turnaround):
```bash
# Run prep locally without cluster submission
cd /home/$USER/leech/pipeline/workflow
snakemake --configfile ../config/bodhi-config.yaml --cores 8 all_prepare
```

## Comparing Bodhi vs Alpine

| Feature | Bodhi (LSF) | Alpine (SLURM) |
|---------|-------------|----------------|
| Scheduler | LSF | SLURM |
| Storage | Unlimited home | Strict quotas |
| Configuration | Simple paths | /scratch + /projects |
| I/O Restrictions | None | Intensive I/O → scratch |
| Best For | Development, testing | Production, large-scale |
| GPU Access | LSF queues | SLURM partitions |
| Job Command | `bjobs` | `squeue` |

### When to Use Bodhi

✅ Development and iterative testing
✅ Small to medium-scale experiments
✅ Rapid prototyping
✅ Frequent I/O operations
✅ No strict storage quotas

### When to Use Alpine

✅ Large-scale production runs
✅ High-performance parallel I/O
✅ Long-running experiments (24h+)
✅ A100 GPU availability
✅ Need for backed-up storage

## Best Practices

### 1. Test Locally First

```bash
# Quick local test with 8 cores
snakemake --configfile config/bodhi-config.yaml --cores 8 -n
```

### 2. Start Small

```bash
# Test with one sample first
snakemake-bodhi results/chunks/sample1/train.json
```

### 3. Monitor Resource Usage

After jobs complete:
```bash
# Check resource usage
bhist -l <job_id>

# Look for:
# - Actual CPU time vs requested
# - Actual memory vs requested
# - Adjust future requests accordingly
```

### 4. Use Dry Runs

Always check what will run:
```bash
snakemake-bodhi -n  # Dry run
snakemake-bodhi -np # Dry run with detailed output
```

### 5. Clean Up Old Results

Bodhi has lots of space, but keep it tidy:
```bash
# Archive old results
tar -czf archive_$(date +%Y%m%d).tar.gz results/
rm -rf results/

# Clean Snakemake metadata
rm -rf .snakemake/
```

## Example: Complete Workflow

```bash
# 1. Setup
cd /home/$USER/leech/pipeline/workflow

# 2. Update config with your data paths
vim ../config/bodhi-config.yaml

# 3. Test with dry run
snakemake-bodhi -n

# 4. Prepare data (can run locally for speed)
snakemake --configfile ../config/bodhi-config.yaml --cores 8 all_prepare

# 5. Train models (submit to GPU queue)
snakemake-bodhi all_train

# 6. Monitor progress
watch -n 10 bjobs

# 7. Run inference
snakemake-bodhi all_infer

# 8. Evaluate
snakemake-bodhi all

# 9. Check results
ls -lh results/models/
ls -lh results/inference/
```

## Getting Help

- **LSF Documentation**: Check your Bodhi cluster documentation
- **LSF Executor Plugin**: https://github.com/BEFH/snakemake-executor-plugin-lsf
- **Snakemake Docs**: https://snakemake.readthedocs.io/
- **Local Support**: Contact your Bodhi cluster administrators

## Quick Reference

```bash
# Setup
uv pip install snakemake-executor-plugin-lsf

# Run (modern)
snakemake --executor lsf --configfile config/bodhi-config.yaml \
  --default-resources lsf_queue=gpuqueue --jobs 100

# Run (legacy)
snakemake --profile profiles/lsf --configfile config/bodhi-config.yaml

# Monitor
bjobs                    # List your jobs
bjobs -l <id>           # Job details
bpeek <id>              # View output
bqueues                 # List queues

# Control
bkill <id>              # Kill job
bstop <id>              # Suspend job
bresume <id>            # Resume job
```
