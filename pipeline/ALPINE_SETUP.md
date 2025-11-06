# Running Leech Pipeline on Alpine Cluster

This guide explains how to run the Leech Snakemake pipeline on CU Boulder's Alpine cluster.

## Alpine Cluster Overview

Alpine is CU Boulder's primary high-performance computing cluster with:
- **CPU Nodes**: `amilan` partition (AMD Milan processors, 64 cores/node)
- **GPU Nodes**:
  - `aa100`: NVIDIA A100 GPUs (40GB or 80GB VRAM, 3 GPUs per node)
  - `ami100`: AMD MI100 GPUs
- **Testing Partitions**:
  - `atesting_a100`: For short A100 GPU test jobs
  - `atesting_mi100`: For short MI100 GPU test jobs

Documentation: https://curc.readthedocs.io/en/latest/clusters/alpine/

## Prerequisites

### 1. Alpine Account and Allocation

You need:
- An active Alpine account
- A computing allocation (account name for billing)

To check your allocations:
```bash
sacctmgr show associations where user=$USER format=account,cluster
```

### 2. Update Configuration

Edit `pipeline/config/config.yaml` and set your allocation account:

```yaml
cluster:
  account: "your_allocation_name"  # REQUIRED: Replace with your Alpine allocation
```

Example accounts: `ucb-general`, `ucb123_asc1`, etc.

## Installation on Alpine

### 1. Load Required Modules

```bash
# Load CUDA for GPU support (optional, but recommended)
module load cuda/12.1.1

# Or load AMD ROCm for MI100 GPUs
# module load rocm
```

### 2. Install uv (Python Package Manager)

```bash
# Install uv in your home directory
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (add this to your ~/.bashrc)
export PATH="$HOME/.cargo/bin:$PATH"

# Verify installation
uv --version
```

### 3. Clone and Setup

```bash
# Clone the repository
cd /projects/$USER  # Or your preferred location

# Install dependencies
cd leech
uv sync --all-extras
```

## Running the Pipeline

### Basic Usage

```bash
cd pipeline/workflow

# Dry run to see what will be executed
snakemake --profile ../profiles/slurm -n

# Execute the pipeline
snakemake --profile ../profiles/slurm
```

### Common Workflows

#### 1. Prepare Training Data Only

```bash
snakemake --profile ../profiles/slurm all_prepare
```

This will:
- Submit jobs to the `amilan` CPU partition
- Process POD5 and BAM files
- Extract training/validation/test chunks

#### 2. Train Models (GPU Jobs)

```bash
snakemake --profile ../profiles/slurm all_train
```

This will:
- Submit jobs to the `aa100` GPU partition
- Request 1 GPU per training job via `--gres=gpu:1`
- Use QoS `normal` (up to 24 hours)
- Allocate 42 tasks per node (optimized for aa100 nodes)

#### 3. Run Full Pipeline with Model Comparison

```bash
# Enable model comparison in config.yaml
# Set: compare_models: true

snakemake --profile ../profiles/slurm all_compare_models
```

#### 4. Run Grid Search (Long Jobs)

```bash
snakemake --profile ../profiles/slurm all_grid_search
```

Grid search jobs run for up to 24 hours with GPU acceleration.

### Resource Configuration

The pipeline automatically configures resources for Alpine:

| Rule Type | Partition | QoS | GPUs | Time | Memory |
|-----------|-----------|-----|------|------|--------|
| Data Prep | `amilan` | `normal` | 0 | 2h | 8GB |
| Training | `aa100` | `normal` | 1 | 8h | 16GB |
| Inference | `aa100` | `normal` | 1 | 4h | 8GB |
| Grid Search | `aa100` | `normal` | 1 | 24h | 16GB |
| Testing | `atesting_a100` | `testing` | 1 | 1h | 4GB |

These are defined in `profiles/slurm/config.yaml` and can be customized.

## Monitoring Jobs

### Check Job Status

```bash
# View your running/pending jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# View accounting info for completed jobs
sacct -u $USER --starttime=today
```

### Monitor GPU Usage

```bash
# SSH to the node running your job
ssh <node_name>

# Check GPU utilization
nvidia-smi

# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```

### View Logs

Logs are stored in the output directories:
```bash
# Training logs
tail -f results/models/charged_vs_uncharged/train.log

# View all logs for a specific rule
ls results/models/*/train.log
```

## Customizing for Alpine

### Using AMD MI100 GPUs Instead of A100

Edit `profiles/slurm/config.yaml`:

```yaml
set-resources:
  # Change partition from aa100 to ami100
  - train_charged_vs_uncharged:partition="ami100"
  - train_charged_vs_uncharged:ntasks=16  # MI100 nodes use 16 tasks
```

### Requesting Specific GPU Memory

For models requiring high VRAM (e.g., 80GB A100s):

Edit the rule in workflow files or use `--resources` flag:
```bash
snakemake --profile ../profiles/slurm --resources gpu_mem_mb=80000
```

### Adjusting Time Limits

For jobs needing >24 hours, use the `long` QoS:

```yaml
set-resources:
  - grid_search_charged_vs_uncharged:qos="long"
  - grid_search_charged_vs_uncharged:runtime=2880  # 48 hours
```

Note: Long QoS has limited availability and may wait longer in queue.

## Troubleshooting

### Common Issues

#### 1. "Invalid account" Error

**Solution**: Update your account in `config/config.yaml`:
```yaml
cluster:
  account: "your_allocation_name"
```

#### 2. GPU Jobs Not Starting

**Symptoms**: Jobs pending with reason `Resources` or `Priority`

**Solution**:
- Check GPU partition availability: `sinfo -p aa100`
- Use testing partition for short tests: `partition="atesting_a100"`
- Reduce GPU memory requirements if possible

#### 3. "uv: command not found"

**Solution**: Install uv or ensure it's in your PATH:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

#### 4. CUDA/GPU Not Detected

**Solution**: Load CUDA module in your job script or uncomment in `profiles/slurm/slurm_submit.sh`:
```bash
module load cuda/12.1.1
```

#### 5. Jobs Timing Out

**Symptoms**: Jobs fail with `TIMEOUT` status

**Solutions**:
- Increase `runtime` in rule resources
- Use testing partition for development/debugging
- Optimize batch size and model complexity

### Getting Help

- **Alpine Documentation**: https://curc.readthedocs.io/en/latest/
- **RC Help**: rc-help@colorado.edu
- **Office Hours**: https://www.colorado.edu/rc/help/support

## Best Practices

### 1. Test First with Small Jobs

Use the testing partition to verify your pipeline works:
```bash
# Run a single small job first
snakemake --profile ../profiles/slurm results/chunks/sample_1/train.json
```

### 2. Use Dry Runs

Always check what will be submitted:
```bash
snakemake --profile ../profiles/slurm -n
```

### 3. Monitor Resource Usage

Check if you're using resources efficiently:
```bash
seff <job_id>  # Shows efficiency statistics after job completes
```

### 4. Optimize Batch Sizes

For GPU jobs, larger batch sizes often improve GPU utilization:
- Start with default (128)
- Increase if GPU memory allows (256, 512)
- Monitor with `nvidia-smi` during training

### 5. Clean Up Old Jobs

Alpine has storage quotas. Clean up completed jobs:
```bash
# Remove old log files
find results/ -name "*.log" -mtime +30 -delete

# Archive completed results
tar -czf results_archive_$(date +%Y%m%d).tar.gz results/
```

## Performance Tips

### Maximize GPU Utilization

1. **Use Multiple GPUs** (for large models):
   ```yaml
   slurm_extra: "--gres=gpu:3"  # Use all 3 GPUs on aa100 node
   ```

2. **Optimize Data Loading**:
   - Store data on `/projects` (faster than `/home`)
   - Use SSD scratch space: `/rc_scratch/$USER`

3. **Parallel Job Submission**:
   The pipeline automatically submits multiple independent jobs in parallel (up to `jobs: 100` in profile config).

### Reduce Queue Wait Times

1. Use testing partitions for development
2. Submit during off-peak hours (late night, weekends)
3. Request only the resources you need
4. Use `--priority` for urgent jobs (limited by allocation)

## Example: Complete Workflow

```bash
# 1. Configure your allocation
vim config/config.yaml  # Set cluster.account

# 2. Test data preparation (CPU, fast)
snakemake --profile ../profiles/slurm all_prepare -j 5

# 3. Train models (GPU)
snakemake --profile ../profiles/slurm all_train

# 4. Run inference (GPU)
snakemake --profile ../profiles/slurm all_infer

# 5. Evaluate and summarize
snakemake --profile ../profiles/slurm all

# 6. Monitor progress
watch -n 10 squeue -u $USER
```

## Contact

For issues specific to this pipeline, open an issue on GitHub.
For Alpine cluster issues, contact RC Help at rc-help@colorado.edu.
