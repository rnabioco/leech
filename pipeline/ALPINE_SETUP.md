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

## Alpine Storage Architecture

Alpine has three primary storage locations with different characteristics:

### 1. Home Directory (`/home/$USER`)
- **Quota**: 2 GB
- **Backed up**: Yes (regular backups)
- **Performance**: Moderate
- **Use for**: Code, scripts, small configuration files
- **Do NOT use for**: Large data files, intensive I/O operations

### 2. Projects Directory (`/projects/$USER`)
- **Quota**: 250 GB
- **Backed up**: Yes (regular backups)
- **Performance**: Moderate
- **Use for**: Raw data (POD5, BAM), trained models, final results, reference files
- **Do NOT use for**: Intensive I/O during compute jobs

### 3. Scratch Directory (`/scratch/alpine/$USER`)
- **Quota**: 10 TB (20 million files)
- **Backed up**: ❌ **NO** - Files are NOT backed up
- **Auto-purge**: ⚠️ **Files deleted after 90 days**
- **Performance**: ⚡ **Very high** - Optimized for parallel I/O
- **Use for**: Intermediate files, chunks, temporary processing, job logs
- **REQUIRED**: **ALL compute jobs with intensive I/O MUST write here**

### ⚠️ CRITICAL WARNINGS

1. **I/O Violations**: Running intensive I/O on `/home` or `/projects` will:
   - Get your jobs **terminated**
   - May result in your account being **temporarily disabled**

2. **90-Day Purge**: Files in `/scratch/alpine` are **automatically deleted 90 days** after creation
   - Set calendar reminders to move important results to `/projects`
   - Intermediate files (chunks) can be regenerated if needed

3. **No Backup**: `/scratch/alpine` is **NOT backed up** - any data loss is permanent

## Multi-Cluster Environment: Bodhi vs Alpine

This pipeline is designed to work in a **mixed cluster environment**:

### Bodhi Cluster (Local)
- **Storage**: Essentially unlimited home directory space
- **Structure**: Everything runs from `/home/$USER`
- **Flexibility**: No strict quotas or I/O restrictions
- **Configuration**: Simple - can use relative paths from home

### Alpine Cluster (CU Boulder)
- **Storage**: Strict quotas and separation of concerns
- **Structure**: Split between `/home`, `/projects`, and `/scratch/alpine`
- **Restrictions**: Intensive I/O MUST use `/scratch/alpine`
- **Configuration**: Requires careful path management

### Configuration Strategy

The pipeline includes **separate config files** for each cluster:

- `config/bodhi-config.yaml` - Bodhi cluster (unlimited home storage)
- `config/alpine-config.yaml` - Alpine cluster (strict quotas)
- `config/config.yaml` - Template/default (local testing)

**Using cluster-specific configs:**
```bash
# On Bodhi
snakemake --configfile ../config/bodhi-config.yaml --cores 8

# On Alpine
snakemake --configfile ../config/alpine-config.yaml --profile ../profiles/slurm
```

**Key Differences:**

| Setting | Bodhi | Alpine |
|---------|-------|--------|
| chunks_dir | `results/chunks` | `/scratch/alpine/$USER/leech/chunks` |
| models_dir | `results/models` | `/projects/$USER/leech/models` |
| Sample paths | Relative: `data/...` | Absolute: `/projects/$USER/...` |
| Account | Not required | **REQUIRED** (allocation) |

### When to Use Each Cluster

**Use Bodhi for:**
- Development and testing (faster turnaround)
- Small-scale experiments
- Iterative model development
- Jobs with frequent I/O patterns

**Use Alpine for:**
- Large-scale production runs
- GPU-intensive training (A100 GPUs)
- Jobs requiring high parallel I/O (scratch filesystem)
- Long-running experiments (up to 24h+ with proper QoS)

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
# Clone the repository to your projects directory
cd /projects/$USER
git clone <repository_url> leech
cd leech

# Install dependencies
uv sync --all-extras
```

## Storage Configuration

### Recommended Directory Structure

Set up your directories on Alpine following this structure:

```bash
# Projects directory (permanent, backed up, 250GB quota)
/projects/$USER/leech/
├── leech/                          # Code repository
├── data/                           # Raw input data (POD5, BAM files)
│   ├── charged/
│   └── uncharged/
├── models/                         # Trained models (keep permanently)
├── inference/                      # Inference results (keep permanently)
└── metrics/                        # Evaluation metrics (keep permanently)

# Scratch directory (temporary, NOT backed up, 10TB quota, 90-day purge)
/scratch/alpine/$USER/leech/
├── chunks/                         # Training chunks (can regenerate)
├── logs/                          # Job logs (temporary)
└── tmp/                           # Temporary processing files
```

### Configure Pipeline for Alpine Storage

Edit `pipeline/config/alpine-config.yaml`:

```yaml
# Use scratch for intermediate files (high I/O)
chunks_dir: "/scratch/alpine/$USER/leech/chunks"

# Use projects for permanent outputs (backed up)
models_dir: "/projects/$USER/leech/models"
inference_dir: "/projects/$USER/leech/inference"
metrics_dir: "/projects/$USER/leech/metrics"

samples:
  sample_charged_ala_rep1:
    # Raw data in projects (permanent)
    pod5: "/projects/$USER/leech/data/charged/ala/rep1.pod5"
    bam: "/projects/$USER/leech/data/charged/ala/rep1.bam"
    label: "charged"
    amino_acid: "Ala"
```

### Create Directories

```bash
# Create projects directory structure
mkdir -p /projects/$USER/leech/{data,models,inference,metrics}

# Create scratch directory structure
mkdir -p /scratch/alpine/$USER/leech/{chunks,logs,tmp}

# Copy or link your data
cp /path/to/your/*.pod5 /projects/$USER/leech/data/
cp /path/to/your/*.bam /projects/$USER/leech/data/
```

## Working Across Bodhi and Alpine

### Transferring Data Between Clusters

**From Bodhi to Alpine:**
```bash
# On Bodhi: Package your data
cd /home/$USER/leech
tar -czf data_for_alpine.tar.gz data/

# Transfer to Alpine
rsync -avz data_for_alpine.tar.gz $USER@login.rc.colorado.edu:/projects/$USER/leech/

# On Alpine: Extract
ssh login.rc.colorado.edu
cd /projects/$USER/leech
tar -xzf data_for_alpine.tar.gz
```

**From Alpine to Bodhi:**
```bash
# On Alpine: Archive results
cd /projects/$USER/leech
tar -czf results_from_alpine.tar.gz models/ inference/ metrics/

# Transfer to Bodhi
rsync -avz results_from_alpine.tar.gz $USER@bodhi:/home/$USER/leech/

# On Bodhi: Extract
cd /home/$USER/leech
tar -xzf results_from_alpine.tar.gz
```

### Syncing Configuration

Keep configuration in sync across clusters:

```bash
# Initialize git repo for configs (if not already)
cd /home/$USER/leech/pipeline
git init

# On Bodhi: Commit config changes
git add config/config.yaml
git commit -m "Update sample paths"
git push

# On Alpine: Pull latest config
cd /projects/$USER/leech/pipeline
git pull

# Then edit paths for Alpine environment
```

### Strategy: Develop on Bodhi, Run on Alpine

A recommended workflow:

```bash
# 1. Develop and test on Bodhi (fast iteration)
# On Bodhi
cd /home/$USER/leech
snakemake --cores 8 -n  # Test locally

# 2. Once working, transfer to Alpine for production
rsync -avz /home/$USER/leech/ \
  $USER@login.rc.colorado.edu:/projects/$USER/leech/ \
  --exclude 'results/' --exclude '.snakemake/'

# 3. Run full pipeline on Alpine with GPUs
# On Alpine
cd /projects/$USER/leech/pipeline/workflow
# Update config.yaml for Alpine paths
snakemake --profile ../profiles/slurm

# 4. Transfer results back to Bodhi for analysis
rsync -avz $USER@login.rc.colorado.edu:/projects/$USER/leech/results/ \
  /home/$USER/leech/results/
```

## Running the Pipeline

### Basic Usage

```bash
cd /projects/$USER/leech/pipeline/workflow

# Dry run to see what will be executed
snakemake --configfile ../config/alpine-config.yaml --profile ../profiles/slurm -n

# Execute the pipeline
snakemake --configfile ../config/alpine-config.yaml --profile ../profiles/slurm
```

**Tip**: Create a shell alias for easier use:
```bash
# Add to your ~/.bashrc on Alpine
alias snakemake-alpine='snakemake --configfile config/alpine-config.yaml --profile profiles/slurm'

# Then use:
cd /projects/$USER/leech/pipeline/workflow
snakemake-alpine -n  # Dry run
snakemake-alpine     # Execute
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

## Managing Data Lifecycle on Alpine

### Understanding File Ages in Scratch

Check when files will be purged from `/scratch/alpine`:

```bash
# Check file creation time and age
stat /scratch/alpine/$USER/leech/chunks/*

# Find files older than 60 days (approaching purge)
find /scratch/alpine/$USER/leech -type f -mtime +60 -ls

# Find files created more than 80 days ago (will be purged soon!)
find /scratch/alpine/$USER/leech -type f -ctime +80 -ls
```

### Preserving Important Results

**Before 90-day purge, move important data to /projects:**

```bash
# Archive and move trained models (if stored in scratch)
tar -czf /projects/$USER/leech/models_backup_$(date +%Y%m%d).tar.gz \
  /scratch/alpine/$USER/leech/models/

# Move inference results to projects
rsync -av /scratch/alpine/$USER/leech/inference/ \
  /projects/$USER/leech/inference/

# Clean up scratch after verification
rm -rf /scratch/alpine/$USER/leech/inference/
```

### Regenerating Chunks

If chunks are purged from scratch, regenerate them:

```bash
cd /projects/$USER/leech/pipeline/workflow

# Regenerate chunks for all samples
snakemake --profile ../profiles/slurm all_prepare
```

Since chunks are derived from POD5/BAM (stored in `/projects`), they can always be regenerated.

### Monitoring Storage Quotas

```bash
# Check your quota usage on all filesystems
curc-quota

# Detailed breakdown
df -h /home/$USER
df -h /projects/$USER
df -h /scratch/alpine/$USER

# Count files in scratch (20M file limit)
find /scratch/alpine/$USER -type f | wc -l
```

### Storage Best Practices

1. **Keep raw data in /projects**: POD5, BAM, reference genomes
2. **Use /scratch for processing**: Chunks, intermediate files, logs
3. **Save final outputs to /projects**: Trained models, inference results, metrics
4. **Set reminders**: Calendar alerts 80 days after starting large jobs
5. **Archive regularly**: Compress and archive completed results
6. **Clean scratch proactively**: Don't wait for auto-purge

### Example Cleanup Script

Create a cleanup script for old scratch data:

```bash
#!/bin/bash
# cleanup_scratch.sh - Run monthly to clean old files

SCRATCH_DIR="/scratch/alpine/$USER/leech"
PROJECTS_DIR="/projects/$USER/leech"

# Archive logs older than 30 days
find $SCRATCH_DIR/logs -name "*.log" -mtime +30 -print0 | \
  tar -czf $PROJECTS_DIR/logs_archive_$(date +%Y%m%d).tar.gz --null -T -

# Remove archived logs
find $SCRATCH_DIR/logs -name "*.log" -mtime +30 -delete

# Archive and remove old chunks (can regenerate if needed)
find $SCRATCH_DIR/chunks -mtime +60 -print0 | \
  tar -czf $PROJECTS_DIR/chunks_archive_$(date +%Y%m%d).tar.gz --null -T -
find $SCRATCH_DIR/chunks -mtime +60 -delete

echo "Cleanup complete. Archived files saved to $PROJECTS_DIR"
```

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

#### 6. Disk Quota Exceeded

**Symptoms**: Jobs fail with "No space left on device" or "Quota exceeded"

**Solutions**:

For `/home` (2GB quota):
```bash
# Check usage
du -sh /home/$USER

# Move data to projects
mv /home/$USER/large_files /projects/$USER/
```

For `/projects` (250GB quota):
```bash
# Check usage
curc-quota

# Archive old results
tar -czf archive_$(date +%Y%m%d).tar.gz old_results/
rm -rf old_results/

# Move to scratch if temporary
mv large_temp_files/ /scratch/alpine/$USER/
```

For `/scratch/alpine` (10TB quota):
```bash
# Check usage and file count
find /scratch/alpine/$USER -type f | wc -l  # Must be < 20M files

# Clean up old files
find /scratch/alpine/$USER -mtime +30 -delete
```

#### 7. Files Missing from Scratch

**Symptoms**: Pipeline can't find files that were in `/scratch/alpine`

**Cause**: Files auto-purged after 90 days or accidentally deleted

**Solutions**:
```bash
# Regenerate chunks from raw data in /projects
snakemake --profile ../profiles/slurm all_prepare

# Restore from backup if you archived
tar -xzf /projects/$USER/leech/chunks_archive_*.tar.gz -C /scratch/alpine/$USER/
```

#### 8. I/O Performance Issues

**Symptoms**: Jobs running slowly, high wait times

**Solutions**:
- Verify you're using `/scratch/alpine` for intensive I/O
- Check if scratch filesystem is under heavy load: `df -h /scratch/alpine`
- Reduce number of concurrent I/O operations
- Optimize batch sizes to reduce I/O frequency

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
