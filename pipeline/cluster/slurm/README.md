# Slurm Executor Profile for Leech Pipeline

This directory contains the Snakemake profile configuration for running the leech pipeline on the CU Boulder Alpine cluster using the Slurm executor.

## Setup

### 1. Install Slurm Executor Plugin

```bash
# Using uv (recommended)
uv pip install snakemake-executor-plugin-slurm

# Or using pip
pip install snakemake-executor-plugin-slurm
```

### 2. Configure Your Alpine Account

Edit `cluster/slurm/config.yaml` and set your Alpine allocation account:

```yaml
default-resources:
  slurm_account: "your-account-here"  # e.g., "ucb-general" or "ucb123_asc1"
```

**Important:** You must set this before running the pipeline, or jobs will fail to submit.

## Usage

### Production Runs

Run the pipeline using the Slurm executor profile with full resources:

```bash
# Full pipeline with Alpine sample configuration
snakemake --profile cluster/slurm --configfile config/samples-alpine.yaml

# Specific target
snakemake --profile cluster/slurm --configfile config/samples-alpine.yaml all_rebasecall
```

This will:
- Submit each rule as a separate Slurm job
- Use production resources (aa100 partition, 32GB memory, etc.)
- Apply rule-specific resource overrides automatically
- Write Slurm logs to `logs/slurm/`

### Testing Runs

For testing with reduced resources and faster turnaround, use the testing profile:

```bash
# Using testing profile (reduced resources, atesting_a100 partition)
snakemake --profile cluster/slurm-testing --configfile config/samples-alpine.yaml <target>
```

**Key differences in testing mode (`cluster/slurm-testing`):**
- ALL GPU rules use `atesting_a100` partition (not just test_* rules)
- Reduced memory: 8GB for training (vs 32GB production), 4GB for inference (vs 16GB)
- Shorter runtimes: 2-4 hours max (vs 8-24 hours)
- Uses `testing` QoS for faster scheduling

### Orchestrated runs

Submit the workflow through the Slurm executor with an orchestrator job that
drives Snakemake (for example, `snakemake --executor slurm --profile
pipeline/cluster/slurm`).

The orchestrator job will:
1. Run a dry-run to show the execution plan
2. Submit actual pipeline rules as separate Slurm jobs
3. Monitor and report completion

## How It Works

### Orchestrator Pattern

The Slurm executor uses an "orchestrator" pattern:

1. **Orchestrator Job**: A lightweight Slurm job that runs Snakemake with the `--executor slurm` flag
2. **Worker Jobs**: Snakemake submits each rule as a separate Slurm job with appropriate resources
3. **Job Management**: The orchestrator monitors worker jobs and manages the DAG execution

### Resource Management

Resources are configured at three levels:

1. **Default Resources** (`config.yaml`): Applied to all rules
   - `slurm_partition: "amilan"` - Default CPU partition
   - `runtime: 120` - Default 2-hour limit
   - `mem_mb: 8000` - Default 8GB memory

2. **Rule-Specific Overrides** (`set-resources` in `config.yaml`): Custom settings per rule
   - GPU jobs automatically use `aa100` partition
   - Training jobs get longer runtime and more memory
   - Lightweight tasks use minimal resources

3. **Snakefile Resources**: Can be overridden in rule definitions

Priority: Snakefile > set-resources > default-resources

## Resource Specifications

### CPU Rules (default)
- Partition: `amilan`
- QoS: `normal` (up to 24 hours)
- Memory: 4-16GB depending on task
- CPUs: 1-8 depending on task

### GPU Rules
- Partition: `aa100` (NVIDIA A100 GPUs)
- QoS: `normal` (or `long` for >24hr jobs)
- Memory: 32GB
- CPUs: 8
- GPUs: 1 A100

GPU rules:
- `basecall_with_dorado`
- `rebasecall_pod5`
- `train_model`
- `grid_search`
- `run_inference`

## Monitoring Jobs

### Check Slurm Queue

```bash
# Your jobs
squeue -u $USER

# Specific job details
scontrol show job <JOB_ID>
```

### Check Logs

```bash
# Orchestrator logs
ls logs/orchestrator_*.out

# Individual rule logs
ls logs/slurm/

# Pipeline rule logs (from Snakemake)
ls logs/
```

### Cancel Jobs

```bash
# Cancel specific job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Cancel by name
scancel --name leech_orchestrator
```

## Troubleshooting

### Jobs Not Submitting

1. **Check account is set**: Verify `slurm_account` in `config.yaml`
2. **Check allocation**: `sacctmgr show assoc user=$USER format=account,qos`
3. **Check partition access**: `sinfo -p amilan` or `sinfo -p aa100`

### Jobs Failing

1. **Check Slurm logs**: `logs/slurm/<rule>-<jobid>.out`
2. **Check rule logs**: `logs/<rule>.log`
3. **Increase resources**: Edit `set-resources` in `config.yaml`

### Out of Memory

Increase `mem_mb` for the specific rule in `set-resources`:

```yaml
set-resources:
  your_rule:
    mem_mb: 32000  # Increase to 32GB
```

### Timeout

Increase `runtime` (in minutes) for the specific rule:

```yaml
set-resources:
  your_rule:
    runtime: 480  # Increase to 8 hours
```

## Customization

### Add New Rule Resources

Edit `cluster/slurm/config.yaml` and add to `set-resources`:

```yaml
set-resources:
  your_new_rule:
    runtime: 240              # 4 hours
    mem_mb: 16000            # 16GB
    cpus_per_task: 8
    slurm_partition: "amilan"
```

### Use Different Partition

For testing or different node types:

```yaml
set-resources:
  your_rule:
    slurm_partition: "amilan-test"  # Testing partition
    slurm_qos: "testing"            # Testing QoS
    runtime: 60                     # 1 hour max for testing
```

## References

- [Snakemake Slurm Executor Plugin](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html)
- [Alpine Documentation](https://curc.readthedocs.io/en/latest/clusters/alpine/)
- [Alpine Slurm Guide](https://curc.readthedocs.io/en/latest/running-jobs/batch-jobs.html)
