#!/bin/bash
# SLURM submission script for Snakemake jobs
# This script is used by Snakemake to submit jobs to SLURM

#SBATCH --job-name={rule}_{wildcards}
#SBATCH --output={log}
#SBATCH --error={log}
#SBATCH --cpus-per-task={threads}
#SBATCH --mem={resources.mem_mb}M
#SBATCH --time={resources.runtime}
#SBATCH --partition={resources.partition}

# GPU allocation (only if gpu > 0)
if [ {resources.gpu} -gt 0 ]; then
    #SBATCH --gres=gpu:{resources.gpu}
    #SBATCH --constraint="{resources.gpu_type}"
fi

# Account/project (if specified in config)
if [ -n "{resources.account}" ]; then
    #SBATCH --account={resources.account}
fi

# Set up environment
set -euo pipefail

# Load modules if needed (uncomment and adjust as needed)
# module purge
# module load cuda/11.8
# module load gcc/11.2.0

# Print job info
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: {rule}_{wildcards}"
echo "Node: $SLURM_NODELIST"
echo "CPUs: {threads}"
echo "Memory: {resources.mem_mb}M"
echo "Time Limit: {resources.runtime} minutes"
echo "GPUs: {resources.gpu}"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo ""

# Check for GPU availability if requested
if [ {resources.gpu} -gt 0 ]; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || echo "nvidia-smi not available"
    echo ""
fi

# Print environment info
echo "Python: $(which python3 || echo 'not found')"
echo "UV: $(which uv || echo 'not found')"
echo ""

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found in PATH"
    echo "Please ensure uv is installed and available"
    exit 1
fi

# Run the job
echo "Starting job execution..."
echo "=========================================="
{exec_job}
