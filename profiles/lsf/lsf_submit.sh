#!/bin/bash
# LSF submission script for Snakemake jobs
# This script is used by Snakemake to submit jobs to LSF

#BSUB -J {rule}_{wildcards}
#BSUB -o {log}
#BSUB -e {log}
#BSUB -n {threads}
#BSUB -M {resources.mem_mb}
#BSUB -R "rusage[mem={resources.mem_mb}]"
#BSUB -W {resources.runtime}
#BSUB -q {resources.queue}

# GPU allocation (only if gpu > 0)
if [ {resources.gpu} -gt 0 ]; then
    #BSUB -gpu "num={resources.gpu}:mode=shared:mps=no:j_exclusive=yes"
    #BSUB -R "select[gpu_model=={resources.gpu_model}]"
    # Alternative GPU resource specification:
    # #BSUB -R "rusage[ngpus_physical={resources.gpu}]"
fi

# Project/group (if specified in config)
if [ -n "{resources.project}" ]; then
    #BSUB -P {resources.project}
fi

# Set up environment
set -euo pipefail

# Load modules if needed (uncomment and adjust as needed)
# module purge
# module load cuda/11.8
# module load gcc/11.2.0

# Print job info
echo "=========================================="
echo "LSF Job Information"
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Job Name: {rule}_{wildcards}"
echo "Execution Host: $LSB_HOSTS"
echo "CPUs: {threads}"
echo "Memory: {resources.mem_mb}MB"
echo "Time Limit: {resources.runtime} minutes"
echo "Queue: {resources.queue}"
echo "GPUs: {resources.gpu}"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo ""

# Check for GPU availability if requested
if [ {resources.gpu} -gt 0 ]; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || echo "nvidia-smi not available"
    echo ""

    # Set CUDA_VISIBLE_DEVICES if LSF provides it
    if [ -n "${LSB_AFFINITY_HOSTFILE:-}" ]; then
        echo "LSF GPU Affinity File: $LSB_AFFINITY_HOSTFILE"
        cat "$LSB_AFFINITY_HOSTFILE" || echo "Could not read affinity file"
        echo ""
    fi
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
