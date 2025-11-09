#!/bin/bash
# SLURM submission script for Snakemake jobs
# Configured for CU Boulder Alpine cluster
# https://curc.readthedocs.io/en/latest/clusters/alpine/

#SBATCH --job-name={rule}_{wildcards}
#SBATCH --output={log}
#SBATCH --error={log}
#SBATCH --nodes=1
#SBATCH --ntasks={resources.ntasks}
#SBATCH --cpus-per-task={threads}
#SBATCH --mem={resources.mem_mb}M
#SBATCH --time={resources.runtime}
#SBATCH --partition={resources.partition}
#SBATCH --qos={resources.qos}

# Set up environment
set -euo pipefail

# Load modules if needed for Alpine cluster
# Uncomment and adjust based on your Alpine configuration
# module purge
# module load cuda/12.1.1  # For GPU jobs - Alpine has CUDA 12.x
# module load gcc/11.2.0

# Print job info
echo "=========================================="
echo "Alpine SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: {rule}_{wildcards}"
echo "Node: $SLURM_NODELIST"
echo "Partition: {resources.partition}"
echo "QoS: {resources.qos}"
echo "CPUs: {threads}"
echo "Tasks: {resources.ntasks}"
echo "Memory: {resources.mem_mb}M"
echo "Time Limit: {resources.runtime} minutes"
echo "GPUs Requested: {resources.gpu}"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo ""

# Check for GPU availability if on GPU partition
if [[ "{resources.partition}" == aa100 ]] || [[ "{resources.partition}" == ami100 ]] || [[ "{resources.partition}" == atesting_a100 ]] || [[ "{resources.partition}" == atesting_mi100 ]]; then
    echo "GPU Partition Detected: {resources.partition}"

    # Display GPU information
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU Information:"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || echo "nvidia-smi query failed"
        echo ""
    elif command -v rocm-smi &> /dev/null; then
        echo "AMD GPU Information:"
        rocm-smi || echo "rocm-smi query failed"
        echo ""
    else
        echo "No GPU monitoring tool found (nvidia-smi or rocm-smi)"
        echo ""
    fi

    # Set CUDA_VISIBLE_DEVICES for PyTorch
    if [ -n "${SLURM_JOB_GPUS:-}" ]; then
        export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
        echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    fi
fi

# Print environment info
echo "Python: $(which python3 || echo 'not found')"
echo "UV: $(which uv || echo 'not found')"
echo ""

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found in PATH"
    echo "Please ensure uv is installed and available on Alpine"
    echo "You may need to install it in your home directory or load a module"
    exit 1
fi

# Run the job
echo "Starting job execution..."
echo "=========================================="
{exec_job}
