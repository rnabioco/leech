#!/bin/bash
#SBATCH --job-name=leech_orchestrator
#SBATCH --output=logs/orchestrator_basecall_%j.out
#SBATCH --error=logs/orchestrator_basecall_%j.err
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00

# Test script: Rebasecall one sample with dorado using Slurm executor
# This orchestrator submits the GPU job as a separate Slurm job
# Can be run from scripts/ directory or project root

set -euo pipefail

# Determine project root (look for workflow/Snakefile)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/workflow/Snakefile" ]]; then
    WORKDIR="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../workflow/Snakefile" ]]; then
    WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "Error: Cannot find workflow/Snakefile. Run from project root or scripts/ directory."
    exit 1
fi

cd "${WORKDIR}"

echo "=========================================="
echo "Leech Pipeline Test - Dorado Rebasecalling"
echo "=========================================="
echo "Orchestrator Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs/slurm

# Target: Rebasecall for one sample
TARGET="results/bam/rebasecall/uncharged_synthetic/uncharged_synthetic.rbc.bam"

echo "Step 1: Testing DAG construction (dry run)..."
snakemake --profile cluster/slurm \
  --dry-run \
  --printshellcmds \
  $TARGET

echo ""
echo "Step 2: Submitting basecalling job via Slurm executor..."
echo "Note: GPU job will be submitted to aa100 partition automatically"
echo ""

# Run with Slurm executor - will submit GPU job to aa100 partition
snakemake --profile cluster/slurm $TARGET

echo ""
echo "=========================================="
echo "Orchestrator completed: $(date)"
echo "Output: results/bam/rebasecall/uncharged_synthetic/"
echo "Check log: results/bam/rebasecall/uncharged_synthetic/rebasecall.log"
echo "Check Slurm logs: logs/slurm/"
echo "=========================================="
