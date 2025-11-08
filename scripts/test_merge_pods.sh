#!/bin/bash
#SBATCH --job-name=leech_orchestrator
#SBATCH --output=logs/orchestrator_merge_%j.out
#SBATCH --error=logs/orchestrator_merge_%j.err
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00

# Test script: Merge POD5 files using Slurm executor
# This orchestrator job submits pipeline rules as separate Slurm jobs
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
echo "Leech Pipeline Test - POD5 Merge (Slurm Executor)"
echo "=========================================="
echo "Orchestrator Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs/slurm

# Test 1: Dry run to check DAG
echo "Step 1: Testing DAG construction (dry run)..."
snakemake --profile cluster/slurm \
  --dry-run \
  --printshellcmds \
  results/pod5/ala_synthetic/ala_synthetic.pod5

echo ""
echo "Step 2: Submitting POD5 merge job via Slurm executor..."
echo "Note: This will submit the actual work as a separate Slurm job"
echo ""

# Test 2: Run with Slurm executor (submits separate jobs)
snakemake --profile cluster/slurm \
  results/pod5/ala_synthetic/ala_synthetic.pod5

echo ""
echo "=========================================="
echo "Orchestrator completed: $(date)"
echo "Check output: results/pod5/ala_synthetic/"
echo "Check log: results/pod5/ala_synthetic/update_and_merge_pods.log"
echo "Check Slurm logs: logs/slurm/"
echo "=========================================="
