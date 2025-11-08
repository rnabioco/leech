#!/bin/bash
#SBATCH --job-name=leech_orchestrator
#SBATCH --output=logs/orchestrator_full_%j.out
#SBATCH --error=logs/orchestrator_full_%j.err
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00

# Full pipeline test using Slurm executor
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
echo "Leech Pipeline - Full Test (1 sample)"
echo "=========================================="
echo "Orchestrator Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs/slurm

# Target: Complete pipeline for one sample (merge → basecall → align → prepare)
TARGET="results/chunks/uncharged_synthetic/{train,val,test}.json"

echo "Running Snakemake with Slurm profile..."
echo "Target: $TARGET"
echo ""

# Test 1: Dry run
echo "Step 1: Testing DAG construction (dry run)..."
snakemake --profile cluster/slurm \
  --dry-run \
  --printshellcmds \
  $TARGET

echo ""
echo "Step 2: Submitting pipeline jobs via Slurm executor..."
echo "Note: This will submit pipeline steps as separate Slurm jobs"
echo ""

# Test 2: Run with Slurm executor
snakemake --profile cluster/slurm $TARGET

echo ""
echo "=========================================="
echo "Orchestrator completed: $(date)"
echo "Results in: results/chunks/uncharged_synthetic/"
echo "Check Slurm logs: logs/slurm/"
echo "=========================================="
