#!/bin/bash
#SBATCH --job-name=leech_orchestrator
#SBATCH --output=logs/orchestrator_inspect_%j.out
#SBATCH --error=logs/orchestrator_inspect_%j.err
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00

# POD5 inspection using Slurm executor
# This orchestrator submits inspection jobs as separate Slurm jobs
# Can be run from scripts/ directory or project root

set -euo pipefail

# Determine project root (look for pipeline/workflow/Snakefile)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/pipeline/workflow/Snakefile" ]]; then
    WORKDIR="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../pipeline/workflow/Snakefile" ]]; then
    WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "Error: Cannot find pipeline/workflow/Snakefile. Run from project root or scripts/ directory."
    exit 1
fi

cd "${WORKDIR}"

echo "=========================================="
echo "Leech Pipeline - POD5 Inspection"
echo "=========================================="
echo "Orchestrator Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs/slurm

# Target: All POD5 inspection reports
TARGET="all_inspect_pod5"

echo "Step 1: Testing DAG construction (dry run)..."
snakemake --profile pipeline/cluster/slurm \
  --dry-run \
  --printshellcmds \
  $TARGET

echo ""
echo "Step 2: Submitting inspection jobs via Slurm executor..."
echo "Note: Each sample will be submitted as a separate Slurm job"
echo ""

# Run with Slurm executor
snakemake --profile pipeline/cluster/slurm $TARGET

echo ""
echo "=========================================="
echo "Orchestrator completed: $(date)"
echo "Check results in: results/pod5_inspection/"
echo "Check Slurm logs: logs/slurm/"
echo "=========================================="
