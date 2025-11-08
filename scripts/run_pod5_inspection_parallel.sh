#!/bin/bash
#
# Run POD5 inspection with parallel SLURM jobs using Slurm executor profile
#
# This script uses Snakemake's SLURM executor plugin to submit each sample
# inspection as a separate SLURM job for maximum parallelization.
#
# NOTE: Run this directly from project root or scripts/ directory
#       DO NOT use sbatch with this script
#
# Usage:
#   From project root:  bash scripts/run_pod5_inspection_parallel.sh
#   From scripts/ dir:  bash run_pod5_inspection_parallel.sh
#

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
echo "Leech Pipeline - POD5 Inspection (Parallel)"
echo "=========================================="
echo "Working directory: $(pwd)"
echo "Started: $(date)"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs/slurm

# Target: All POD5 inspection reports
TARGET="all_inspect_pod5"

echo "Running POD5 inspection with Slurm profile..."
echo "Each sample will be submitted as a separate Slurm job"
echo ""

# Run with Slurm executor profile
snakemake --profile cluster/slurm $TARGET

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "Check results in: results/pod5_inspection/"
echo "Check Slurm logs: logs/slurm/"
echo "=========================================="
