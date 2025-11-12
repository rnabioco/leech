#!/bin/bash
#SBATCH --job-name=leech_all_align
#SBATCH --output=logs/all_align_%j.out
#SBATCH --error=logs/all_align_%j.err
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00

# Orchestrator script: Run production pipeline through alignment
# This script runs the full workflow through all_align:
#   1. Merge POD5 files (all_merge_pods)
#   2. Rebasecall with dorado (all_rebasecall)
#   3. Align to reference (all_align)
#
# IMPORTANT: Submit this script from the project root directory with:
#   sbatch scripts/run_all_align.sh

set -euo pipefail

# Get the submission directory (where sbatch was run from)
# This is stored in SLURM_SUBMIT_DIR by Slurm
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    # Running via sbatch - use submission directory
    WORKDIR="${SLURM_SUBMIT_DIR}"
else
    # Running directly (testing) - detect from script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    if [[ -f "${SCRIPT_DIR}/pipeline/workflow/Snakefile" ]]; then
        WORKDIR="${SCRIPT_DIR}"
    elif [[ -f "${SCRIPT_DIR}/../pipeline/workflow/Snakefile" ]]; then
        WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
    else
        echo "Error: Cannot find pipeline/workflow/Snakefile."
        echo "When running via sbatch, submit from the project root directory:"
        echo "  cd /path/to/leech && sbatch scripts/run_all_align.sh"
        exit 1
    fi
fi

cd "${WORKDIR}"

# Verify we're in the right place
if [[ ! -f "pipeline/workflow/Snakefile" ]]; then
    echo "Error: Not in project root (cannot find pipeline/workflow/Snakefile)"
    echo "Current directory: $(pwd)"
    echo "Please submit from project root: sbatch scripts/run_all_align.sh"
    exit 1
fi

echo "=========================================="
echo "Leech Pipeline - Run Through Alignment"
echo "=========================================="
echo "Orchestrator Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="
echo ""
echo "Workflow stages:"
echo "  1. Merge POD5 files"
echo "  2. Rebasecall with dorado (RNA modifications enabled)"
echo "  3. Align to reference with minimap2"
echo ""

# Create logs directory
mkdir -p logs/slurm

echo "Samples to process:"
echo "-------------------"
grep -A 1 "^  [a-z]" pipeline/config/samples-alpine.yaml | grep -v "^--$" | grep "^  [a-z]" | sed 's/://g' | nl
echo ""
echo "Total: $(grep -A 1 "^  [a-z]" pipeline/config/samples-alpine.yaml | grep -v "^--$" | grep "^  [a-z]" | wc -l) samples"
echo ""

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found in PATH"
    echo "Please ensure uv is installed and available"
    exit 1
fi

echo "Using uv from: $(which uv)"
echo ""

# Unlock directory in case of previous interrupted run
echo "Step 0: Unlocking workflow directory..."
echo "---------------------------------------"
uv run snakemake --profile=pipeline/cluster/slurm --configfile=pipeline/config/samples-alpine.yaml --unlock

echo ""
# Dry run first to check DAG
echo "Step 1: Testing DAG construction (dry run)..."
echo "-------------------------------------------"
uv run snakemake --profile=pipeline/cluster/slurm \
  --configfile=pipeline/config/samples-alpine.yaml \
  --dry-run \
  --printshellcmds \
  all_align

echo ""
echo "Step 2: Submitting all jobs via Slurm executor..."
echo "-------------------------------------------------"
echo "GPU jobs (rebasecall): aa100 partition"
echo "CPU jobs (merge, align): amilan partition"
echo ""
echo "You can monitor jobs with: squeue -u $USER"
echo ""

# Run with Slurm executor - will submit jobs for each stage
uv run snakemake --profile=pipeline/cluster/slurm \
  --configfile=pipeline/config/samples-alpine.yaml \
  all_align

echo ""
echo "=========================================="
echo "Orchestrator completed: $(date)"
echo "=========================================="
echo "Output directories:"
echo "  POD5: /scratch/alpine/jhesselberth@xsede.org/leech/synthetic-trna/pod5/"
echo "  BAM:  /scratch/alpine/jhesselberth@xsede.org/leech/synthetic-trna/bam/rebasecall/"
echo ""
echo "Check individual sample logs:"
echo "  Merge:      /scratch/alpine/.../pod5/{sample}/merge_pods.log"
echo "  Rebasecall: /scratch/alpine/.../bam/rebasecall/{sample}/rebasecall.log"
echo "  Align:      /scratch/alpine/.../bam/rebasecall/{sample}/align.log"
echo ""
echo "Check Slurm logs: logs/slurm/"
echo ""
echo "To check job status:"
echo "  squeue -u $USER"
echo ""
echo "To check failed jobs:"
echo "  sacct -j $SLURM_JOB_ID --format=JobID,JobName,State,ExitCode"
echo "=========================================="
