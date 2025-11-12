#!/bin/bash
#SBATCH --job-name=leech_full_pipeline
#SBATCH --output=logs/full_pipeline_%j.out
#SBATCH --error=logs/full_pipeline_%j.err
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jay.hesselberth@cuanschutz.edu

# Orchestrator script: Run the complete leech pipeline
# This script runs the full workflow from data preparation to inference:
#   1. Prepare training chunks (all_prepare)
#   2. Merge chunks and split into train/val/test (all_merge)
#   3. Train charged vs uncharged model (all_train)
#   4. Test trained model (all_test)
#   5. Run inference on held-out data (all_infer)
#
# IMPORTANT: Submit this script from the project root directory with:
#   sbatch scripts/run_full_pipeline.sh

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
        echo "  cd /path/to/leech && sbatch scripts/run_full_pipeline.sh"
        exit 1
    fi
fi

cd "${WORKDIR}"

# Verify we're in the right place
if [[ ! -f "pipeline/workflow/Snakefile" ]]; then
    echo "Error: Not in project root (cannot find pipeline/workflow/Snakefile)"
    echo "Current directory: $(pwd)"
    echo "Please submit from project root: sbatch scripts/run_full_pipeline.sh"
    exit 1
fi

echo "=========================================="
echo "Leech Pipeline - FULL PIPELINE"
echo "=========================================="
echo "Orchestrator Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="
echo ""
echo "Workflow stages:"
echo "  1. Prepare training chunks from aligned BAM + POD5 (per-sample)"
echo "  2. Merge chunks from all samples and split into train/val/test"
echo "  3. Train charged vs uncharged classification model"
echo "  4. Test trained model on held-out test set"
echo "  5. Run inference on new data"
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
  all_infer

echo ""
echo "Step 2: Submitting all jobs via Slurm executor..."
echo "-------------------------------------------------"
echo "Prepare jobs:   amilan partition (CPU)"
echo "Merge jobs:     amilan partition (CPU)"
echo "Training jobs:  aa100 partition (GPU)"
echo "Testing jobs:   aa100 partition (GPU)"
echo "Inference jobs: amilan partition (CPU)"
echo ""
echo "You can monitor jobs with: squeue -u $USER"
echo ""

# Run with Slurm executor - will submit jobs for all stages
# Target 'all_infer' includes all dependencies: prepare → merge → train → test → infer
uv run snakemake --profile=pipeline/cluster/slurm \
  --configfile=pipeline/config/samples-alpine.yaml \
  all_infer

echo ""
echo "=========================================="
echo "Orchestrator completed: $(date)"
echo "=========================================="
echo "Output directories:"
echo "  Chunks (per-sample): /scratch/alpine/jhesselberth@xsede.org/leech/synthetic-trna/chunks/{sample}/"
echo "  Chunks (merged):     /scratch/alpine/jhesselberth@xsede.org/leech/synthetic-trna/chunks/merged/"
echo "  Models:              /scratch/alpine/jhesselberth@xsede.org/leech/synthetic-trna/models/"
echo "  Test results:        /scratch/alpine/jhesselberth@xsede.org/leech/synthetic-trna/test/"
echo "  Inference results:   /scratch/alpine/jhesselberth@xsede.org/leech/synthetic-trna/infer/"
echo ""
echo "Check individual logs:"
echo "  Prepare:   /scratch/alpine/.../chunks/{sample}/prepare.log"
echo "  Seeds:     /scratch/alpine/.../chunks/{sample}/seed.txt"
echo "  Merge:     /scratch/alpine/.../chunks/merged/charged_vs_uncharged/merge.log"
echo "  Train:     /scratch/alpine/.../models/charged_vs_uncharged/train.log"
echo "  Test:      /scratch/alpine/.../test/charged_vs_uncharged/test.log"
echo "  Inference: /scratch/alpine/.../infer/{sample}/infer.log"
echo ""
echo "Check Slurm logs: logs/slurm/"
echo ""
echo "To check job status:"
echo "  squeue -u $USER"
echo ""
echo "To check failed jobs:"
echo "  sacct -j $SLURM_JOB_ID --format=JobID,JobName,State,ExitCode"
echo ""
echo "Key outputs:"
echo "  - Best model:     models/charged_vs_uncharged/model_best.pt"
echo "  - Test metrics:   test/charged_vs_uncharged/metrics.json"
echo "  - Predictions:    infer/{sample}/predictions.bam"
echo "=========================================="
