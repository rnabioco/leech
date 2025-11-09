#!/bin/bash
#SBATCH --job-name=leech_test_prep_train
#SBATCH --output=logs/test_prep_train_%j.out
#SBATCH --error=logs/test_prep_train_%j.err
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jhesselberth@xsede.org

# Test script: Run data preparation and model training on a small subset
# This script runs prep and training on just a few test samples:
#   - uncharged_synthetic (uncharged)
#   - ala_synthetic (charged - Alanine)
#   - gly_synthetic (charged - Glycine)
#
# IMPORTANT: Submit this script from the project root directory with:
#   sbatch scripts/test_prep_and_train.sh

set -euo pipefail

# Get the submission directory (where sbatch was run from)
# This is stored in SLURM_SUBMIT_DIR by Slurm
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    # Running via sbatch - use submission directory
    WORKDIR="${SLURM_SUBMIT_DIR}"
else
    # Running directly (testing) - detect from script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    if [[ -f "${SCRIPT_DIR}/workflow/Snakefile" ]]; then
        WORKDIR="${SCRIPT_DIR}"
    elif [[ -f "${SCRIPT_DIR}/../workflow/Snakefile" ]]; then
        WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
    else
        echo "Error: Cannot find workflow/Snakefile."
        echo "When running via sbatch, submit from the project root directory:"
        echo "  cd /path/to/leech && sbatch scripts/test_prep_and_train.sh"
        exit 1
    fi
fi

cd "${WORKDIR}"

# Verify we're in the right place
if [[ ! -f "workflow/Snakefile" ]]; then
    echo "Error: Not in project root (cannot find workflow/Snakefile)"
    echo "Current directory: $(pwd)"
    echo "Please submit from project root: sbatch scripts/test_prep_and_train.sh"
    exit 1
fi

echo "=========================================="
echo "Leech Pipeline - TEST Prepare and Train"
echo "=========================================="
echo "Orchestrator Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="
echo ""
echo "TEST MODE: Processing 3 samples only"
echo "  - uncharged_synthetic (uncharged)"
echo "  - ala_synthetic (charged - Alanine)"
echo "  - gly_synthetic (charged - Glycine)"
echo ""
echo "Workflow stages:"
echo "  1. Prepare training chunks from aligned BAM + POD5"
echo "  2. Train charged vs uncharged classification model"
echo ""

# Create logs directory
mkdir -p logs/slurm

# Define test samples
TEST_SAMPLES="uncharged_synthetic ala_synthetic gly_synthetic"

echo "Test samples:"
echo "-------------"
for sample in $TEST_SAMPLES; do
    echo "  - $sample"
done
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
uv run snakemake --profile=cluster/slurm --configfile=config/samples-alpine.yaml --unlock

echo ""
# Dry run first to check DAG
echo "Step 1: Testing DAG construction (dry run)..."
echo "-------------------------------------------"
uv run snakemake --profile=cluster/slurm \
  --configfile=config/samples-alpine.yaml \
  --dry-run \
  --printshellcmds \
  all_prepare -- $TEST_SAMPLES

echo ""
echo "Step 2: Submitting preparation jobs via Slurm executor..."
echo "---------------------------------------------------------"
echo "Prepare jobs: amilan partition (CPU)"
echo ""
echo "NOTE: This test script only runs data preparation for 3 samples."
echo "      To test training, you need to either:"
echo "      1. Run 'all_prepare' for ALL samples first, then 'all_train'"
echo "      2. Use scripts/run_prep_and_train.sh for the full pipeline"
echo ""
echo "You can monitor jobs with: squeue -u $USER"
echo ""

# Run preparation only for test samples
uv run snakemake --profile=cluster/slurm \
  --configfile=config/samples-alpine.yaml \
  all_prepare -- $TEST_SAMPLES

echo ""
echo "=========================================="
echo "TEST preparation completed: $(date)"
echo "=========================================="
echo "Output directories:"
echo "  Chunks: /scratch/alpine/jhesselberth@xsede.org/leech/synthetic-trna/chunks/"
echo "    (prepared for: $TEST_SAMPLES)"
echo ""
echo "Check individual logs:"
echo "  Prepare: /scratch/alpine/.../chunks/{sample}/prepare.log"
echo "  Seeds:   /scratch/alpine/.../chunks/{sample}/seed.txt"
echo ""
echo "Check Slurm logs: logs/slurm/"
echo ""
echo "To check job status:"
echo "  squeue -u $USER"
echo ""
echo "To check failed jobs:"
echo "  sacct -j $SLURM_JOB_ID --format=JobID,JobName,State,ExitCode"
echo ""
echo "Next steps:"
echo "  - To prepare all samples: sbatch scripts/run_prep_and_train.sh"
echo "  - Or prepare remaining samples with: uv run snakemake --profile=cluster/slurm --configfile=config/samples-alpine.yaml all_prepare"
echo "=========================================="
