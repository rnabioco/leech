#!/usr/bin/env bash
#SBATCH --job-name=leech-orchestrator
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/orchestrator-%j.out
#SBATCH --error=logs/orchestrator-%j.err
#
# Leech Model Comparison Orchestrator (Batch Mode) - TSV-Based Workflow
# ======================================================================
#
# IMPORTANT: The pairwise Snakemake rules have been removed.
# Use the TSV-based comparison spec approach instead (see examples below).
#
# This batch script orchestrates Snakemake for charged vs uncharged only.
# For pairwise/multi-label comparisons, use the CLI directly.
#
# Usage:
#   sbatch scripts/run_model_comparison_batch.sh [charged|dry-run]
#
# Examples:
#   sbatch scripts/run_model_comparison_batch.sh dry-run    # Preview (safe default)
#   sbatch scripts/run_model_comparison_batch.sh charged    # Train charged vs uncharged
#
# For pairwise comparisons, use the TSV spec workflow directly:
#   uv run leech merge-and-split \
#     --input-chunks results/chunks/*_synthetic/*.npz \
#     --comparison-spec config/comparisons_chemical_properties.tsv \
#     --output-dir results/chunks/merged

set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROFILE="cluster/slurm"
SAMPLES_CONFIG="config/samples-alpine.yaml"
MAIN_CONFIG="config/config.yaml"

# Print with color
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage
show_usage() {
    cat << EOF

${GREEN}Leech Model Comparison Orchestrator - TSV-Based Workflow${NC}

${YELLOW}IMPORTANT:${NC} Pairwise Snakemake rules have been removed.

${GREEN}This Script (Snakemake for Charged vs Uncharged):${NC}
  sbatch $0 charged      # Train 6 architectures on charged vs uncharged
  sbatch $0 dry-run      # Preview (default if no arg)

${GREEN}For Pairwise/Multi-Label (Use CLI Directly):${NC}

  ${YELLOW}Step 1:${NC} Prepare comparisons from spec file
    uv run leech merge-and-split \\
      --input-chunks results/chunks/*_synthetic/*.npz \\
      --comparison-spec config/comparisons_chemical_properties.tsv \\
      --output-dir results/chunks/merged \\
      --seed 42

  ${YELLOW}Step 2:${NC} Train models (example with bash loop)
    for comp_dir in results/chunks/merged/*/; do
      comp_name=\$(basename "\$comp_dir")
      uv run leech train \\
        --train-data "\$comp_dir/train.npz" \\
        --val-data "\$comp_dir/val.npz" \\
        --model ConvLSTMDwell \\
        --output-dir "results/models/\$comp_name" \\
        --epochs 50
    done

${GREEN}Available Comparison Specs:${NC}
  config/comparisons_all_pairwise.tsv          # All 190 AA pairs
  config/comparisons_chemical_properties.tsv    # 9 chemical property comparisons

${GREEN}Output:${NC}
  Orchestrator: logs/orchestrator-{jobid}.{out,err}
  Comparisons:  results/chunks/merged/<comparison_name>/metadata.json
  Models:       results/models/<comparison_name>/model_best.pt

EOF
}

# Run Snakemake
run_snakemake() {
    local mode=$1

    if [[ "$mode" == "dry-run" ]]; then
        print_info "Running dry-run for charged vs uncharged..."
        snakemake \
            --profile "$PROFILE" \
            --configfile "$SAMPLES_CONFIG" \
            --dryrun \
            --printshellcmds \
            compare_architectures_charged
        print_success "Dry-run complete. Check logs/orchestrator-${SLURM_JOB_ID}.out"
        exit 0
    fi

    print_info "Starting training: Charged vs Uncharged (6 jobs)"
    print_info "Orchestrator job ID: $SLURM_JOB_ID"

    snakemake \
        --profile "$PROFILE" \
        --configfile "$SAMPLES_CONFIG" \
        --printshellcmds \
        compare_architectures_charged

    if [[ $? -eq 0 ]]; then
        print_success "Pipeline submitted successfully!"
        print_info "Monitor: squeue -u \$USER"
        print_info "Logs: logs/orchestrator-${SLURM_JOB_ID}.out"
    else
        print_error "Pipeline failed. Check logs/orchestrator-${SLURM_JOB_ID}.err"
        exit 1
    fi
}

# Main
main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     Leech Model Comparison Orchestrator (TSV-Based)       ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Node: $(hostname)"
    echo "Start time: $(date)"
    echo ""

    # Default to dry-run if no argument (safe)
    MODE="${1:-dry-run}"

    if [[ $# -eq 0 ]]; then
        print_info "No mode specified, defaulting to 'dry-run'"
        print_info "To submit jobs: sbatch $0 charged"
        echo ""
    fi

    case "$MODE" in
        charged|dry-run)
            run_snakemake "$MODE"
            ;;
        *)
            print_error "Unknown mode: $MODE"
            show_usage
            exit 1
            ;;
    esac

    echo ""
    echo "End time: $(date)"
}

main "$@"
