#!/usr/bin/env bash
#SBATCH --job-name=leech-orchestrator
#SBATCH --comment=leech-orchestrator
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
# For custom comparisons, use the TSV spec workflow directly:
#   uv run leech merge-and-split \
#     --input-chunks results/chunks/*_synthetic/*.npz \
#     --comparison-spec pipeline/config/comparisons_chemical_properties.tsv \
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
PROFILE="pipeline/cluster/slurm"
SAMPLES_CONFIG="pipeline/config/samples-alpine.yaml"
MAIN_CONFIG="pipeline/config/config.yaml"
SNAKEFILE="pipeline/workflow/Snakefile"
CHARGED_SPEC="pipeline/config/comparisons_charged_uncharged.tsv"

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
    echo ""
    echo -e "${GREEN}Leech Model Comparison Orchestrator - TSV-Based Workflow${NC}"
    echo ""
    echo -e "${YELLOW}IMPORTANT:${NC} Pairwise Snakemake rules have been removed."
    echo ""
    echo -e "${GREEN}Usage:${NC}"
    echo "  sbatch $0 charged              # Compare 6 architectures on charged vs uncharged (6 jobs)"
    echo "  sbatch $0 pairwise             # Compare 6 architectures on 190 AA pairs (1,140 jobs)"
    echo "  sbatch $0 chemical-properties  # Compare 6 architectures on 9 chemical comparisons (54 jobs)"
    echo "  sbatch $0 dry-run              # Preview (default if no arg)"
    echo ""
    echo -e "${GREEN}Available Comparison Specs:${NC}"
    echo "  pipeline/config/comparisons_charged_uncharged.tsv   # Charged vs uncharged tRNA"
    echo "  pipeline/config/comparisons_all_pairwise.tsv         # All 190 AA pairs"
    echo "  pipeline/config/comparisons_chemical_properties.tsv  # 9 chemical property comparisons"
    echo ""
    echo -e "${GREEN}Output:${NC}"
    echo "  Orchestrator: logs/orchestrator-{jobid}.{out,err}"
    echo "  Comparisons:  results/chunks/merged/<comparison_name>/metadata.json"
    echo "  Models:       results/models/<comparison_name>/model_best.pt"
    echo ""
}

# Run Snakemake
run_snakemake() {
    local mode=$1

    case "$mode" in
        dry-run)
            print_info "Running dry-run for charged vs uncharged..."
            snakemake \
                --snakefile "$SNAKEFILE" \
                --profile "$PROFILE" \
                --configfile "$SAMPLES_CONFIG" \
                --config comparison_spec_file="$CHARGED_SPEC" \
                --dryrun \
                --printshellcmds \
                all_compare_models
            print_success "Dry-run complete. Check logs/orchestrator-${SLURM_JOB_ID}.out"
            exit 0
            ;;

        charged)
            print_info "Starting training: Charged vs Uncharged (6 architectures)"
            print_info "Orchestrator job ID: $SLURM_JOB_ID"

            snakemake \
                --snakefile "$SNAKEFILE" \
                --profile "$PROFILE" \
                --configfile "$SAMPLES_CONFIG" \
                --config comparison_spec_file="$CHARGED_SPEC" \
                --printshellcmds \
                all_compare_models
            ;;

        pairwise)
            print_info "Starting pairwise comparisons: All 190 AA pairs"
            print_info "This will merge data and train all architectures for each pair"
            print_info "Total jobs: 190 pairs × 6 architectures = 1,140 training jobs"
            print_info "Orchestrator job ID: $SLURM_JOB_ID"

            snakemake \
                --snakefile "$SNAKEFILE" \
                --profile "$PROFILE" \
                --configfile "$SAMPLES_CONFIG" \
                --config comparison_spec_file=pipeline/config/comparisons_all_pairwise.tsv \
                --printshellcmds \
                all_compare_models
            ;;

        chemical-properties)
            print_info "Starting chemical property comparisons: 9 comparisons"
            print_info "This will merge data and train all architectures for each comparison"
            print_info "Total jobs: 9 comparisons × 6 architectures = 54 training jobs"
            print_info "Orchestrator job ID: $SLURM_JOB_ID"

            snakemake \
                --snakefile "$SNAKEFILE" \
                --profile "$PROFILE" \
                --configfile "$SAMPLES_CONFIG" \
                --config comparison_spec_file=pipeline/config/comparisons_chemical_properties.tsv \
                --printshellcmds \
                all_compare_models
            ;;
    esac

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
        charged|pairwise|chemical-properties|dry-run)
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
