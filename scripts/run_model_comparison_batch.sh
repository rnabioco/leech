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
# Leech Model Comparison Orchestrator (Batch Mode)
# =================================================
# This is a non-interactive batch script that orchestrates Snakemake job submission.
# The orchestrator itself runs as a low-resource SLURM job and submits all training
# jobs via Snakemake's SLURM executor.
#
# This script trains all 6 model architectures on:
# 1. Charged vs uncharged tRNA classification (6 jobs)
# 2. Pairwise amino acid classification (1,140 jobs = 6 models × 190 pairs)
#
# Total: 1,146 training jobs
#
# Usage:
#   sbatch scripts/run_model_comparison_batch.sh [charged|pairwise|all|dry-run]
#
# Examples:
#   sbatch scripts/run_model_comparison_batch.sh dry-run      # See what would run
#   sbatch scripts/run_model_comparison_batch.sh charged      # Just charged vs uncharged (6 jobs)
#   sbatch scripts/run_model_comparison_batch.sh pairwise     # Just pairwise AA (1,140 jobs)
#   sbatch scripts/run_model_comparison_batch.sh all          # Everything (1,146 jobs)
#
# Default mode if no argument provided: dry-run (safe default)

set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs

# Colors for output (works in log files too)
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

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check if we're in the right directory
    if [[ ! -f "workflow/Snakefile" ]]; then
        print_error "Snakefile not found. Script must run from leech root directory."
        exit 1
    fi

    # Check if configs exist
    if [[ ! -f "$MAIN_CONFIG" ]]; then
        print_error "Main config not found: $MAIN_CONFIG"
        exit 1
    fi

    if [[ ! -f "$SAMPLES_CONFIG" ]]; then
        print_error "Samples config not found: $SAMPLES_CONFIG"
        exit 1
    fi

    # Check if compare_models is enabled
    if ! grep -q "^compare_models: true" "$MAIN_CONFIG"; then
        print_error "compare_models is not set to 'true' in $MAIN_CONFIG"
        print_info "Please set: compare_models: true"
        exit 1
    fi

    # Check if cluster profile exists
    if [[ ! -d "$PROFILE" ]]; then
        print_error "Cluster profile not found: $PROFILE"
        exit 1
    fi

    print_success "All prerequisites satisfied"
}

# Show configuration summary
show_config() {
    print_info "Configuration Summary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Extract key config values
    local cpu_mode=$(grep "^use_cpu_training:" "$MAIN_CONFIG" | awk '{print $2}')
    local models=$(grep -A 6 "^models_to_compare:" "$MAIN_CONFIG" | grep '  - ' | wc -l)
    local amino_acids=$(grep -A 25 "^amino_acids:" "$MAIN_CONFIG" | grep '  - ' | wc -l)
    local pairs=$((amino_acids * (amino_acids - 1) / 2))

    echo "  Orchestrator Job: $SLURM_JOB_ID"
    echo "  Cluster Profile:  $PROFILE"
    echo "  Samples Config:   $SAMPLES_CONFIG"
    echo "  CPU Mode:         $cpu_mode"
    echo "  Models:           $models architectures"
    echo "  Amino Acids:      $amino_acids"
    echo "  Pairwise Pairs:   $pairs combinations"
    echo ""
    echo "  Training Jobs:"
    echo "    - Charged vs Uncharged:  $models jobs"
    echo "    - Pairwise AA:           $((models * pairs)) jobs"
    echo "    - TOTAL:                 $((models + models * pairs)) jobs"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Run Snakemake command
run_snakemake() {
    local mode=$1
    local targets=""
    local description=""

    case $mode in
        charged)
            targets="compare_architectures_charged"
            description="Charged vs Uncharged (6 jobs)"
            ;;
        pairwise)
            targets="compare_architectures_pairwise"
            description="Pairwise Amino Acids (1,140 jobs)"
            ;;
        all)
            targets="compare_architectures_charged compare_architectures_pairwise"
            description="All Tasks (1,146 jobs)"
            ;;
        dry-run)
            print_info "Running dry-run for ALL targets..."
            snakemake \
                --profile "$PROFILE" \
                --configfile "$SAMPLES_CONFIG" \
                --dryrun \
                --printshellcmds \
                compare_architectures_charged \
                compare_architectures_pairwise
            print_success "Dry-run complete. Review the output in logs/orchestrator-${SLURM_JOB_ID}.out"
            exit 0
            ;;
        *)
            print_error "Invalid mode: $mode"
            print_error "Valid modes: dry-run, charged, pairwise, all"
            exit 1
            ;;
    esac

    print_info "Starting training: $description"
    print_warning "This will submit $description to the cluster!"
    print_info "Orchestrator job ID: $SLURM_JOB_ID"
    print_info "Logs: logs/orchestrator-${SLURM_JOB_ID}.{out,err}"

    # Run Snakemake
    snakemake \
        --profile "$PROFILE" \
        --configfile "$SAMPLES_CONFIG" \
        --printshellcmds \
        $targets

    if [[ $? -eq 0 ]]; then
        print_success "Pipeline submitted successfully!"
        print_info "Monitor progress with: squeue -u \$USER"
        print_info "Check logs in: logs/ directory"
        print_info "Orchestrator log: logs/orchestrator-${SLURM_JOB_ID}.out"
    else
        print_error "Pipeline failed. Check error messages above and in logs/orchestrator-${SLURM_JOB_ID}.err"
        exit 1
    fi
}

# Show usage
show_usage() {
    cat << EOF

${GREEN}Usage:${NC} sbatch $0 [MODE]

${GREEN}Modes:${NC}
  ${YELLOW}dry-run${NC}   - Show what would be executed (no jobs submitted) [DEFAULT]
  ${YELLOW}charged${NC}   - Train charged vs uncharged only (6 jobs, ~45 min)
  ${YELLOW}pairwise${NC}  - Train pairwise amino acids only (1,140 jobs, ~4 days)
  ${YELLOW}all${NC}       - Train everything (1,146 jobs, ~4 days)

${GREEN}Examples:${NC}
  sbatch $0                # Dry-run (safe default)
  sbatch $0 dry-run        # Review execution plan
  sbatch $0 charged        # Start with charged vs uncharged
  sbatch $0 all            # Run full comparison

${GREEN}Output Locations:${NC}
  Models:       /scratch/alpine/\$USER/leech/synthetic-trna/models/comparison/
  Metrics:      /scratch/alpine/\$USER/leech/synthetic-trna/metrics/comparison/
  Logs:         logs/
  Orchestrator: logs/orchestrator-{jobid}.{out,err}

${GREEN}Monitoring:${NC}
  Queue:        squeue -u \$USER
  Orchestrator: tail -f logs/orchestrator-{jobid}.out
  Training:     tail -f logs/leech-train_architecture_charged-*.out
  Models:       ls /scratch/alpine/\$USER/leech/synthetic-trna/models/comparison/charged_vs_uncharged/*/model_best.pt

EOF
}

# Main script
main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     Leech Multi-Architecture Model Comparison Pipeline     ║"
    echo "║                    (Batch Mode)                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Node: $(hostname)"
    echo "Start time: $(date)"
    echo ""

    # Default to dry-run if no argument provided (safe default)
    MODE="${1:-dry-run}"

    # If dry-run was default, notify user
    if [[ $# -eq 0 ]]; then
        print_info "No mode specified, defaulting to 'dry-run' (safe mode)"
        print_info "To actually submit jobs, use: sbatch $0 [charged|pairwise|all]"
        echo ""
    fi

    # Validate and run
    check_prerequisites
    show_config
    echo ""

    run_snakemake "$MODE"

    echo ""
    echo "End time: $(date)"
    echo "Orchestrator job completed."
}

# Run main with all arguments
main "$@"
