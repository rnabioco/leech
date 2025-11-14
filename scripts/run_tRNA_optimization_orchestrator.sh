#!/usr/bin/env bash
#SBATCH --job-name=trna-opt-orchestrator
#SBATCH --comment=trna-optimization-orchestrator
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --account=amc-general
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/trna_optimization_orchestrator-%j.out
#SBATCH --error=logs/trna_optimization_orchestrator-%j.err
#
# tRNA Optimization Full Analysis Orchestrator
# ============================================
#
# This script orchestrates the complete tRNA optimization workflow for pairwise
# amino acid comparisons, including:
#   1. Training 4 model variants (base, dwell, tcn_signal_features, dwell_masked)
#   2. Testing all variants on held-out test sets
#   3. Model comparison analysis
#   4. Feature importance analysis
#   5. Sequence ablation tests
#   6. Aggregate results across all pairs
#
# See docs/guides/09-TRNA_OPTIMIZATION_IMPLEMENTATION.md for details.
# See docs/guides/11-TRNA_OPTIMIZATION_WORKFLOW.md for workflow documentation.
#
# Usage:
#   sbatch scripts/run_tRNA_optimization_orchestrator.sh [MODE]
#
# Modes:
#   dry-run              Preview workflow (safe default)
#   charged              Full analysis for charged vs uncharged only (1 comparison)
#   pairwise             Full analysis for all 190 AA pairwise comparisons
#   chemical-properties  Full analysis for 9 chemical property comparisons
#
# Examples:
#   sbatch scripts/run_tRNA_optimization_orchestrator.sh dry-run
#   sbatch scripts/run_tRNA_optimization_orchestrator.sh charged
#   sbatch scripts/run_tRNA_optimization_orchestrator.sh pairwise
#
# Expected Job Counts:
#   charged:             4 training + 4 testing + 1 comparison + 1 feature importance + 1 ablation = ~11 jobs
#   pairwise:            190×4 training + 190×4 testing + 190 comparisons + 190 feature importance + 190 ablation + 1 aggregate = ~1,521 jobs
#   chemical-properties: 9×4 training + 9×4 testing + 9 comparisons + 9 feature importance + 9 ablation + 1 aggregate = ~91 jobs
#
# Output Structure (with project_name: synthetic-trna-opt):
#   /scratch/alpine/user/leech/synthetic-trna-opt/models/tRNA_optimization/pairwise/{pair}/{variant}/model_best.pt
#   /scratch/alpine/user/leech/synthetic-trna-opt/metrics/tRNA_optimization/pairwise/{pair}/comparison_results.json
#   /scratch/alpine/user/leech/synthetic-trna-opt/metrics/tRNA_optimization/pairwise/{pair}/feature_importance/
#   /scratch/alpine/user/leech/synthetic-trna-opt/metrics/tRNA_optimization/pairwise/{pair}/sequence_ablation/
#   /scratch/alpine/user/leech/synthetic-trna-opt/metrics/tRNA_optimization/aggregate/variant_comparison.tsv
#

set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROFILE="pipeline/cluster/slurm"
SAMPLES_CONFIG="pipeline/config/samples-alpine.yaml"
MAIN_CONFIG="pipeline/config/config.yaml"
SNAKEFILE="pipeline/workflow/Snakefile"
CHARGED_SPEC="pipeline/config/comparisons_charged_uncharged.tsv"
PAIRWISE_SPEC="pipeline/config/comparisons_all_pairwise.tsv"
CHEMICAL_SPEC="pipeline/config/comparisons_chemical_properties.tsv"

# Project name for organizing outputs
PROJECT_NAME="synthetic-trna-opt"

# Snakemake target
TARGET_RULE="tRNA_optimization_all"

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

print_header() {
    echo -e "${MAGENTA}[HEADER]${NC} $1"
}

print_detail() {
    echo -e "${CYAN}[DETAIL]${NC} $1"
}

# Show usage
show_usage() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║      tRNA Optimization Full Analysis Orchestrator                     ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Description:${NC}"
    echo "  Runs complete tRNA optimization workflow for pairwise comparisons:"
    echo "    • Trains 4 model variants per comparison"
    echo "    • Evaluates all variants on test sets"
    echo "    • Compares model performance"
    echo "    • Analyzes feature importance"
    echo "    • Tests sequence ablation"
    echo "    • Aggregates results across all pairs"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo "  sbatch $0 [MODE]"
    echo ""
    echo -e "${YELLOW}Modes:${NC}"
    echo "  dry-run              Preview workflow (default, safe)"
    echo "  charged              Full analysis for charged vs uncharged (1 comparison)"
    echo "  pairwise             Full analysis for all 190 AA pairs"
    echo "  chemical-properties  Full analysis for 9 chemical property comparisons"
    echo ""
    echo -e "${YELLOW}Model Variants Trained:${NC}"
    echo "  1. base               ConvLSTMBase (no features, baseline)"
    echo "  2. dwell              ConvLSTMDwell (full model)"
    echo "  3. tcn_signal_features TCNSignalFeatures (optimized for constant sequence)"
    echo "  4. dwell_masked       ConvLSTMDwell + 50% sequence masking"
    echo ""
    echo -e "${YELLOW}Expected Job Counts:${NC}"
    echo "  charged:             ~11 jobs (4 train + 4 test + 3 analysis)"
    echo "  pairwise:            ~1,521 jobs (190 pairs × 8 tasks + aggregate)"
    echo "  chemical-properties: ~91 jobs (9 comparisons × 10 tasks + aggregate)"
    echo ""
    echo -e "${YELLOW}Output:${NC}"
    echo "  Orchestrator logs:   logs/trna_optimization_orchestrator-{jobid}.{out,err}"
    echo "  Models:              /scratch/alpine/.../leech/synthetic-trna-opt/models/tRNA_optimization/..."
    echo "  Metrics:             /scratch/alpine/.../leech/synthetic-trna-opt/metrics/tRNA_optimization/..."
    echo "  Aggregate:           /scratch/alpine/.../leech/synthetic-trna-opt/metrics/tRNA_optimization/aggregate/"
    echo ""
    echo -e "${YELLOW}Reference Documentation:${NC}"
    echo "  Implementation:      docs/guides/09-TRNA_OPTIMIZATION_IMPLEMENTATION.md"
    echo "  Workflow Guide:      docs/guides/11-TRNA_OPTIMIZATION_WORKFLOW.md"
    echo "  Model Selection:     docs/guides/10-MODEL_SELECTION_GUIDE.md"
    echo ""
}

# Print job statistics
print_job_stats() {
    local mode=$1
    local spec_file=$2

    # Count comparisons from spec file
    local num_comparisons=$(wc -l < "$spec_file")
    local num_variants=4

    # Calculate job counts
    local train_jobs=$((num_comparisons * num_variants))
    local test_jobs=$((num_comparisons * num_variants))
    local compare_jobs=$num_comparisons
    local feature_jobs=$num_comparisons
    local ablation_jobs=$num_comparisons
    local aggregate_jobs=1
    local total_jobs=$((train_jobs + test_jobs + compare_jobs + feature_jobs + ablation_jobs + aggregate_jobs))

    print_header "Expected Job Statistics"
    echo "  Comparisons:         $num_comparisons"
    echo "  Variants per comp:   $num_variants"
    echo "  Training jobs:       $train_jobs"
    echo "  Testing jobs:        $test_jobs"
    echo "  Comparison jobs:     $compare_jobs"
    echo "  Feature imp. jobs:   $feature_jobs"
    echo "  Ablation jobs:       $ablation_jobs"
    echo "  Aggregate jobs:      $aggregate_jobs"
    echo "  ─────────────────────────────"
    echo "  Total jobs:          ~$total_jobs"
    echo ""
}

# Print workflow summary
print_workflow_summary() {
    local mode=$1
    local spec_file=$2

    print_header "Workflow Summary"
    echo "  Mode:                $mode"
    echo "  Comparison spec:     $spec_file"
    echo "  Project name:        $PROJECT_NAME"
    echo "  Target rule:         $TARGET_RULE"
    echo "  Profile:             $PROFILE"
    echo "  Samples config:      $SAMPLES_CONFIG"
    echo ""
}

# Run Snakemake workflow
run_snakemake() {
    local mode=$1
    local spec_file=$2
    local is_dry_run=$3

    print_workflow_summary "$mode" "$spec_file"

    if [[ $is_dry_run == "true" ]]; then
        print_info "Running dry-run (no jobs will be submitted)..."
        print_warning "This preview shows what WOULD be executed"
        echo ""
    else
        print_job_stats "$mode" "$spec_file"
        print_info "Starting tRNA optimization workflow..."
        print_info "Orchestrator job ID: $SLURM_JOB_ID"
        print_info "This will submit hundreds/thousands of jobs - be patient!"
        echo ""
    fi

    # Build snakemake command
    local snakemake_cmd=(
        snakemake
        --snakefile "$SNAKEFILE"
        --profile "$PROFILE"
        --configfile "$SAMPLES_CONFIG"
        --config "comparison_spec_file=$spec_file" "project_name=$PROJECT_NAME"
        --printshellcmds
    )

    # Add dry-run flag if requested
    if [[ $is_dry_run == "true" ]]; then
        snakemake_cmd+=(--dryrun)
    fi

    # Add target rule
    snakemake_cmd+=("$TARGET_RULE")

    # Execute
    if "${snakemake_cmd[@]}"; then
        echo ""
        if [[ $is_dry_run == "true" ]]; then
            print_success "Dry-run complete!"
            print_info "To execute: sbatch $0 $mode"
        else
            print_success "Workflow submitted successfully!"
            print_info "Project name: $PROJECT_NAME"
            print_info "Monitor jobs: squeue -u \$USER"
            print_info "View logs: tail -f logs/trna_optimization_orchestrator-${SLURM_JOB_ID}.out"
            print_info "Results will be in: /scratch/alpine/\$USER/leech/$PROJECT_NAME/metrics/tRNA_optimization/"
        fi
        return 0
    else
        echo ""
        print_error "Workflow failed!"
        print_error "Check logs: logs/trna_optimization_orchestrator-${SLURM_JOB_ID}.err"
        return 1
    fi
}

# Main
main() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════════╗"
    echo "║      tRNA Optimization Full Analysis Orchestrator                     ║"
    echo "╚═══════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "Node: $(hostname)"
    echo "Start time: $(date)"
    echo ""

    # Default to dry-run if no argument (safe)
    MODE="${1:-dry-run}"

    if [[ $# -eq 0 ]]; then
        print_warning "No mode specified, defaulting to 'dry-run'"
        print_info "To submit jobs: sbatch $0 <mode>"
        show_usage
        echo ""
    fi

    # Determine spec file and dry-run flag
    local spec_file=""
    local is_dry_run="false"

    case "$MODE" in
        dry-run)
            spec_file="$CHARGED_SPEC"
            is_dry_run="true"
            ;;
        charged)
            spec_file="$CHARGED_SPEC"
            ;;
        pairwise)
            spec_file="$PAIRWISE_SPEC"
            ;;
        chemical-properties)
            spec_file="$CHEMICAL_SPEC"
            ;;
        help|--help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown mode: $MODE"
            show_usage
            exit 1
            ;;
    esac

    # Validate spec file exists
    if [[ ! -f "$spec_file" ]]; then
        print_error "Comparison spec file not found: $spec_file"
        exit 1
    fi

    # Run workflow
    if run_snakemake "$MODE" "$spec_file" "$is_dry_run"; then
        echo ""
        echo "End time: $(date)"
        exit 0
    else
        echo ""
        echo "End time: $(date)"
        exit 1
    fi
}

main "$@"
