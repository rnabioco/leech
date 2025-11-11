#!/usr/bin/env bash
#
# Example script for running model comparisons with the NEW TSV-based workflow
#
# IMPORTANT: The pairwise Snakemake rules have been removed. Use the TSV-based
# comparison spec approach instead:
#
# 1. Create/use a comparison spec TSV file (see config/*.tsv)
# 2. Run: leech merge-and-split --comparison-spec <file.tsv>
# 3. Train models on each comparison directory
#
# This script demonstrates the new workflow for charged vs uncharged only.
# For pairwise/multi-label comparisons, use the TSV spec approach directly.

set -euo pipefail

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

${GREEN}Leech Model Comparison - NEW TSV-Based Workflow${NC}

${YELLOW}IMPORTANT:${NC} Pairwise Snakemake rules have been removed.
Use the comparison spec TSV approach instead.

${GREEN}For Charged vs Uncharged (Snakemake):${NC}
  $0 charged      # Train 6 model architectures on charged vs uncharged
  $0 dry-run      # Preview what would run

${GREEN}For Pairwise/Multi-Label Comparisons (TSV Spec):${NC}

  ${YELLOW}Step 1:${NC} Prepare all comparisons from spec file
    uv run leech merge-and-split \\
      --input-chunks results/chunks/*_synthetic/*.npz \\
      --comparison-spec config/comparisons_chemical_properties.tsv \\
      --output-dir results/chunks/merged \\
      --seed 42

  ${YELLOW}Step 2:${NC} Train models on each comparison
    for comp_dir in results/chunks/merged/*/; do
      comp_name=\$(basename "\$comp_dir")
      echo "Training: \$comp_name"

      uv run leech train \\
        --train-data "\$comp_dir/train.npz" \\
        --val-data "\$comp_dir/val.npz" \\
        --model ConvLSTMDwell \\
        --output-dir "results/models/\$comp_name" \\
        --epochs 50 \\
        --batch-size 128
    done

${GREEN}Available Comparison Specs:${NC}
  config/comparisons_all_pairwise.tsv          # All 190 AA pairs
  config/comparisons_chemical_properties.tsv    # Chemical property groups (basic vs acidic, etc.)

${GREEN}Output Locations:${NC}
  Comparisons: results/chunks/merged/<comparison_name>/
  Models:      results/models/<comparison_name>/
  Metadata:    results/chunks/merged/<comparison_name>/metadata.json

${GREEN}Monitoring:${NC}
  Comparisons: ls -l results/chunks/merged/*/metadata.json
  Models:      ls -l results/models/*/model_best.pt

EOF
}

# Run charged vs uncharged comparison (Snakemake)
run_charged() {
    local mode=$1

    if [[ "$mode" == "dry-run" ]]; then
        print_info "Running dry-run for charged vs uncharged..."
        snakemake \
            --profile "$PROFILE" \
            --configfile "$SAMPLES_CONFIG" \
            --dryrun \
            --printshellcmds \
            compare_architectures_charged
        print_success "Dry-run complete."
        exit 0
    fi

    print_info "Starting training: Charged vs Uncharged (6 jobs)"
    print_warning "This will submit jobs to the cluster!"

    snakemake \
        --profile "$PROFILE" \
        --configfile "$SAMPLES_CONFIG" \
        --printshellcmds \
        compare_architectures_charged

    if [[ $? -eq 0 ]]; then
        print_success "Pipeline submitted successfully!"
        print_info "Monitor with: squeue -u \$USER"
    else
        print_error "Pipeline failed."
        exit 1
    fi
}

# Main
main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     Leech Model Comparison (TSV-Based Workflow)           ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    MODE="${1:-}"

    if [[ -z "$MODE" ]]; then
        show_usage
        exit 0
    fi

    case "$MODE" in
        charged|dry-run)
            run_charged "$MODE"
            ;;
        *)
            print_error "Unknown mode: $MODE"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
