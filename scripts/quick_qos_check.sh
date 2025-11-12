#!/bin/bash
# Quick check to verify QoS configuration is working
# This does NOT submit jobs, just verifies configuration

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=================================================="
echo "Quick QoS Configuration Check"
echo "=================================================="
echo ""

# Check 1: Verify config file has correct parameter
echo "1. Checking cluster/slurm/config.yaml..."
if grep -q "^slurm-qos:" cluster/slurm/config.yaml; then
    echo "   ✓ Found 'slurm-qos' (hyphen) - CORRECT"
    grep "^slurm-qos:" cluster/slurm/config.yaml
else
    echo "   ✗ Missing 'slurm-qos' parameter"
    exit 1
fi

# Check 2: Make sure old incorrect format isn't there
echo ""
echo "2. Checking for incorrect 'slurm_qos' (underscore) in resources..."
if grep -q "slurm_qos:" cluster/slurm/config.yaml; then
    echo "   ⚠ WARNING: Found 'slurm_qos' (underscore) - this should be removed"
    grep "slurm_qos:" cluster/slurm/config.yaml
else
    echo "   ✓ No incorrect 'slurm_qos' found - GOOD"
fi

# Check 3: Verify Snakemake version
echo ""
echo "3. Checking Snakemake version..."
SNAKE_VERSION=$(snakemake --version 2>/dev/null || echo "not found")
if [ "$SNAKE_VERSION" != "not found" ]; then
    echo "   ✓ Snakemake version: $SNAKE_VERSION"
else
    echo "   ✗ Snakemake not found in PATH"
    exit 1
fi

# Check 4: Dry run to see if config loads correctly
echo ""
echo "4. Testing Snakemake dry-run (checks if config loads)..."
if snakemake --profile pipeline/cluster/slurm --dry-run \
    results/pod5/uncharged_synthetic/uncharged_synthetic.pod5 \
    >/dev/null 2>&1; then
    echo "   ✓ Dry-run successful - config loads correctly"
else
    echo "   ✗ Dry-run failed - check configuration"
    exit 1
fi

echo ""
echo "=================================================="
echo "✓ All checks passed!"
echo "=================================================="
echo ""
echo "The QoS configuration is correct."
echo ""
echo "To actually test job submission (will fail fast if QoS still missing):"
echo "  snakemake --profile pipeline/cluster/slurm --jobs 1 \\"
echo "    results/pod5/uncharged_synthetic/uncharged_synthetic.pod5"
echo ""
echo "To see the sbatch command with --qos flag:"
echo "  bash scripts/test_real_submit.sh 2>&1 | grep 'sbatch call:'"
echo ""
