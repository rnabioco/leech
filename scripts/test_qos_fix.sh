#!/bin/bash
# Test script to verify QoS fix
# This does a dry-run and checks if --qos appears in the output

set -euo pipefail

cd /projects/jhesselberth@xsede.org/devel/rnabioco/leech

echo "========================================"
echo "Testing QoS Fix"
echo "========================================"
echo ""

echo "Running dry-run with new config..."
snakemake --profile cluster/slurm \
    --dry-run \
    --verbose \
    results/pod5/uncharged_synthetic/uncharged_synthetic.pod5 \
    2>&1 | tee /tmp/qos_test_output.txt

echo ""
echo "========================================"
echo "Checking for --qos in output..."
echo "========================================"

if grep -q "slurm-qos" /tmp/qos_test_output.txt; then
    echo "✓ Found slurm-qos parameter in Snakemake output"
    grep "slurm-qos" /tmp/qos_test_output.txt | head -5
else
    echo "✗ slurm-qos not found in output"
fi

echo ""
echo "Checking Snakemake configuration loaded:"
grep -A 5 "slurm-qos\|executor.*slurm" /tmp/qos_test_output.txt | head -10

rm -f /tmp/qos_test_output.txt

echo ""
echo "========================================"
echo "Test complete!"
echo "The fix should now pass --qos=normal to sbatch"
echo "========================================"
