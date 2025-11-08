#!/bin/bash
# Simple verification that QoS fix is working
# Checks if --qos appears in the sbatch command

set -euo pipefail

cd /projects/jhesselberth@xsede.org/devel/rnabioco/leech

echo "============================================"
echo "Verifying QoS Fix"
echo "============================================"
echo ""
echo "This script verifies that --qos is passed to sbatch"
echo ""

# Unlock in case there's a stale lock
snakemake --unlock 2>/dev/null || true

# Run with verbose to capture sbatch command
# Use timeout and || true to handle potential failures gracefully
timeout 30 snakemake --profile cluster/slurm \
    --verbose \
    --jobs 1 \
    results/pod5/uncharged_synthetic/uncharged_synthetic.pod5 \
    2>&1 | head -200 > /tmp/qos_verify.txt || true

echo "Checking sbatch command..."
if grep "sbatch call:" /tmp/qos_verify.txt >/dev/null 2>&1; then
    echo ""
    echo "Found sbatch command:"
    grep "sbatch call:" /tmp/qos_verify.txt | head -1
    echo ""

    if grep "sbatch call:" /tmp/qos_verify.txt | grep -q -- "--qos=normal"; then
        echo "✓ SUCCESS: --qos=normal is present!"
        echo ""
        echo "The fix is working correctly."
    else
        echo "✗ FAILED: --qos flag not found"
        echo ""
        echo "The configuration may need adjustment."
    fi
else
    echo "Could not find sbatch command in output"
    echo "Check /tmp/qos_verify.txt for details"
fi

rm -f /tmp/qos_verify.txt

echo ""
echo "============================================"
