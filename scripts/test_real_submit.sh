#!/bin/bash
# Test real submission to see actual sbatch command
# Will catch and display the error to see if --qos is included

set -euo pipefail

cd /projects/jhesselberth@xsede.org/devel/rnabioco/leech

echo "========================================"
echo "Testing Real SLURM Submission"
echo "========================================"
echo ""

# Make sure we capture both stdout and stderr
echo "Attempting to run merge_pods..."
echo "This should submit to SLURM and we can check the sbatch command"
echo ""

snakemake --profile pipeline/cluster/slurm \
    --verbose \
    --jobs 1 \
    results/pod5/uncharged_synthetic/uncharged_synthetic.pod5 \
    2>&1 | tee /tmp/real_submit_output.txt || true

echo ""
echo "========================================"
echo "Checking output for sbatch command..."
echo "========================================"

if grep -q "sbatch" /tmp/real_submit_output.txt; then
    echo "Found sbatch command:"
    echo ""
    grep "sbatch" /tmp/real_submit_output.txt
    echo ""

    if grep "sbatch" /tmp/real_submit_output.txt | grep -q -- "--qos\|-q "; then
        echo "✓ SUCCESS: --qos flag IS present in sbatch command!"
    else
        echo "✗ FAILED: --qos flag is MISSING from sbatch command"
    fi
else
    echo "No sbatch command found"
    echo "Showing error output:"
    tail -30 /tmp/real_submit_output.txt
fi

rm -f /tmp/real_submit_output.txt
