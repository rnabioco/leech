#!/bin/bash
# Test actual SLURM submission to see the sbatch command
# This will try to submit but fail with QoS error, showing us the command

set -euo pipefail

cd /projects/jhesselberth@xsede.org/devel/rnabioco/leech

echo "Testing actual Snakemake SLURM submission..."
echo "This will fail but show the sbatch command being used"
echo ""

# Run with verbosity to see sbatch command
snakemake --profile pipeline/cluster/slurm \
    --verbose \
    results/pod5/uncharged_synthetic/uncharged_synthetic.pod5 \
    2>&1 | grep -A 20 "sbatch" || true

echo ""
echo "Looking for QoS in the command..."
snakemake --profile pipeline/cluster/slurm \
    --verbose \
    results/pod5/uncharged_synthetic/uncharged_synthetic.pod5 \
    2>&1 | grep -i "qos" || echo "No QoS found in output"
