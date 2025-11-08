#!/bin/bash
# Diagnostic script to debug SLURM QoS issue with Snakemake
# Tests various configurations to identify the problem

set -euo pipefail

echo "============================================"
echo "SLURM QoS Debug Script"
echo "============================================"
echo ""

# Step 1: Check Snakemake version
echo "1. Checking Snakemake version..."
snakemake --version || echo "Snakemake not in PATH"
echo ""

# Step 2: Check if we can submit test jobs with QoS
echo "2. Testing direct sbatch with QoS..."
cat > /tmp/test_qos_$$.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=test_qos
#SBATCH --partition=amilan
#SBATCH --account=amc-general
#SBATCH --qos=normal
#SBATCH --time=00:01:00
#SBATCH --mem=100M
#SBATCH --output=/dev/null

echo "QoS test successful"
EOF

sbatch --test-only /tmp/test_qos_$$.sh && echo "✓ Direct sbatch with --qos works" || echo "✗ Direct sbatch with --qos failed"
rm /tmp/test_qos_$$.sh
echo ""

# Step 3: Check SLURM environment
echo "3. Checking SLURM configuration..."
echo "Available QoS:"
sacctmgr show qos format=name -n | head -10
echo ""
echo "Account QoS for amc-general:"
sacctmgr show assoc where account=amc-general format=account,qos -n | head -5
echo ""

# Step 4: Test Snakemake with verbose output
echo "4. Testing Snakemake dry-run with verbose SLURM logging..."
cd /projects/jhesselberth@xsede.org/devel/rnabioco/leech

# Create minimal test Snakefile
cat > /tmp/test_snakefile_$$.smk << 'EOF'
rule test_qos:
    output: "/tmp/test_qos_output.txt"
    resources:
        slurm_partition="amilan",
        slurm_account="amc-general",
        slurm_qos="normal",
        runtime=1,
        mem_mb=100,
        cpus_per_task=1
    shell:
        "echo 'test' > {output}"
EOF

echo "Running minimal Snakefile with SLURM executor..."
snakemake --snakefile /tmp/test_snakefile_$$.smk \
    --executor slurm \
    --jobs 1 \
    --default-resources slurm_partition=amilan slurm_account=amc-general slurm_qos=normal \
    --dry-run \
    --printshellcmds \
    /tmp/test_qos_output.txt 2>&1 | tee /tmp/snakemake_debug_$$.log

echo ""
echo "5. Checking if --qos appears in sbatch command..."
if grep -q "qos" /tmp/snakemake_debug_$$.log; then
    echo "✓ Found QoS in Snakemake output:"
    grep -i "qos" /tmp/snakemake_debug_$$.log
else
    echo "✗ No QoS found in sbatch command - THIS IS THE PROBLEM"
fi

# Cleanup
rm -f /tmp/test_snakefile_$$.smk /tmp/snakemake_debug_$$.log

echo ""
echo "============================================"
echo "Debug complete"
echo "============================================"
