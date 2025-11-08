#!/usr/bin/env python3
"""
Simple test to check if Snakemake SLURM executor properly handles QoS.
This script tests the resource mapping from slurm_qos to --qos flag.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

def check_snakemake_version():
    """Check Snakemake version."""
    result = subprocess.run(
        ["snakemake", "--version"],
        capture_output=True,
        text=True
    )
    version = result.stdout.strip()
    print(f"Snakemake version: {version}")

    # Parse version
    try:
        major = int(version.split('.')[0])
        if major < 8:
            print(f"⚠️  Warning: Snakemake {version} may have different SLURM resource handling")
        else:
            print(f"✓ Snakemake {version} uses new executor API")
    except:
        print(f"Could not parse version: {version}")

    return version

def create_test_snakefile(path):
    """Create a minimal test Snakefile."""
    content = """
rule test_qos:
    output: touch("test_output.txt")
    resources:
        slurm_partition="amilan",
        slurm_account="amc-general",
        slurm_qos="normal",
        runtime=1,
        mem_mb=100
    shell:
        "echo 'test'"
"""
    path.write_text(content)

def test_dry_run(snakefile_path):
    """Run Snakemake dry-run and capture output."""
    print("\nRunning Snakemake dry-run...")

    cmd = [
        "snakemake",
        "--snakefile", str(snakefile_path),
        "--executor", "slurm",
        "--jobs", "1",
        "--dry-run",
        "--printshellcmds",
        "test_output.txt"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=snakefile_path.parent
    )

    return result.stdout + result.stderr

def analyze_output(output):
    """Analyze Snakemake output for QoS handling."""
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    # Check for sbatch command
    if "sbatch" in output:
        print("✓ Found sbatch command")

        # Extract sbatch line
        for line in output.split('\n'):
            if 'sbatch' in line.lower():
                print(f"\nsbatch command:\n{line}")

                # Check for --qos or -q
                if '--qos' in line or '-q ' in line:
                    print("\n✓ QoS flag IS present in sbatch command")
                    return True
                else:
                    print("\n✗ QoS flag is MISSING from sbatch command")
                    print("  This is the problem!")
                    return False
    else:
        print("✗ No sbatch command found in output")
        return False

def main():
    print("="*60)
    print("Snakemake SLURM QoS Test")
    print("="*60)
    print()

    # Check version
    check_snakemake_version()

    # Create temporary test
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        snakefile = tmppath / "Snakefile"
        create_test_snakefile(snakefile)

        # Run test
        output = test_dry_run(snakefile)

        # Analyze
        qos_present = analyze_output(output)

        # Show full output for debugging
        print("\n" + "="*60)
        print("FULL OUTPUT")
        print("="*60)
        print(output)

        return 0 if qos_present else 1

if __name__ == "__main__":
    sys.exit(main())
