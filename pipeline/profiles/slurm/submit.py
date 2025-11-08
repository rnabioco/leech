#!/usr/bin/env python3
"""
SLURM job submission wrapper for Alpine cluster.

This script generates proper SBATCH directives for the Alpine cluster,
including conditional GPU allocation with --gres.

Usage: submit.py <jobscript>
"""

import os
import sys
import subprocess
from pathlib import Path


def submit_job(jobscript):
    """Submit a job to SLURM with proper Alpine directives."""

    # Read the jobscript
    with open(jobscript) as f:
        script_content = f.read()

    # Extract SBATCH directives from the script
    lines = script_content.split('\n')
    sbatch_lines = [line for line in lines if line.strip().startswith('#SBATCH')]

    # Extract resource values from the script
    # Parse values from #SBATCH directives
    resources = {}
    for line in sbatch_lines:
        if '--partition=' in line:
            resources['partition'] = line.split('=')[1].strip()
        elif '--qos=' in line:
            resources['qos'] = line.split('=')[1].strip()

    # Check if this is a GPU job based on partition
    gpu_partitions = ['aa100', 'ami100', 'atesting_a100', 'atesting_mi100']
    is_gpu_job = resources.get('partition', '') in gpu_partitions

    # Build sbatch command
    sbatch_cmd = ['sbatch']

    # Add GPU directive if needed
    if is_gpu_job:
        # Extract GPU count from environment or default to 1
        gpu_count = os.environ.get('SNAKEMAKE_GPU', '1')
        sbatch_cmd.extend(['--gres', f'gpu:{gpu_count}'])

    # Add account if specified
    account = os.environ.get('SLURM_ACCOUNT', '')
    if account:
        sbatch_cmd.extend(['--account', account])

    # Add the jobscript
    sbatch_cmd.append(jobscript)

    # Submit the job
    try:
        result = subprocess.run(
            sbatch_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout, end='')
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return 1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: submit.py <jobscript>", file=sys.stderr)
        sys.exit(1)

    jobscript = sys.argv[1]
    sys.exit(submit_job(jobscript))
