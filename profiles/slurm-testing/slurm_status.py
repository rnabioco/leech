#!/usr/bin/env python3
"""
Check the status of a SLURM job.

This script is called by Snakemake to check job status.
It should print one of: running, failed, success

Usage: slurm_status.py <job_id>
"""

import subprocess
import sys
import time


def get_job_status(job_id):
    """
    Query SLURM for job status using sacct.

    Returns:
        str: 'running', 'success', or 'failed'
    """
    # Try sacct first (works for completed jobs)
    try:
        result = subprocess.run(
            ["sacct", "-j", str(job_id), "-n", "-o", "State", "-P"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            status = result.stdout.strip().split("\n")[0]
            return parse_slurm_status(status)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Try squeue for running jobs
    try:
        result = subprocess.run(
            ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            status = result.stdout.strip()
            return parse_slurm_status(status)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # If we can't find the job, assume it failed
    return "failed"


def parse_slurm_status(status):
    """
    Convert SLURM status to Snakemake status.

    Args:
        status: SLURM status string

    Returns:
        str: 'running', 'success', or 'failed'
    """
    status = status.upper()

    # Running states
    if status in ["PENDING", "CONFIGURING", "RUNNING", "COMPLETING"]:
        return "running"

    # Success states
    if status in ["COMPLETED"]:
        return "success"

    # Failed states
    # FAILED, TIMEOUT, CANCELLED, NODE_FAIL, PREEMPTED, OUT_OF_MEMORY, etc.
    return "failed"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: slurm_status.py <job_id>", file=sys.stderr)
        sys.exit(1)

    job_id = sys.argv[1]

    # Retry a few times with exponential backoff
    # Sometimes there's a delay in job status propagation
    max_retries = 3
    for attempt in range(max_retries):
        status = get_job_status(job_id)

        # If we got a definitive status (not failed due to query issues), return it
        if status in ["running", "success"]:
            print(status)
            sys.exit(0)

        # If this isn't our last attempt, wait before retrying
        if attempt < max_retries - 1:
            time.sleep(2**attempt)  # 1s, 2s, 4s

    # After all retries, return the last status
    print(status)
    sys.exit(0)
