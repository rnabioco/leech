#!/usr/bin/env python3
"""
Check the status of an LSF job.

This script is called by Snakemake to check job status.
It should print one of: running, failed, success

Usage: lsf_status.py <job_id>
"""

import subprocess
import sys
import time


def get_job_status(job_id):
    """
    Query LSF for job status using bjobs.

    Returns:
        str: 'running', 'success', or 'failed'
    """
    # Try bjobs for current/recent jobs
    try:
        result = subprocess.run(
            ["bjobs", "-noheader", "-o", "stat", str(job_id)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            status = result.stdout.strip()
            return parse_lsf_status(status)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Try bjobs -a for finished jobs (includes history)
    try:
        result = subprocess.run(
            ["bjobs", "-a", "-noheader", "-o", "stat", str(job_id)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            status = result.stdout.strip()
            return parse_lsf_status(status)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # If we can't find the job, assume it completed successfully
    # (LSF purges completed jobs from bjobs after a while)
    return "success"


def parse_lsf_status(status):
    """
    Convert LSF status to Snakemake status.

    LSF statuses:
    - PEND: pending
    - RUN: running
    - DONE: completed successfully
    - EXIT: exited with error
    - ZOMBI: zombie (usually means done)
    - UNKWN: unknown
    - PSUSP: pending, user suspended
    - USUSP: user suspended
    - SSUSP: system suspended

    Args:
        status: LSF status string

    Returns:
        str: 'running', 'success', or 'failed'
    """
    status = status.upper()

    # Running/pending states
    if status in ["PEND", "RUN", "PSUSP", "USUSP", "SSUSP"]:
        return "running"

    # Success states
    if status in ["DONE", "ZOMBI"]:
        return "success"

    # Failed states (EXIT, UNKWN, etc.)
    return "failed"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: lsf_status.py <job_id>", file=sys.stderr)
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
