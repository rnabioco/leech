# Scripts Directory

This directory contains test and diagnostic scripts for the Leech pipeline.

## QoS Debugging Scripts

These scripts were created to diagnose and verify the SLURM QoS configuration issue:

### Quick Checks

**`quick_qos_check.sh`** - Fast configuration verification
```bash
bash scripts/quick_qos_check.sh
```
- Verifies `slurm-qos` parameter is correctly set
- Checks Snakemake configuration loads
- Does NOT submit any jobs
- Run this first to verify the fix

### Diagnostic Scripts

**`debug_slurm_qos.sh`** - Comprehensive diagnostic
```bash
bash scripts/debug_slurm_qos.sh
```
- Checks Snakemake version
- Tests direct sbatch with QoS
- Shows available QoS options
- Creates and tests minimal Snakefile

**`test_qos_simple.py`** - Python-based test
```bash
uv run python scripts/test_qos_simple.py
```
- Creates temporary test Snakefile
- Analyzes Snakemake dry-run output
- Checks if QoS mapping works

**`verify_qos_fix.sh`** - Verify the fix works
```bash
bash scripts/verify_qos_fix.sh
```
- Runs Snakemake with verbose output
- Captures sbatch command
- Confirms `--qos=normal` is present

**`test_real_submit.sh`** - Test actual submission
```bash
bash scripts/test_real_submit.sh 2>&1 | grep "sbatch call"
```
- Attempts real SLURM submission
- Shows the exact sbatch command generated
- Demonstrates QoS flag is included

## Pipeline Test Scripts

**`test_merge_pods.sh`** - Test POD5 merging
```bash
sbatch scripts/test_merge_pods.sh
```
- Orchestrator job that submits merge_pods via Snakemake
- Tests the full SLURM executor integration
- Output: `results/pod5/uncharged_synthetic/uncharged_synthetic.pod5`

## Usage

### First-time verification after QoS fix
```bash
# 1. Quick check
bash scripts/quick_qos_check.sh

# 2. See the sbatch command (optional)
timeout 30 bash scripts/test_real_submit.sh 2>&1 | grep -A 1 "sbatch call" | head -5

# 3. Run actual pipeline
snakemake --profile cluster/slurm --jobs 1 \
  results/pod5/uncharged_synthetic/uncharged_synthetic.pod5
```

### If issues persist
```bash
# Full diagnostic
bash scripts/debug_slurm_qos.sh

# Python test
uv run python scripts/test_qos_simple.py

# Check Snakemake logs
tail -100 logs/slurm/rule_merge_pods/uncharged_synthetic/*.log
```

## What Was Fixed

The SLURM QoS configuration was updated in `cluster/slurm/config.yaml`:

**Before (incorrect):**
```yaml
default-resources:
  slurm_qos: "normal"  # Wrong - underscore, treated as resource
```

**After (correct):**
```yaml
slurm-qos: "normal"  # Correct - hyphen, top-level plugin parameter
```

This ensures all sbatch commands include `--qos=normal`, which is now required by Alpine cluster.

## Documentation

- **TESTING.md** - Pipeline testing guide with QoS troubleshooting
- **docs/troubleshooting.md** - Comprehensive troubleshooting guide
- **docs/qos-fix-summary.md** - Detailed explanation of the QoS fix

## Notes

- All diagnostic scripts are safe to run - they don't modify data
- Scripts use timeouts to prevent hanging
- Snakemake lock files are cleaned up automatically
- The fix applies to all pipeline rules automatically
