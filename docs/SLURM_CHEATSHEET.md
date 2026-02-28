# SLURM Job Monitoring Cheatsheet

## Submit Jobs
```bash title="Bash" linenums="1"
sbatch test_merge_pods.sh              # Submit single job
sbatch --array=1-10 script.sh          # Submit job array
```

## Monitor Queue (Real-time)
```bash title="Bash" linenums="1"
watch -n 2 'squeue -u $USER'           # Auto-refresh every 2 seconds (Ctrl+C to exit)
watch -n 5 'squeue -u $USER'           # Auto-refresh every 5 seconds
squeue -u $USER                        # One-time check
```

## Monitor Specific Job
```bash title="Bash" linenums="1"
squeue -j <JOB_ID>                     # Check specific job
scontrol show job <JOB_ID>             # Detailed job info
watch -n 2 "squeue -j <JOB_ID>"        # Auto-refresh specific job
```

## Monitor Log Files
```bash title="Bash" linenums="1"
tail -f logs/test_merge_*.out          # Follow latest log file
tail -f logs/*.out                     # Follow all logs
tail -n 100 logs/test_merge_12345.out  # View last 100 lines
```

## Job History
```bash title="Bash" linenums="1"
sacct -u $USER --starttime=today       # Jobs from today
sacct -j <JOB_ID>                      # Specific job history
sacct -u $USER --starttime=now-1day --format=JobID,JobName,State,Elapsed,ExitCode
```

## Job Control
```bash title="Bash" linenums="1"
scancel <JOB_ID>                       # Cancel specific job
scancel -u $USER                       # Cancel all your jobs
scancel -n <JOB_NAME>                  # Cancel jobs by name
scontrol hold <JOB_ID>                 # Hold job (prevent from running)
scontrol release <JOB_ID>              # Release held job
```

## Job States
- **PD** (Pending): Job waiting in queue
- **R** (Running): Job is running
- **CG** (Completing): Job is finishing
- **CD** (Completed): Job finished successfully
- **F** (Failed): Job failed
- **CA** (Cancelled): Job was cancelled
- **TO** (Timeout): Job exceeded time limit
- **OOM** (Out of Memory): Job ran out of memory

## Pending Reasons
```bash title="Bash" linenums="1"
squeue -u $USER -t PD -o "%.18i %.9P %.50j %.8u %.10r"
```
Common reasons:
- **Resources**: Waiting for resources to become available
- **Priority**: Other jobs have higher priority
- **QOSMaxCpuPerUserLimit**: You've hit CPU limit for your QOS
- **AssocGrpCPULimit**: Account/allocation CPU limit reached
- **ReqNodeNotAvail**: Requested node not available

## Cluster Info
```bash title="Bash" linenums="1"
sinfo                                  # Partition/node status
sinfo -o "%20P %10a %10l %10c %10G %10m %N"  # Detailed partition info
sinfo -p aa100                         # GPU partition info
```

## Resource Usage
```bash title="Bash" linenums="1"
# Check job efficiency after completion
seff <JOB_ID>

# Resource usage during run
sstat -j <JOB_ID> --format=JobID,MaxRSS,AveCPU,AvePages

# Detailed accounting
sacct -j <JOB_ID> --format=JobID,MaxRSS,Elapsed,CPUTime,TotalCPU,ReqMem,MaxVMSize
```

## Useful One-Liners
```bash title="Bash" linenums="1"
# Count running jobs
squeue -u $USER -t R | wc -l

# Count pending jobs
squeue -u $USER -t PD | wc -l

# Show only GPU jobs
squeue -u $USER -p aa100,ami100

# Monitor and grep for errors
tail -f logs/*.out | grep -i error

# Watch Snakemake progress
tail -f logs/controller_*.out | grep -E "rule|Finished|Error"
```

## Snakemake-Specific Monitoring
```bash title="Bash" linenums="1"
# Watch Snakemake controller log
tail -f logs/controller_*.out

# See all jobs submitted by Snakemake
squeue -u $USER -o "%.18i %.9P %.30j %.8T %.10M %.6D"

# Count jobs by state
squeue -u $USER | awk '{print $5}' | sort | uniq -c
```

## Helpful Aliases (add to ~/.bashrc)
```bash title="Bash" linenums="1"
alias sq='squeue -u $USER'
alias sqw='watch -n 2 "squeue -u $USER"'
alias sqall='squeue -u $USER -o "%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R"'
alias sa='sacct -u $USER --starttime=today --format=JobID,JobName,Partition,State,Elapsed,ExitCode'
alias scl='scancel'
alias seff='seff'  # Job efficiency
alias logs='tail -f logs/*.out'
```

## GPU-Specific Commands
```bash title="Bash" linenums="1"
# Check GPU partition availability
sinfo -p aa100,ami100 -o "%20P %10a %10l %10c %10G %10m %N"

# Monitor GPU jobs
squeue -u $USER -p aa100,ami100

# SSH to running job node and check GPU (if allowed)
squeue -u $USER -j <JOB_ID> -h -o %N  # Get node name
ssh <NODE_NAME> nvidia-smi            # Check GPU usage (if SSH allowed)
```

## Emergency Actions
```bash title="Bash" linenums="1"
# Cancel all your jobs immediately
scancel -u $USER

# Hold all pending jobs (pause queue)
squeue -u $USER -t PD -h -o %i | xargs -n1 scontrol hold

# Release all held jobs
squeue -u $USER -t H -h -o %i | xargs -n1 scontrol release
```

## Troubleshooting Failed Jobs
```bash title="Bash" linenums="1"
# Find failed jobs from today
sacct -u $USER --starttime=today --state=FAILED

# Get exit code of failed job
sacct -j <JOB_ID> --format=JobID,State,ExitCode

# View error log
cat logs/test_merge_<JOB_ID>.err

# Check why job failed
scontrol show job <JOB_ID> | grep -i reason
```
