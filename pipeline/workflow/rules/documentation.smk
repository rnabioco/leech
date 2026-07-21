"""
Pipeline Run Documentation
==========================

Generates self-documenting summary files for each pipeline run.
"""


rule generate_run_summary:
    """Generate a run summary document with config, timestamp, and version info.

    This rule creates a self-documenting summary at the project root that includes:
    - Timestamp of the run
    - Git version (tag or commit)
    - Full configuration used for the run

    The summary is generated at the start of each pipeline run and provides
    a permanent record of the exact parameters and version used.
    """
    output:
        RUN_SUMMARY_FILE,
    run:
        from pathlib import Path
        import subprocess
        import datetime
        import yaml

        # Ensure output directory exists
        output_path = Path(output[0])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Get git version info
        git_version = "unknown"
        git_commit = "unknown"
        git_dirty = False
        try:
            # Try to get git tag
            result = subprocess.run(
                ["git", "describe", "--tags", "--exact-match"],
                capture_output=True,
                text=True,
                cwd=workflow.basedir,
                check=False,
            )
            if result.returncode == 0:
                git_version = result.stdout.strip()
            else:
                # No exact tag, get commit hash
                result = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=workflow.basedir,
                    check=True,
                )
                git_commit = result.stdout.strip()
                git_version = f"commit-{git_commit}"
                # Check if working directory is dirty
            result = subprocess.run(
                ["git", "diff", "--quiet"], cwd=workflow.basedir, check=False
            )
            git_dirty = result.returncode != 0
            if git_dirty:
                git_version += " (dirty)"
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Git not available or not a git repository
            pass
            # Write summary file
        with open(output[0], "w") as f:
            f.write("# Leech Pipeline Run Summary\n\n")
            f.write(f"**Generated:** {timestamp}\n\n")
            f.write(f"**Pipeline Version:** {git_version}\n\n")
            if git_commit != "unknown" and not git_dirty:
                f.write(f"**Git Commit:** {git_commit}\n\n")
            if git_dirty:
                f.write(
                    "**Warning:** Working directory has uncommitted changes\n\n"
                )
            f.write("---\n\n")
            f.write("## Configuration\n\n")
            f.write("This run was executed with the following configuration:\n\n")
            f.write("```yaml\n")
            # Create a clean config dict without sensitive/internal data
            clean_config = dict(config)
            # Format and write config as YAML
            yaml_str = yaml.dump(
                clean_config,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
            f.write(yaml_str)
            f.write("```\n\n")
            f.write("---\n\n")
            f.write("## Pipeline Outputs\n\n")
            f.write("The following directories contain pipeline outputs:\n\n")
            f.write(f"- **Chunks:** `{CHUNKS_DIR}`\n")
            f.write(f"- **Models:** `{MODELS_DIR}`\n")
            f.write(f"- **Inference:** `{INFER_DIR}`\n")
            f.write(f"- **Metrics:** `{METRICS_DIR}`\n\n")
            if config.get("project_name"):
                f.write(f"**Project Name:** {config['project_name']}\n\n")
            f.write("---\n\n")
            f.write("## Sample Information\n\n")
            f.write(f"**Number of samples:** {len(SAMPLES)}\n\n")
            f.write("**Samples:**\n\n")
            for sample in SAMPLES:
                sample_data = config["samples"][sample]
                label = sample_data.get("label", "unknown")
                f.write(f"- `{sample}` (label: {label})\n")
            f.write("\n---\n\n")
            f.write("## Comparison Specification\n\n")
            if AA_PAIRS:
                f.write(
                    f"**Comparison spec file:** `{config.get('comparison_spec_file')}`\n\n"
                )
                f.write(f"**Number of comparisons:** {len(AA_PAIRS)}\n\n")
                f.write("**Comparisons:**\n\n")
                for pair in AA_PAIRS:
                    if pair in AA_PAIR_TO_LABELS:
                        labels1, labels2 = AA_PAIR_TO_LABELS[pair]
                        f.write(
                            f"- `{pair}`: {', '.join(labels1)} vs {', '.join(labels2)}\n"
                        )
                    else:
                        f.write(f"- `{pair}`\n")
            else:
                f.write("No comparisons configured.\n")
            f.write("\n---\n\n")
            f.write("## Model Configuration\n\n")
            if COMPARE_MODELS:
                f.write("**Mode:** Multi-architecture comparison\n\n")
                f.write(f"**Architectures:** {', '.join(MODEL_ARCHITECTURES)}\n\n")
            else:
                f.write("**Mode:** Single model\n\n")
                f.write(
                    f"**Architecture:** {config.get('model', 'ConvLSTMDwell')}\n\n"
                )
            f.write(
                f"**Use grid search:** {config.get('use_grid_search', False)}\n\n"
            )
            if config.get("use_grid_search"):
                f.write("**Grid search parameters:**\n\n")
                f.write("```yaml\n")
                grid_config = config.get("grid_search", {})
                f.write(yaml.dump(grid_config, default_flow_style=False))
                f.write("```\n\n")
            else:
                f.write("**Training hyperparameters:**\n\n")
                f.write(f"- Epochs: {config.get('epochs',50)}\n")
                f.write(f"- Batch size: {config.get('batch_size',128)}\n")
                f.write(f"- Learning rate: {config.get('learning_rate',0.001)}\n")
                f.write(
                    f"- Early stopping patience: {config.get('early_stopping_patience',5)}\n"
                )
            f.write("\n---\n\n")
            f.write(
                f"*This summary was automatically generated by the Leech pipeline at {timestamp}*\n"
            )
