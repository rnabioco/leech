"""
Signal orientation diagnostic rules.

Diagnoses potential signal reversal issues in RNA sequencing data
by checking move table monotonicity and signal-sequence alignment.
"""


rule diagnose_signal_orientation:
    """
    Run signal orientation diagnostics on a sample.

    This standalone rule checks if signal and sequence are properly aligned
    for RNA sequencing data. It performs three tests:
    1. Move table monotonicity (should increase, not decrease)
    2. Visual signal-sequence alignment at motifs
    3. Base-signal correlation patterns

    Outputs:
    - PNG plot showing signal traces with base annotations
    - Log file with diagnostic results and interpretation

    Usage:
        # Run on single sample
        snakemake --configfile config/samples-alpine.yaml \
            results/diagnostics/ala_synthetic/signal_orientation.png

        # Run on all AA synthetic samples
        snakemake --configfile config/samples-alpine.yaml all_diagnose
    """
    input:
        pod5=get_project_path(config.get("pod5_dir", "results/pod5"))
        + "/{sample}/{sample}.pod5",
        bam=get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall"))
        + "/{sample}/{sample}.aligned.bam",
    output:
        plot=get_project_path(config.get("metrics_dir", "results/metrics"))
        + "/diagnostics/{sample}/signal_orientation.png",
    wildcard_constraints:
        sample="[^/]+",  # Sample name cannot contain slashes
    params:
        motif=config.get("motif", "CCATGGC"),
        output_dir=get_project_path(config.get("metrics_dir", "results/metrics"))
        + "/diagnostics/{sample}",
    log:
        get_project_path(config.get("metrics_dir", "results/metrics"))
        + "/diagnostics/{sample}/signal_orientation.log",
    conda:
        "../envs/leech.yaml"
    shell:
        """
        # Create output directory
        mkdir -p {params.output_dir}

        # Run diagnostic script
        uv run python scripts/diagnose_signal_orientation.py \
            --bam {input.bam} \
            --pod5 {input.pod5} \
            --motif {params.motif} \
            --output {output.plot} \
            > {log} 2>&1
        """
