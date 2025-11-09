"""
POD5 rebasecalling rules using dorado.
"""

import os
from pathlib import Path

# Note: get_project_path() is defined in rules/common.smk which is included before this file


# ============================================================================
# Helper functions
# ============================================================================


def get_scratch_or_output_dir(base_dir, scratch_enabled=None):
    """Get scratch directory if enabled, otherwise return base directory."""
    if scratch_enabled is None:
        scratch_enabled = config.get("scratch_dir")

    if scratch_enabled and scratch_enabled not in ["", "null"]:
        # Expand environment variables in scratch path
        scratch_path = os.path.expandvars(scratch_enabled)
        return os.path.join(scratch_path, base_dir)
    return base_dir


def get_raw_pod5_inputs(wildcards):
    """Get raw POD5 inputs for a sample.

    Expects sample config to have either:
    - raw_pod5: single file path or directory containing POD5 files
    - pod5: single file path or directory containing POD5 files (alias for raw_pod5)
    - raw_pod5_list: list of POD5 file paths
    """
    sample_config = config["samples"][wildcards.sample]

    if "raw_pod5_list" in sample_config:
        # List of POD5 files provided
        return sample_config["raw_pod5_list"]
    elif "raw_pod5" in sample_config:
        # Single file or directory
        raw_pod5 = sample_config["raw_pod5"]
        if os.path.isdir(raw_pod5):
            # Directory: find all POD5 files
            from pathlib import Path
            return list(Path(raw_pod5).rglob("*.pod5"))
        else:
            # Single file
            return [raw_pod5]
    elif "pod5" in sample_config:
        # Support 'pod5' as an alias for 'raw_pod5'
        raw_pod5 = sample_config["pod5"]
        if os.path.isdir(raw_pod5):
            # Directory: find all POD5 files
            from pathlib import Path
            return list(Path(raw_pod5).rglob("*.pod5"))
        else:
            # Single file
            return [raw_pod5]
    else:
        raise ValueError(
            f"Sample {wildcards.sample} must have 'raw_pod5', 'pod5', or 'raw_pod5_list' in config"
        )


# ============================================================================
# Rules
# ============================================================================


rule merge_pods:
    """
    Merge raw POD5 files.

    The pod5 merge command automatically handles v3→v4 migration,
    so no separate update step is needed.
    """
    input:
        get_raw_pod5_inputs
    output:
        pod5=get_project_path(config.get("pod5_dir", "results/pod5")) + "/{sample}/{sample}.pod5"
    threads: config.get("merge_pods_threads", 12)
    resources:
        mem_mb=config.get("merge_pods_mem", 16000),
        runtime=config.get("merge_pods_time", 120)
    log:
        get_project_path(config.get("pod5_dir", "results/pod5")) + "/{sample}/merge_pods.log"
    shell:
        """
        pod5 merge -t {threads} -f -o {output.pod5} {input} &> {log}
        rm -rf .tmp_pod5_v3_v4_migration_*
        """


rule rebasecall:
    """
    Rebasecall POD5 files using dorado basecaller.

    Outputs BAM file with basecalls and move tables (mv tag) needed for leech feature extraction.
    """
    input:
        rules.merge_pods.output.pod5
    output:
        bam=protected(get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall")) + "/{sample}/{sample}.rbc.bam")
    params:
        dorado_bin=config.get("dorado_bin", "dorado"),
        model=config.get("base_calling_model", "rna004_130bps_sup@v5.2.0"),
        modifications=lambda wildcards: (
            config.get("modifications", "").replace(",", " ")
            if config.get("modifications", "")
            else ""
        ),
        dorado_opts=config.get("dorado_opts", "--emit-moves")
    threads: config.get("rebasecall_threads", 4)
    resources:
        mem_mb=config.get("rebasecall_mem", 16000),
        runtime=config.get("rebasecall_time", 480),
        gpu=0,  # GPU controlled by cluster profile
    log:
        get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall")) + "/{sample}/rebasecall.log"
    shell:
        """
        if [[ "${{CUDA_VISIBLE_DEVICES:-}}" ]]; then
            echo "CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES"
            export CUDA_VISIBLE_DEVICES
        fi

        if [[ -n "{params.modifications}" ]]; then
            {params.dorado_bin} basecaller {params.model} {input} {params.dorado_opts} --modified-bases {params.modifications} > {output.bam} 2> {log}
        else
            {params.dorado_bin} basecaller {params.model} {input} {params.dorado_opts} > {output.bam} 2> {log}
        fi
        """


rule align_rebasecalled:
    """
    Align rebasecalled reads to reference using minimap2.

    CRITICAL: This step preserves all tags from the rebasecalled BAM:
    - mv: Move table (required for leech dwell time features)
    - ns: Number of samples per base
    - MM: Modified base positions/types (if modification calling enabled)
    - ML: Modified base probabilities (if modification calling enabled)

    The -T '*' flag in samtools fastq preserves all auxiliary tags.
    The -y flag in minimap2 copies tags from input to aligned output.
    """
    input:
        bam=rules.rebasecall.output.bam,
        reference=config.get("reference", "references/reference.fasta")
    output:
        bam=get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall")) + "/{sample}/{sample}.aligned.bam",
        bai=get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall")) + "/{sample}/{sample}.aligned.bam.bai"
    params:
        samtools_bin=config.get("samtools_bin", "samtools"),
        minimap2_bin=config.get("minimap2_bin", "minimap2")
    threads: config.get("align_threads", 8)
    resources:
        mem_mb=config.get("align_mem", 16000),
        runtime=config.get("align_time", 240)
    log:
        get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall")) + "/{sample}/align.log"
    shell:
        """
        # Convert BAM to FASTQ preserving all tags (-T '*')
        # Align with minimap2 copying tags to output (-y)
        # Sort and index the aligned BAM
        {params.samtools_bin} fastq -T '*' {input.bam} | \
        {params.minimap2_bin} -ax map-ont -y {input.reference} - | \
        {params.samtools_bin} sort -@ {threads} -o {output.bam} - 2> {log}

        # Index the aligned BAM
        {params.samtools_bin} index {output.bam}

        # Verify critical tags are present
        echo "Verifying critical tags (mv, ns, MM, ML) in aligned BAM..." >> {log}
        {params.samtools_bin} view {output.bam} | head -n 1 | grep -o "mv:B:[^[:space:]]*" >> {log} || echo "WARNING: mv tag not found" >> {log}
        {params.samtools_bin} view {output.bam} | head -n 1 | grep -o "ns:i:[^[:space:]]*" >> {log} || echo "WARNING: ns tag not found" >> {log}
        """
