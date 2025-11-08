"""
POD5 rebasecalling rules using dorado.
"""

import os


# ============================================================================
# Helper functions
# ============================================================================


def get_raw_pod5_inputs(wildcards):
    """Get raw POD5 inputs for a sample.

    Expects sample config to have either:
    - raw_pod5: single file path or directory containing POD5 files
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
    else:
        raise ValueError(
            f"Sample {wildcards.sample} must have 'raw_pod5' or 'raw_pod5_list' in config"
        )


# ============================================================================
# Rules
# ============================================================================


rule merge_pods:
    """
    Merge multiple POD5 files into a single file.

    This is useful for consolidating raw POD5 files before basecalling,
    which can improve efficiency and organization.
    """
    input:
        get_raw_pod5_inputs
    output:
        pod5=config.get("pod5_dir", "results/pod5") + "/{sample}/{sample}.pod5"
    threads: config.get("merge_pods_threads", 12)
    resources:
        mem_mb=config.get("merge_pods_mem", 8000),
        runtime=config.get("merge_pods_time", 120)
    log:
        config.get("pod5_dir", "results/pod5") + "/{sample}/merge_pods.log"
    shell:
        """
        pod5 merge -t {threads} -f -o {output.pod5} {input} 2>&1 | tee {log}
        """


rule rebasecall:
    """
    Rebasecall POD5 files using dorado basecaller.

    Requires:
    - Merged POD5 file (from merge_pods rule)
    - Dorado model (specified in config)

    Outputs:
    - BAM file with basecalls and move tables (mv tag)

    The output BAM will have the required mv (move table) and ns (num samples)
    tags needed for leech feature extraction.
    """
    input:
        pod5=rules.merge_pods.output.pod5
    output:
        bam=protected(config.get("rebasecall_dir", "results/bam/rebasecall") + "/{sample}/{sample}.rbc.bam")
    params:
        dorado_bin=config.get("dorado_bin", "dorado"),
        model=config.get("base_calling_model", "dna_r10.4.1_e8.2_400bps_hac@v4.2.0"),
        dorado_opts=config.get("dorado_opts", "--emit-moves"),
        cuda_devices=lambda wildcards, resources: os.environ.get("CUDA_VISIBLE_DEVICES", "")
    threads: config.get("rebasecall_threads", 4)
    resources:
        mem_mb=config.get("rebasecall_mem", 16000),
        runtime=config.get("rebasecall_time", 480),
        gpu=config.get("rebasecall_gpu", 1)
    log:
        config.get("rebasecall_dir", "results/bam/rebasecall") + "/{sample}/rebasecall.log"
    shell:
        """
        # Set CUDA devices if available
        if [ -n "{params.cuda_devices}" ]; then
            export CUDA_VISIBLE_DEVICES={params.cuda_devices}
        fi

        # Run dorado basecaller
        {params.dorado_bin} basecaller {params.dorado_opts} {params.model} {input.pod5} > {output.bam} 2> {log}
        """


rule align_rebasecalled:
    """
    Align rebasecalled reads to reference using minimap2.

    This is an optional step if you want to align the rebasecalled reads
    to a reference genome/transcriptome before running leech.
    """
    input:
        bam=rules.rebasecall.output.bam,
        reference=lambda wildcards: config["samples"][wildcards.sample].get("reference")
    output:
        bam=config.get("rebasecall_dir", "results/bam/rebasecall") + "/{sample}/{sample}.aligned.bam",
        bai=config.get("rebasecall_dir", "results/bam/rebasecall") + "/{sample}/{sample}.aligned.bam.bai"
    threads: config.get("align_threads", 8)
    resources:
        mem_mb=config.get("align_mem", 16000),
        runtime=config.get("align_time", 240)
    log:
        config.get("rebasecall_dir", "results/bam/rebasecall") + "/{sample}/align.log"
    shell:
        """
        # Convert BAM to FASTQ, align with minimap2, sort and index
        samtools fastq -T '*' {input.bam} | \
        minimap2 -ax map-ont -y {input.reference} - | \
        samtools sort -@ {threads} -o {output.bam} - 2> {log}

        samtools index {output.bam}
        """
