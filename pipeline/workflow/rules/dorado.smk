"""
POD5 rebasecalling rules using dorado.
"""

import os
from pathlib import Path


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

    Uses scratch directory if configured for better I/O performance.
    """
    input:
        get_raw_pod5_inputs
    output:
        pod5=config.get("pod5_dir", "results/pod5") + "/{sample}/{sample}.pod5"
    params:
        scratch_dir=lambda wildcards: get_scratch_or_output_dir(
            config.get("pod5_dir", "results/pod5") + f"/{wildcards.sample}"
        ),
        use_scratch=lambda wildcards: config.get("scratch_dir") not in [None, "", "null"]
    threads: config.get("merge_pods_threads", 12)
    resources:
        mem_mb=config.get("merge_pods_mem", 8000),
        runtime=config.get("merge_pods_time", 120)
    log:
        config.get("pod5_dir", "results/pod5") + "/{sample}/merge_pods.log"
    shell:
        """
        # Determine output location (scratch or final)
        if [ "{params.use_scratch}" = "True" ]; then
            OUT_DIR="{params.scratch_dir}"
            FINAL_DIR="$(dirname {output.pod5})"
            mkdir -p "$OUT_DIR" "$FINAL_DIR"
            OUT_FILE="$OUT_DIR/{wildcards.sample}.pod5"
        else
            OUT_DIR="$(dirname {output.pod5})"
            mkdir -p "$OUT_DIR"
            OUT_FILE="{output.pod5}"
        fi

        # Merge POD5 files
        pod5 merge -t {threads} -f -o "$OUT_FILE" {input} 2>&1 | tee {log}

        # Move from scratch to final location if needed
        if [ "{params.use_scratch}" = "True" ]; then
            mv "$OUT_FILE" {output.pod5}
        fi
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

    Uses scratch directory if configured for better I/O performance.
    """
    input:
        pod5=rules.merge_pods.output.pod5
    output:
        bam=protected(config.get("rebasecall_dir", "results/bam/rebasecall") + "/{sample}/{sample}.rbc.bam")
    params:
        dorado_bin=config.get("dorado_bin", "dorado"),
        base_model=config.get("base_calling_model", "rna004_130bps_sup@v5.2.0"),
        modifications=config.get("modifications", ""),
        model=lambda wildcards: (
            f"{config.get('base_calling_model', 'rna004_130bps_sup@v5.2.0')},"
            f"{config.get('modifications', '')}"
            if config.get("modifications", "")
            else config.get("base_calling_model", "rna004_130bps_sup@v5.2.0")
        ),
        dorado_opts=config.get("dorado_opts", "--emit-moves"),
        scratch_dir=lambda wildcards: get_scratch_or_output_dir(
            config.get("rebasecall_dir", "results/bam/rebasecall") + f"/{wildcards.sample}"
        ),
        use_scratch=lambda wildcards: config.get("scratch_dir") not in [None, "", "null"],
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

        # Determine output location (scratch or final)
        if [ "{params.use_scratch}" = "True" ]; then
            OUT_DIR="{params.scratch_dir}"
            FINAL_DIR="$(dirname {output.bam})"
            mkdir -p "$OUT_DIR" "$FINAL_DIR"
            OUT_FILE="$OUT_DIR/{wildcards.sample}.rbc.bam"
        else
            OUT_DIR="$(dirname {output.bam})"
            mkdir -p "$OUT_DIR"
            OUT_FILE="{output.bam}"
        fi

        # Run dorado basecaller
        {params.dorado_bin} basecaller {params.dorado_opts} {params.model} {input.pod5} > "$OUT_FILE" 2> {log}

        # Move from scratch to final location if needed
        if [ "{params.use_scratch}" = "True" ]; then
            mv "$OUT_FILE" {output.bam}
        fi
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
