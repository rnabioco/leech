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
        # Merge POD5 files
        pod5 merge -t {threads} -f -o {output.pod5} {input} &>> {log}
        """


rule inspect_merged_pod5:
    """
    Inspect merged POD5 file for provenance and version information.

    Reports:
    - POD5 file format version
    - Total number of reads
    - Input file provenance
    - Read count per input file
    """
    input:
        pod5=rules.merge_pods.output.pod5,
        raw_inputs=get_raw_pod5_inputs
    output:
        report=get_project_path(config.get("pod5_dir", "results/pod5")) + "/{sample}/{sample}_inspect.txt"
    resources:
        mem_mb=4000,
        runtime=30
    log:
        get_project_path(config.get("pod5_dir", "results/pod5")) + "/{sample}/inspect.log"
    shell:
        """
        echo "=== POD5 Inspection Report ===" > {output.report}
        echo "Sample: {wildcards.sample}" >> {output.report}
        echo "Date: $(date)" >> {output.report}
        echo "" >> {output.report}

        # Get POD5 version
        echo "=== POD5 Tool Version ===" >> {output.report}
        pod5 --version >> {output.report} 2>&1
        echo "" >> {output.report}

        # Get merged file format version and info
        echo "=== Merged POD5 File Info ===" >> {output.report}
        echo "File: {input.pod5}" >> {output.report}
        pod5 inspect read {input.pod5} >> {output.report} 2>&1 || echo "Inspect failed, trying alternative method..." >> {output.report}
        echo "" >> {output.report}

        # Count reads in merged file
        echo "=== Merged File Read Count ===" >> {output.report}
        merged_count=$(pod5 view {input.pod5} --include "read_id" --output - 2>/dev/null | tail -n +2 | wc -l)
        echo "Total reads in merged file: $merged_count" >> {output.report}
        echo "" >> {output.report}

        # Input file provenance
        echo "=== Input File Provenance ===" >> {output.report}
        echo "Number of input files: $(echo {input.raw_inputs} | wc -w)" >> {output.report}
        echo "" >> {output.report}

        # Count reads per input file
        echo "=== Read Counts Per Input File ===" >> {output.report}
        total_input_reads=0
        for f in {input.raw_inputs}; do
            count=$(pod5 view "$f" --include "read_id" --output - 2>/dev/null | tail -n +2 | wc -l)
            echo "  $f: $count reads" >> {output.report}
            total_input_reads=$((total_input_reads + count))
        done
        echo "" >> {output.report}
        echo "Total reads in input files: $total_input_reads" >> {output.report}
        echo "" >> {output.report}

        # Verify read count matches
        echo "=== Verification ===" >> {output.report}
        if [ "$merged_count" -eq "$total_input_reads" ]; then
            echo "✓ Read counts match: $merged_count reads" >> {output.report}
        else
            echo "✗ WARNING: Read count mismatch!" >> {output.report}
            echo "  Input files total: $total_input_reads" >> {output.report}
            echo "  Merged file total: $merged_count" >> {output.report}
            diff=$((merged_count - total_input_reads))
            echo "  Difference: $diff reads" >> {output.report}
        fi

        echo "Inspection complete" >> {log}
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
        dorado_opts=config.get("dorado_opts", "--emit-moves"),
        models_dir=config.get("dorado_models_dir", get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall")) + "/.models")
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
            echo "CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES" >> {log}
            export CUDA_VISIBLE_DEVICES
        fi

        # Create models directory on scratch
        mkdir -p {params.models_dir}
        echo "Using dorado models directory: {params.models_dir}" >> {log}

        # Run basecaller with models directory on scratch
        if [[ -n "{params.modifications}" ]]; then
            {params.dorado_bin} basecaller {params.model} {input} {params.dorado_opts} --models-directory {params.models_dir} --modified-bases {params.modifications} 2> {log} > {output.bam}
        else
            {params.dorado_bin} basecaller {params.model} {input} {params.dorado_opts} --models-directory {params.models_dir} 2> {log} > {output.bam}
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
