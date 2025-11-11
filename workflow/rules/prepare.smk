"""
Data preparation rules for extracting training chunks.
"""


rule prepare_chunks:
    """Extract training chunks from POD5/BAM files without splitting.

    Chunks are extracted with labels but not split into train/val/test.
    Splitting happens later at the merge step to prevent data leakage across samples.

    Uses parallel processing with workers matching the number of allocated CPUs.
    """
    input:
        pod5=get_project_path(config.get("pod5_dir", "results/pod5"))
        + "/{sample}/{sample}.pod5",
        bam=get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall"))
        + "/{sample}/{sample}.aligned.bam",
    output:
        all=CHUNKS_DIR + "/{sample}/all.npz",
    wildcard_constraints:
        sample="[^/]+",  # Sample name cannot contain slashes (excludes merged/*/)
    threads: config.get("workers", 4)  # Match threads to workers from config
    params:
        output_dir=CHUNKS_DIR + "/{sample}",
        motif=config.get("motif", "CCA"),
        motif_offset=config.get("motif_offset", 2),
        motif_reference=config.get("motif_reference", "fasta"),
        reference_fasta=config.get("reference_fasta", None),
        skip_motif_indels=config.get("skip_motif_indels", True),
        workers=config.get("workers", 4),
        chunk_size=config.get("chunk_size", 100),
        label=lambda wildcards: (
            1 if config["samples"][wildcards.sample].get("label") == "charged" else 0
        ),
        label_type=lambda wildcards: config["samples"][wildcards.sample].get("amino_acid", None),
        # Conditional arguments using lambda functions
        ref_fasta_arg=lambda wildcards: (
            f"--reference-fasta {config.get('reference_fasta', None)}"
            if config.get("reference_fasta", None) and config.get("reference_fasta", None) != "None"
            else ""
        ),
        skip_indels_arg=lambda wildcards: (
            "--skip-motif-indels" if config.get("skip_motif_indels", True) else ""
        ),
        label_type_arg=lambda wildcards: (
            f"--label-type {config['samples'][wildcards.sample].get('amino_acid', None)}"
            if config['samples'][wildcards.sample].get('amino_acid', None)
            else ""
        ),
        slurm_extra="",  # No GPU needed for data preparation
    log:
        CHUNKS_DIR + "/{sample}/prepare.log",
    shell:
        """
        uv run leech prepare \
            --pod5 {input.pod5} \
            --bam {input.bam} \
            --output-dir {params.output_dir} \
            --motif {params.motif} \
            --motif-offset {params.motif_offset} \
            --motif-reference {params.motif_reference} \
            {params.ref_fasta_arg} \
            {params.skip_indels_arg} \
            --label {params.label} \
            {params.label_type_arg} \
            --workers {params.workers} \
            --chunk-size {params.chunk_size} \
            --no-split \
            2>&1 | tee {log}
        """
