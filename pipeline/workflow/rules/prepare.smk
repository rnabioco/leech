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
    log:
        CHUNKS_DIR + "/{sample}/prepare.log",
    # `sample` is constrained globally (Snakefile) to exclude "/", so this
    # rule's output can't be mistaken for a path under merged/, optimized/, etc.
    threads: config.get("workers", 4)  # Match threads to workers from config
    params:
        output_dir=CHUNKS_DIR + "/{sample}",
        motif=require_config("motif"),
        motif_offset=config.get("motif_offset", 0),
        motif_reference=config.get("motif_reference", "fasta"),
        reference_fasta=config.get("reference_fasta", None),
        skip_motif_indels=config.get("skip_motif_indels", True),
        workers=config.get("workers", 4),
        chunk_size=config.get("chunk_size", 100),
        # Shared with the grid-search re-prepare rules (grid_search.smk,
        # compare_models.smk) so the arg-building logic lives in one place.
        ref_fasta_arg=build_reference_fasta_arg(),
        skip_indels_arg=build_skip_indels_arg(),
        label_arg=lambda wildcards: build_label_arg(wildcards),
        signal_context_arg=build_signal_context_arg(),
        slurm_extra="",  # No GPU needed for data preparation
    shell:
        """
        uv run leech data prepare \
            --pod5 {input.pod5} \
            --bam {input.bam} \
            --output-dir {params.output_dir} \
            --motif {params.motif} \
            --motif-offset {params.motif_offset} \
            --motif-reference {params.motif_reference} \
            {params.ref_fasta_arg} \
            {params.skip_indels_arg} \
            {params.label_arg} \
            {params.signal_context_arg} \
            --workers {params.workers} \
            --chunk-size {params.chunk_size} \
            --no-split \
            2>&1 | tee {log}
        """
