"""
Data preparation rules for extracting training chunks.
"""


rule prepare_chunks:
    """Extract training chunks from POD5/BAM files."""
    input:
        pod5=get_project_path(config.get("pod5_dir", "results/pod5"))
        + "/{sample}/{sample}.pod5",
        bam=get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall"))
        + "/{sample}/{sample}.aligned.bam",
    output:
        train=CHUNKS_DIR + "/{sample}/train.npz",
        val=CHUNKS_DIR + "/{sample}/val.npz",
        test=CHUNKS_DIR + "/{sample}/test.npz",
    wildcard_constraints:
        sample="[^/]+",  # Sample name cannot contain slashes (excludes merged/*/)
    params:
        output_dir=CHUNKS_DIR + "/{sample}",
        motif=config.get("motif", "CCA"),
        motif_offset=config.get("motif_offset", 2),
        motif_reference=config.get("motif_reference", "fasta"),
        reference_fasta=config.get("reference_fasta", None),
        skip_motif_indels=config.get("skip_motif_indels", True),
        train_split=config.get("train_split", 0.7),
        val_split=config.get("val_split", 0.15),
        seed=config.get("seed", ""),  # Empty string = no seed (will generate random)
        label=lambda wildcards: (
            1 if config["samples"][wildcards.sample].get("label") == "charged" else 0
        ),
        # Conditional arguments using lambda functions
        seed_arg=lambda wildcards: (
            f"--seed {config.get('seed', '')}" if config.get("seed", "") else ""
        ),
        ref_fasta_arg=lambda wildcards: (
            f"--reference-fasta {config.get('reference_fasta', None)}"
            if config.get("reference_fasta", None) and config.get("reference_fasta", None) != "None"
            else ""
        ),
        skip_indels_arg=lambda wildcards: (
            "--skip-motif-indels" if config.get("skip_motif_indels", True) else ""
        ),
        slurm_extra="",  # No GPU needed for data preparation
    threads: 4
    resources:
        mem_mb=8000,
        runtime=60,
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
            --train-split {params.train_split} \
            --val-split {params.val_split} \
            {params.seed_arg} \
            2>&1 | tee {log}
        """
