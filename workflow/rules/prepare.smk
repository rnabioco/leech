"""
Data preparation rules for extracting training chunks.
"""


rule prepare_chunks:
    """Extract training chunks from POD5/BAM files."""
    input:
        pod5=config.get("pod5_dir", "results/pod5") + "/{sample}/{sample}.pod5",
        bam=config.get("rebasecall_dir", "results/bam/rebasecall") + "/{sample}/{sample}.aligned.bam",
    output:
        train=CHUNKS_DIR + "/{sample}/train.json",
        val=CHUNKS_DIR + "/{sample}/val.json",
        test=CHUNKS_DIR + "/{sample}/test.json",
    params:
        output_dir=CHUNKS_DIR + "/{sample}",
        motif=config.get("motif", "CCA"),
        motif_offset=config.get("motif_offset", 2),
        kmer_len=config.get("kmer_len", 5),
        chunk_context=config.get("chunk_context", [200, 200]),
        train_split=config.get("train_split", 0.7),
        val_split=config.get("val_split", 0.15),
        slurm_extra="",  # No GPU needed for data preparation
    threads: 4
    resources:
        mem_mb=8000,
        runtime=120,
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
            --kmer-len {params.kmer_len} \
            --chunk-context {params.chunk_context[0]} {params.chunk_context[1]} \
            --train-split {params.train_split} \
            --val-split {params.val_split} \
            2>&1 | tee {log}
        """
