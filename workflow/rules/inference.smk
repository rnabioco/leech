"""
Inference rules for applying trained models to data.
"""


rule infer_charged_vs_uncharged:
    """Run inference on test data for charged vs uncharged classification."""
    input:
        model=MODELS_DIR + "/charged_vs_uncharged/model_best.pt",
        pod5=lambda wildcards: config["samples"][wildcards.sample]["pod5"],
        bam=lambda wildcards: config["samples"][wildcards.sample]["bam"],
    output:
        bam=INFER_DIR + "/charged_vs_uncharged/{sample}_predictions.bam",
        bai=INFER_DIR + "/charged_vs_uncharged/{sample}_predictions.bam.bai",
    params:
        batch_size=config.get("infer_batch_size", 256),
        samtools_bin=config.get("samtools_bin", "samtools"),
    threads: 4
    resources:
        mem_mb=8000,
        runtime=240,
        gpu=0,  # GPU controlled by cluster profile
        gpu_mem_mb=4000,
    log:
        INFER_DIR + "/charged_vs_uncharged/{sample}_infer.log",
    shell:
        """
        uv run leech infer \
            --model {input.model} \
            --pod5 {input.pod5} \
            --bam {input.bam} \
            --output {output.bam} \
            --batch-size {params.batch_size} \
            2>&1 | tee {log}

        {params.samtools_bin} index {output.bam}
        """


rule infer_pairwise_aa:
    """Run inference for pairwise amino acid classification."""
    input:
        model=MODELS_DIR + "/pairwise/{pair}/model_best.pt",
        pod5=lambda wildcards: config["samples"][wildcards.sample]["pod5"],
        bam=lambda wildcards: config["samples"][wildcards.sample]["bam"],
    output:
        bam=INFER_DIR + "/pairwise/{pair}/{sample}_predictions.bam",
        bai=INFER_DIR + "/pairwise/{pair}/{sample}_predictions.bam.bai",
    params:
        batch_size=config.get("infer_batch_size", 256),
        samtools_bin=config.get("samtools_bin", "samtools"),
    threads: 4
    resources:
        mem_mb=8000,
        runtime=240,
        gpu=0,  # GPU controlled by cluster profile
        gpu_mem_mb=4000,
    log:
        INFER_DIR + "/pairwise/{pair}/{sample}_infer.log",
    shell:
        """
        uv run leech infer \
            --model {input.model} \
            --pod5 {input.pod5} \
            --bam {input.bam} \
            --output {output.bam} \
            --batch-size {params.batch_size} \
            2>&1 | tee {log}

        {params.samtools_bin} index {output.bam}
        """
