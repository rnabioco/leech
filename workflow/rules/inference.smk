"""
Inference rules for applying trained models to data.
"""


rule infer_charged_vs_uncharged:
    """Run inference on test data for charged vs uncharged classification."""
    input:
        model=MODELS_DIR + "/charged_vs_uncharged/model_best.pt",
        pod5=get_project_path(config.get("pod5_dir", "results/pod5")) + "/{sample}/{sample}.pod5",
        bam=get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall")) + "/{sample}/{sample}.aligned.bam",
        bai=get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall")) + "/{sample}/{sample}.aligned.bam.bai",
    output:
        bam=INFER_DIR + "/charged_vs_uncharged/{sample}_predictions.bam",
        bai=INFER_DIR + "/charged_vs_uncharged/{sample}_predictions.bam.bai",
    params:
        batch_size=config.get("infer_batch_size", 256),
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
        samtools_bin=config.get("samtools_bin", "samtools"),
    resources:
        slurm_partition=lambda wildcards, attempt: "amilan" if config.get("use_cpu_training", False) else "aa100",
        runtime=lambda wildcards, attempt: 960 if config.get("use_cpu_training", False) else 240,
        cpus_per_task=lambda wildcards, attempt: 16 if config.get("use_cpu_training", False) else 4,
        mem_mb=8000,
        gres=lambda wildcards, attempt: "" if config.get("use_cpu_training", False) else "gpu:1",
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
            --device {params.device} \
            2>&1 | tee {log}

        {params.samtools_bin} index {output.bam}
        """


rule infer_pairwise_aa:
    """Run inference for pairwise amino acid classification."""
    input:
        model=MODELS_DIR + "/pairwise/{pair}/model_best.pt",
        pod5=get_project_path(config.get("pod5_dir", "results/pod5")) + "/{sample}/{sample}.pod5",
        bam=get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall")) + "/{sample}/{sample}.aligned.bam",
        bai=get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall")) + "/{sample}/{sample}.aligned.bam.bai",
    output:
        bam=INFER_DIR + "/pairwise/{pair}/{sample}_predictions.bam",
        bai=INFER_DIR + "/pairwise/{pair}/{sample}_predictions.bam.bai",
    params:
        batch_size=config.get("infer_batch_size", 256),
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
        samtools_bin=config.get("samtools_bin", "samtools"),
    resources:
        slurm_partition=lambda wildcards, attempt: "amilan" if config.get("use_cpu_training", False) else "aa100",
        runtime=lambda wildcards, attempt: 960 if config.get("use_cpu_training", False) else 240,
        cpus_per_task=lambda wildcards, attempt: 16 if config.get("use_cpu_training", False) else 4,
        mem_mb=8000,
        gres=lambda wildcards, attempt: "" if config.get("use_cpu_training", False) else "gpu:1",
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
            --device {params.device} \
            2>&1 | tee {log}

        {params.samtools_bin} index {output.bam}
        """
