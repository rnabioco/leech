"""
Inference rules for applying trained models to data.
All comparisons (including charged vs uncharged) are handled as pairwise comparisons.
"""


rule infer_pairwise_aa:
    """Run inference for pairwise amino acid classification."""
    input:
        model=MODELS_DIR + "/pairwise/{pair}/model_best.pt",
        pod5=get_project_path(config.get("pod5_dir", "results/pod5"))
        + "/{sample}/{sample}.pod5",
        bam=get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall"))
        + "/{sample}/{sample}.aligned.bam",
        bai=get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall"))
        + "/{sample}/{sample}.aligned.bam.bai",
    output:
        bam=INFER_DIR + "/pairwise/{pair}/{sample}_predictions.bam",
        bai=INFER_DIR + "/pairwise/{pair}/{sample}_predictions.bam.bai",
    log:
        INFER_DIR + "/pairwise/{pair}/{sample}_infer.log",
    # slurm_partition/runtime/cpus_per_task/mem_mb/gres for this rule live in
    # the cluster profile (pipeline/cluster/slurm{,-cpu}/config.yaml) -- see
    # train.smk's train_pairwise_aa for why.
    params:
        model_dir=MODELS_DIR + "/pairwise/{pair}",
        batch_size=config.get("infer_batch_size", 256),
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
        samtools_bin=config.get("samtools_bin", "samtools"),
    shell:
        """
        uv run leech predict \
            --model {params.model_dir} \
            --pod5 {input.pod5} \
            --bam {input.bam} \
            --output {output.bam} \
            --batch-size {params.batch_size} \
            --device {params.device} \
            2>&1 | tee {log}

        {params.samtools_bin} index {output.bam}
        """
