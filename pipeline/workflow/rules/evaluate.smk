"""
Evaluation rules for testing models and aggregating results.
All comparisons (including charged vs uncharged) are handled as pairwise comparisons.
"""


rule test_pairwise_aa:
    """Evaluate pairwise amino acid model on merged test set."""
    input:
        model=MODELS_DIR + "/pairwise/{pair}/model_best.pt",
        test=CHUNKS_DIR + "/merged/pairwise/{pair}/test.npz",
    output:
        metrics=METRICS_DIR + "/pairwise/{pair}/test_metrics.json",
    params:
        model_dir=MODELS_DIR + "/pairwise/{pair}",
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
    resources:
        slurm_partition=lambda wildcards, attempt: (
            "amilan" if config.get("use_cpu_training", False) else "atesting_a100"
        ),
        runtime=lambda wildcards, attempt: (
            240 if config.get("use_cpu_training", False) else 60
        ),
        cpus_per_task=lambda wildcards, attempt: (
            16 if config.get("use_cpu_training", False) else 2
        ),
        mem_mb=4000,
        gres=lambda wildcards, attempt: (
            "" if config.get("use_cpu_training", False) else "gpu:1"
        ),
    log:
        METRICS_DIR + "/pairwise/{pair}/test.log",
    shell:
        """
        uv run leech test \
            --model {params.model_dir} \
            --test-data {input.test} \
            --output {output.metrics} \
            --device {params.device} \
            2>&1 | tee {log}
        """


rule summarize_pairwise:
    """Aggregate metrics across all pairwise classifiers."""
    input:
        expand(
            METRICS_DIR + "/pairwise/{pair}/test_metrics.json",
            pair=AA_PAIRS,
        )
        if AA_PAIRS
        else [],
    output:
        summary=METRICS_DIR + "/pairwise_summary.tsv",
    params:
        metrics_dir=METRICS_DIR + "/pairwise",
    script:
        "../scripts/summarize_metrics.py"
