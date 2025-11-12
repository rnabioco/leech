"""
Evaluation rules for testing models and aggregating results.
"""


rule test_charged_vs_uncharged:
    """Evaluate model on merged test set."""
    input:
        model=MODELS_DIR + "/charged_vs_uncharged/model_best.pt",
        test=CHUNKS_DIR + "/merged/charged_vs_uncharged/test.npz",
    output:
        metrics=METRICS_DIR + "/charged_vs_uncharged/test_metrics.json",
    params:
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
        METRICS_DIR + "/charged_vs_uncharged/test.log",
    shell:
        """
        uv run leech test \
            --model {input.model} \
            --test-data {input.test} \
            --output {output.metrics} \
            --device {params.device} \
            2>&1 | tee {log}
        """


rule test_pairwise_aa:
    """Evaluate pairwise amino acid model on merged test set."""
    input:
        model=MODELS_DIR + "/pairwise/{pair}/model_best.pt",
        test=CHUNKS_DIR + "/merged/pairwise/{pair}/test.npz",
    output:
        metrics=METRICS_DIR + "/pairwise/{pair}/test_metrics.json",
    params:
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
            --model {input.model} \
            --test-data {input.test} \
            --output {output.metrics} \
            --device {params.device} \
            2>&1 | tee {log}
        """


rule summarize_charged_vs_uncharged:
    """Create summary report for charged vs uncharged test metrics."""
    input:
        METRICS_DIR + "/charged_vs_uncharged/test_metrics.json",
    output:
        summary=METRICS_DIR + "/charged_vs_uncharged_summary.tsv",
    params:
        metrics_dir=METRICS_DIR + "/charged_vs_uncharged",
    script:
        "../scripts/summarize_metrics.py"


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
