"""
Evaluation rules for testing models and aggregating results.
"""


rule test_charged_vs_uncharged:
    """Evaluate model on test set."""
    input:
        model=MODELS_DIR + "/charged_vs_uncharged/model_best.pt",
        test=CHUNKS_DIR + "/{sample}/test.json",
    output:
        metrics=METRICS_DIR + "/charged_vs_uncharged/{sample}_metrics.json",
    threads: 2
    resources:
        mem_mb=4000,
        runtime=60,
        gpu=0,  # GPU controlled by cluster profile
        gpu_mem_mb=4000,
    log:
        METRICS_DIR + "/charged_vs_uncharged/{sample}_test.log",
    shell:
        """
        uv run leech test \
            --model {input.model} \
            --test-data {input.test} \
            --output {output.metrics} \
            2>&1 | tee {log}
        """


rule test_pairwise_aa:
    """Evaluate pairwise amino acid model on test set."""
    input:
        model=MODELS_DIR + "/pairwise/{pair}/model_best.pt",
        test=CHUNKS_DIR + "/{sample}/test.json",
    output:
        metrics=METRICS_DIR + "/pairwise/{pair}/{sample}_metrics.json",
    threads: 2
    resources:
        mem_mb=4000,
        runtime=60,
        gpu=0,  # GPU controlled by cluster profile
        gpu_mem_mb=4000,
    log:
        METRICS_DIR + "/pairwise/{pair}/{sample}_test.log",
    shell:
        """
        uv run leech test \
            --model {input.model} \
            --test-data {input.test} \
            --output {output.metrics} \
            2>&1 | tee {log}
        """


rule summarize_charged_vs_uncharged:
    """Aggregate metrics across all samples for charged vs uncharged."""
    input:
        expand(
            METRICS_DIR + "/charged_vs_uncharged/{sample}_metrics.json", sample=SAMPLES
        ),
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
            METRICS_DIR + "/pairwise/{pair}/{sample}_metrics.json",
            pair=AA_PAIRS,
            sample=SAMPLES,
        )
        if AA_PAIRS
        else [],
    output:
        summary=METRICS_DIR + "/pairwise_summary.tsv",
    params:
        metrics_dir=METRICS_DIR + "/pairwise",
    script:
        "../scripts/summarize_metrics.py"
