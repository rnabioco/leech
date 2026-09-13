"""
Evaluation rules for testing models and aggregating results.
All comparisons (including charged vs uncharged) are handled as pairwise comparisons.
"""


rule test_pairwise_aa:
    """Evaluate pairwise amino acid model on merged test set.

    Evaluates against the same geometry `train_pairwise_aa` trained on: the
    grid-search-optimized test split when `use_grid_search` is set, the base
    one otherwise. Mismatching these raises a signal_len shape error (or
    worse, silently scores a model against the wrong window).
    """
    input:
        model=MODELS_DIR + "/pairwise/{pair}/model_best.pt",
        test=(
            CHUNKS_DIR + "/merged/pairwise/{pair}/optimized/test.npz"
            if config.get("use_grid_search", False)
            else CHUNKS_DIR + "/merged/pairwise/{pair}/test.npz"
        ),
    output:
        metrics=METRICS_DIR + "/pairwise/{pair}/test_metrics.json",
    log:
        METRICS_DIR + "/pairwise/{pair}/test.log",
    # slurm_partition/runtime/cpus_per_task/mem_mb/gres for this rule live in
    # the cluster profile (pipeline/cluster/slurm{,-cpu}/config.yaml) -- see
    # train.smk's train_pairwise_aa for why.
    params:
        model_dir=MODELS_DIR + "/pairwise/{pair}",
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
    shell:
        """
        uv run leech eval test \
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
        summary=METRICS_DIR + "/pairwise_summary.tsv.gz",
    params:
        metrics_dir=METRICS_DIR + "/pairwise",
    script:
        "../scripts/summarize_metrics.py"
