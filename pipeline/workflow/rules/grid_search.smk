"""
Grid search rules for hyperparameter tuning.
All comparisons (including charged vs uncharged) are handled as pairwise comparisons.
"""


rule grid_search_pairwise_aa:
    """Perform grid search for pairwise amino acid classifier."""
    input:
        train=CHUNKS_DIR + "/merged/pairwise/{pair}/train.npz",
        val=CHUNKS_DIR + "/merged/pairwise/{pair}/val.npz",
    output:
        results=MODELS_DIR + "/grid_search/pairwise/{pair}/grid_search_results.json",
        best_params=MODELS_DIR + "/grid_search/pairwise/{pair}/best_params.json",
    log:
        MODELS_DIR + "/grid_search/pairwise/{pair}/grid_search.log",
    resources:
        slurm_partition=lambda wildcards, attempt: (
            "amilan" if config.get("use_cpu_training", False) else "aa100"
        ),
        runtime=240,
        cpus_per_task=lambda wildcards, attempt: (
            16 if config.get("use_cpu_training", False) else 4
        ),
        mem_mb=16000,
        gres=lambda wildcards, attempt: (
            "" if config.get("use_cpu_training", False) else "gpu:1"
        ),
    params:
        output_dir=MODELS_DIR + "/grid_search/pairwise/{pair}",
        # `leech model optimize` searches signal context windows and dwell
        # offset -- not learning rate / batch size / layer sizes.
        context_grid=get_grid_search_setting("context_grid", "200:1000:200"),
        left_contexts=optional_flag(
            "--left-contexts", get_grid_search_setting("left_contexts", None)
        ),
        right_contexts=optional_flag(
            "--right-contexts", get_grid_search_setting("right_contexts", None)
        ),
        dwell_offsets=get_grid_search_setting("dwell_offsets", "0"),
        model=config.get("model", "ConvLSTMDwell"),
        epochs=config.get("grid_search_epochs", 20),
        parallel=config.get("grid_search_parallel", 1),
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
    shell:
        """
        uv run leech model optimize \
            --train-data {input.train} \
            --val-data {input.val} \
            --output-dir {params.output_dir} \
            --model {params.model} \
            --epochs {params.epochs} \
            --context-grid '{params.context_grid}' \
            {params.left_contexts} \
            {params.right_contexts} \
            --dwell-offsets '{params.dwell_offsets}' \
            --parallel {params.parallel} \
            --device {params.device} \
            2>&1 | tee {log}
        """
