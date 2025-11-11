"""
Grid search rules for hyperparameter tuning.
"""


rule grid_search_charged_vs_uncharged:
    """Perform grid search for hyperparameter tuning on charged vs uncharged."""
    input:
        train=CHUNKS_DIR + "/merged/charged_vs_uncharged/train.npz",
        val=CHUNKS_DIR + "/merged/charged_vs_uncharged/val.npz",
    output:
        results=MODELS_DIR
        + "/grid_search/charged_vs_uncharged/grid_search_results.json",
        best_params=MODELS_DIR + "/grid_search/charged_vs_uncharged/best_params.json",
    params:
        output_dir=MODELS_DIR + "/grid_search/charged_vs_uncharged",
        param_grid=config.get(
            "grid_search",
            {
                "learning_rate": [0.0001, 0.001, 0.01],
                "batch_size": [64, 128, 256],
                "hidden_size": [128, 256, 512],
                "num_layers": [1, 2, 3],
            },
        ),
        max_epochs=config.get("grid_search_epochs", 20),
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
    resources:
        slurm_partition=lambda wildcards, attempt: "amilan" if config.get("use_cpu_training", False) else "aa100",
        runtime=lambda wildcards, attempt: 5760 if config.get("use_cpu_training", False) else 1440,
        cpus_per_task=lambda wildcards, attempt: 16 if config.get("use_cpu_training", False) else 4,
        mem_mb=16000,
        gres=lambda wildcards, attempt: "" if config.get("use_cpu_training", False) else "gpu:1",
    log:
        MODELS_DIR + "/grid_search/charged_vs_uncharged/grid_search.log",
    shell:
        """
        uv run leech grid-search \
            --train-data {input.train} \
            --val-data {input.val} \
            --output-dir {params.output_dir} \
            --max-epochs {params.max_epochs} \
            --param-grid '{params.param_grid}' \
            --device {params.device} \
            2>&1 | tee {log}
        """


rule grid_search_pairwise_aa:
    """Perform grid search for pairwise amino acid classifier."""
    input:
        train=CHUNKS_DIR + "/merged/pairwise/{pair}/train.npz",
        val=CHUNKS_DIR + "/merged/pairwise/{pair}/val.npz",
    output:
        results=MODELS_DIR + "/grid_search/pairwise/{pair}/grid_search_results.json",
        best_params=MODELS_DIR + "/grid_search/pairwise/{pair}/best_params.json",
    params:
        output_dir=MODELS_DIR + "/grid_search/pairwise/{pair}",
        param_grid=config.get(
            "grid_search",
            {
                "learning_rate": [0.0001, 0.001, 0.01],
                "batch_size": [64, 128, 256],
                "hidden_size": [128, 256, 512],
                "num_layers": [1, 2, 3],
            },
        ),
        max_epochs=config.get("grid_search_epochs", 20),
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
    resources:
        slurm_partition=lambda wildcards, attempt: "amilan" if config.get("use_cpu_training", False) else "aa100",
        runtime=lambda wildcards, attempt: 5760 if config.get("use_cpu_training", False) else 1440,
        cpus_per_task=lambda wildcards, attempt: 16 if config.get("use_cpu_training", False) else 4,
        mem_mb=16000,
        gres=lambda wildcards, attempt: "" if config.get("use_cpu_training", False) else "gpu:1",
    log:
        MODELS_DIR + "/grid_search/pairwise/{pair}/grid_search.log",
    shell:
        """
        uv run leech grid-search \
            --train-data {input.train} \
            --val-data {input.val} \
            --output-dir {params.output_dir} \
            --max-epochs {params.max_epochs} \
            --param-grid '{params.param_grid}' \
            --device {params.device} \
            2>&1 | tee {log}
        """
