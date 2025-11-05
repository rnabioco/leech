"""
Grid search rules for hyperparameter tuning.
"""

rule grid_search_charged_vs_uncharged:
    """Perform grid search for hyperparameter tuning on charged vs uncharged."""
    input:
        train = expand(CHUNKS_DIR + "/{sample}/train.json",
                      sample=get_charged_samples()),
        val = expand(CHUNKS_DIR + "/{sample}/val.json",
                    sample=get_charged_samples()),
    output:
        results = MODELS_DIR + "/grid_search/charged_vs_uncharged/grid_search_results.json",
        best_params = MODELS_DIR + "/grid_search/charged_vs_uncharged/best_params.json",
    params:
        output_dir = MODELS_DIR + "/grid_search/charged_vs_uncharged",
        param_grid = config.get("grid_search", {
            "learning_rate": [0.0001, 0.001, 0.01],
            "batch_size": [64, 128, 256],
            "hidden_size": [128, 256, 512],
            "num_layers": [1, 2, 3],
        }),
        max_epochs = config.get("grid_search_epochs", 20),
    threads: 4
    resources:
        mem_mb = 16000,
        runtime = 1440,  # 24 hours for grid search
        gpu = 1,
        gpu_mem_mb = 10000,
    log:
        MODELS_DIR + "/grid_search/charged_vs_uncharged/grid_search.log"
    shell:
        """
        uv run leech grid-search \
            --train-data {input.train} \
            --val-data {input.val} \
            --output-dir {params.output_dir} \
            --max-epochs {params.max_epochs} \
            --param-grid '{params.param_grid}' \
            2>&1 | tee {log}
        """


rule grid_search_pairwise_aa:
    """Perform grid search for pairwise amino acid classifier."""
    input:
        train = lambda wildcards: expand(
            CHUNKS_DIR + "/{sample}/train.json",
            sample=get_samples_for_aa_pair(wildcards.pair)
        ),
        val = lambda wildcards: expand(
            CHUNKS_DIR + "/{sample}/val.json",
            sample=get_samples_for_aa_pair(wildcards.pair)
        ),
    output:
        results = MODELS_DIR + "/grid_search/pairwise/{pair}/grid_search_results.json",
        best_params = MODELS_DIR + "/grid_search/pairwise/{pair}/best_params.json",
    params:
        output_dir = MODELS_DIR + "/grid_search/pairwise/{pair}",
        param_grid = config.get("grid_search", {
            "learning_rate": [0.0001, 0.001, 0.01],
            "batch_size": [64, 128, 256],
            "hidden_size": [128, 256, 512],
            "num_layers": [1, 2, 3],
        }),
        max_epochs = config.get("grid_search_epochs", 20),
    threads: 4
    resources:
        mem_mb = 16000,
        runtime = 1440,
        gpu = 1,
        gpu_mem_mb = 10000,
    log:
        MODELS_DIR + "/grid_search/pairwise/{pair}/grid_search.log"
    shell:
        """
        uv run leech grid-search \
            --train-data {input.train} \
            --val-data {input.val} \
            --output-dir {params.output_dir} \
            --max-epochs {params.max_epochs} \
            --param-grid '{params.param_grid}' \
            2>&1 | tee {log}
        """
