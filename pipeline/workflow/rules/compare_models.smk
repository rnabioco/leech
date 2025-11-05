"""
Model comparison rules for evaluating multiple architectures.

These rules train and evaluate multiple model architectures on the same
data splits to enable fair comparison.
"""

# ============================================================================
# Multi-Architecture Training: Charged vs Uncharged
# ============================================================================

rule train_architecture_charged:
    """Train a specific architecture for charged vs uncharged classification."""
    input:
        train = expand(CHUNKS_DIR + "/{sample}/train.json",
                      sample=get_charged_samples()),
        val = expand(CHUNKS_DIR + "/{sample}/val.json",
                    sample=get_charged_samples()),
        grid_search = ancient(MODELS_DIR + "/grid_search/charged_vs_uncharged/{architecture}/best_params.json") if config.get("use_grid_search", False) else [],
    output:
        model = MODELS_DIR + "/comparison/charged_vs_uncharged/{architecture}/model_best.pt",
        checkpoint = MODELS_DIR + "/comparison/charged_vs_uncharged/{architecture}/model_checkpoint.pt",
        history = MODELS_DIR + "/comparison/charged_vs_uncharged/{architecture}/training_history.json",
    params:
        output_dir = MODELS_DIR + "/comparison/charged_vs_uncharged/{architecture}",
        epochs = config.get("epochs", 50),
        batch_size = config.get("batch_size", 128),
        lr = config.get("learning_rate", 0.001),
        early_stopping = config.get("early_stopping_patience", 5),
        config_flag = lambda wildcards, input: f"--config {input.grid_search}" if config.get("use_grid_search", False) else "",
    threads: 4
    resources:
        mem_mb = 16000,
        runtime = 480,
        gpu = 1,
        gpu_mem_mb = 8000,
    log:
        MODELS_DIR + "/comparison/charged_vs_uncharged/{architecture}/train.log"
    shell:
        """
        uv run leech train \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {wildcards.architecture} \
            --output-dir {params.output_dir} \
            --epochs {params.epochs} \
            --batch-size {params.batch_size} \
            --learning-rate {params.lr} \
            --early-stopping {params.early_stopping} \
            {params.config_flag} \
            2>&1 | tee {log}
        """


rule test_architecture_charged:
    """Test a specific architecture on the test set."""
    input:
        model = MODELS_DIR + "/comparison/charged_vs_uncharged/{architecture}/model_best.pt",
        test = CHUNKS_DIR + "/{sample}/test.json",
    output:
        metrics = METRICS_DIR + "/comparison/charged_vs_uncharged/{architecture}/{sample}_metrics.json",
    threads: 2
    resources:
        mem_mb = 4000,
        runtime = 60,
        gpu = 1,
        gpu_mem_mb = 4000,
    log:
        METRICS_DIR + "/comparison/charged_vs_uncharged/{architecture}/{sample}_test.log"
    shell:
        """
        uv run leech test \
            --model {input.model} \
            --test-data {input.test} \
            --output {output.metrics} \
            2>&1 | tee {log}
        """


rule compare_architectures_charged:
    """Compare all architectures on charged vs uncharged task."""
    input:
        expand(METRICS_DIR + "/comparison/charged_vs_uncharged/{arch}/{sample}_metrics.json",
               arch=MODEL_ARCHITECTURES, sample=SAMPLES)
    output:
        comparison = METRICS_DIR + "/comparison/charged_vs_uncharged_architecture_comparison.tsv",
        summary = METRICS_DIR + "/comparison/charged_vs_uncharged_architecture_summary.txt",
    params:
        metrics_dir = METRICS_DIR + "/comparison/charged_vs_uncharged",
        architectures = MODEL_ARCHITECTURES,
    script:
        "../scripts/compare_architectures.py"


# ============================================================================
# Multi-Architecture Training: Pairwise Amino Acids
# ============================================================================

rule train_architecture_pairwise:
    """Train a specific architecture for pairwise AA classification."""
    input:
        train = lambda wildcards: expand(
            CHUNKS_DIR + "/{sample}/train.json",
            sample=get_samples_for_aa_pair(wildcards.pair)
        ),
        val = lambda wildcards: expand(
            CHUNKS_DIR + "/{sample}/val.json",
            sample=get_samples_for_aa_pair(wildcards.pair)
        ),
        grid_search = ancient(MODELS_DIR + "/grid_search/pairwise/{pair}/{architecture}/best_params.json") if config.get("use_grid_search", False) else [],
    output:
        model = MODELS_DIR + "/comparison/pairwise/{pair}/{architecture}/model_best.pt",
        checkpoint = MODELS_DIR + "/comparison/pairwise/{pair}/{architecture}/model_checkpoint.pt",
        history = MODELS_DIR + "/comparison/pairwise/{pair}/{architecture}/training_history.json",
    params:
        output_dir = MODELS_DIR + "/comparison/pairwise/{pair}/{architecture}",
        epochs = config.get("epochs", 50),
        batch_size = config.get("batch_size", 128),
        lr = config.get("learning_rate", 0.001),
        early_stopping = config.get("early_stopping_patience", 5),
        config_flag = lambda wildcards, input: f"--config {input.grid_search}" if config.get("use_grid_search", False) else "",
    threads: 4
    resources:
        mem_mb = 16000,
        runtime = 480,
        gpu = 1,
        gpu_mem_mb = 8000,
    log:
        MODELS_DIR + "/comparison/pairwise/{pair}/{architecture}/train.log"
    shell:
        """
        uv run leech train \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {wildcards.architecture} \
            --output-dir {params.output_dir} \
            --epochs {params.epochs} \
            --batch-size {params.batch_size} \
            --learning-rate {params.lr} \
            --early-stopping {params.early_stopping} \
            {params.config_flag} \
            2>&1 | tee {log}
        """


rule test_architecture_pairwise:
    """Test a specific architecture on pairwise AA test set."""
    input:
        model = MODELS_DIR + "/comparison/pairwise/{pair}/{architecture}/model_best.pt",
        test = CHUNKS_DIR + "/{sample}/test.json",
    output:
        metrics = METRICS_DIR + "/comparison/pairwise/{pair}/{architecture}/{sample}_metrics.json",
    threads: 2
    resources:
        mem_mb = 4000,
        runtime = 60,
        gpu = 1,
        gpu_mem_mb = 4000,
    log:
        METRICS_DIR + "/comparison/pairwise/{pair}/{architecture}/{sample}_test.log"
    shell:
        """
        uv run leech test \
            --model {input.model} \
            --test-data {input.test} \
            --output {output.metrics} \
            2>&1 | tee {log}
        """


rule compare_architectures_pairwise:
    """Compare all architectures on pairwise AA task."""
    input:
        expand(METRICS_DIR + "/comparison/pairwise/{pair}/{arch}/{sample}_metrics.json",
               pair=AA_PAIRS, arch=MODEL_ARCHITECTURES, sample=SAMPLES) if AA_PAIRS else []
    output:
        comparison = METRICS_DIR + "/comparison/pairwise_architecture_comparison.tsv",
        summary = METRICS_DIR + "/comparison/pairwise_architecture_summary.txt",
    params:
        metrics_dir = METRICS_DIR + "/comparison/pairwise",
        architectures = MODEL_ARCHITECTURES,
    script:
        "../scripts/compare_architectures.py"


# ============================================================================
# Grid Search per Architecture (optional)
# ============================================================================

rule grid_search_architecture_charged:
    """Perform grid search for a specific architecture on charged vs uncharged."""
    input:
        train = expand(CHUNKS_DIR + "/{sample}/train.json",
                      sample=get_charged_samples()),
        val = expand(CHUNKS_DIR + "/{sample}/val.json",
                    sample=get_charged_samples()),
    output:
        results = MODELS_DIR + "/grid_search/charged_vs_uncharged/{architecture}/grid_search_results.json",
        best_params = MODELS_DIR + "/grid_search/charged_vs_uncharged/{architecture}/best_params.json",
    params:
        output_dir = MODELS_DIR + "/grid_search/charged_vs_uncharged/{architecture}",
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
        MODELS_DIR + "/grid_search/charged_vs_uncharged/{architecture}/grid_search.log"
    shell:
        """
        uv run leech grid-search \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {wildcards.architecture} \
            --output-dir {params.output_dir} \
            --max-epochs {params.max_epochs} \
            --param-grid '{params.param_grid}' \
            2>&1 | tee {log}
        """


rule grid_search_architecture_pairwise:
    """Perform grid search for a specific architecture on pairwise AA."""
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
        results = MODELS_DIR + "/grid_search/pairwise/{pair}/{architecture}/grid_search_results.json",
        best_params = MODELS_DIR + "/grid_search/pairwise/{pair}/{architecture}/best_params.json",
    params:
        output_dir = MODELS_DIR + "/grid_search/pairwise/{pair}/{architecture}",
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
        MODELS_DIR + "/grid_search/pairwise/{pair}/{architecture}/grid_search.log"
    shell:
        """
        uv run leech grid-search \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {wildcards.architecture} \
            --output-dir {params.output_dir} \
            --max-epochs {params.max_epochs} \
            --param-grid '{params.param_grid}' \
            2>&1 | tee {log}
        """
