"""
Training rules for charged vs uncharged and pairwise amino acid classification.
"""

rule train_charged_vs_uncharged:
    """Train classifier to distinguish charged vs uncharged tRNAs."""
    input:
        train = expand(CHUNKS_DIR + "/{sample}/train.json",
                      sample=get_charged_samples()),
        val = expand(CHUNKS_DIR + "/{sample}/val.json",
                    sample=get_charged_samples()),
        grid_search = ancient(MODELS_DIR + "/grid_search/charged_vs_uncharged/best_params.json") if config.get("use_grid_search", False) else [],
    output:
        model = MODELS_DIR + "/charged_vs_uncharged/model_best.pt",
        checkpoint = MODELS_DIR + "/charged_vs_uncharged/model_checkpoint.pt",
        history = MODELS_DIR + "/charged_vs_uncharged/training_history.json",
    params:
        output_dir = MODELS_DIR + "/charged_vs_uncharged",
        model_type = config.get("model", "ConvLSTMDwell"),
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
        MODELS_DIR + "/charged_vs_uncharged/train.log"
    shell:
        """
        uv run leech train \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {params.model_type} \
            --output-dir {params.output_dir} \
            --epochs {params.epochs} \
            --batch-size {params.batch_size} \
            --learning-rate {params.lr} \
            --early-stopping {params.early_stopping} \
            {params.config_flag} \
            2>&1 | tee {log}
        """


rule train_pairwise_aa:
    """Train pairwise amino acid classifier."""
    input:
        train = lambda wildcards: expand(
            CHUNKS_DIR + "/{sample}/train.json",
            sample=get_samples_for_aa_pair(wildcards.pair)
        ),
        val = lambda wildcards: expand(
            CHUNKS_DIR + "/{sample}/val.json",
            sample=get_samples_for_aa_pair(wildcards.pair)
        ),
        grid_search = ancient(MODELS_DIR + "/grid_search/pairwise/{pair}/best_params.json") if config.get("use_grid_search", False) else [],
    output:
        model = MODELS_DIR + "/pairwise/{pair}/model_best.pt",
        checkpoint = MODELS_DIR + "/pairwise/{pair}/model_checkpoint.pt",
        history = MODELS_DIR + "/pairwise/{pair}/training_history.json",
    params:
        output_dir = MODELS_DIR + "/pairwise/{pair}",
        model_type = config.get("model", "ConvLSTMDwell"),
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
        MODELS_DIR + "/pairwise/{pair}/train.log"
    shell:
        """
        uv run leech train \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {params.model_type} \
            --output-dir {params.output_dir} \
            --epochs {params.epochs} \
            --batch-size {params.batch_size} \
            --learning-rate {params.lr} \
            --early-stopping {params.early_stopping} \
            {params.config_flag} \
            2>&1 | tee {log}
        """
