"""
Training rules for charged vs uncharged and pairwise amino acid classification.
"""


rule merge_chunks_charged:
    """Merge all charged/uncharged chunks and split at read level to prevent data leakage.

    This rule implements the correct workflow:
    1. Merge all chunks from different samples (charged and uncharged)
    2. Split merged data at the READ level into train/val/test

    This prevents data leakage that occurs when splitting each sample independently
    and then merging the splits.
    """
    input:
        chunks=expand(CHUNKS_DIR + "/{sample}/all.npz", sample=get_charged_samples()),
    output:
        train=CHUNKS_DIR + "/merged/charged_vs_uncharged/train.npz",
        val=CHUNKS_DIR + "/merged/charged_vs_uncharged/val.npz",
        test=CHUNKS_DIR + "/merged/charged_vs_uncharged/test.npz",
    params:
        output_dir=CHUNKS_DIR + "/merged/charged_vs_uncharged",
        train_split=config.get("train_split", 0.7),
        val_split=config.get("val_split", 0.15),
        seed=config.get("seed", 42),
        # Build input arguments for merge-and-split command
        input_args=lambda wildcards, input: " ".join([f"-i {f}" for f in input.chunks]),
    threads: 4
    resources:
        mem_mb=16000,
        runtime=60,
    log:
        CHUNKS_DIR + "/merged/charged_vs_uncharged/merge_and_split.log",
    shell:
        """
        uv run leech merge-and-split \
            {params.input_args} \
            --output-dir {params.output_dir} \
            --train-split {params.train_split} \
            --val-split {params.val_split} \
            --seed {params.seed} \
            2>&1 | tee {log}
        """


rule train_charged_vs_uncharged:
    """Train classifier to distinguish charged vs uncharged tRNAs."""
    input:
        train=rules.merge_chunks_charged.output.train,
        val=rules.merge_chunks_charged.output.val,
        grid_search=(
            ancient(MODELS_DIR + "/grid_search/charged_vs_uncharged/best_params.json")
            if config.get("use_grid_search", False)
            else []
        ),
    output:
        model=MODELS_DIR + "/charged_vs_uncharged/model_best.pt",
        checkpoint=MODELS_DIR + "/charged_vs_uncharged/model_checkpoint.pt",
        history=MODELS_DIR + "/charged_vs_uncharged/training_history.json",
    params:
        output_dir=MODELS_DIR + "/charged_vs_uncharged",
        model_type=config.get("model", "ConvLSTMDwell"),
        epochs=config.get("epochs", 50),
        batch_size=config.get("batch_size", 128),
        lr=config.get("learning_rate", 0.001),
        early_stopping=config.get("early_stopping_patience", 5),
        config_flag=lambda wildcards, input: (
            f"--config {input.grid_search}"
            if config.get("use_grid_search", False)
            else ""
        ),
    threads: 4
    resources:
        mem_mb=76000,
        runtime=60,
        gpu=0,  # GPU controlled by cluster profile
        gpu_mem_mb=32000,
    log:
        MODELS_DIR + "/charged_vs_uncharged/train.log",
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


rule merge_chunks_pairwise:
    """Merge pairwise amino acid chunks and split at read level to prevent data leakage.

    This rule implements the correct workflow:
    1. Merge all chunks from the two amino acid samples
    2. Split merged data at the READ level into train/val/test

    This prevents data leakage that occurs when splitting each sample independently
    and then merging the splits.
    """
    input:
        chunks=lambda wildcards: expand(
            CHUNKS_DIR + "/{sample}/all.npz",
            sample=get_samples_for_aa_pair(wildcards.pair),
        ),
    output:
        train=CHUNKS_DIR + "/merged/pairwise/{pair}/train.npz",
        val=CHUNKS_DIR + "/merged/pairwise/{pair}/val.npz",
        test=CHUNKS_DIR + "/merged/pairwise/{pair}/test.npz",
    params:
        output_dir=CHUNKS_DIR + "/merged/pairwise/{pair}",
        train_split=config.get("train_split", 0.7),
        val_split=config.get("val_split", 0.15),
        seed=config.get("seed", 42),
        # Build input arguments for merge-and-split command
        input_args=lambda wildcards, input: " ".join([f"-i {f}" for f in input.chunks]),
    threads: 4
    resources:
        mem_mb=16000,
        runtime=60,
    log:
        CHUNKS_DIR + "/merged/pairwise/{pair}/merge_and_split.log",
    shell:
        """
        uv run leech merge-and-split \
            {params.input_args} \
            --output-dir {params.output_dir} \
            --train-split {params.train_split} \
            --val-split {params.val_split} \
            --seed {params.seed} \
            2>&1 | tee {log}
        """


rule train_pairwise_aa:
    """Train pairwise amino acid classifier."""
    input:
        train=CHUNKS_DIR + "/merged/pairwise/{pair}/train.npz",
        val=CHUNKS_DIR + "/merged/pairwise/{pair}/val.npz",
        grid_search=(
            ancient(MODELS_DIR + "/grid_search/pairwise/{pair}/best_params.json")
            if config.get("use_grid_search", False)
            else []
        ),
    output:
        model=MODELS_DIR + "/pairwise/{pair}/model_best.pt",
        checkpoint=MODELS_DIR + "/pairwise/{pair}/model_checkpoint.pt",
        history=MODELS_DIR + "/pairwise/{pair}/training_history.json",
    params:
        output_dir=MODELS_DIR + "/pairwise/{pair}",
        model_type=config.get("model", "ConvLSTMDwell"),
        epochs=config.get("epochs", 50),
        batch_size=config.get("batch_size", 128),
        lr=config.get("learning_rate", 0.001),
        early_stopping=config.get("early_stopping_patience", 5),
        config_flag=lambda wildcards, input: (
            f"--config {input.grid_search}"
            if config.get("use_grid_search", False)
            else ""
        ),
    threads: 4
    resources:
        mem_mb=76000,
        runtime=60,
        gpu=0,  # GPU controlled by cluster profile
        gpu_mem_mb=32000,
    log:
        MODELS_DIR + "/pairwise/{pair}/train.log",
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
