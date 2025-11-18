"""
Training rules for pairwise comparisons (including charged vs uncharged).
All comparisons are defined in TSV spec files and handled uniformly.
"""


rule merge_chunks_pairwise:
    """Merge pairwise amino acid chunks and split at read level to prevent data leakage.

    This rule implements the correct workflow:
    1. Merge all chunks from the two amino acid samples
    2. Relabel chunks for pairwise comparison (aa1=0, aa2=1)
    3. Split merged data at the READ level into train/val/test

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
        # Build input arguments with label=file format
        input_args=lambda wildcards, input: build_merge_input_args(
            wildcards.pair, input.chunks
        ),
    log:
        CHUNKS_DIR + "/merged/pairwise/{pair}/merge_and_split.log",
    shell:
        """
        uv run leech data merge \
            {params.input_args} \
            --output-dir {params.output_dir} \
            --train-split {params.train_split} \
            --val-split {params.val_split} \
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
        checkpoint=MODELS_DIR + "/pairwise/{pair}/model_last.pt",
        history=MODELS_DIR + "/pairwise/{pair}/metrics.json",
    params:
        output_dir=MODELS_DIR + "/pairwise/{pair}",
        model_type=config.get("model", "ConvLSTMDwell"),
        epochs=config.get("epochs", 50),
        batch_size=config.get("batch_size", 128),
        lr=config.get("learning_rate", 0.001),
        early_stopping=config.get("early_stopping_patience", 5),
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
        config_flag=lambda wildcards, input: (
            f"--config {input.grid_search}"
            if config.get("use_grid_search", False)
            else ""
        ),
    resources:
        slurm_partition=lambda wildcards, attempt: (
            "amilan" if config.get("use_cpu_training", False) else "aa100"
        ),
        runtime=lambda wildcards, attempt: (
            480 if config.get("use_cpu_training", False) else 60
        ),
        cpus_per_task=lambda wildcards, attempt: (
            16 if config.get("use_cpu_training", False) else 10
        ),
        mem_mb=18000,
        gres=lambda wildcards, attempt: (
            "" if config.get("use_cpu_training", False) else "gpu:1"
        ),
    log:
        MODELS_DIR + "/pairwise/{pair}/train.log",
    shell:
        """
        uv run leech model train \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {params.model_type} \
            --output-dir {params.output_dir} \
            --epochs {params.epochs} \
            --batch-size {params.batch_size} \
            --learning-rate {params.lr} \
            --early-stopping {params.early_stopping} \
            --device {params.device} \
            {params.config_flag} \
            2>&1 | tee {log}
        """
