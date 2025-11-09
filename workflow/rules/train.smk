"""
Training rules for charged vs uncharged and pairwise amino acid classification.
"""


rule merge_chunks_charged:
    """Merge training chunks from all charged/uncharged samples."""
    input:
        train=expand(CHUNKS_DIR + "/{sample}/train.npz", sample=get_charged_samples()),
        val=expand(CHUNKS_DIR + "/{sample}/val.npz", sample=get_charged_samples()),
    output:
        train=CHUNKS_DIR + "/merged/charged_vs_uncharged/train.npz",
        val=CHUNKS_DIR + "/merged/charged_vs_uncharged/val.npz",
    threads: 4
    resources:
        mem_mb=16000,
        runtime=60,
    log:
        CHUNKS_DIR + "/merged/charged_vs_uncharged/merge.log",
    run:
        import numpy as np
        from pathlib import Path

        # Merge training chunks
        all_train_chunks = []
        for train_file in input.train:
            data = np.load(train_file, allow_pickle=True)
            all_train_chunks.append(data)

            # Concatenate all arrays
        merged_train = {
            "signals": np.concatenate([d["signals"] for d in all_train_chunks]),
            "sequences": np.concatenate([d["sequences"] for d in all_train_chunks]),
            "dwells": np.concatenate([d["dwells"] for d in all_train_chunks]),
            "features": np.concatenate([d["features"] for d in all_train_chunks]),
            "labels": np.concatenate([d["labels"] for d in all_train_chunks]),
            "read_ids": np.concatenate([d["read_ids"] for d in all_train_chunks]),
            "base_indices": np.concatenate(
                [d["base_indices"] for d in all_train_chunks]
            ),
        }

        # Merge validation chunks
        all_val_chunks = []
        for val_file in input.val:
            data = np.load(val_file, allow_pickle=True)
            all_val_chunks.append(data)

        merged_val = {
            "signals": np.concatenate([d["signals"] for d in all_val_chunks]),
            "sequences": np.concatenate([d["sequences"] for d in all_val_chunks]),
            "dwells": np.concatenate([d["dwells"] for d in all_val_chunks]),
            "features": np.concatenate([d["features"] for d in all_val_chunks]),
            "labels": np.concatenate([d["labels"] for d in all_val_chunks]),
            "read_ids": np.concatenate([d["read_ids"] for d in all_val_chunks]),
            "base_indices": np.concatenate([d["base_indices"] for d in all_val_chunks]),
        }

        # Save merged chunks
        Path(output.train).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output.train, **merged_train)
        np.savez_compressed(output.val, **merged_val)

        with open(log[0], "w") as f:
            f.write(f"Merged {len(input.train)} training files\n")
            f.write(f"Total training chunks: {len(merged_train['labels'])}\n")
            f.write(f"Merged {len(input.val)} validation files\n")
            f.write(f"Total validation chunks: {len(merged_val['labels'])}\n")


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
        mem_mb=16000,
        runtime=480,
        gpu=0,  # GPU controlled by cluster profile
        gpu_mem_mb=8000,
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
    """Merge training chunks for pairwise amino acid classification."""
    input:
        train=lambda wildcards: expand(
            CHUNKS_DIR + "/{sample}/train.npz",
            sample=get_samples_for_aa_pair(wildcards.pair),
        ),
        val=lambda wildcards: expand(
            CHUNKS_DIR + "/{sample}/val.npz",
            sample=get_samples_for_aa_pair(wildcards.pair),
        ),
    output:
        train=CHUNKS_DIR + "/merged/pairwise/{pair}/train.npz",
        val=CHUNKS_DIR + "/merged/pairwise/{pair}/val.npz",
    threads: 4
    resources:
        mem_mb=16000,
        runtime=60,
    log:
        CHUNKS_DIR + "/merged/pairwise/{pair}/merge.log",
    run:
        import numpy as np
        from pathlib import Path

        # Merge training chunks
        all_train_chunks = []
        for train_file in input.train:
            data = np.load(train_file, allow_pickle=True)
            all_train_chunks.append(data)

        merged_train = {
            "signals": np.concatenate([d["signals"] for d in all_train_chunks]),
            "sequences": np.concatenate([d["sequences"] for d in all_train_chunks]),
            "dwells": np.concatenate([d["dwells"] for d in all_train_chunks]),
            "features": np.concatenate([d["features"] for d in all_train_chunks]),
            "labels": np.concatenate([d["labels"] for d in all_train_chunks]),
            "read_ids": np.concatenate([d["read_ids"] for d in all_train_chunks]),
            "base_indices": np.concatenate(
                [d["base_indices"] for d in all_train_chunks]
            ),
        }

        # Merge validation chunks
        all_val_chunks = []
        for val_file in input.val:
            data = np.load(val_file, allow_pickle=True)
            all_val_chunks.append(data)

        merged_val = {
            "signals": np.concatenate([d["signals"] for d in all_val_chunks]),
            "sequences": np.concatenate([d["sequences"] for d in all_val_chunks]),
            "dwells": np.concatenate([d["dwells"] for d in all_val_chunks]),
            "features": np.concatenate([d["features"] for d in all_val_chunks]),
            "labels": np.concatenate([d["labels"] for d in all_val_chunks]),
            "read_ids": np.concatenate([d["read_ids"] for d in all_val_chunks]),
            "base_indices": np.concatenate([d["base_indices"] for d in all_val_chunks]),
        }

        # Save merged chunks
        Path(output.train).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output.train, **merged_train)
        np.savez_compressed(output.val, **merged_val)

        with open(log[0], "w") as f:
            f.write(
                f"Merged {len(input.train)} training files for pair {wildcards.pair}\n"
            )
            f.write(f"Total training chunks: {len(merged_train['labels'])}\n")
            f.write(f"Merged {len(input.val)} validation files\n")
            f.write(f"Total validation chunks: {len(merged_val['labels'])}\n")


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
        mem_mb=16000,
        runtime=480,
        gpu=0,  # GPU controlled by cluster profile
        gpu_mem_mb=8000,
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
