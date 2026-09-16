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
    log:
        CHUNKS_DIR + "/merged/pairwise/{pair}/merge_and_split.log",
    params:
        output_dir=CHUNKS_DIR + "/merged/pairwise/{pair}",
        train_split=config.get("train_split", 0.7),
        val_split=config.get("val_split", 0.15),
        # Build input arguments with label=file format
        input_args=lambda wildcards, input: build_merge_input_args(
            wildcards.pair, input.chunks
        ),
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
    """Train pairwise amino acid classifier.

    When `use_grid_search` is set, trains on the chunks
    `reprepare_chunks_optimized_pairwise` / `merge_chunks_optimized_pairwise`
    re-extracted at the grid-search-selected signal context, instead of on
    `--model-config` from `best_params.json` -- that file holds data
    preparation parameters (signal context window), not architecture kwargs,
    so it was never a valid `model train` input (see grid_search.smk).
    """
    input:
        train=(
            CHUNKS_DIR + "/merged/pairwise/{pair}/optimized/train.npz"
            if config.get("use_grid_search", False)
            else CHUNKS_DIR + "/merged/pairwise/{pair}/train.npz"
        ),
        val=(
            CHUNKS_DIR + "/merged/pairwise/{pair}/optimized/val.npz"
            if config.get("use_grid_search", False)
            else CHUNKS_DIR + "/merged/pairwise/{pair}/val.npz"
        ),
    output:
        model=MODELS_DIR + "/pairwise/{pair}/model_best.pt",
        checkpoint=MODELS_DIR + "/pairwise/{pair}/model_last.pt",
        history=MODELS_DIR + "/pairwise/{pair}/metrics.json",
    log:
        MODELS_DIR + "/pairwise/{pair}/train.log",
    resources:
        # slurm_partition/runtime/cpus_per_task for this rule live in the
        # cluster profile (pipeline/cluster/slurm{,-cpu}/config.yaml) since
        # profile `set-resources` always wins over a rule's own `resources:`
        # -- see that file for why CPU vs. GPU mode is a profile choice.
        # mem_mb/gres stay here because they scale with `train_gpus`, a
        # per-run count no static profile entry can express.
        mem_mb=lambda wildcards, attempt: (
            config.get("train_mem_mb_per_gpu", 45000) * config.get("train_gpus", 1)
        ),
        gres=lambda wildcards, attempt: (
            ""
            if config.get("use_cpu_training", False)
            else f"gpu:{config.get('train_gpus',1)}"
        ),
    params:
        output_dir=MODELS_DIR + "/pairwise/{pair}",
        model_type=config.get("model", "ConvLSTMDwell"),
        # --motif is required=True on `model train` (recorded in config.json
        # for inference provenance, matching the geometry the data was
        # prepared with) -- not passing it fails every run at CLI parsing.
        motif=require_config("motif"),
        motif_offset=config.get("motif_offset", 0),
        epochs=config.get("epochs", 50),
        batch_size=config.get("batch_size", 128),
        lr=config.get("learning_rate", 0.001),
        early_stopping=config.get("early_stopping_patience", 5),
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
        gpus=config.get("train_gpus", 1),
    shell:
        # --resume points at the undeclared rolling checkpoint
        # (model_resume.pt), not either declared output -- Snakemake deletes
        # a failed attempt's declared outputs, so a --resume wired to
        # model_last.pt or model_best.pt would find nothing on the retry a
        # SLURM walltime kill triggers. Ignored by `leech model train` when
        # absent, so the first attempt is unaffected. See CLAUDE.md
        # ("Interrupt-safe resume") and leech#330.
        """
        uv run leech model train \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {params.model_type} \
            --output-dir {params.output_dir} \
            --motif {params.motif} \
            --motif-offset {params.motif_offset} \
            --epochs {params.epochs} \
            --batch-size {params.batch_size} \
            --learning-rate {params.lr} \
            --early-stopping {params.early_stopping} \
            --device {params.device} \
            --gpus {params.gpus} \
            --resume {params.output_dir}/model_resume.pt \
            2>&1 | tee {log}
        """
