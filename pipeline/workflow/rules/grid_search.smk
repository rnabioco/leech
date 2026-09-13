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
    # slurm_partition/runtime/cpus_per_task/mem_mb/gres for this rule live in
    # the cluster profile (pipeline/cluster/slurm{,-cpu}/config.yaml) -- see
    # train.smk's train_pairwise_aa for why.
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
        # --motif is required=True on `model optimize` (recorded in
        # config.json for provenance) -- not passing it fails every grid
        # search job at CLI parsing.
        motif=require_config("motif"),
        motif_offset=config.get("motif_offset", 0),
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
            --motif {params.motif} \
            --motif-offset {params.motif_offset} \
            --epochs {params.epochs} \
            --context-grid '{params.context_grid}' \
            {params.left_contexts} \
            {params.right_contexts} \
            --dwell-offsets '{params.dwell_offsets}' \
            --parallel {params.parallel} \
            --device {params.device} \
            2>&1 | tee {log}
        """


rule reprepare_chunks_optimized_pairwise:
    """Re-extract one sample's chunks at the geometry grid search selected.

    `leech model optimize` only narrows the signal window it is handed --
    it cannot search wider than `merge_chunks_pairwise`'s input was
    extracted at. Its winning `left_context`/`right_context` (best_params.json)
    are therefore DATA PREPARATION parameters (the `--signal-context` window),
    not `leech model train` kwargs. This rule is the handoff: re-run
    `data prepare` from the original pod5/bam at the selected geometry, so
    `train_pairwise_aa` trains on chunks whose signal_len already matches the
    model it builds, instead of the corpus at the original (possibly wider)
    exploration window.

    `best_params.json` also carries a `dwell_offset`, but neither `data
    prepare` nor `model train` currently exposes a flag for it (it is a
    grid-search-internal parameter -- see rnabioco/leech#266); this rule logs
    the value and moves on rather than inventing a CLI hook out of scope here.
    """
    input:
        pod5=get_project_path(config.get("pod5_dir", "results/pod5"))
        + "/{sample}/{sample}.pod5",
        bam=get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall"))
        + "/{sample}/{sample}.aligned.bam",
        best_params=ancient(
            MODELS_DIR + "/grid_search/pairwise/{pair}/best_params.json"
        ),
    output:
        all=CHUNKS_DIR + "/optimized/pairwise/{pair}/{sample}/all.npz",
    log:
        CHUNKS_DIR + "/optimized/pairwise/{pair}/{sample}/prepare.log",
    # sample/pair are constrained globally (Snakefile).
    threads: config.get("workers", 4)
    params:
        output_dir=CHUNKS_DIR + "/optimized/pairwise/{pair}/{sample}",
        motif=require_config("motif"),
        motif_offset=config.get("motif_offset", 0),
        motif_reference=config.get("motif_reference", "fasta"),
        workers=config.get("workers", 4),
        chunk_size=config.get("chunk_size", 100),
        ref_fasta_arg=build_reference_fasta_arg(),
        skip_indels_arg=build_skip_indels_arg(),
        label_arg=lambda wildcards: build_label_arg(wildcards),
    shell:
        """
        LEFT=$(uv run python -c "import json; print(json.load(open('{input.best_params}'))['left_context'])")
        RIGHT=$(uv run python -c "import json; print(json.load(open('{input.best_params}'))['right_context'])")
        DWELL_OFFSET=$(uv run python -c "import json; print(json.load(open('{input.best_params}')).get('dwell_offset', 0))")
        if [ -z "$LEFT" ] || [ -z "$RIGHT" ]; then
            echo "ERROR: failed to read left_context/right_context from {input.best_params}" >&2
            exit 1
        fi
        echo "Re-preparing {wildcards.sample} for pair {wildcards.pair} at grid-search geometry: left=$LEFT right=$RIGHT (from {input.best_params})" >{log}
        echo "Note: grid-search dwell_offset=$DWELL_OFFSET has no 'data prepare' or 'model train' CLI equivalent yet; only left/right signal context is re-applied here (see rnabioco/leech#266)." >>{log}

        uv run leech data prepare \
            --pod5 {input.pod5} \
            --bam {input.bam} \
            --output-dir {params.output_dir} \
            --motif {params.motif} \
            --motif-offset {params.motif_offset} \
            --motif-reference {params.motif_reference} \
            {params.ref_fasta_arg} \
            {params.skip_indels_arg} \
            {params.label_arg} \
            --workers {params.workers} \
            --chunk-size {params.chunk_size} \
            --signal-context "$LEFT" "$RIGHT" \
            --no-split \
            2>&1 | tee -a {log}
        """


rule merge_chunks_optimized_pairwise:
    """Merge grid-search-optimized pairwise chunks and split at read level.

    Mirrors `merge_chunks_pairwise` exactly, but over the re-prepared chunks
    from `reprepare_chunks_optimized_pairwise` instead of the original ones.
    """
    input:
        chunks=lambda wildcards: expand(
            CHUNKS_DIR + "/optimized/pairwise/{{pair}}/{sample}/all.npz",
            sample=get_samples_for_aa_pair(wildcards.pair),
        ),
    output:
        train=CHUNKS_DIR + "/merged/pairwise/{pair}/optimized/train.npz",
        val=CHUNKS_DIR + "/merged/pairwise/{pair}/optimized/val.npz",
        test=CHUNKS_DIR + "/merged/pairwise/{pair}/optimized/test.npz",
    log:
        CHUNKS_DIR + "/merged/pairwise/{pair}/optimized/merge_and_split.log",
    params:
        output_dir=CHUNKS_DIR + "/merged/pairwise/{pair}/optimized",
        train_split=config.get("train_split", 0.7),
        val_split=config.get("val_split", 0.15),
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
