"""
Model comparison rules for evaluating multiple architectures.

These rules train and evaluate multiple model architectures on the same
data splits to enable fair comparison. All comparisons (including charged
vs uncharged) are handled uniformly as pairwise comparisons.
"""

# ============================================================================
# Multi-Architecture Training: Pairwise Comparisons (including charged vs uncharged)
# ============================================================================


rule train_architecture_pairwise:
    """Train a specific architecture for pairwise amino acid classification.

    When `use_grid_search` is set, trains on the chunks
    `reprepare_chunks_optimized_architecture` / `merge_chunks_optimized_architecture`
    re-extracted at that architecture's grid-search-selected signal context,
    instead of passing `best_params.json` to `--model-config` -- see
    `grid_search_architecture_pairwise` below for why that was never valid.
    """
    input:
        train=(
            CHUNKS_DIR + "/merged/pairwise/{pair}/{architecture}/optimized/train.npz"
            if config.get("use_grid_search", False)
            else CHUNKS_DIR + "/merged/pairwise/{pair}/train.npz"
        ),
        val=(
            CHUNKS_DIR + "/merged/pairwise/{pair}/{architecture}/optimized/val.npz"
            if config.get("use_grid_search", False)
            else CHUNKS_DIR + "/merged/pairwise/{pair}/val.npz"
        ),
    output:
        model=MODELS_DIR + "/comparison/pairwise/{pair}/{architecture}/model_best.pt",
        checkpoint=MODELS_DIR
        + "/comparison/pairwise/{pair}/{architecture}/model_last.pt",
        history=MODELS_DIR + "/comparison/pairwise/{pair}/{architecture}/metrics.json",
    log:
        MODELS_DIR + "/comparison/pairwise/{pair}/{architecture}/train.log",
    # slurm_partition/runtime/cpus_per_task/mem_mb/gres for this rule live in
    # the cluster profile (pipeline/cluster/slurm{,-cpu}/config.yaml) -- see
    # train.smk's train_pairwise_aa for why (profile `set-resources` always
    # wins over a rule's own `resources:`). Unlike train_pairwise_aa, this
    # rule is always single-GPU, so nothing here needs a per-run lambda.
    params:
        output_dir=MODELS_DIR + "/comparison/pairwise/{pair}/{architecture}",
        # --motif is required=True on `model train` (recorded in config.json
        # for inference provenance) -- not passing it fails every run at CLI
        # parsing.
        motif=require_config("motif"),
        motif_offset=config.get("motif_offset", 0),
        epochs=config.get("epochs", 50),
        batch_size=config.get("batch_size", 128),
        lr=config.get("learning_rate", 0.001),
        early_stopping=config.get("early_stopping_patience", 5),
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
    shell:
        # --resume points at the undeclared rolling checkpoint
        # (model_resume.pt), not either declared output -- see
        # train_pairwise_aa in train.smk and leech#330 for why.
        """
        uv run leech model train \
            --train-data {input.train} \
            --val-data {input.val} \
            --model {wildcards.architecture} \
            --output-dir {params.output_dir} \
            --motif {params.motif} \
            --motif-offset {params.motif_offset} \
            --epochs {params.epochs} \
            --batch-size {params.batch_size} \
            --learning-rate {params.lr} \
            --early-stopping {params.early_stopping} \
            --device {params.device} \
            --resume {params.output_dir}/model_resume.pt \
            2>&1 | tee {log}
        """


rule test_architecture_pairwise:
    """Test a specific architecture on pairwise test set.

    Evaluates against the same geometry `train_architecture_pairwise` trained
    on: the grid-search-optimized test split when `use_grid_search` is set,
    the base one otherwise. Mismatching these raises a signal_len shape error
    (or worse, silently scores a model against the wrong window).
    """
    input:
        model=MODELS_DIR + "/comparison/pairwise/{pair}/{architecture}/model_best.pt",
        test=(
            CHUNKS_DIR + "/merged/pairwise/{pair}/{architecture}/optimized/test.npz"
            if config.get("use_grid_search", False)
            else CHUNKS_DIR + "/merged/pairwise/{pair}/test.npz"
        ),
    output:
        metrics=METRICS_DIR
        + "/comparison/pairwise/{pair}/{architecture}/test_metrics.json",
    log:
        METRICS_DIR + "/comparison/pairwise/{pair}/{architecture}/test.log",
    # Resources for this rule live in the cluster profile -- see
    # train_architecture_pairwise above.
    params:
        model_dir=MODELS_DIR + "/comparison/pairwise/{pair}/{architecture}",
        device="cpu" if config.get("use_cpu_training", False) else "cuda",
    shell:
        """
        uv run leech eval test \
            --model {params.model_dir} \
            --test-data {input.test} \
            --output {output.metrics} \
            --device {params.device} \
            2>&1 | tee {log}
        """


rule compare_architectures_pairwise:
    """Compare all architectures on a specific pairwise task."""
    input:
        expand(
            METRICS_DIR + "/comparison/pairwise/{{pair}}/{arch}/test_metrics.json",
            arch=MODEL_ARCHITECTURES,
        ),
    output:
        comparison=METRICS_DIR
        + "/comparison/pairwise/{pair}/architecture_comparison.tsv.gz",
        summary=METRICS_DIR + "/comparison/pairwise/{pair}/architecture_summary.txt",
    params:
        metrics_dir=METRICS_DIR + "/comparison/pairwise/{pair}",
        architectures=MODEL_ARCHITECTURES,
    script:
        "../scripts/compare_architectures.py"


rule aggregate_pairwise_comparisons:
    """Aggregate architecture comparisons across all pairwise tasks."""
    input:
        expand(
            METRICS_DIR + "/comparison/pairwise/{pair}/architecture_comparison.tsv.gz",
            pair=AA_PAIRS,
        )
        if AA_PAIRS
        else [],
    output:
        comparison=METRICS_DIR
        + "/comparison/aggregate/pairwise_architecture_comparison.tsv.gz",
        summary=METRICS_DIR + "/comparison/aggregate/pairwise_architecture_summary.txt",
    run:
        import pandas as pd
        from pathlib import Path

        if not AA_PAIRS:
            # No pairwise comparisons, create empty files
            Path(output.comparison).touch()
            Path(output.summary).write_text("No pairwise comparisons configured.\n")
        else:
            # Read and concatenate all pairwise comparison results
            dfs = []
            for pair_file in input:
                pair_name = Path(pair_file).parent.name
                df = pd.read_csv(pair_file, sep="\t", compression="gzip")
                df.insert(0, "pair", pair_name)
                dfs.append(df)
                # Concatenate and save
            combined = pd.concat(dfs, ignore_index=True)
            combined.to_csv(
                output.comparison, sep="\t", index=False, compression="gzip"
            )
            # Create summary
            with open(output.summary, "w") as f:
                f.write(
                    "Architecture Comparison Summary Across All Pairwise Tasks\n"
                )
                f.write("=" * 70 + "\n\n")
                f.write(f"Total pairs evaluated: {len(AA_PAIRS)}\n")
                f.write(
                    f"Architectures compared: {', '.join(MODEL_ARCHITECTURES)}\n\n"
                )
                # Best architecture per pair
                f.write("Best Architecture by Pair (by accuracy):\n")
                f.write("-" * 70 + "\n")
                for pair in AA_PAIRS:
                    pair_df = combined[combined["pair"] == pair]
                    if not pair_df.empty:
                        best = pair_df.loc[pair_df["accuracy_mean"].idxmax()]
                        f.write(
                            f"{pair:20s} {best['architecture']:20s} acc={best['accuracy_mean']:.4f}\n"
                        )
                f.write("\n")
                # Overall best architecture (average across pairs)
                f.write("Overall Best Architecture (average accuracy):\n")
                f.write("-" * 70 + "\n")
                avg_by_arch = (
                    combined.groupby("architecture")["accuracy_mean"]
                    .mean()
                    .sort_values(ascending=False)
                )
                for arch, acc in avg_by_arch.items():
                    f.write(f"{arch:20s} avg_acc={acc:.4f}\n")


rule grid_search_architecture_pairwise:
    """Perform grid search for a specific architecture on pairwise comparison.

    Mirrors `grid_search_pairwise_aa` (grid_search.smk) with an
    `{architecture}` wildcard added: `leech model optimize` searches signal
    context windows and dwell offset -- not learning rate / batch size /
    layer sizes, which is what this rule passed as `--param-grid` to a
    `--max-epochs`/`--param-grid` pair that `optimize` has never had.
    """
    input:
        train=CHUNKS_DIR + "/merged/pairwise/{pair}/train.npz",
        val=CHUNKS_DIR + "/merged/pairwise/{pair}/val.npz",
    output:
        results=MODELS_DIR
        + "/grid_search/pairwise/{pair}/{architecture}/grid_search_results.json",
        best_params=MODELS_DIR
        + "/grid_search/pairwise/{pair}/{architecture}/best_params.json",
    log:
        MODELS_DIR + "/grid_search/pairwise/{pair}/{architecture}/grid_search.log",
    # Resources for this rule live in the cluster profile -- see
    # train_architecture_pairwise above.
    params:
        output_dir=MODELS_DIR + "/grid_search/pairwise/{pair}/{architecture}",
        context_grid=get_grid_search_setting("context_grid", "200:1000:200"),
        left_contexts=optional_flag(
            "--left-contexts", get_grid_search_setting("left_contexts", None)
        ),
        right_contexts=optional_flag(
            "--right-contexts", get_grid_search_setting("right_contexts", None)
        ),
        dwell_offsets=get_grid_search_setting("dwell_offsets", "0"),
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
            --model {wildcards.architecture} \
            --output-dir {params.output_dir} \
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


rule reprepare_chunks_optimized_architecture:
    """Re-extract one sample's chunks at the geometry `{architecture}`'s grid search selected.

    The `{architecture}`-parameterized twin of
    `reprepare_chunks_optimized_pairwise` (grid_search.smk) -- see that rule's
    docstring for why this handoff re-runs `data prepare` instead of feeding
    `best_params.json` to `model train`. Kept separate per architecture
    because two architectures' grid searches can pick different geometry.
    """
    input:
        pod5=get_project_path(config.get("pod5_dir", "results/pod5"))
        + "/{sample}/{sample}.pod5",
        bam=get_project_path(config.get("rebasecall_dir", "results/bam/rebasecall"))
        + "/{sample}/{sample}.aligned.bam",
        best_params=ancient(
            MODELS_DIR + "/grid_search/pairwise/{pair}/{architecture}/best_params.json"
        ),
    output:
        all=CHUNKS_DIR + "/optimized/pairwise/{pair}/{architecture}/{sample}/all.npz",
    log:
        CHUNKS_DIR + "/optimized/pairwise/{pair}/{architecture}/{sample}/prepare.log",
    # sample/pair/architecture are constrained globally (Snakefile).
    threads: config.get("workers", 4)
    params:
        output_dir=CHUNKS_DIR + "/optimized/pairwise/{pair}/{architecture}/{sample}",
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
        echo "Re-preparing {wildcards.sample} for pair {wildcards.pair} / {wildcards.architecture} at grid-search geometry: left=$LEFT right=$RIGHT (from {input.best_params})" >{log}
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


rule merge_chunks_optimized_architecture:
    """Merge grid-search-optimized per-architecture chunks and split at read level."""
    input:
        chunks=lambda wildcards: expand(
            CHUNKS_DIR
            + "/optimized/pairwise/{{pair}}/{{architecture}}/{sample}/all.npz",
            sample=get_samples_for_aa_pair(wildcards.pair),
        ),
    output:
        train=CHUNKS_DIR + "/merged/pairwise/{pair}/{architecture}/optimized/train.npz",
        val=CHUNKS_DIR + "/merged/pairwise/{pair}/{architecture}/optimized/val.npz",
        test=CHUNKS_DIR + "/merged/pairwise/{pair}/{architecture}/optimized/test.npz",
    log:
        CHUNKS_DIR
        + "/merged/pairwise/{pair}/{architecture}/optimized/merge_and_split.log",
    params:
        output_dir=CHUNKS_DIR + "/merged/pairwise/{pair}/{architecture}/optimized",
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
